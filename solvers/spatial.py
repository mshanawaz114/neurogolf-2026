from __future__ import annotations

"""
spatial.py — Solver for spatial transformation tasks (flip, rotate, transpose).

These are tasks where pixel colours are unchanged but positions are permuted.
The 30×30 canvas complicates this because grids are in the top-left corner.

Strategy per transform:
  identity    → 1×1 identity conv (100 params)
  flip_h/v    → Slice + Pad (zero params — just constants)
  rotate_90   → transpose_hw + flip_h   (zero params)
  rotate_180  → flip_h + flip_v         (zero params)
  rotate_270  → transpose_hw + flip_v   (zero params)
  transpose   → transpose_hw            (zero params)

For transforms that change H↔W (rotate 90/270, transpose), the grid moves
to a different region of the canvas. We handle this by extracting the grid,
applying the transform, and re-padding to [1,10,30,30].
"""

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.onnx_builder import (
    identity_net, flip_h_net, flip_v_net, build_graph,
    conv1x1, save, _make_int64, _t, CANVAS, CHANNELS as C,
)

INT_MAX =  2**31 - 1
INT_MIN = -(2**31)


def _slice_dim(inputs: list[str], output: str, axis: int, start: int, end: int, prefix: str):
    inits = [
        _make_int64(f"{prefix}_s", [start]),
        _make_int64(f"{prefix}_e", [end]),
        _make_int64(f"{prefix}_a", [axis]),
        _make_int64(f"{prefix}_st", [1]),
    ]
    node = helper.make_node(
        "Slice", inputs=inputs + [f"{prefix}_s", f"{prefix}_e", f"{prefix}_a", f"{prefix}_st"], outputs=[output]
    )
    return [node], inits


def _flip_h_branch(prefix: str, grid_w: int):
    nodes = []
    inits = []
    start = CANVAS - grid_w

    n, i = _slice_dim(["input"], f"{prefix}_flipped", 3, INT_MAX, INT_MIN, f"{prefix}_flip")
    nodes += n
    inits += i
    inits[-1] = _make_int64(f"{prefix}_flip_st", [-1])
    nodes[-1] = helper.make_node(
        "Slice",
        inputs=["input", f"{prefix}_flip_s", f"{prefix}_flip_e", f"{prefix}_flip_a", f"{prefix}_flip_st"],
        outputs=[f"{prefix}_flipped"],
    )

    n, i = _slice_dim([f"{prefix}_flipped"], f"{prefix}_sliced", 3, start, CANVAS, f"{prefix}_crop")
    nodes += n
    inits += i

    inits.append(_make_int64(f"{prefix}_pad", [0, 0, 0, 0, 0, 0, 0, CANVAS - grid_w]))
    nodes.append(
        helper.make_node("Pad", inputs=[f"{prefix}_sliced", f"{prefix}_pad"], outputs=[f"{prefix}_out"], mode="constant")
    )
    return nodes, inits, f"{prefix}_out"


def _flip_v_branch(prefix: str, grid_h: int):
    nodes = []
    inits = []
    start = CANVAS - grid_h

    n, i = _slice_dim(["input"], f"{prefix}_flipped", 2, INT_MAX, INT_MIN, f"{prefix}_flip")
    nodes += n
    inits += i
    inits[-1] = _make_int64(f"{prefix}_flip_st", [-1])
    nodes[-1] = helper.make_node(
        "Slice",
        inputs=["input", f"{prefix}_flip_s", f"{prefix}_flip_e", f"{prefix}_flip_a", f"{prefix}_flip_st"],
        outputs=[f"{prefix}_flipped"],
    )

    n, i = _slice_dim([f"{prefix}_flipped"], f"{prefix}_sliced", 2, start, CANVAS, f"{prefix}_crop")
    nodes += n
    inits += i

    inits.append(_make_int64(f"{prefix}_pad", [0, 0, 0, 0, 0, 0, CANVAS - grid_h, 0]))
    nodes.append(
        helper.make_node("Pad", inputs=[f"{prefix}_sliced", f"{prefix}_pad"], outputs=[f"{prefix}_out"], mode="constant")
    )
    return nodes, inits, f"{prefix}_out"


def _transpose_branch(prefix: str, H: int, W: int):
    nodes = []
    inits = []
    n, i = _slice_dim(["input"], f"{prefix}_rows", 2, 0, H, f"{prefix}_row_slice")
    nodes += n
    inits += i
    n, i = _slice_dim([f"{prefix}_rows"], f"{prefix}_crop", 3, 0, W, f"{prefix}_col_slice")
    nodes += n
    inits += i
    nodes.append(helper.make_node("Transpose", inputs=[f"{prefix}_crop"], outputs=[f"{prefix}_tr"], perm=[0, 1, 3, 2]))
    inits.append(_make_int64(f"{prefix}_pad", [0, 0, 0, 0, 0, 0, CANVAS - W, CANVAS - H]))
    nodes.append(
        helper.make_node("Pad", inputs=[f"{prefix}_tr", f"{prefix}_pad"], outputs=[f"{prefix}_out"], mode="constant")
    )
    return nodes, inits, f"{prefix}_out"


def _shape_selector(prefix: str, H: int, W: int):
    nodes = []
    inits = []

    nodes.append(helper.make_node("ReduceMax", inputs=["input"], outputs=[f"{prefix}_cmax"], axes=[1], keepdims=0))
    nodes.append(
        helper.make_node("ReduceMax", inputs=[f"{prefix}_cmax"], outputs=[f"{prefix}_col_presence"], axes=[1], keepdims=0)
    )
    nodes.append(
        helper.make_node("ReduceMax", inputs=[f"{prefix}_cmax"], outputs=[f"{prefix}_row_presence"], axes=[2], keepdims=0)
    )

    def pick(vec_name: str, axis: int, idx: int, name: str):
        inits_local = [
            _make_int64(f"{name}_s", [idx]),
            _make_int64(f"{name}_e", [idx + 1]),
            _make_int64(f"{name}_a", [axis]),
            _make_int64(f"{name}_st", [1]),
        ]
        node = helper.make_node(
            "Slice", inputs=[vec_name, f"{name}_s", f"{name}_e", f"{name}_a", f"{name}_st"], outputs=[f"{name}_out"]
        )
        return node, inits_local, f"{name}_out"

    node, init, h_prev = pick(f"{prefix}_row_presence", 1, H - 1, f"{prefix}_hp")
    nodes.append(node)
    inits += init
    node, init, w_prev = pick(f"{prefix}_col_presence", 1, W - 1, f"{prefix}_wp")
    nodes.append(node)
    inits += init

    h_sel = h_prev
    if H < CANVAS:
        node, init, h_next = pick(f"{prefix}_row_presence", 1, H, f"{prefix}_hn")
        nodes.append(node)
        inits += init
        inits.append(_t(f"{prefix}_one", np.array([1.0], dtype=np.float32)))
        nodes.append(helper.make_node("Sub", inputs=[f"{prefix}_one", h_next], outputs=[f"{prefix}_hgap"]))
        nodes.append(helper.make_node("Mul", inputs=[h_prev, f"{prefix}_hgap"], outputs=[f"{prefix}_hsel"]))
        h_sel = f"{prefix}_hsel"

    w_sel = w_prev
    if W < CANVAS:
        node, init, w_next = pick(f"{prefix}_col_presence", 1, W, f"{prefix}_wn")
        nodes.append(node)
        inits += init
        if not any(t.name == f"{prefix}_one" for t in inits):
            inits.append(_t(f"{prefix}_one", np.array([1.0], dtype=np.float32)))
        nodes.append(helper.make_node("Sub", inputs=[f"{prefix}_one", w_next], outputs=[f"{prefix}_wgap"]))
        nodes.append(helper.make_node("Mul", inputs=[w_prev, f"{prefix}_wgap"], outputs=[f"{prefix}_wsel"]))
        w_sel = f"{prefix}_wsel"

    nodes.append(helper.make_node("Mul", inputs=[h_sel, w_sel], outputs=[f"{prefix}_sel"]))
    return nodes, inits, f"{prefix}_sel"


def _multi_shape_spatial_net(transform: str, shapes: list[tuple[int, int]]) -> onnx.ModelProto:
    branch_builder = {
        "flip_h": lambda prefix, H, W: _flip_h_branch(prefix, W),
        "flip_v": lambda prefix, H, W: _flip_v_branch(prefix, H),
        "transpose": _transpose_branch,
    }[transform]

    nodes = []
    inits = []
    weighted = []
    unique_shapes = sorted(set(shapes))

    for idx, (H, W) in enumerate(unique_shapes):
        prefix = f"m{idx}"
        bn, bi, bout = branch_builder(prefix, H, W)
        sn, si, sel = _shape_selector(prefix, H, W)
        nodes += bn + sn
        inits += bi + si
        nodes.append(helper.make_node("Mul", inputs=[bout, sel], outputs=[f"{prefix}_weighted"]))
        weighted.append(f"{prefix}_weighted")

    current = weighted[0]
    for idx, name in enumerate(weighted[1:], start=1):
        out = "output" if idx == len(weighted) - 1 else f"m_add_{idx}"
        nodes.append(helper.make_node("Add", inputs=[current, name], outputs=[out]))
        current = out

    graph = helper.make_graph(
        nodes,
        f"{transform}_multi",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _rotate_90_net(H: int, W: int) -> onnx.ModelProto:
    """
    Rotate 90° CCW: [H,W] → [W,H].  Maps (h,w) → (W-1-w, h).
    = transpose then flip_v.
    """
    # Extract top-left [H,W] from canvas, transpose to [W,H], pad back.
    # Step 1: Slice rows 0..H, cols 0..W
    # Step 2: Transpose HW
    # Step 3: Flip H (now of size W)
    # Step 4: Pad to 30×30

    nodes = []
    inits = []

    # Slice [1,C,H,W]
    inits += [_make_int64("r90_rs", [0]), _make_int64("r90_re", [H]),
              _make_int64("r90_rax",[2]), _make_int64("r90_rst",[1])]
    nodes.append(helper.make_node("Slice",
        inputs=["input","r90_rs","r90_re","r90_rax","r90_rst"], outputs=["r90_rows"]))

    inits += [_make_int64("r90_cs", [0]), _make_int64("r90_ce", [W]),
              _make_int64("r90_cax",[3]), _make_int64("r90_cst",[1])]
    nodes.append(helper.make_node("Slice",
        inputs=["r90_rows","r90_cs","r90_ce","r90_cax","r90_cst"], outputs=["r90_crop"]))

    # Transpose [1,C,H,W] → [1,C,W,H]
    nodes.append(helper.make_node("Transpose",
        inputs=["r90_crop"], outputs=["r90_tr"], perm=[0,1,3,2]))

    # Flip_v on the [1,C,W,H] tensor (flip the H dim, which is now W)
    inits += [_make_int64("r90_fvs",[INT_MAX]), _make_int64("r90_fve",[INT_MIN]),
              _make_int64("r90_fva",[2]),       _make_int64("r90_fvst",[-1])]
    nodes.append(helper.make_node("Slice",
        inputs=["r90_tr","r90_fvs","r90_fve","r90_fva","r90_fvst"], outputs=["r90_flipped"]))

    # Pad to [1,C,30,30]
    pad_h = CANVAS - W   # new H is W
    pad_w = CANVAS - H   # new W is H
    inits.append(_make_int64("r90_pv", [0,0,0,0, 0,0,pad_h,pad_w]))
    nodes.append(helper.make_node("Pad",
        inputs=["r90_flipped","r90_pv"], outputs=["output"], mode="constant"))

    graph = helper.make_graph(nodes, "rotate_90",
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _rotate_270_net(H: int, W: int) -> onnx.ModelProto:
    """Rotate 270° CCW (= 90° CW): [H,W] → [W,H]. = flip_h then transpose."""
    nodes, inits = [], []

    # Crop
    inits += [_make_int64("r27_rs",[0]),_make_int64("r27_re",[H]),
              _make_int64("r27_ra",[2]),_make_int64("r27_rst",[1])]
    nodes.append(helper.make_node("Slice",
        inputs=["input","r27_rs","r27_re","r27_ra","r27_rst"], outputs=["r27_rows"]))
    inits += [_make_int64("r27_cs",[0]),_make_int64("r27_ce",[W]),
              _make_int64("r27_ca",[3]),_make_int64("r27_cst",[1])]
    nodes.append(helper.make_node("Slice",
        inputs=["r27_rows","r27_cs","r27_ce","r27_ca","r27_cst"], outputs=["r27_crop"]))

    # Flip H (flip rows before transpose)
    inits += [_make_int64("r27_fvs",[INT_MAX]),_make_int64("r27_fve",[INT_MIN]),
              _make_int64("r27_fva",[2]),       _make_int64("r27_fvst",[-1])]
    nodes.append(helper.make_node("Slice",
        inputs=["r27_crop","r27_fvs","r27_fve","r27_fva","r27_fvst"], outputs=["r27_flipped"]))

    # Transpose
    nodes.append(helper.make_node("Transpose",
        inputs=["r27_flipped"], outputs=["r27_tr"], perm=[0,1,3,2]))

    # Pad
    pad_h = CANVAS - W
    pad_w = CANVAS - H
    inits.append(_make_int64("r27_pv",[0,0,0,0, 0,0,pad_h,pad_w]))
    nodes.append(helper.make_node("Pad",
        inputs=["r27_tr","r27_pv"], outputs=["output"], mode="constant"))

    graph = helper.make_graph(nodes, "rotate_270",
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _transpose_net(H: int, W: int) -> onnx.ModelProto:
    """Diagonal transpose: [H,W] → [W,H]."""
    nodes, inits = [], []
    inits += [_make_int64("tp_rs",[0]),_make_int64("tp_re",[H]),
              _make_int64("tp_ra",[2]),_make_int64("tp_rst",[1])]
    nodes.append(helper.make_node("Slice",
        inputs=["input","tp_rs","tp_re","tp_ra","tp_rst"], outputs=["tp_rows"]))
    inits += [_make_int64("tp_cs",[0]),_make_int64("tp_ce",[W]),
              _make_int64("tp_ca",[3]),_make_int64("tp_cst",[1])]
    nodes.append(helper.make_node("Slice",
        inputs=["tp_rows","tp_cs","tp_ce","tp_ca","tp_cst"], outputs=["tp_crop"]))
    nodes.append(helper.make_node("Transpose",
        inputs=["tp_crop"], outputs=["tp_tr"], perm=[0,1,3,2]))
    pad_h = CANVAS - W
    pad_w = CANVAS - H
    inits.append(_make_int64("tp_pv",[0,0,0,0, 0,0,pad_h,pad_w]))
    nodes.append(helper.make_node("Pad",
        inputs=["tp_tr","tp_pv"], outputs=["output"], mode="constant"))

    graph = helper.make_graph(nodes, "transpose",
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _rotate_180_net(H: int, W: int) -> onnx.ModelProto:
    """Rotate 180°: flip_h + flip_v, then re-pad correctly."""
    nodes, inits = [], []
    # Crop
    inits += [_make_int64("r18_rs",[0]),_make_int64("r18_re",[H]),
              _make_int64("r18_ra",[2]),_make_int64("r18_rst",[1])]
    nodes.append(helper.make_node("Slice",
        inputs=["input","r18_rs","r18_re","r18_ra","r18_rst"], outputs=["r18_rows"]))
    inits += [_make_int64("r18_cs",[0]),_make_int64("r18_ce",[W]),
              _make_int64("r18_ca",[3]),_make_int64("r18_cst",[1])]
    nodes.append(helper.make_node("Slice",
        inputs=["r18_rows","r18_cs","r18_ce","r18_ca","r18_cst"], outputs=["r18_crop"]))
    # Flip H
    inits += [_make_int64("r18_fhs",[INT_MAX]),_make_int64("r18_fhe",[INT_MIN]),
              _make_int64("r18_fha",[2]),       _make_int64("r18_fhst",[-1])]
    nodes.append(helper.make_node("Slice",
        inputs=["r18_crop","r18_fhs","r18_fhe","r18_fha","r18_fhst"], outputs=["r18_fh"]))
    # Flip W
    inits += [_make_int64("r18_fws",[INT_MAX]),_make_int64("r18_fwe",[INT_MIN]),
              _make_int64("r18_fwa",[3]),       _make_int64("r18_fwst",[-1])]
    nodes.append(helper.make_node("Slice",
        inputs=["r18_fh","r18_fws","r18_fwe","r18_fwa","r18_fwst"], outputs=["r18_flipped"]))
    # Pad back
    pad_h = CANVAS - H
    pad_w = CANVAS - W
    inits.append(_make_int64("r18_pv",[0,0,0,0, 0,0,pad_h,pad_w]))
    nodes.append(helper.make_node("Pad",
        inputs=["r18_flipped","r18_pv"], outputs=["output"], mode="constant"))

    graph = helper.make_graph(nodes, "rotate_180",
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class SpatialSolver(BaseSolver):
    PRIORITY = 5   # Highest: spatial-only solutions have near-zero cost

    _BUILDERS = {
        "identity":       lambda H, W: identity_net(),
        "flip_h":         lambda H, W: flip_h_net(W),
        "flip_v":         lambda H, W: flip_v_net(H),
        "rotate_90":      _rotate_90_net,
        "rotate_180":     _rotate_180_net,
        "rotate_270":     _rotate_270_net,
        "transpose":      _transpose_net,
        "anti_transpose": lambda H, W: None,   # TODO
    }

    def can_solve(self, analysis: dict) -> bool:
        t = analysis.get("spatial_transform")
        return t is not None and t in self._BUILDERS and self._BUILDERS[t] is not None

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        t = analysis.get("spatial_transform")
        builder = self._BUILDERS.get(t)
        if builder is None:
            return None

        # identity needs no size — build immediately
        if t == "identity":
            try:
                model = builder(1, 1)
                path = out_dir / f"{task_id}.onnx"
                save(model, str(path))
                return path
            except Exception as e:
                print(f"    SpatialSolver(identity) failed: {e}")
                return None

        # For size-dependent transforms, try each unique train input shape
        in_shapes = sorted(
            {
                (len(pair["input"]), len(pair["input"][0]))
                for split in ["train", "test", "arc-gen"]
                for pair in task.get(split, [])
            }
        )
        if not in_shapes:
            return None

        path = out_dir / f"{task_id}.onnx"
        if t in {"flip_h", "flip_v", "transpose"} and len(in_shapes) > 1:
            try:
                model = _multi_shape_spatial_net(t, in_shapes)
                save(model, str(path))
                return path
            except Exception as e:
                print(f"    SpatialSolver({t}, multi-shape) failed: {e}")

        for shape in in_shapes:
            H, W = shape
            try:
                model = builder(H, W)
                save(model, str(path))
                return path   # caller validates; if wrong size fails, it removes the file
            except Exception as e:
                print(f"    SpatialSolver({t}) H={H} W={W} failed: {e}")
                continue

        return None
