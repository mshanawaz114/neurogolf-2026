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


def _out_src_matrix(out_len: int, src_fn) -> np.ndarray:
    mat = np.zeros((CANVAS, CANVAS), dtype=np.float32)
    for out_idx in range(out_len):
        src_idx = int(src_fn(out_idx))
        if 0 <= src_idx < CANVAS:
            mat[out_idx, src_idx] = 1.0
    return mat


def _identity_matrix(length: int) -> np.ndarray:
    return _out_src_matrix(length, lambda out_idx: out_idx)


def _reverse_matrix(length: int) -> np.ndarray:
    return _out_src_matrix(length, lambda out_idx: length - 1 - out_idx)


def _matmul_shift_nodes(prefix: str, row_mat: np.ndarray, col_mat: np.ndarray):
    nodes = [
        helper.make_node("Reshape", inputs=["input", f"{prefix}_shape_cnn"], outputs=[f"{prefix}_in2d"]),
        helper.make_node("MatMul", inputs=[f"{prefix}_rows", f"{prefix}_in2d"], outputs=[f"{prefix}_rows_out"]),
        helper.make_node("MatMul", inputs=[f"{prefix}_rows_out", f"{prefix}_cols"], outputs=[f"{prefix}_shifted"]),
        helper.make_node("Reshape", inputs=[f"{prefix}_shifted", f"{prefix}_shape_1cnn"], outputs=[f"{prefix}_out"]),
    ]
    inits = [
        _t(f"{prefix}_rows", row_mat),
        _t(f"{prefix}_cols", col_mat),
        _make_int64(f"{prefix}_shape_cnn", [C, CANVAS, CANVAS]),
        _make_int64(f"{prefix}_shape_1cnn", [1, C, CANVAS, CANVAS]),
    ]
    return nodes, inits, f"{prefix}_out"


def _single_entry_selector(prefix: str, vec_name: str, idx: int):
    vec = np.zeros((CANVAS, 1), dtype=np.float32)
    vec[idx, 0] = 1.0
    return [
        helper.make_node("MatMul", inputs=[vec_name, f"{prefix}_pick"], outputs=[f"{prefix}_out"])
    ], [
        _t(f"{prefix}_pick", vec)
    ], f"{prefix}_out"


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
    return _matmul_shift_nodes(prefix, _identity_matrix(CANVAS), _reverse_matrix(grid_w).T)


def _flip_v_branch(prefix: str, grid_h: int):
    return _matmul_shift_nodes(prefix, _reverse_matrix(grid_h), _identity_matrix(CANVAS).T)


def _transpose_branch(prefix: str, H: int, W: int):
    nodes, inits, shifted = _matmul_shift_nodes(prefix, _identity_matrix(H), _identity_matrix(W).T)
    transposed = f"{prefix}_tr_out"
    nodes.append(helper.make_node("Transpose", inputs=[shifted], outputs=[transposed], perm=[0, 1, 3, 2]))
    return nodes, inits, transposed


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

    n, i, h_prev = _single_entry_selector(f"{prefix}_hp", f"{prefix}_row_presence", H - 1)
    nodes += n
    inits += i
    n, i, w_prev = _single_entry_selector(f"{prefix}_wp", f"{prefix}_col_presence", W - 1)
    nodes += n
    inits += i

    h_sel = h_prev
    if H < CANVAS:
        n, i, h_next = _single_entry_selector(f"{prefix}_hn", f"{prefix}_row_presence", H)
        nodes += n
        inits += i
        inits.append(_t(f"{prefix}_one", np.array([1.0], dtype=np.float32)))
        nodes.append(helper.make_node("Sub", inputs=[f"{prefix}_one", h_next], outputs=[f"{prefix}_hgap"]))
        nodes.append(helper.make_node("Mul", inputs=[h_prev, f"{prefix}_hgap"], outputs=[f"{prefix}_hsel"]))
        h_sel = f"{prefix}_hsel"

    w_sel = w_prev
    if W < CANVAS:
        n, i, w_next = _single_entry_selector(f"{prefix}_wn", f"{prefix}_col_presence", W)
        nodes += n
        inits += i
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
    nodes, inits, shifted = _matmul_shift_nodes("r90", _identity_matrix(H), _identity_matrix(W).T)
    nodes.append(helper.make_node("Transpose", inputs=[shifted], outputs=["r90_tr"], perm=[0, 1, 3, 2]))
    n, i, flipped = _matmul_shift_nodes("r90_post", _reverse_matrix(W), _identity_matrix(CANVAS).T)
    for node in n:
        for idx, inp in enumerate(node.input):
            if inp == "input":
                node.input[idx] = "r90_tr"
    nodes += n
    inits += i
    nodes[-1].output[0] = "output"

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
    nodes, inits, shifted = _matmul_shift_nodes("r27", _reverse_matrix(H), _identity_matrix(W).T)
    nodes.append(helper.make_node("Transpose", inputs=[shifted], outputs=["output"], perm=[0, 1, 3, 2]))

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
    nodes, inits, shifted = _matmul_shift_nodes("tp", _identity_matrix(H), _identity_matrix(W).T)
    nodes.append(helper.make_node("Transpose", inputs=[shifted], outputs=["output"], perm=[0, 1, 3, 2]))

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
    nodes, inits, shifted = _matmul_shift_nodes("r18", _reverse_matrix(H), _reverse_matrix(W).T)
    nodes[-1].output[0] = "output"

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
                save(model, str(path), try_simplify=False)
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
                save(model, str(path), try_simplify=False)
                return path
            except Exception as e:
                print(f"    SpatialSolver({t}, multi-shape) failed: {e}")

        for shape in in_shapes:
            H, W = shape
            try:
                model = builder(H, W)
                save(model, str(path), try_simplify=False)
                return path   # caller validates; if wrong size fails, it removes the file
            except Exception as e:
                print(f"    SpatialSolver({t}) H={H} W={W} failed: {e}")
                continue

        return None
