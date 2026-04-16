from __future__ import annotations

"""
onnx_builder.py — Build minimal ONNX graphs analytically for NeuroGolf 2026.

COMPETITION FORMAT: Input is ALWAYS [1, 10, 30, 30] (fixed static shape).
Output must also be [1, 10, 30, 30].

Competitive advantage: hand-crafted ONNX is 10-100× smaller than PyTorch exports.

Score formula:  max(1, 25 - ln(params + mem_bytes + MACs))

Key operations and their costs on 30×30 canvas:
  conv1x1 (10→10):    100 params, 400 B, 90,000 MACs   → score ~13.6
  flip (Slice+math):  ~10 params, ~80 B, 0 MACs         → score ~21.4
  identity (1×1 eye): 100 params, 400 B, 90,000 MACs   → score ~13.6
  tiling conv:        small kernel, proportional MACs
"""

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper


CANVAS   = 30
CHANNELS = 10
C        = 10   # alias used throughout

# ── Tensor helpers ────────────────────────────────────────────────────────────

def _t(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(arr, name=name)


def _make_int64(name: str, vals) -> onnx.TensorProto:
    return _t(name, np.array(vals, dtype=np.int64))


def _make_float(name: str, vals) -> onnx.TensorProto:
    return _t(name, np.array(vals, dtype=np.float32))


# ── Operation builders ────────────────────────────────────────────────────────

def conv1x1(W: np.ndarray, name: str = "c", bias: np.ndarray | None = None) -> dict:
    """
    1×1 conv: colour-space transform.
    W: [C_out, C_in] or [C_out, C_in, 1, 1]
    Cost on 30×30: C_out*C_in params, C_out*C_in*4B, C_out*C_in*900 MACs
    """
    W = W.reshape(W.shape[0], -1, 1, 1).astype(np.float32)
    inputs = ["x", f"{name}_W"]
    inits  = [_t(f"{name}_W", W)]
    if bias is not None:
        inputs.append(f"{name}_b")
        inits.append(_t(f"{name}_b", bias.reshape(-1).astype(np.float32)))
    node = helper.make_node("Conv", inputs=inputs, outputs=[f"{name}_out"],
                            kernel_shape=[1,1], pads=[0,0,0,0], name=name)
    return {"nodes": [node], "inits": inits, "out": f"{name}_out"}


def conv2d(W: np.ndarray, name: str = "c2d", bias: np.ndarray | None = None,
           pads: list | None = None) -> dict:
    """
    General 2-D conv. W: [C_out, C_in, kH, kW].
    Default pads: same-size padding.
    """
    W = W.astype(np.float32)
    kH, kW = W.shape[2], W.shape[3]
    if pads is None:
        ph, pw = kH // 2, kW // 2
        pads = [ph, pw, kH-1-ph, kW-1-pw]
    inputs = ["x", f"{name}_W"]
    inits  = [_t(f"{name}_W", W)]
    if bias is not None:
        inputs.append(f"{name}_b")
        inits.append(_t(f"{name}_b", bias.reshape(-1).astype(np.float32)))
    node = helper.make_node("Conv", inputs=inputs, outputs=[f"{name}_out"],
                            kernel_shape=[kH, kW], pads=pads, name=name)
    return {"nodes": [node], "inits": inits, "out": f"{name}_out"}


def flip_h(name: str = "fh") -> dict:
    """
    Horizontal flip of the FULL 30×30 canvas. Zero MACs.
    ⚠ Only correct when grid fills the full width or when combined with a
    colour-correction pass to zero out flipped padding.
    """
    inits = [
        _make_int64(f"{name}_s",  [2**31-1]),
        _make_int64(f"{name}_e",  [-(2**31)]),
        _make_int64(f"{name}_ax", [3]),
        _make_int64(f"{name}_st", [-1]),
    ]
    node = helper.make_node("Slice",
        inputs=["x", f"{name}_s", f"{name}_e", f"{name}_ax", f"{name}_st"],
        outputs=[f"{name}_out"], name=name)
    return {"nodes": [node], "inits": inits, "out": f"{name}_out"}


def flip_v(name: str = "fv") -> dict:
    """Vertical flip of the full 30×30 canvas. Zero MACs."""
    inits = [
        _make_int64(f"{name}_s",  [2**31-1]),
        _make_int64(f"{name}_e",  [-(2**31)]),
        _make_int64(f"{name}_ax", [2]),
        _make_int64(f"{name}_st", [-1]),
    ]
    node = helper.make_node("Slice",
        inputs=["x", f"{name}_s", f"{name}_e", f"{name}_ax", f"{name}_st"],
        outputs=[f"{name}_out"], name=name)
    return {"nodes": [node], "inits": inits, "out": f"{name}_out"}


def transpose_hw(name: str = "tr") -> dict:
    """Transpose H and W dims: [1,C,H,W]→[1,C,W,H]. Zero MACs."""
    node = helper.make_node("Transpose", inputs=["x"], outputs=[f"{name}_out"],
                            perm=[0,1,3,2], name=name)
    return {"nodes": [node], "inits": [], "out": f"{name}_out"}


def mul_scalar(val: float, name: str = "ms") -> dict:
    c = _make_float(f"{name}_c", [val])
    node = helper.make_node("Mul", inputs=["x", f"{name}_c"], outputs=[f"{name}_out"], name=name)
    return {"nodes": [node], "inits": [c], "out": f"{name}_out"}


def add_const(arr: np.ndarray, name: str = "ac") -> dict:
    c = _t(f"{name}_c", arr.astype(np.float32))
    node = helper.make_node("Add", inputs=["x", f"{name}_c"], outputs=[f"{name}_out"], name=name)
    return {"nodes": [node], "inits": [c], "out": f"{name}_out"}


def relu(name: str = "relu") -> dict:
    node = helper.make_node("Relu", inputs=["x"], outputs=[f"{name}_out"], name=name)
    return {"nodes": [node], "inits": [], "out": f"{name}_out"}


def clip_op(lo: float, hi: float, name: str = "clip") -> dict:
    inits = [_make_float(f"{name}_lo", [lo]), _make_float(f"{name}_hi", [hi])]
    node = helper.make_node("Clip", inputs=["x", f"{name}_lo", f"{name}_hi"],
                            outputs=[f"{name}_out"], name=name)
    return {"nodes": [node], "inits": inits, "out": f"{name}_out"}


# ── Composite helpers ─────────────────────────────────────────────────────────

def identity_net() -> onnx.ModelProto:
    """Cheapest correct identity: 10×10 identity 1×1 conv."""
    return build_graph([conv1x1(np.eye(10, dtype=np.float32), name="id")])


def color_perm_net(mapping: dict[int,int]) -> onnx.ModelProto:
    """
    Build minimal ONNX for a colour-permutation task.
    mapping: {src_color: dst_color}  (must cover all 10 channels for correctness)

    Implementation: 1×1 conv with a [10,10,1,1] weight matrix where
    W[dst, src] = 1 for the mapping, 0 elsewhere.
    Correctly passes through zero-hot pixels (outside grid) since all-zero
    input → all-zero output through a linear conv.
    """
    W = np.zeros((10, 10), dtype=np.float32)
    for src in range(10):
        dst = mapping.get(src, src)   # default: identity for unmapped
        W[dst, src] = 1.0
    return build_graph([conv1x1(W, name="cp")])


def flip_h_net(grid_W: int) -> onnx.ModelProto:
    """
    Horizontal flip for a grid of known width grid_W (placed at top-left of 30×30).
    Strategy: build a 1×1 weight matrix that zero-pads the right side,
    then use a full-canvas flip + mask to correct for the padding issue.

    Since grid is at top-left, flip_h on full canvas puts the zero-padding
    at the LEFT, which is wrong.  We correct this by:
      1. Flip the full canvas
      2. Slice the flipped canvas to take only columns [30-grid_W : 30]
         and paste them back to columns [0 : grid_W]

    This is done analytically with Slice + Pad ops.
    """
    # Step 1: flip entire 30×30 canvas horizontally
    # Step 2: slice off the right grid_W columns (which now hold the flipped grid)
    # Step 3: pad them back to [1,10,30,30] on the right with zeros

    # Slice: take columns [30-grid_W : 30]
    start = CANVAS - grid_W

    inits_flip = [
        _make_int64("fh_s",  [2**31-1]),
        _make_int64("fh_e",  [-(2**31)]),
        _make_int64("fh_ax", [3]),
        _make_int64("fh_st", [-1]),
    ]
    flip_node = helper.make_node("Slice",
        inputs=["input", "fh_s", "fh_e", "fh_ax", "fh_st"],
        outputs=["flipped"])

    inits_slc = [
        _make_int64("slc_s",  [start]),
        _make_int64("slc_e",  [CANVAS]),
        _make_int64("slc_ax", [3]),
        _make_int64("slc_st", [1]),
    ]
    slice_node = helper.make_node("Slice",
        inputs=["flipped", "slc_s", "slc_e", "slc_ax", "slc_st"],
        outputs=["sliced"])

    # Pad: add (CANVAS - grid_W) zeros on the right of dim 3
    pad_amt = CANVAS - grid_W
    pads_val = _make_int64("pad_v", [0,0,0,0, 0,0,0,pad_amt])
    pad_node = helper.make_node("Pad",
        inputs=["sliced", "pad_v"],
        outputs=["output"],
        mode="constant")

    graph = helper.make_graph(
        [flip_node, slice_node, pad_node],
        "flip_h",
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        inits_flip + inits_slc + [pads_val],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def flip_v_net(grid_H: int) -> onnx.ModelProto:
    """Vertical flip for a grid of known height grid_H."""
    start = CANVAS - grid_H
    inits_flip = [
        _make_int64("fv_s",  [2**31-1]),
        _make_int64("fv_e",  [-(2**31)]),
        _make_int64("fv_ax", [2]),
        _make_int64("fv_st", [-1]),
    ]
    flip_node = helper.make_node("Slice",
        inputs=["input","fv_s","fv_e","fv_ax","fv_st"], outputs=["flipped"])
    inits_slc = [
        _make_int64("sv_s",  [start]),
        _make_int64("sv_e",  [CANVAS]),
        _make_int64("sv_ax", [2]),
        _make_int64("sv_st", [1]),
    ]
    slice_node = helper.make_node("Slice",
        inputs=["flipped","sv_s","sv_e","sv_ax","sv_st"], outputs=["sliced"])
    pad_amt = CANVAS - grid_H
    pads_val = _make_int64("pad_v", [0,0,0,0, 0,0,pad_amt,0])
    pad_node  = helper.make_node("Pad",
        inputs=["sliced","pad_v"], outputs=["output"], mode="constant")

    graph = helper.make_graph(
        [flip_node, slice_node, pad_node],
        "flip_v",
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        inits_flip + inits_slc + [pads_val],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


# ── Graph assembler (for chaining ops) ───────────────────────────────────────

def build_graph(operations: list[dict]) -> onnx.ModelProto:
    """
    Chain a list of operation dicts into a [1,10,30,30]→[1,10,30,30] model.
    """
    all_nodes, all_inits = [], []
    current = "input"

    for i, op in enumerate(operations):
        nodes = []
        for n in op["nodes"]:
            new_in = [current if inp == "x" else inp for inp in n.input]
            new_node = helper.make_node(
                n.op_type, inputs=new_in, outputs=list(n.output),
                name=n.name or f"n{i}",
                **{a.name: helper.get_attribute_value(a) for a in n.attribute},
            )
            nodes.append(new_node)
        all_nodes.extend(nodes)
        all_inits.extend(op["inits"])
        current = op["out"]

    graph = helper.make_graph(
        all_nodes, "neurogolf",
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        [helper.make_tensor_value_info(current, TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        all_inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


# ── Save + simplify ───────────────────────────────────────────────────────────

def save(model: onnx.ModelProto, path: str, try_simplify: bool = True):
    onnx.save(model, path)
    if try_simplify:
        try:
            from onnxsim import simplify as _sim
            sim, ok = _sim(model)
            if ok:
                onnx.save(sim, path)
        except Exception:
            pass
    kb = __import__("os").path.getsize(path) / 1024
    print(f"    saved {path}  ({kb:.1f} KB)")
