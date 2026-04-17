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
    Always names the graph output tensor 'output' for competition compliance.
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

    # Always rename final output tensor to "output" for competition compliance
    if current != "output":
        for node in all_nodes:
            node.output[:] = ["output" if o == current else o for o in node.output]
        current = "output"

    graph = helper.make_graph(
        all_nodes, "neurogolf",
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        all_inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


# ── Static crop-and-shift (MatMul approach — ALL shapes stay [30,30]) ─────────

def static_crop_shift_nodes(
    content_in: str,   # input tensor name: [1, 1, 30, 30] (single channel mask)
    content_out: str,  # output tensor name: [1, 1, 30, 30]
    y0_name: str,      # scalar int64 [1] — first row (inclusive)
    y1_name: str,      # scalar int64 [1] — last row (exclusive)
    x0_name: str,      # scalar int64 [1] — first col (inclusive)
    x1_name: str,      # scalar int64 [1] — last col (exclusive)
    prefix: str = "sc",
) -> tuple[list, list]:
    """
    Crop [y0:y1, x0:x1] from content_in and place at top-left of [1,1,30,30].
    Uses MatMul P_rows @ content @ P_cols^T so ALL intermediate shapes are static.

    P_rows[r, j] = 1 if j == r + y0 AND j < y1  (shifts row y0 → row 0)
    P_cols[c, k] = 1 if k == c + x0 AND k < x1  (shifts col x0 → col 0)

    Uses ONLY: Cast, Sub, Abs, Relu, Mul, Transpose, MatMul, Reshape
    — avoids Equal/Less/And/Not/Cast-from-bool which may not be supported.

    Arithmetic equivalents (all float, for integer-valued inputs):
      j==r+y0  ↔  Relu(1 - |jmr_f - y0_f|) = 1 iff jmr==y0
      j<y1     ↔  Relu(1 - Relu(j+1 - y1_f)) = 1 iff j<y1

    Cost: 2 MatMul ops on [30,30]×[30,30] = 2×27000 MACs, plus small constants.
    """
    nodes, inits = [], []
    N = CANVAS  # 30

    # Static float index matrices (avoids int64 Equal/Less issues)
    r_vec = np.arange(N, dtype=np.float32).reshape(N, 1)   # [30, 1]
    j_vec = np.arange(N, dtype=np.float32).reshape(1, N)   # [1, 30]
    j_minus_r_f = j_vec - r_vec                             # [30, 30] float
    j_p1_f = np.arange(1, N + 1, dtype=np.float32).reshape(1, N)  # [1,30]: j+1

    inits += [
        _t(f"{prefix}_jmr_f",  j_minus_r_f),
        _t(f"{prefix}_jp1_f",  j_p1_f),
        _t(f"{prefix}_ones_f", np.ones((1, 1), dtype=np.float32)),
    ]

    # Cast bounds (int64 [1]) → float [1] → reshape to [1,1] for broadcasting
    inits.append(_make_int64(f"{prefix}_shape11", [1, 1]))
    for tag, src in [("y0", y0_name), ("y1", y1_name), ("x0", x0_name), ("x1", x1_name)]:
        nodes.append(helper.make_node("Cast",
            inputs=[src], outputs=[f"{prefix}_{tag}_fflat"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Reshape",
            inputs=[f"{prefix}_{tag}_fflat", f"{prefix}_shape11"],
            outputs=[f"{prefix}_{tag}_f"]))

    # ── P_rows[r,j] = Relu(1 - |jmr_f - y0_f|) * Relu(1 - Relu(j+1 - y1_f)) ──
    # Row indicator: 1 iff j - r == y0 (i.e. j == r + y0)
    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_jmr_f", f"{prefix}_y0_f"], outputs=[f"{prefix}_diff_r"]))
    nodes.append(helper.make_node("Abs",
        inputs=[f"{prefix}_diff_r"], outputs=[f"{prefix}_abs_r"]))
    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_ones_f", f"{prefix}_abs_r"], outputs=[f"{prefix}_pr_pre"]))
    nodes.append(helper.make_node("Relu",
        inputs=[f"{prefix}_pr_pre"], outputs=[f"{prefix}_Pr_ind"]))   # [30,30]

    # j < y1 mask: 1 iff j+1 <= y1  (i.e. j < y1 for integers)
    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_jp1_f", f"{prefix}_y1_f"], outputs=[f"{prefix}_jp1_m_y1"]))
    nodes.append(helper.make_node("Relu",
        inputs=[f"{prefix}_jp1_m_y1"], outputs=[f"{prefix}_relu_jp1_m_y1"]))
    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_ones_f", f"{prefix}_relu_jp1_m_y1"], outputs=[f"{prefix}_lt_y1_raw"]))
    nodes.append(helper.make_node("Relu",
        inputs=[f"{prefix}_lt_y1_raw"], outputs=[f"{prefix}_lt_y1"]))  # [1,30]

    nodes.append(helper.make_node("Mul",
        inputs=[f"{prefix}_Pr_ind", f"{prefix}_lt_y1"], outputs=[f"{prefix}_P_rows"]))

    # ── P_cols[c,k] = Relu(1 - |jmr_f - x0_f|) * Relu(1 - Relu(k+1 - x1_f)) ──
    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_jmr_f", f"{prefix}_x0_f"], outputs=[f"{prefix}_diff_c"]))
    nodes.append(helper.make_node("Abs",
        inputs=[f"{prefix}_diff_c"], outputs=[f"{prefix}_abs_c"]))
    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_ones_f", f"{prefix}_abs_c"], outputs=[f"{prefix}_pc_pre"]))
    nodes.append(helper.make_node("Relu",
        inputs=[f"{prefix}_pc_pre"], outputs=[f"{prefix}_Pc_ind"]))    # [30,30]

    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_jp1_f", f"{prefix}_x1_f"], outputs=[f"{prefix}_jp1_m_x1"]))
    nodes.append(helper.make_node("Relu",
        inputs=[f"{prefix}_jp1_m_x1"], outputs=[f"{prefix}_relu_jp1_m_x1"]))
    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_ones_f", f"{prefix}_relu_jp1_m_x1"], outputs=[f"{prefix}_lt_x1_raw"]))
    nodes.append(helper.make_node("Relu",
        inputs=[f"{prefix}_lt_x1_raw"], outputs=[f"{prefix}_lt_x1"]))  # [1,30]

    nodes.append(helper.make_node("Mul",
        inputs=[f"{prefix}_Pc_ind", f"{prefix}_lt_x1"], outputs=[f"{prefix}_P_cols"]))
    nodes.append(helper.make_node("Transpose",
        inputs=[f"{prefix}_P_cols"], outputs=[f"{prefix}_P_cols_T"], perm=[1, 0]))

    # Reshape content [1,1,30,30] -> [1,30,30]
    inits.append(_make_int64(f"{prefix}_shape1nn", [1, N, N]))
    nodes.append(helper.make_node("Reshape",
        inputs=[content_in, f"{prefix}_shape1nn"], outputs=[f"{prefix}_c2d"]))

    # Apply row shift: P_rows [30,30] @ content [1,30,30] -> [1,30,30]
    nodes.append(helper.make_node("MatMul",
        inputs=[f"{prefix}_P_rows", f"{prefix}_c2d"], outputs=[f"{prefix}_rows_shifted"]))

    # Apply col shift: rows_shifted [1,30,30] @ P_cols^T [30,30] -> [1,30,30]
    nodes.append(helper.make_node("MatMul",
        inputs=[f"{prefix}_rows_shifted", f"{prefix}_P_cols_T"],
        outputs=[f"{prefix}_shifted"]))

    # Reshape back [1,30,30] -> [1,1,30,30]
    inits.append(_make_int64(f"{prefix}_shape1c", [1, 1, N, N]))
    nodes.append(helper.make_node("Reshape",
        inputs=[f"{prefix}_shifted", f"{prefix}_shape1c"], outputs=[content_out]))

    return nodes, inits


def static_crop_flip_shift_nodes(
    content_in: str,
    content_out: str,
    y0_name: str, y1_name: str,
    x0_name: str, x1_name: str,
    prefix: str = "scf",
) -> tuple[list, list]:
    """
    Like static_crop_shift_nodes but horizontally flips the crop before placing at top-left.
    Row shift: same (j == r + y0, j < y1)
    Col shift reversed: P_flip[c, k] = 1 if k == x1-1-c AND k >= x0 AND k < x1
      Equivalently: c + k == x1 - 1, and x0 <= k < x1
    ALL intermediate shapes remain [30, 30].

    Uses ONLY: Cast, Sub, Abs, Relu, Mul, Transpose, MatMul, Reshape
    — avoids Equal/Less/And/Not/Cast-from-bool.
    """
    nodes, inits = [], []
    N = CANVAS

    r_vec = np.arange(N, dtype=np.float32).reshape(N, 1)
    j_vec = np.arange(N, dtype=np.float32).reshape(1, N)
    j_minus_r_f = j_vec - r_vec   # [30,30] float
    c_plus_k_f  = r_vec + j_vec   # [30,30] float (c rows, k cols)
    j_p1_f = np.arange(1, N + 1, dtype=np.float32).reshape(1, N)  # [1,30]: j+1

    inits += [
        _t(f"{prefix}_jmr_f",  j_minus_r_f),
        _t(f"{prefix}_cpk_f",  c_plus_k_f),
        _t(f"{prefix}_jp1_f",  j_p1_f),
        _t(f"{prefix}_jvec_f", j_vec),                               # [1,30]: k values
        _t(f"{prefix}_ones_f", np.ones((1, 1), dtype=np.float32)),
    ]

    # Cast bounds (int64 [1]) → float [1] → reshape to [1,1]
    inits.append(_make_int64(f"{prefix}_shape11", [1, 1]))
    for tag, src in [("y0", y0_name), ("y1", y1_name), ("x0", x0_name), ("x1", x1_name)]:
        nodes.append(helper.make_node("Cast",
            inputs=[src], outputs=[f"{prefix}_{tag}_fflat"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Reshape",
            inputs=[f"{prefix}_{tag}_fflat", f"{prefix}_shape11"],
            outputs=[f"{prefix}_{tag}_f"]))

    # x1 - 1.0 for flip
    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_x1_f", f"{prefix}_ones_f"], outputs=[f"{prefix}_x1m1_f"]))

    # ── P_rows[r,j] = Relu(1 - |jmr_f - y0_f|) * Relu(1 - Relu(j+1 - y1_f)) ──
    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_jmr_f", f"{prefix}_y0_f"], outputs=[f"{prefix}_diff_r"]))
    nodes.append(helper.make_node("Abs",
        inputs=[f"{prefix}_diff_r"], outputs=[f"{prefix}_abs_r"]))
    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_ones_f", f"{prefix}_abs_r"], outputs=[f"{prefix}_pr_pre"]))
    nodes.append(helper.make_node("Relu",
        inputs=[f"{prefix}_pr_pre"], outputs=[f"{prefix}_Pr_ind"]))

    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_jp1_f", f"{prefix}_y1_f"], outputs=[f"{prefix}_jp1_m_y1"]))
    nodes.append(helper.make_node("Relu",
        inputs=[f"{prefix}_jp1_m_y1"], outputs=[f"{prefix}_relu_jp1_m_y1"]))
    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_ones_f", f"{prefix}_relu_jp1_m_y1"], outputs=[f"{prefix}_lt_y1_raw"]))
    nodes.append(helper.make_node("Relu",
        inputs=[f"{prefix}_lt_y1_raw"], outputs=[f"{prefix}_lt_y1"]))

    nodes.append(helper.make_node("Mul",
        inputs=[f"{prefix}_Pr_ind", f"{prefix}_lt_y1"], outputs=[f"{prefix}_Pr"]))

    # ── P_flip[c,k]: (c+k == x1-1) AND (k >= x0) AND (k < x1) ──
    # c+k == x1-1 indicator
    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_cpk_f", f"{prefix}_x1m1_f"], outputs=[f"{prefix}_diff_ck"]))
    nodes.append(helper.make_node("Abs",
        inputs=[f"{prefix}_diff_ck"], outputs=[f"{prefix}_abs_ck"]))
    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_ones_f", f"{prefix}_abs_ck"], outputs=[f"{prefix}_eq_ck_raw"]))
    nodes.append(helper.make_node("Relu",
        inputs=[f"{prefix}_eq_ck_raw"], outputs=[f"{prefix}_eq_ck"]))  # [30,30]

    # k >= x0: Relu(1 - Relu(x0_f - jvec_f))
    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_x0_f", f"{prefix}_jvec_f"], outputs=[f"{prefix}_x0_m_k"]))
    nodes.append(helper.make_node("Relu",
        inputs=[f"{prefix}_x0_m_k"], outputs=[f"{prefix}_relu_x0_m_k"]))
    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_ones_f", f"{prefix}_relu_x0_m_k"], outputs=[f"{prefix}_ge_x0_raw"]))
    nodes.append(helper.make_node("Relu",
        inputs=[f"{prefix}_ge_x0_raw"], outputs=[f"{prefix}_ge_x0"]))  # [1,30]

    # k < x1: Relu(1 - Relu(k+1 - x1_f))
    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_jp1_f", f"{prefix}_x1_f"], outputs=[f"{prefix}_jp1_m_x1"]))
    nodes.append(helper.make_node("Relu",
        inputs=[f"{prefix}_jp1_m_x1"], outputs=[f"{prefix}_relu_jp1_m_x1"]))
    nodes.append(helper.make_node("Sub",
        inputs=[f"{prefix}_ones_f", f"{prefix}_relu_jp1_m_x1"], outputs=[f"{prefix}_lt_x1_raw"]))
    nodes.append(helper.make_node("Relu",
        inputs=[f"{prefix}_lt_x1_raw"], outputs=[f"{prefix}_lt_x1"]))  # [1,30]

    # Combine: eq_ck [30,30] * ge_x0 [1,30] * lt_x1 [1,30]
    nodes.append(helper.make_node("Mul",
        inputs=[f"{prefix}_ge_x0", f"{prefix}_lt_x1"], outputs=[f"{prefix}_k_range"]))
    nodes.append(helper.make_node("Mul",
        inputs=[f"{prefix}_eq_ck", f"{prefix}_k_range"], outputs=[f"{prefix}_Pfc"]))
    nodes.append(helper.make_node("Transpose",
        inputs=[f"{prefix}_Pfc"], outputs=[f"{prefix}_Pfc_T"], perm=[1, 0]))

    inits.append(_make_int64(f"{prefix}_shape1nn", [1, N, N]))
    nodes.append(helper.make_node("Reshape",
        inputs=[content_in, f"{prefix}_shape1nn"], outputs=[f"{prefix}_c2d"]))
    nodes.append(helper.make_node("MatMul",
        inputs=[f"{prefix}_Pr", f"{prefix}_c2d"], outputs=[f"{prefix}_rows_sh"]))
    nodes.append(helper.make_node("MatMul",
        inputs=[f"{prefix}_rows_sh", f"{prefix}_Pfc_T"], outputs=[f"{prefix}_shifted"]))

    inits.append(_make_int64(f"{prefix}_shape1c", [1, 1, N, N]))
    nodes.append(helper.make_node("Reshape",
        inputs=[f"{prefix}_shifted", f"{prefix}_shape1c"], outputs=[content_out]))

    return nodes, inits


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
