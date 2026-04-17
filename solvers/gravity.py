from __future__ import annotations

"""
gravity.py — Deterministic solver for column-wise gravity tasks.

Gravity tasks preserve each column independently:
  - `gravity_down`: non-background cells compact to the bottom
  - `gravity_up`:   non-background cells compact to the top

Key observation:
For each source cell we can compute its order among non-background cells in the
same column using `CumSum`. That order determines the target row exactly.

We then place each source row into the correct target row with a broadcasted
`Equal` mask over the 30 possible row indices.
"""

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_int64, _t, save


def _gravity_net(direction: str) -> onnx.ModelProto:
    nodes = []
    inits = []

    def reduce_max(inp: str, out: str, axes_vals: list[int], keepdims: int = 1):
        nodes.append(helper.make_node("ReduceMax", inputs=[inp], outputs=[out], axes=axes_vals, keepdims=keepdims))

    def reduce_sum(inp: str, out: str, axes_vals: list[int], keepdims: int = 1):
        axes_name = f"{out}_axes"
        inits.append(_make_int64(axes_name, axes_vals))
        nodes.append(
            helper.make_node("ReduceSum", inputs=[inp, axes_name], outputs=[out], keepdims=keepdims)
        )

    # Non-background channels only. ARC uses colour 0 as background.
    inits += [
        _make_int64("g_cs", [1]),
        _make_int64("g_ce", [C]),
        _make_int64("g_ca", [1]),
        _make_int64("g_cst", [1]),
    ]
    nodes.append(
        helper.make_node(
            "Slice", inputs=["input", "g_cs", "g_ce", "g_ca", "g_cst"], outputs=["nonbg"]
        )
    )

    # inside_mask = 1 inside the original grid rectangle, 0 on zero-hot padding.
    reduce_max("input", "inside_mask", [1], keepdims=1)

    # presence[source_row, col] = whether this source cell is non-background.
    reduce_max("nonbg", "presence_f", [1], keepdims=1)
    nodes.append(helper.make_node("Cast", inputs=["presence_f"], outputs=["presence"], to=TensorProto.INT64))

    # Grid height H from the inside-grid support, not from non-background cells.
    reduce_max("inside_mask", "row_support_f", [3], keepdims=1)
    nodes.append(helper.make_node("Cast", inputs=["row_support_f"], outputs=["row_support"], to=TensorProto.INT64))
    reduce_sum("row_support", "grid_h", [2], keepdims=1)

    # rank[source_row, col] = 1-based order of this non-background cell in its column.
    inits.append(_make_int64("g_row_axis", [2]))
    nodes.append(helper.make_node("CumSum", inputs=["presence", "g_row_axis"], outputs=["rank"]))
    reduce_sum("presence", "count_per_col", [2], keepdims=1)

    if direction == "down":
        nodes.append(helper.make_node("Sub", inputs=["grid_h", "count_per_col"], outputs=["base_row"]))
        nodes.append(helper.make_node("Add", inputs=["base_row", "rank"], outputs=["target_row_1"]))
        inits.append(_make_int64("g_one", [1]))
        nodes.append(helper.make_node("Sub", inputs=["target_row_1", "g_one"], outputs=["target_row"]))
    else:
        inits.append(_make_int64("g_one", [1]))
        nodes.append(helper.make_node("Sub", inputs=["rank", "g_one"], outputs=["target_row"]))

    # Broadcast target rows against all possible output row indices 0..29.
    inits.append(_t("g_row_ids", np.arange(CANVAS, dtype=np.int64).reshape(1, CANVAS, 1, 1)))
    nodes.append(helper.make_node("Equal", inputs=["g_row_ids", "target_row"], outputs=["row_mask_b"]))
    nodes.append(helper.make_node("Cast", inputs=["row_mask_b"], outputs=["row_mask"], to=TensorProto.FLOAT))

    # Broadcast input channels and row mask to [1, 9, target_row, source_row, col].
    inits.append(_make_int64("g_unsq_axis", [2]))
    inits.append(_make_int64("g_mask_unsq_axis", [1]))
    nodes.append(helper.make_node("Unsqueeze", inputs=["nonbg", "g_unsq_axis"], outputs=["nonbg_5d"]))
    nodes.append(helper.make_node("Unsqueeze", inputs=["row_mask", "g_mask_unsq_axis"], outputs=["row_mask_5d"]))
    nodes.append(helper.make_node("Mul", inputs=["nonbg_5d", "row_mask_5d"], outputs=["placed_5d"]))
    reduce_sum("placed_5d", "nonbg_out", [3], keepdims=0)

    # Rebuild background channel inside the original grid rectangle.
    reduce_max("nonbg_out", "nonbg_presence", [1], keepdims=1)
    nodes.append(helper.make_node("Sub", inputs=["inside_mask", "nonbg_presence"], outputs=["bg_out"]))
    nodes.append(helper.make_node("Concat", inputs=["bg_out", "nonbg_out"], outputs=["output"], axis=1))

    graph = helper.make_graph(
        nodes,
        f"gravity_{direction}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class GravitySolver(BaseSolver):
    PRIORITY = 15

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("gravity_down") or analysis.get("gravity_up"))

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        direction = "down" if analysis.get("gravity_down") else "up" if analysis.get("gravity_up") else None
        if direction is None:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{task_id}.onnx"
        try:
            save(_gravity_net(direction), str(path), try_simplify=False)
            return path
        except Exception as e:
            print(f"    GravitySolver({direction}) failed: {e}")
            return None
