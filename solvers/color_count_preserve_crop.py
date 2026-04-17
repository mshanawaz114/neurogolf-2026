from __future__ import annotations

from pathlib import Path
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.arc_utils import detect_color_count_preserve_crop, grid_to_array
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_float, _make_int64, save


def _color_count_preserve_crop_net(mode: str) -> onnx.ModelProto:
    nodes = []
    inits = []

    def reduce_sum(inp: str, out: str, axes_vals: list[int], keepdims: int = 1):
        axes_name = f"{out}_axes"
        inits.append(_make_int64(axes_name, axes_vals))
        nodes.append(helper.make_node("ReduceSum", inputs=[inp, axes_name], outputs=[out], keepdims=keepdims))

    def reduce_max(inp: str, out: str, axes_vals: list[int], keepdims: int = 1):
        nodes.append(helper.make_node("ReduceMax", inputs=[inp], outputs=[out], axes=axes_vals, keepdims=keepdims))

    def reduce_min(inp: str, out: str, axes_vals: list[int], keepdims: int = 1):
        nodes.append(helper.make_node("ReduceMin", inputs=[inp], outputs=[out], axes=axes_vals, keepdims=keepdims))

    reduce_sum("input", "ccp_counts", [2, 3], keepdims=0)
    inits += [
        _make_int64("ccp_cstart", [1]),
        _make_int64("ccp_cend", [C]),
        _make_int64("ccp_caxis", [1]),
        _make_int64("ccp_cstep", [1]),
    ]
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["ccp_counts", "ccp_cstart", "ccp_cend", "ccp_caxis", "ccp_cstep"],
            outputs=["ccp_nonbg_counts"],
        )
    )

    if mode == "max":
        nodes.append(helper.make_node("ArgMax", inputs=["ccp_nonbg_counts"], outputs=["ccp_idx0"], axis=1, keepdims=0))
    elif mode == "min":
        inits.append(_make_float("ccp_zero_f", [0.0]))
        inits.append(_make_float("ccp_big_f", [[1e9] * (C - 1)]))
        nodes.append(helper.make_node("Greater", inputs=["ccp_nonbg_counts", "ccp_zero_f"], outputs=["ccp_present_b"]))
        nodes.append(
            helper.make_node("Where", inputs=["ccp_present_b", "ccp_nonbg_counts", "ccp_big_f"], outputs=["ccp_masked_counts"])
        )
        nodes.append(helper.make_node("ArgMin", inputs=["ccp_masked_counts"], outputs=["ccp_idx0"], axis=1, keepdims=0))
    else:
        raise ValueError(f"unknown mode: {mode}")

    inits.append(_make_int64("ccp_one_i", [1]))
    nodes.append(helper.make_node("Add", inputs=["ccp_idx0", "ccp_one_i"], outputs=["ccp_color_idx"]))
    nodes.append(helper.make_node("Gather", inputs=["input", "ccp_color_idx"], outputs=["ccp_sel"], axis=1))

    reduce_max("ccp_sel", "ccp_sel_hw", [1], keepdims=0)
    reduce_max("ccp_sel_hw", "ccp_row_presence", [2], keepdims=0)
    reduce_max("ccp_sel_hw", "ccp_col_presence", [1], keepdims=0)

    inits += [
        _make_float("ccp_zero_mask", [[0.0] * CANVAS]),
        _make_int64("ccp_ids", [list(range(CANVAS))]),
        _make_int64("ccp_ids_p1", [list(range(1, CANVAS + 1))]),
        _make_int64("ccp_big", [[CANVAS] * CANVAS]),
        _make_int64("ccp_zeros", [[0] * CANVAS]),
    ]
    nodes.append(helper.make_node("Greater", inputs=["ccp_row_presence", "ccp_zero_mask"], outputs=["ccp_row_present_b"]))
    nodes.append(helper.make_node("Where", inputs=["ccp_row_present_b", "ccp_ids", "ccp_big"], outputs=["ccp_y0_candidates"]))
    reduce_min("ccp_y0_candidates", "ccp_y0", [1], keepdims=0)
    nodes.append(helper.make_node("Where", inputs=["ccp_row_present_b", "ccp_ids_p1", "ccp_zeros"], outputs=["ccp_y1_candidates"]))
    reduce_max("ccp_y1_candidates", "ccp_y1", [1], keepdims=0)

    nodes.append(helper.make_node("Greater", inputs=["ccp_col_presence", "ccp_zero_mask"], outputs=["ccp_col_present_b"]))
    nodes.append(helper.make_node("Where", inputs=["ccp_col_present_b", "ccp_ids", "ccp_big"], outputs=["ccp_x0_candidates"]))
    reduce_min("ccp_x0_candidates", "ccp_x0", [1], keepdims=0)
    nodes.append(helper.make_node("Where", inputs=["ccp_col_present_b", "ccp_ids_p1", "ccp_zeros"], outputs=["ccp_x1_candidates"]))
    reduce_max("ccp_x1_candidates", "ccp_x1", [1], keepdims=0)

    inits += [
        _make_int64("ccp_row_axis", [2]),
        _make_int64("ccp_col_axis", [3]),
        _make_int64("ccp_step", [1]),
    ]
    nodes.append(helper.make_node("Slice", inputs=["input", "ccp_y0", "ccp_y1", "ccp_row_axis", "ccp_step"], outputs=["ccp_rows"]))
    nodes.append(helper.make_node("Slice", inputs=["ccp_rows", "ccp_x0", "ccp_x1", "ccp_col_axis", "ccp_step"], outputs=["ccp_crop"]))

    nodes.append(helper.make_node("Sub", inputs=["ccp_y1", "ccp_y0"], outputs=["ccp_h"]))
    nodes.append(helper.make_node("Sub", inputs=["ccp_x1", "ccp_x0"], outputs=["ccp_w"]))
    inits += [
        _make_int64("ccp_canvas_i", [CANVAS]),
        _make_int64("ccp_pad_prefix", [0, 0, 0, 0, 0, 0]),
    ]
    nodes.append(helper.make_node("Sub", inputs=["ccp_canvas_i", "ccp_h"], outputs=["ccp_bottom"]))
    nodes.append(helper.make_node("Sub", inputs=["ccp_canvas_i", "ccp_w"], outputs=["ccp_right"]))
    nodes.append(helper.make_node("Concat", inputs=["ccp_pad_prefix", "ccp_bottom", "ccp_right"], outputs=["ccp_pads"], axis=0))
    nodes.append(helper.make_node("Pad", inputs=["ccp_crop", "ccp_pads"], outputs=["output"], mode="constant"))

    graph = helper.make_graph(
        nodes,
        f"color_count_preserve_crop_{mode}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class ColorCountPreserveCropSolver(BaseSolver):
    PRIORITY = 14

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("color_count_preserve_crop")) and analysis.get("color_count_preserve_crop_mode") in {"min", "max"}

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        mode = analysis.get("color_count_preserve_crop_mode")
        if mode not in {"min", "max"}:
            return None

        for split in ["train", "test", "arc-gen"]:
            for pair in task.get(split, []):
                if not detect_color_count_preserve_crop(grid_to_array(pair["input"]), grid_to_array(pair["output"]), mode):
                    return None

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{task_id}.onnx"
        try:
            save(_color_count_preserve_crop_net(mode), str(path), try_simplify=False)
            return path
        except Exception as e:
            print(f"    ColorCountPreserveCropSolver({mode}) failed: {e}")
            return None
