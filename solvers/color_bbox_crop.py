from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.arc_utils import detect_color_bbox_crop, grid_to_array
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_float, _make_int64, _t, save


def _color_bbox_crop_net(mode: str) -> onnx.ModelProto:
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

    # Per-colour row/col support.
    reduce_max("input", "cbc_row_presence", [3], keepdims=0)  # [1, C, 30]
    reduce_max("input", "cbc_col_presence", [2], keepdims=0)  # [1, C, 30]

    inits += [
        _t("cbc_ids", np.arange(CANVAS, dtype=np.int64).reshape(1, 1, CANVAS)),
        _t("cbc_ids_p1", np.arange(1, CANVAS + 1, dtype=np.int64).reshape(1, 1, CANVAS)),
        _make_int64("cbc_big", np.full((1, 1, CANVAS), CANVAS, dtype=np.int64)),
        _make_int64("cbc_zeros", np.zeros((1, 1, CANVAS), dtype=np.int64)),
        _t("cbc_zero_mask", np.zeros((1, 1, CANVAS), dtype=np.float32)),
    ]

    nodes.append(helper.make_node("Greater", inputs=["cbc_row_presence", "cbc_zero_mask"], outputs=["cbc_row_present_b"]))
    nodes.append(
        helper.make_node("Where", inputs=["cbc_row_present_b", "cbc_ids", "cbc_big"], outputs=["cbc_row_first_c"])
    )
    reduce_min("cbc_row_first_c", "cbc_y0_all", [2], keepdims=0)
    nodes.append(
        helper.make_node("Where", inputs=["cbc_row_present_b", "cbc_ids_p1", "cbc_zeros"], outputs=["cbc_row_last_c"])
    )
    reduce_max("cbc_row_last_c", "cbc_y1_all", [2], keepdims=0)

    nodes.append(helper.make_node("Greater", inputs=["cbc_col_presence", "cbc_zero_mask"], outputs=["cbc_col_present_b"]))
    nodes.append(
        helper.make_node("Where", inputs=["cbc_col_present_b", "cbc_ids", "cbc_big"], outputs=["cbc_col_first_c"])
    )
    reduce_min("cbc_col_first_c", "cbc_x0_all", [2], keepdims=0)
    nodes.append(
        helper.make_node("Where", inputs=["cbc_col_present_b", "cbc_ids_p1", "cbc_zeros"], outputs=["cbc_col_last_c"])
    )
    reduce_max("cbc_col_last_c", "cbc_x1_all", [2], keepdims=0)

    # area per colour = (y1-y0) * (x1-x0)
    nodes.append(helper.make_node("Sub", inputs=["cbc_y1_all", "cbc_y0_all"], outputs=["cbc_h_all"]))
    nodes.append(helper.make_node("Sub", inputs=["cbc_x1_all", "cbc_x0_all"], outputs=["cbc_w_all"]))
    nodes.append(helper.make_node("Mul", inputs=["cbc_h_all", "cbc_w_all"], outputs=["cbc_area_all"]))
    reduce_sum("input", "cbc_counts_all", [2, 3], keepdims=0)

    # Ignore background by slicing channels 1..9.
    inits += [
        _make_int64("cbc_cstart", [1]),
        _make_int64("cbc_cend", [C]),
        _make_int64("cbc_caxis", [1]),
        _make_int64("cbc_cstep", [1]),
    ]
    for src, dst in [
        ("cbc_area_all", "cbc_area_nonbg"),
        ("cbc_counts_all", "cbc_counts_nonbg"),
        ("cbc_y0_all", "cbc_y0_nonbg"),
        ("cbc_y1_all", "cbc_y1_nonbg"),
        ("cbc_x0_all", "cbc_x0_nonbg"),
        ("cbc_x1_all", "cbc_x1_nonbg"),
    ]:
        nodes.append(
            helper.make_node("Slice", inputs=[src, "cbc_cstart", "cbc_cend", "cbc_caxis", "cbc_cstep"], outputs=[dst])
        )
    inits.append(_make_float("cbc_zero_f", [0.0]))
    nodes.append(helper.make_node("Greater", inputs=["cbc_counts_nonbg", "cbc_zero_f"], outputs=["cbc_present_b"]))
    nodes.append(helper.make_node("Cast", inputs=["cbc_area_nonbg"], outputs=["cbc_area_nonbg_f"], to=TensorProto.FLOAT))

    if mode == "min_bbox":
        inits.append(_make_float("cbc_big_f", np.full((1, C - 1), 1e9, dtype=np.float32)))
        nodes.append(helper.make_node("Where", inputs=["cbc_present_b", "cbc_area_nonbg_f", "cbc_big_f"], outputs=["cbc_area_masked_f"]))
        nodes.append(helper.make_node("ArgMin", inputs=["cbc_area_masked_f"], outputs=["cbc_idx0"], axis=1, keepdims=0))
    elif mode == "max_bbox":
        inits.append(_make_float("cbc_neg_f", np.zeros((1, C - 1), dtype=np.float32)))
        nodes.append(helper.make_node("Where", inputs=["cbc_present_b", "cbc_area_nonbg_f", "cbc_neg_f"], outputs=["cbc_area_masked_f"]))
        nodes.append(helper.make_node("ArgMax", inputs=["cbc_area_masked_f"], outputs=["cbc_idx0"], axis=1, keepdims=0))
    else:
        raise ValueError(f"unknown mode: {mode}")

    inits.append(_make_int64("cbc_one_i", [1]))
    nodes.append(helper.make_node("Add", inputs=["cbc_idx0", "cbc_one_i"], outputs=["cbc_color_idx"]))
    nodes.append(helper.make_node("Gather", inputs=["input", "cbc_color_idx"], outputs=["cbc_sel"], axis=1))

    # Recompute bbox for the selected colour mask.
    reduce_max("cbc_sel", "cbc_sel_hw", [1], keepdims=0)
    reduce_max("cbc_sel_hw", "cbc_row_presence_sel", [2], keepdims=0)
    reduce_max("cbc_sel_hw", "cbc_col_presence_sel", [1], keepdims=0)

    inits += [
        _t("cbc_ids_2d", np.arange(CANVAS, dtype=np.int64).reshape(1, CANVAS)),
        _t("cbc_ids_p1_2d", np.arange(1, CANVAS + 1, dtype=np.int64).reshape(1, CANVAS)),
        _make_int64("cbc_big_2d", np.full((1, CANVAS), CANVAS, dtype=np.int64)),
        _make_int64("cbc_zeros_2d", np.zeros((1, CANVAS), dtype=np.int64)),
        _t("cbc_zero_mask_2d", np.zeros((1, CANVAS), dtype=np.float32)),
    ]
    nodes.append(
        helper.make_node("Greater", inputs=["cbc_row_presence_sel", "cbc_zero_mask_2d"], outputs=["cbc_row_present_sel_b"])
    )
    nodes.append(
        helper.make_node(
            "Where", inputs=["cbc_row_present_sel_b", "cbc_ids_2d", "cbc_big_2d"], outputs=["cbc_y0_candidates"]
        )
    )
    reduce_min("cbc_y0_candidates", "cbc_y0", [1], keepdims=0)
    nodes.append(
        helper.make_node(
            "Where", inputs=["cbc_row_present_sel_b", "cbc_ids_p1_2d", "cbc_zeros_2d"], outputs=["cbc_y1_candidates"]
        )
    )
    reduce_max("cbc_y1_candidates", "cbc_y1", [1], keepdims=0)

    nodes.append(
        helper.make_node("Greater", inputs=["cbc_col_presence_sel", "cbc_zero_mask_2d"], outputs=["cbc_col_present_sel_b"])
    )
    nodes.append(
        helper.make_node(
            "Where", inputs=["cbc_col_present_sel_b", "cbc_ids_2d", "cbc_big_2d"], outputs=["cbc_x0_candidates"]
        )
    )
    reduce_min("cbc_x0_candidates", "cbc_x0", [1], keepdims=0)
    nodes.append(
        helper.make_node(
            "Where", inputs=["cbc_col_present_sel_b", "cbc_ids_p1_2d", "cbc_zeros_2d"], outputs=["cbc_x1_candidates"]
        )
    )
    reduce_max("cbc_x1_candidates", "cbc_x1", [1], keepdims=0)

    inits += [
        _make_int64("cbc_row_axis", [2]),
        _make_int64("cbc_col_axis", [3]),
        _make_int64("cbc_step", [1]),
    ]
    nodes.append(
        helper.make_node("Slice", inputs=["cbc_sel", "cbc_y0", "cbc_y1", "cbc_row_axis", "cbc_step"], outputs=["cbc_rows"])
    )
    nodes.append(
        helper.make_node("Slice", inputs=["cbc_rows", "cbc_x0", "cbc_x1", "cbc_col_axis", "cbc_step"], outputs=["cbc_crop"])
    )

    nodes.append(helper.make_node("Sub", inputs=["cbc_y1", "cbc_y0"], outputs=["cbc_h"]))
    nodes.append(helper.make_node("Sub", inputs=["cbc_x1", "cbc_x0"], outputs=["cbc_w"]))
    inits += [
        _make_int64("cbc_canvas_i", [CANVAS]),
        _make_int64("cbc_pad_prefix", [0, 0, 0, 0, 0, 0]),
    ]
    nodes.append(helper.make_node("Sub", inputs=["cbc_canvas_i", "cbc_h"], outputs=["cbc_bottom"]))
    nodes.append(helper.make_node("Sub", inputs=["cbc_canvas_i", "cbc_w"], outputs=["cbc_right"]))
    nodes.append(helper.make_node("Concat", inputs=["cbc_pad_prefix", "cbc_bottom", "cbc_right"], outputs=["cbc_pads"], axis=0))
    nodes.append(helper.make_node("Pad", inputs=["cbc_crop", "cbc_pads"], outputs=["cbc_mask_padded"], mode="constant"))

    inits += [
        _make_int64("cbc_color_ids", list(range(C))),
        _make_int64("cbc_color_shape", [1, C, 1, 1]),
    ]
    nodes.append(helper.make_node("Equal", inputs=["cbc_color_ids", "cbc_color_idx"], outputs=["cbc_color_b"]))
    nodes.append(helper.make_node("Cast", inputs=["cbc_color_b"], outputs=["cbc_color_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Reshape", inputs=["cbc_color_f", "cbc_color_shape"], outputs=["cbc_color_mask"]))
    nodes.append(helper.make_node("Mul", inputs=["cbc_color_mask", "cbc_mask_padded"], outputs=["output"]))

    graph = helper.make_graph(
        nodes,
        f"color_bbox_crop_{mode}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class ColorBBoxCropSolver(BaseSolver):
    PRIORITY = 14

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("color_bbox_crop")) and analysis.get("color_bbox_crop_mode") in {"min_bbox", "max_bbox"}

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        mode = analysis.get("color_bbox_crop_mode")
        if mode not in {"min_bbox", "max_bbox"}:
            return None

        for split in ["train", "test", "arc-gen"]:
            for pair in task.get(split, []):
                if not detect_color_bbox_crop(grid_to_array(pair["input"]), grid_to_array(pair["output"]), mode):
                    return None

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{task_id}.onnx"
        try:
            save(_color_bbox_crop_net(mode), str(path), try_simplify=False)
            return path
        except Exception as e:
            print(f"    ColorBBoxCropSolver({mode}) failed: {e}")
            return None
