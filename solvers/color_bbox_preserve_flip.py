from __future__ import annotations

from pathlib import Path
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.arc_utils import detect_color_bbox_preserve_flip, grid_to_array
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_float, _make_int64, save


def _color_bbox_preserve_flip_net(mode: str) -> onnx.ModelProto:
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

    # Per-colour row/col support to estimate bbox area for each channel.
    reduce_max("input", "cbpf_row_presence", [3], keepdims=0)  # [1, C, 30]
    reduce_max("input", "cbpf_col_presence", [2], keepdims=0)  # [1, C, 30]

    inits += [
        _make_int64("cbpf_ids", [list(range(CANVAS))]),
        _make_int64("cbpf_ids_p1", [list(range(1, CANVAS + 1))]),
        _make_int64("cbpf_big", [[CANVAS] * CANVAS]),
        _make_int64("cbpf_zeros", [[0] * CANVAS]),
        _make_float("cbpf_zero_mask", [[0.0] * CANVAS]),
    ]

    nodes.append(helper.make_node("Greater", inputs=["cbpf_row_presence", "cbpf_zero_mask"], outputs=["cbpf_row_present_b"]))
    nodes.append(helper.make_node("Where", inputs=["cbpf_row_present_b", "cbpf_ids", "cbpf_big"], outputs=["cbpf_y0_cand"]))
    reduce_min("cbpf_y0_cand", "cbpf_y0_all", [2], keepdims=0)
    nodes.append(helper.make_node("Where", inputs=["cbpf_row_present_b", "cbpf_ids_p1", "cbpf_zeros"], outputs=["cbpf_y1_cand"]))
    reduce_max("cbpf_y1_cand", "cbpf_y1_all", [2], keepdims=0)

    nodes.append(helper.make_node("Greater", inputs=["cbpf_col_presence", "cbpf_zero_mask"], outputs=["cbpf_col_present_b"]))
    nodes.append(helper.make_node("Where", inputs=["cbpf_col_present_b", "cbpf_ids", "cbpf_big"], outputs=["cbpf_x0_cand"]))
    reduce_min("cbpf_x0_cand", "cbpf_x0_all", [2], keepdims=0)
    nodes.append(helper.make_node("Where", inputs=["cbpf_col_present_b", "cbpf_ids_p1", "cbpf_zeros"], outputs=["cbpf_x1_cand"]))
    reduce_max("cbpf_x1_cand", "cbpf_x1_all", [2], keepdims=0)

    nodes.append(helper.make_node("Sub", inputs=["cbpf_y1_all", "cbpf_y0_all"], outputs=["cbpf_h_all"]))
    nodes.append(helper.make_node("Sub", inputs=["cbpf_x1_all", "cbpf_x0_all"], outputs=["cbpf_w_all"]))
    nodes.append(helper.make_node("Mul", inputs=["cbpf_h_all", "cbpf_w_all"], outputs=["cbpf_area_all"]))
    reduce_sum("input", "cbpf_counts_all", [2, 3], keepdims=0)

    inits += [
        _make_int64("cbpf_cstart", [1]),
        _make_int64("cbpf_cend", [C]),
        _make_int64("cbpf_caxis", [1]),
        _make_int64("cbpf_cstep", [1]),
    ]
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["cbpf_area_all", "cbpf_cstart", "cbpf_cend", "cbpf_caxis", "cbpf_cstep"],
            outputs=["cbpf_area_nonbg"],
        )
    )
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["cbpf_counts_all", "cbpf_cstart", "cbpf_cend", "cbpf_caxis", "cbpf_cstep"],
            outputs=["cbpf_counts_nonbg"],
        )
    )
    inits.append(_make_float("cbpf_zero_f", [0.0]))
    nodes.append(helper.make_node("Greater", inputs=["cbpf_counts_nonbg", "cbpf_zero_f"], outputs=["cbpf_present_b"]))
    nodes.append(helper.make_node("Cast", inputs=["cbpf_area_nonbg"], outputs=["cbpf_area_nonbg_f"], to=TensorProto.FLOAT))

    if mode == "min_bbox":
        inits.append(_make_float("cbpf_big_f", [[1e9] * (C - 1)]))
        nodes.append(helper.make_node("Where", inputs=["cbpf_present_b", "cbpf_area_nonbg_f", "cbpf_big_f"], outputs=["cbpf_area_masked_f"]))
        nodes.append(helper.make_node("ArgMin", inputs=["cbpf_area_masked_f"], outputs=["cbpf_idx0"], axis=1, keepdims=0))
    elif mode == "max_bbox":
        inits.append(_make_float("cbpf_neg_f", [[0.0] * (C - 1)]))
        nodes.append(helper.make_node("Where", inputs=["cbpf_present_b", "cbpf_area_nonbg_f", "cbpf_neg_f"], outputs=["cbpf_area_masked_f"]))
        nodes.append(helper.make_node("ArgMax", inputs=["cbpf_area_masked_f"], outputs=["cbpf_idx0"], axis=1, keepdims=0))
    else:
        raise ValueError(f"unknown mode: {mode}")

    inits.append(_make_int64("cbpf_one_i", [1]))
    nodes.append(helper.make_node("Add", inputs=["cbpf_idx0", "cbpf_one_i"], outputs=["cbpf_color_idx"]))
    nodes.append(helper.make_node("Gather", inputs=["input", "cbpf_color_idx"], outputs=["cbpf_sel"], axis=1))

    # Recompute bbox for the selected colour mask using the already-working dynamic path.
    reduce_max("cbpf_sel", "cbpf_sel_hw", [1], keepdims=0)
    reduce_max("cbpf_sel_hw", "cbpf_row_presence_sel", [2], keepdims=0)
    reduce_max("cbpf_sel_hw", "cbpf_col_presence_sel", [1], keepdims=0)

    inits += [
        _make_int64("cbpf_ids_2d", [list(range(CANVAS))]),
        _make_int64("cbpf_ids_p1_2d", [list(range(1, CANVAS + 1))]),
        _make_int64("cbpf_big_2d", [[CANVAS] * CANVAS]),
        _make_int64("cbpf_zeros_2d", [[0] * CANVAS]),
        _make_float("cbpf_zero_mask_2d", [[0.0] * CANVAS]),
    ]
    nodes.append(helper.make_node("Greater", inputs=["cbpf_row_presence_sel", "cbpf_zero_mask_2d"], outputs=["cbpf_row_present_sel_b"]))
    nodes.append(helper.make_node("Where", inputs=["cbpf_row_present_sel_b", "cbpf_ids_2d", "cbpf_big_2d"], outputs=["cbpf_y0_candidates"]))
    reduce_min("cbpf_y0_candidates", "cbpf_y0", [1], keepdims=0)
    nodes.append(helper.make_node("Where", inputs=["cbpf_row_present_sel_b", "cbpf_ids_p1_2d", "cbpf_zeros_2d"], outputs=["cbpf_y1_candidates"]))
    reduce_max("cbpf_y1_candidates", "cbpf_y1", [1], keepdims=0)

    nodes.append(helper.make_node("Greater", inputs=["cbpf_col_presence_sel", "cbpf_zero_mask_2d"], outputs=["cbpf_col_present_sel_b"]))
    nodes.append(helper.make_node("Where", inputs=["cbpf_col_present_sel_b", "cbpf_ids_2d", "cbpf_big_2d"], outputs=["cbpf_x0_candidates"]))
    reduce_min("cbpf_x0_candidates", "cbpf_x0", [1], keepdims=0)
    nodes.append(helper.make_node("Where", inputs=["cbpf_col_present_sel_b", "cbpf_ids_p1_2d", "cbpf_zeros_2d"], outputs=["cbpf_x1_candidates"]))
    reduce_max("cbpf_x1_candidates", "cbpf_x1", [1], keepdims=0)

    inits += [
        _make_int64("cbpf_row_axis", [2]),
        _make_int64("cbpf_col_axis", [3]),
        _make_int64("cbpf_step", [1]),
    ]
    nodes.append(helper.make_node("Slice", inputs=["input", "cbpf_y0", "cbpf_y1", "cbpf_row_axis", "cbpf_step"], outputs=["cbpf_rows"]))
    nodes.append(helper.make_node("Slice", inputs=["cbpf_rows", "cbpf_x0", "cbpf_x1", "cbpf_col_axis", "cbpf_step"], outputs=["cbpf_crop"]))

    inits += [
        _make_int64("cbpf_fs", [2**31 - 1]),
        _make_int64("cbpf_fe", [-(2**31)]),
        _make_int64("cbpf_fa", [3]),
        _make_int64("cbpf_fst", [-1]),
    ]
    nodes.append(helper.make_node("Slice", inputs=["cbpf_crop", "cbpf_fs", "cbpf_fe", "cbpf_fa", "cbpf_fst"], outputs=["cbpf_flipped"]))

    nodes.append(helper.make_node("Sub", inputs=["cbpf_y1", "cbpf_y0"], outputs=["cbpf_h"]))
    nodes.append(helper.make_node("Sub", inputs=["cbpf_x1", "cbpf_x0"], outputs=["cbpf_w"]))
    inits += [
        _make_int64("cbpf_canvas_i", [CANVAS]),
        _make_int64("cbpf_pad_prefix", [0, 0, 0, 0, 0, 0]),
    ]
    nodes.append(helper.make_node("Sub", inputs=["cbpf_canvas_i", "cbpf_h"], outputs=["cbpf_bottom"]))
    nodes.append(helper.make_node("Sub", inputs=["cbpf_canvas_i", "cbpf_w"], outputs=["cbpf_right"]))
    nodes.append(helper.make_node("Concat", inputs=["cbpf_pad_prefix", "cbpf_bottom", "cbpf_right"], outputs=["cbpf_pads"], axis=0))
    nodes.append(helper.make_node("Pad", inputs=["cbpf_flipped", "cbpf_pads"], outputs=["output"], mode="constant"))

    graph = helper.make_graph(
        nodes,
        f"color_bbox_preserve_flip_{mode}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class ColorBBoxPreserveFlipSolver(BaseSolver):
    PRIORITY = 14

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("color_bbox_preserve_flip")) and analysis.get("color_bbox_preserve_flip_mode") in {"min_bbox", "max_bbox"}

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        mode = analysis.get("color_bbox_preserve_flip_mode")
        if mode not in {"min_bbox", "max_bbox"}:
            return None

        for split in ["train", "test", "arc-gen"]:
            for pair in task.get(split, []):
                if not detect_color_bbox_preserve_flip(grid_to_array(pair["input"]), grid_to_array(pair["output"]), mode):
                    return None

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{task_id}.onnx"
        try:
            save(_color_bbox_preserve_flip_net(mode), str(path), try_simplify=False)
            return path
        except Exception as e:
            print(f"    ColorBBoxPreserveFlipSolver({mode}) failed: {e}")
            return None
