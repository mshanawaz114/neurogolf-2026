from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.arc_utils import detect_color_count_crop, grid_to_array
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_float, _make_int64, _t, save, static_crop_shift_nodes


def _color_count_crop_net(mode: str) -> onnx.ModelProto:
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

    reduce_sum("input", "ccc_counts", [2, 3], keepdims=0)
    inits += [
        _make_int64("ccc_cstart", [1]),
        _make_int64("ccc_cend", [C]),
        _make_int64("ccc_caxis", [1]),
        _make_int64("ccc_cstep", [1]),
    ]
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["ccc_counts", "ccc_cstart", "ccc_cend", "ccc_caxis", "ccc_cstep"],
            outputs=["ccc_nonbg_counts"],
        )
    )

    if mode == "max":
        nodes.append(helper.make_node("ArgMax", inputs=["ccc_nonbg_counts"], outputs=["ccc_idx0"], axis=1, keepdims=0))
    else:
        inits.append(_make_float("ccc_zero_f", [0.0]))
        inits.append(_make_float("ccc_big_f", np.full((1, C - 1), 1e9, dtype=np.float32)))
        nodes.append(helper.make_node("Greater", inputs=["ccc_nonbg_counts", "ccc_zero_f"], outputs=["ccc_present_b"]))
        nodes.append(
            helper.make_node(
                "Where", inputs=["ccc_present_b", "ccc_nonbg_counts", "ccc_big_f"], outputs=["ccc_masked_counts"]
            )
        )
        nodes.append(
            helper.make_node("ArgMin", inputs=["ccc_masked_counts"], outputs=["ccc_idx0"], axis=1, keepdims=0)
        )

    inits.append(_make_int64("ccc_one_i", [1]))
    nodes.append(helper.make_node("Add", inputs=["ccc_idx0", "ccc_one_i"], outputs=["ccc_color_idx"]))
    nodes.append(helper.make_node("Gather", inputs=["input", "ccc_color_idx"], outputs=["ccc_sel"], axis=1))

    reduce_max("ccc_sel", "ccc_sel_hw", [1], keepdims=0)
    reduce_max("ccc_sel_hw", "ccc_row_presence", [2], keepdims=0)
    reduce_max("ccc_sel_hw", "ccc_col_presence", [1], keepdims=0)

    inits += [
        _make_float("ccc_zero_mask", np.zeros((1, CANVAS), dtype=np.float32)),
        _t("ccc_row_ids", np.arange(CANVAS, dtype=np.int64).reshape(1, CANVAS)),
        _t("ccc_row_ids_p1", np.arange(1, CANVAS + 1, dtype=np.int64).reshape(1, CANVAS)),
        _make_int64("ccc_row_zeros", np.zeros((1, CANVAS), dtype=np.int64)),
        _make_int64("ccc_row_big", np.full((1, CANVAS), CANVAS, dtype=np.int64)),
    ]
    nodes.append(helper.make_node("Greater", inputs=["ccc_row_presence", "ccc_zero_mask"], outputs=["ccc_row_present_b"]))
    nodes.append(
        helper.make_node(
            "Where", inputs=["ccc_row_present_b", "ccc_row_ids", "ccc_row_big"], outputs=["ccc_row_first_candidates"]
        )
    )
    reduce_min("ccc_row_first_candidates", "ccc_y0", [1], keepdims=0)
    nodes.append(
        helper.make_node(
            "Where", inputs=["ccc_row_present_b", "ccc_row_ids_p1", "ccc_row_zeros"], outputs=["ccc_row_last_candidates"]
        )
    )
    reduce_max("ccc_row_last_candidates", "ccc_y1", [1], keepdims=0)

    nodes.append(helper.make_node("Greater", inputs=["ccc_col_presence", "ccc_zero_mask"], outputs=["ccc_col_present_b"]))
    nodes.append(
        helper.make_node(
            "Where", inputs=["ccc_col_present_b", "ccc_row_ids", "ccc_row_big"], outputs=["ccc_col_first_candidates"]
        )
    )
    reduce_min("ccc_col_first_candidates", "ccc_x0", [1], keepdims=0)
    nodes.append(
        helper.make_node(
            "Where", inputs=["ccc_col_present_b", "ccc_row_ids_p1", "ccc_row_zeros"], outputs=["ccc_col_last_candidates"]
        )
    )
    reduce_max("ccc_col_last_candidates", "ccc_x1", [1], keepdims=0)

    # Static crop-and-shift via MatMul (all shapes stay [30,30])
    sn, si = static_crop_shift_nodes(
        "ccc_sel", "ccc_mask_padded",
        "ccc_y0", "ccc_y1", "ccc_x0", "ccc_x1",
        prefix="ccc_sc",
    )
    nodes += sn
    inits += si

    # One-hot color mask using Relu arithmetic (avoids Equal + Cast-from-bool)
    inits += [
        _make_float("ccc_color_ids_f", list(range(C))),   # [0.0, 1.0, ..., 9.0]
        _make_float("ccc_color_ones_f", [1.0]),             # scalar 1.0
        _make_int64("ccc_color_shape", [1, C, 1, 1]),
    ]
    nodes.append(helper.make_node("Cast", inputs=["ccc_color_idx"], outputs=["ccc_color_idx_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Sub", inputs=["ccc_color_ids_f", "ccc_color_idx_f"], outputs=["ccc_color_diff"]))
    nodes.append(helper.make_node("Abs", inputs=["ccc_color_diff"], outputs=["ccc_color_abs"]))
    nodes.append(helper.make_node("Sub", inputs=["ccc_color_ones_f", "ccc_color_abs"], outputs=["ccc_color_hot_raw"]))
    nodes.append(helper.make_node("Relu", inputs=["ccc_color_hot_raw"], outputs=["ccc_color_f"]))
    nodes.append(helper.make_node("Reshape", inputs=["ccc_color_f", "ccc_color_shape"], outputs=["ccc_color_mask"]))
    nodes.append(helper.make_node("Mul", inputs=["ccc_color_mask", "ccc_mask_padded"], outputs=["output"]))

    graph = helper.make_graph(
        nodes,
        f"color_count_crop_{mode}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class ColorCountCropSolver(BaseSolver):
    PRIORITY = 14

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("color_count_crop")) and analysis.get("color_count_crop_mode") in {"min", "max"}

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        mode = analysis.get("color_count_crop_mode")
        if mode not in {"min", "max"}:
            return None

        for split in ["train", "test", "arc-gen"]:
            for pair in task.get(split, []):
                if not detect_color_count_crop(grid_to_array(pair["input"]), grid_to_array(pair["output"]), mode):
                    return None

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{task_id}.onnx"
        try:
            save(_color_count_crop_net(mode), str(path), try_simplify=False)
            return path
        except Exception as e:
            print(f"    ColorCountCropSolver({mode}) failed: {e}")
            return None
