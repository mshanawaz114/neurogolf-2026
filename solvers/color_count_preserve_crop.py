from __future__ import annotations

from pathlib import Path
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.arc_utils import detect_color_count_preserve_crop, grid_to_array
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_float, _make_int64, _t, save, static_crop_shift_nodes


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

    # Static crop-and-shift via MatMul — input [1,C,30,30] → need single channel for crop
    # For preserve_crop we crop ALL channels, not just selected color.
    # Apply shift to each channel: sum over channels of per-channel shifts.
    # Easier: reshape input to [C,30,30], apply P_rows @ x @ P_cols^T, reshape back.
    # We reuse static_crop_shift_nodes per-channel by treating [1,C,30,30] as [C] channels.

    # Reshape [1,C,30,30] -> [C,30,30]
    inits.append(_make_int64("ccp_shapeC2", [C, CANVAS, CANVAS]))
    nodes.append(helper.make_node("Reshape", inputs=["input", "ccp_shapeC2"], outputs=["ccp_inp_2d"]))

    # Build P_rows and P_cols inline using Relu arithmetic (avoids Equal/Less/And/Cast-from-bool)
    # P_rows[r,j] = Relu(1 - |jmr_f - y0_f|) * Relu(1 - Relu(j+1 - y1_f))
    # P_cols[c,k] = Relu(1 - |jmr_f - x0_f|) * Relu(1 - Relu(k+1 - x1_f))
    import numpy as np
    r_vec = np.arange(CANVAS, dtype=np.float32).reshape(CANVAS, 1)
    j_vec = np.arange(CANVAS, dtype=np.float32).reshape(1, CANVAS)
    j_minus_r_f = j_vec - r_vec          # [30,30] float
    j_p1_f = np.arange(1, CANVAS + 1, dtype=np.float32).reshape(1, CANVAS)  # [1,30]

    inits.append(_t("ccp_jmr_f", j_minus_r_f))
    inits.append(_t("ccp_jp1_f", j_p1_f))
    inits.append(_t("ccp_ones_f", np.ones((1, 1), dtype=np.float32)))

    inits.append(_make_int64("ccp_shape11", [1, 1]))
    # Cast y0/y1/x0/x1 (int64 [1]) → float → [1,1]
    for tag in ["y0", "y1", "x0", "x1"]:
        nodes.append(helper.make_node("Cast",
            inputs=[f"ccp_{tag}"], outputs=[f"ccp_{tag}_fflat"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Reshape",
            inputs=[f"ccp_{tag}_fflat", "ccp_shape11"], outputs=[f"ccp_{tag}_f"]))

    # P_rows: row indicator * j<y1 mask
    nodes.append(helper.make_node("Sub", inputs=["ccp_jmr_f", "ccp_y0_f"], outputs=["ccp_diff_r"]))
    nodes.append(helper.make_node("Abs", inputs=["ccp_diff_r"], outputs=["ccp_abs_r"]))
    nodes.append(helper.make_node("Sub", inputs=["ccp_ones_f", "ccp_abs_r"], outputs=["ccp_pr_pre"]))
    nodes.append(helper.make_node("Relu", inputs=["ccp_pr_pre"], outputs=["ccp_Pr_ind"]))

    nodes.append(helper.make_node("Sub", inputs=["ccp_jp1_f", "ccp_y1_f"], outputs=["ccp_jp1_m_y1"]))
    nodes.append(helper.make_node("Relu", inputs=["ccp_jp1_m_y1"], outputs=["ccp_relu_jp1_m_y1"]))
    nodes.append(helper.make_node("Sub", inputs=["ccp_ones_f", "ccp_relu_jp1_m_y1"], outputs=["ccp_lt_y1_raw"]))
    nodes.append(helper.make_node("Relu", inputs=["ccp_lt_y1_raw"], outputs=["ccp_lt_y1"]))
    nodes.append(helper.make_node("Mul", inputs=["ccp_Pr_ind", "ccp_lt_y1"], outputs=["ccp_Pr"]))

    # P_cols: col indicator * k<x1 mask
    nodes.append(helper.make_node("Sub", inputs=["ccp_jmr_f", "ccp_x0_f"], outputs=["ccp_diff_c"]))
    nodes.append(helper.make_node("Abs", inputs=["ccp_diff_c"], outputs=["ccp_abs_c"]))
    nodes.append(helper.make_node("Sub", inputs=["ccp_ones_f", "ccp_abs_c"], outputs=["ccp_pc_pre"]))
    nodes.append(helper.make_node("Relu", inputs=["ccp_pc_pre"], outputs=["ccp_Pc_ind"]))

    nodes.append(helper.make_node("Sub", inputs=["ccp_jp1_f", "ccp_x1_f"], outputs=["ccp_jp1_m_x1"]))
    nodes.append(helper.make_node("Relu", inputs=["ccp_jp1_m_x1"], outputs=["ccp_relu_jp1_m_x1"]))
    nodes.append(helper.make_node("Sub", inputs=["ccp_ones_f", "ccp_relu_jp1_m_x1"], outputs=["ccp_lt_x1_raw"]))
    nodes.append(helper.make_node("Relu", inputs=["ccp_lt_x1_raw"], outputs=["ccp_lt_x1"]))
    nodes.append(helper.make_node("Mul", inputs=["ccp_Pc_ind", "ccp_lt_x1"], outputs=["ccp_Pc"]))
    nodes.append(helper.make_node("Transpose", inputs=["ccp_Pc"], outputs=["ccp_Pc_T"], perm=[1, 0]))

    # Apply: P_rows @ inp_2d [C,30,30] -> [C,30,30]
    nodes.append(helper.make_node("MatMul", inputs=["ccp_Pr", "ccp_inp_2d"], outputs=["ccp_rows_sh"]))
    # Then: rows_sh [C,30,30] @ P_cols^T [30,30] -> [C,30,30]
    nodes.append(helper.make_node("MatMul", inputs=["ccp_rows_sh", "ccp_Pc_T"], outputs=["ccp_shifted"]))

    # Reshape back [C,30,30] -> [1,C,30,30]
    inits.append(_make_int64("ccp_shape_out", [1, C, CANVAS, CANVAS]))
    nodes.append(helper.make_node("Reshape", inputs=["ccp_shifted", "ccp_shape_out"], outputs=["output"]))

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
