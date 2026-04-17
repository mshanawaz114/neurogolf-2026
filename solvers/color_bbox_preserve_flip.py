from __future__ import annotations

from pathlib import Path
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.arc_utils import detect_color_bbox_preserve_flip, grid_to_array
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_float, _make_int64, _t, save, static_crop_flip_shift_nodes


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

    # Static crop + horizontal flip + shift to top-left on ALL 10 channels.
    # Reshape [1,C,30,30] -> [C,30,30], apply P_rows @ x @ P_flip_cols^T, reshape back.
    # Uses Relu arithmetic (avoids Equal/Less/And/Not/Cast-from-bool).
    import numpy as np
    N = CANVAS
    r_vec = np.arange(N, dtype=np.float32).reshape(N, 1)
    j_vec = np.arange(N, dtype=np.float32).reshape(1, N)
    j_minus_r_f = j_vec - r_vec           # [30,30] float
    c_plus_k_f  = r_vec + j_vec           # [30,30] float
    j_p1_f = np.arange(1, N + 1, dtype=np.float32).reshape(1, N)  # [1,30]

    inits.append(_t("cbpf_jmr_f",  j_minus_r_f))
    inits.append(_t("cbpf_cpk_f",  c_plus_k_f))
    inits.append(_t("cbpf_jp1_f",  j_p1_f))
    inits.append(_t("cbpf_jvec_f", j_vec))          # [1,30]: k values
    inits.append(_t("cbpf_ones_f", np.ones((1, 1), dtype=np.float32)))

    inits.append(_make_int64("cbpf_shape11", [1, 1]))
    # Cast y0/y1/x0/x1 (int64 [1]) → float → [1,1]
    for tag in ["y0", "y1", "x0", "x1"]:
        nodes.append(helper.make_node("Cast",
            inputs=[f"cbpf_{tag}"], outputs=[f"cbpf_{tag}_fflat"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Reshape",
            inputs=[f"cbpf_{tag}_fflat", "cbpf_shape11"], outputs=[f"cbpf_{tag}_f"]))

    # x1 - 1.0 for flip column formula
    nodes.append(helper.make_node("Sub",
        inputs=["cbpf_x1_f", "cbpf_ones_f"], outputs=["cbpf_x1m1_f"]))

    # P_rows: Relu(1 - |jmr_f - y0_f|) * Relu(1 - Relu(j+1 - y1_f))
    nodes.append(helper.make_node("Sub", inputs=["cbpf_jmr_f", "cbpf_y0_f"], outputs=["cbpf_diff_r"]))
    nodes.append(helper.make_node("Abs", inputs=["cbpf_diff_r"], outputs=["cbpf_abs_r"]))
    nodes.append(helper.make_node("Sub", inputs=["cbpf_ones_f", "cbpf_abs_r"], outputs=["cbpf_pr_pre"]))
    nodes.append(helper.make_node("Relu", inputs=["cbpf_pr_pre"], outputs=["cbpf_Pr_ind"]))

    nodes.append(helper.make_node("Sub", inputs=["cbpf_jp1_f", "cbpf_y1_f"], outputs=["cbpf_jp1_m_y1"]))
    nodes.append(helper.make_node("Relu", inputs=["cbpf_jp1_m_y1"], outputs=["cbpf_relu_jp1_m_y1"]))
    nodes.append(helper.make_node("Sub", inputs=["cbpf_ones_f", "cbpf_relu_jp1_m_y1"], outputs=["cbpf_lt_y1_raw"]))
    nodes.append(helper.make_node("Relu", inputs=["cbpf_lt_y1_raw"], outputs=["cbpf_lt_y1"]))
    nodes.append(helper.make_node("Mul", inputs=["cbpf_Pr_ind", "cbpf_lt_y1"], outputs=["cbpf_Pr"]))

    # P_flip_cols: Relu(1 - |cpk_f - x1m1_f|) * Relu(1-Relu(x0_f-jvec_f)) * Relu(1-Relu(j+1-x1_f))
    # (c+k == x1-1) indicator
    nodes.append(helper.make_node("Sub", inputs=["cbpf_cpk_f", "cbpf_x1m1_f"], outputs=["cbpf_diff_ck"]))
    nodes.append(helper.make_node("Abs", inputs=["cbpf_diff_ck"], outputs=["cbpf_abs_ck"]))
    nodes.append(helper.make_node("Sub", inputs=["cbpf_ones_f", "cbpf_abs_ck"], outputs=["cbpf_eq_ck_raw"]))
    nodes.append(helper.make_node("Relu", inputs=["cbpf_eq_ck_raw"], outputs=["cbpf_eq_ck"]))  # [30,30]

    # k >= x0: Relu(1 - Relu(x0_f - jvec_f))
    nodes.append(helper.make_node("Sub", inputs=["cbpf_x0_f", "cbpf_jvec_f"], outputs=["cbpf_x0_m_k"]))
    nodes.append(helper.make_node("Relu", inputs=["cbpf_x0_m_k"], outputs=["cbpf_relu_x0_m_k"]))
    nodes.append(helper.make_node("Sub", inputs=["cbpf_ones_f", "cbpf_relu_x0_m_k"], outputs=["cbpf_ge_x0_raw"]))
    nodes.append(helper.make_node("Relu", inputs=["cbpf_ge_x0_raw"], outputs=["cbpf_ge_x0"]))  # [1,30]

    # k < x1: Relu(1 - Relu(j+1 - x1_f))
    nodes.append(helper.make_node("Sub", inputs=["cbpf_jp1_f", "cbpf_x1_f"], outputs=["cbpf_jp1_m_x1"]))
    nodes.append(helper.make_node("Relu", inputs=["cbpf_jp1_m_x1"], outputs=["cbpf_relu_jp1_m_x1"]))
    nodes.append(helper.make_node("Sub", inputs=["cbpf_ones_f", "cbpf_relu_jp1_m_x1"], outputs=["cbpf_lt_x1_raw"]))
    nodes.append(helper.make_node("Relu", inputs=["cbpf_lt_x1_raw"], outputs=["cbpf_lt_x1"]))  # [1,30]

    # Combine
    nodes.append(helper.make_node("Mul", inputs=["cbpf_ge_x0", "cbpf_lt_x1"], outputs=["cbpf_k_range"]))
    nodes.append(helper.make_node("Mul", inputs=["cbpf_eq_ck", "cbpf_k_range"], outputs=["cbpf_Pfc"]))
    nodes.append(helper.make_node("Transpose", inputs=["cbpf_Pfc"], outputs=["cbpf_Pfc_T"], perm=[1, 0]))

    # Reshape input [1,C,30,30] -> [C,30,30]
    inits.append(_make_int64("cbpf_shC2", [C, N, N]))
    nodes.append(helper.make_node("Reshape", inputs=["input", "cbpf_shC2"], outputs=["cbpf_inp_2d"]))
    nodes.append(helper.make_node("MatMul", inputs=["cbpf_Pr", "cbpf_inp_2d"], outputs=["cbpf_rows_sh"]))
    nodes.append(helper.make_node("MatMul", inputs=["cbpf_rows_sh", "cbpf_Pfc_T"], outputs=["cbpf_shifted"]))

    inits.append(_make_int64("cbpf_sh_out", [1, C, N, N]))
    nodes.append(helper.make_node("Reshape", inputs=["cbpf_shifted", "cbpf_sh_out"], outputs=["output"]))

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
