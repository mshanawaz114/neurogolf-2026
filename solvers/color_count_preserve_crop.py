from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.arc_utils import detect_color_count_preserve_crop, grid_to_array
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_float, _make_int64, _t, save


def _color_count_preserve_crop_net(mode: str) -> onnx.ModelProto:
    nodes = []
    inits = []

    def reduce_sum(inp: str, out: str, axes_vals: list[int], keepdims: int = 1):
        axes_name = f"{out}_axes"
        inits.append(_make_int64(axes_name, axes_vals))
        nodes.append(helper.make_node("ReduceSum", inputs=[inp, axes_name], outputs=[out], keepdims=keepdims))

    def reduce_max(inp: str, out: str, axes_vals: list[int], keepdims: int = 1):
        nodes.append(helper.make_node("ReduceMax", inputs=[inp], outputs=[out], axes=axes_vals, keepdims=keepdims))

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

    inits += [
        _make_float("ccp_zero_f", [0.0]),
        _make_float("ccp_one_f", [1.0]),
        _make_int64("ccp_shape1911", [1, C - 1, 1, 1]),
        _make_int64("ccp_shape1", [1]),
    ]

    def slice_count(idx: int, out: str) -> None:
        inits.extend([
            _make_int64(f"{out}_s", [idx]),
            _make_int64(f"{out}_e", [idx + 1]),
            _make_int64(f"{out}_a", [1]),
            _make_int64(f"{out}_st", [1]),
        ])
        nodes.append(
            helper.make_node(
                "Slice",
                inputs=["ccp_nonbg_counts", f"{out}_s", f"{out}_e", f"{out}_a", f"{out}_st"],
                outputs=[out],
            )
        )

    indicators = []
    for idx in range(C - 1):
        cur = f"ccp_count_{idx}"
        slice_count(idx, cur)
        nodes.append(helper.make_node("Clip", inputs=[cur, "ccp_zero_f", "ccp_one_f"], outputs=[f"ccp_pos_{idx}"]))
        active = f"ccp_pos_{idx}"
        for jdx in range(C - 1):
            if idx == jdx:
                continue
            other = f"ccp_count_{idx}_{jdx}"
            slice_count(jdx, other)
            if mode == "max":
                nodes.append(helper.make_node("Sub", inputs=[cur, other], outputs=[f"ccp_diff_{idx}_{jdx}"]))
                nodes.append(helper.make_node("Relu", inputs=[f"ccp_diff_{idx}_{jdx}"], outputs=[f"ccp_relu_{idx}_{jdx}"]))
                nodes.append(
                    helper.make_node("Clip", inputs=[f"ccp_relu_{idx}_{jdx}", "ccp_zero_f", "ccp_one_f"], outputs=[f"ccp_gt_{idx}_{jdx}"])
                )
                nodes.append(helper.make_node("Equal", inputs=[cur, other], outputs=[f"ccp_eqb_{idx}_{jdx}"]))
                nodes.append(
                    helper.make_node("Cast", inputs=[f"ccp_eqb_{idx}_{jdx}"], outputs=[f"ccp_eq_{idx}_{jdx}"], to=TensorProto.FLOAT)
                )
                nodes.append(helper.make_node("Add", inputs=[f"ccp_gt_{idx}_{jdx}", f"ccp_eq_{idx}_{jdx}"], outputs=[f"ccp_ge_raw_{idx}_{jdx}"]))
                nodes.append(
                    helper.make_node("Clip", inputs=[f"ccp_ge_raw_{idx}_{jdx}", "ccp_zero_f", "ccp_one_f"], outputs=[f"ccp_ge_{idx}_{jdx}"])
                )
                cmp_name = f"ccp_gt_{idx}_{jdx}" if jdx < idx else f"ccp_ge_{idx}_{jdx}"
            else:
                nodes.append(helper.make_node("Clip", inputs=[other, "ccp_zero_f", "ccp_one_f"], outputs=[f"ccp_pos_{idx}_{jdx}"]))
                nodes.append(helper.make_node("Sub", inputs=["ccp_one_f", f"ccp_pos_{idx}_{jdx}"], outputs=[f"ccp_other_absent_{idx}_{jdx}"]))
                nodes.append(helper.make_node("Sub", inputs=[other, cur], outputs=[f"ccp_diff_{idx}_{jdx}"]))
                nodes.append(helper.make_node("Relu", inputs=[f"ccp_diff_{idx}_{jdx}"], outputs=[f"ccp_relu_{idx}_{jdx}"]))
                nodes.append(
                    helper.make_node("Clip", inputs=[f"ccp_relu_{idx}_{jdx}", "ccp_zero_f", "ccp_one_f"], outputs=[f"ccp_lt_{idx}_{jdx}"])
                )
                nodes.append(helper.make_node("Equal", inputs=[cur, other], outputs=[f"ccp_eqb_{idx}_{jdx}"]))
                nodes.append(
                    helper.make_node("Cast", inputs=[f"ccp_eqb_{idx}_{jdx}"], outputs=[f"ccp_eq_{idx}_{jdx}"], to=TensorProto.FLOAT)
                )
                nodes.append(helper.make_node("Add", inputs=[f"ccp_lt_{idx}_{jdx}", f"ccp_eq_{idx}_{jdx}"], outputs=[f"ccp_le_raw_{idx}_{jdx}"]))
                nodes.append(
                    helper.make_node("Clip", inputs=[f"ccp_le_raw_{idx}_{jdx}", "ccp_zero_f", "ccp_one_f"], outputs=[f"ccp_le_{idx}_{jdx}"])
                )
                base_cmp = f"ccp_lt_{idx}_{jdx}" if jdx < idx else f"ccp_le_{idx}_{jdx}"
                nodes.append(
                    helper.make_node("Add", inputs=[base_cmp, f"ccp_other_absent_{idx}_{jdx}"], outputs=[f"ccp_cmp_raw_{idx}_{jdx}"])
                )
                nodes.append(
                    helper.make_node("Clip", inputs=[f"ccp_cmp_raw_{idx}_{jdx}", "ccp_zero_f", "ccp_one_f"], outputs=[f"ccp_cmp_{idx}_{jdx}"])
                )
                cmp_name = f"ccp_cmp_{idx}_{jdx}"
            next_active = f"ccp_sel_{idx}_{jdx}"
            nodes.append(helper.make_node("Mul", inputs=[active, cmp_name], outputs=[next_active]))
            active = next_active
        indicators.append(active)

    nodes.append(helper.make_node("Concat", inputs=indicators, outputs=["ccp_selector"], axis=1))
    nodes.append(helper.make_node("Reshape", inputs=["ccp_selector", "ccp_shape1911"], outputs=["ccp_selector_4d"]))

    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["input", "ccp_cstart", "ccp_cend", "ccp_caxis", "ccp_cstep"],
            outputs=["ccp_nonbg_input"],
        )
    )
    nodes.append(helper.make_node("Mul", inputs=["ccp_nonbg_input", "ccp_selector_4d"], outputs=["ccp_selected_all"]))
    reduce_max("ccp_selected_all", "ccp_selected_mask", [1], keepdims=1)

    reduce_max("ccp_selected_mask", "ccp_row_presence", [1, 3], keepdims=0)
    reduce_max("ccp_selected_mask", "ccp_col_presence", [1, 2], keepdims=0)

    inits += [
        _make_int64("ccp_row_axis", [1]),
        _t("ccp_ids_f", np.arange(CANVAS, dtype=np.float32).reshape(1, CANVAS)),
        _t("ccp_ids_p1_f", np.arange(1, CANVAS + 1, dtype=np.float32).reshape(1, CANVAS)),
        _make_int64("ccp_rev_s", [2**31 - 1]),
        _make_int64("ccp_rev_e", [-(2**31)]),
        _make_int64("ccp_rev_a", [1]),
        _make_int64("ccp_rev_st", [-1]),
    ]

    nodes.append(helper.make_node("CumSum", inputs=["ccp_row_presence", "ccp_row_axis"], outputs=["ccp_row_csum"]))
    nodes.append(helper.make_node("Equal", inputs=["ccp_row_csum", "ccp_one_f"], outputs=["ccp_row_first_b"]))
    nodes.append(helper.make_node("Cast", inputs=["ccp_row_first_b"], outputs=["ccp_row_first_eq"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", inputs=["ccp_row_first_eq", "ccp_row_presence"], outputs=["ccp_row_first"]))
    nodes.append(helper.make_node("Mul", inputs=["ccp_row_first", "ccp_ids_f"], outputs=["ccp_y0_weighted"]))
    reduce_sum("ccp_y0_weighted", "ccp_y0_sum_f", [1], keepdims=0)
    nodes.append(helper.make_node("Reshape", inputs=["ccp_y0_sum_f", "ccp_shape1"], outputs=["ccp_y0_scalar_f"]))
    nodes.append(helper.make_node("Cast", inputs=["ccp_y0_scalar_f"], outputs=["ccp_y0"], to=TensorProto.INT64))

    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["ccp_row_presence", "ccp_rev_s", "ccp_rev_e", "ccp_rev_a", "ccp_rev_st"],
            outputs=["ccp_row_rev"],
        )
    )
    nodes.append(helper.make_node("CumSum", inputs=["ccp_row_rev", "ccp_row_axis"], outputs=["ccp_row_rev_csum"]))
    nodes.append(helper.make_node("Equal", inputs=["ccp_row_rev_csum", "ccp_one_f"], outputs=["ccp_row_last_b"]))
    nodes.append(helper.make_node("Cast", inputs=["ccp_row_last_b"], outputs=["ccp_row_last_eq"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", inputs=["ccp_row_last_eq", "ccp_row_rev"], outputs=["ccp_row_last_rev"]))
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["ccp_row_last_rev", "ccp_rev_s", "ccp_rev_e", "ccp_rev_a", "ccp_rev_st"],
            outputs=["ccp_row_last"],
        )
    )
    nodes.append(helper.make_node("Mul", inputs=["ccp_row_last", "ccp_ids_p1_f"], outputs=["ccp_y1_weighted"]))
    reduce_sum("ccp_y1_weighted", "ccp_y1_sum_f", [1], keepdims=0)
    nodes.append(helper.make_node("Reshape", inputs=["ccp_y1_sum_f", "ccp_shape1"], outputs=["ccp_y1_scalar_f"]))
    nodes.append(helper.make_node("Cast", inputs=["ccp_y1_scalar_f"], outputs=["ccp_y1"], to=TensorProto.INT64))

    nodes.append(helper.make_node("CumSum", inputs=["ccp_col_presence", "ccp_row_axis"], outputs=["ccp_col_csum"]))
    nodes.append(helper.make_node("Equal", inputs=["ccp_col_csum", "ccp_one_f"], outputs=["ccp_col_first_b"]))
    nodes.append(helper.make_node("Cast", inputs=["ccp_col_first_b"], outputs=["ccp_col_first_eq"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", inputs=["ccp_col_first_eq", "ccp_col_presence"], outputs=["ccp_col_first"]))
    nodes.append(helper.make_node("Mul", inputs=["ccp_col_first", "ccp_ids_f"], outputs=["ccp_x0_weighted"]))
    reduce_sum("ccp_x0_weighted", "ccp_x0_sum_f", [1], keepdims=0)
    nodes.append(helper.make_node("Reshape", inputs=["ccp_x0_sum_f", "ccp_shape1"], outputs=["ccp_x0_scalar_f"]))
    nodes.append(helper.make_node("Cast", inputs=["ccp_x0_scalar_f"], outputs=["ccp_x0"], to=TensorProto.INT64))

    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["ccp_col_presence", "ccp_rev_s", "ccp_rev_e", "ccp_rev_a", "ccp_rev_st"],
            outputs=["ccp_col_rev"],
        )
    )
    nodes.append(helper.make_node("CumSum", inputs=["ccp_col_rev", "ccp_row_axis"], outputs=["ccp_col_rev_csum"]))
    nodes.append(helper.make_node("Equal", inputs=["ccp_col_rev_csum", "ccp_one_f"], outputs=["ccp_col_last_b"]))
    nodes.append(helper.make_node("Cast", inputs=["ccp_col_last_b"], outputs=["ccp_col_last_eq"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", inputs=["ccp_col_last_eq", "ccp_col_rev"], outputs=["ccp_col_last_rev"]))
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["ccp_col_last_rev", "ccp_rev_s", "ccp_rev_e", "ccp_rev_a", "ccp_rev_st"],
            outputs=["ccp_col_last"],
        )
    )
    nodes.append(helper.make_node("Mul", inputs=["ccp_col_last", "ccp_ids_p1_f"], outputs=["ccp_x1_weighted"]))
    reduce_sum("ccp_x1_weighted", "ccp_x1_sum_f", [1], keepdims=0)
    nodes.append(helper.make_node("Reshape", inputs=["ccp_x1_sum_f", "ccp_shape1"], outputs=["ccp_x1_scalar_f"]))
    nodes.append(helper.make_node("Cast", inputs=["ccp_x1_scalar_f"], outputs=["ccp_x1"], to=TensorProto.INT64))

    # Static crop-and-shift via MatMul on the full input tensor.
    inits.append(_make_int64("ccp_shapeC2", [C, CANVAS, CANVAS]))
    nodes.append(helper.make_node("Reshape", inputs=["input", "ccp_shapeC2"], outputs=["ccp_inp_2d"]))
    r_vec = np.arange(CANVAS, dtype=np.float32).reshape(CANVAS, 1)
    j_vec = np.arange(CANVAS, dtype=np.float32).reshape(1, CANVAS)
    j_minus_r_f = j_vec - r_vec
    j_p1_f = np.arange(1, CANVAS + 1, dtype=np.float32).reshape(1, CANVAS)

    inits.append(_t("ccp_jmr_f", j_minus_r_f))
    inits.append(_t("ccp_jp1_f", j_p1_f))
    inits.append(_t("ccp_ones_f", np.ones((1, 1), dtype=np.float32)))

    inits.append(_make_int64("ccp_shape11", [1, 1]))
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

    nodes.append(helper.make_node("MatMul", inputs=["ccp_Pr", "ccp_inp_2d"], outputs=["ccp_rows_sh"]))
    nodes.append(helper.make_node("MatMul", inputs=["ccp_rows_sh", "ccp_Pc_T"], outputs=["ccp_shifted"]))
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
