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

    inits += [
        _make_float("ccc_zero_f", [0.0]),
        _make_float("ccc_one_f", [1.0]),
        _make_int64("ccc_shape1911", [1, C - 1, 1, 1]),
        _make_int64("ccc_shape1", [1]),
        _t("ccc_ones_canvas", np.ones((1, 1, CANVAS, CANVAS), dtype=np.float32)),
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
                inputs=["ccc_nonbg_counts", f"{out}_s", f"{out}_e", f"{out}_a", f"{out}_st"],
                outputs=[out],
            )
        )

    def build_selector(sel_prefix: str, sel_mode: str) -> str:
        indicators = []
        for idx in range(C - 1):
            cur = f"{sel_prefix}_count_{idx}"
            slice_count(idx, cur)
            nodes.append(helper.make_node("Clip", inputs=[cur, "ccc_zero_f", "ccc_one_f"], outputs=[f"{sel_prefix}_pos_{idx}"]))
            active = f"{sel_prefix}_pos_{idx}"
            for jdx in range(C - 1):
                if idx == jdx:
                    continue
                other = f"{sel_prefix}_count_{idx}_{jdx}"
                slice_count(jdx, other)
                if sel_mode == "max":
                    nodes.append(helper.make_node("Sub", inputs=[cur, other], outputs=[f"{sel_prefix}_diff_{idx}_{jdx}"]))
                    nodes.append(helper.make_node("Relu", inputs=[f"{sel_prefix}_diff_{idx}_{jdx}"], outputs=[f"{sel_prefix}_relu_{idx}_{jdx}"]))
                    nodes.append(
                        helper.make_node("Clip", inputs=[f"{sel_prefix}_relu_{idx}_{jdx}", "ccc_zero_f", "ccc_one_f"], outputs=[f"{sel_prefix}_gt_{idx}_{jdx}"])
                    )
                    nodes.append(helper.make_node("Equal", inputs=[cur, other], outputs=[f"{sel_prefix}_eqb_{idx}_{jdx}"]))
                    nodes.append(
                        helper.make_node("Cast", inputs=[f"{sel_prefix}_eqb_{idx}_{jdx}"], outputs=[f"{sel_prefix}_eq_{idx}_{jdx}"], to=TensorProto.FLOAT)
                    )
                    nodes.append(helper.make_node("Add", inputs=[f"{sel_prefix}_gt_{idx}_{jdx}", f"{sel_prefix}_eq_{idx}_{jdx}"], outputs=[f"{sel_prefix}_ge_raw_{idx}_{jdx}"]))
                    nodes.append(
                        helper.make_node("Clip", inputs=[f"{sel_prefix}_ge_raw_{idx}_{jdx}", "ccc_zero_f", "ccc_one_f"], outputs=[f"{sel_prefix}_ge_{idx}_{jdx}"])
                    )
                    cmp_name = f"{sel_prefix}_gt_{idx}_{jdx}" if jdx < idx else f"{sel_prefix}_ge_{idx}_{jdx}"
                else:
                    nodes.append(helper.make_node("Clip", inputs=[other, "ccc_zero_f", "ccc_one_f"], outputs=[f"{sel_prefix}_pos_{idx}_{jdx}"]))
                    nodes.append(helper.make_node("Sub", inputs=["ccc_one_f", f"{sel_prefix}_pos_{idx}_{jdx}"], outputs=[f"{sel_prefix}_other_absent_{idx}_{jdx}"]))
                    nodes.append(helper.make_node("Sub", inputs=[other, cur], outputs=[f"{sel_prefix}_diff_{idx}_{jdx}"]))
                    nodes.append(helper.make_node("Relu", inputs=[f"{sel_prefix}_diff_{idx}_{jdx}"], outputs=[f"{sel_prefix}_relu_{idx}_{jdx}"]))
                    nodes.append(
                        helper.make_node("Clip", inputs=[f"{sel_prefix}_relu_{idx}_{jdx}", "ccc_zero_f", "ccc_one_f"], outputs=[f"{sel_prefix}_lt_{idx}_{jdx}"])
                    )
                    nodes.append(helper.make_node("Equal", inputs=[cur, other], outputs=[f"{sel_prefix}_eqb_{idx}_{jdx}"]))
                    nodes.append(
                        helper.make_node("Cast", inputs=[f"{sel_prefix}_eqb_{idx}_{jdx}"], outputs=[f"{sel_prefix}_eq_{idx}_{jdx}"], to=TensorProto.FLOAT)
                    )
                    nodes.append(helper.make_node("Add", inputs=[f"{sel_prefix}_lt_{idx}_{jdx}", f"{sel_prefix}_eq_{idx}_{jdx}"], outputs=[f"{sel_prefix}_le_raw_{idx}_{jdx}"]))
                    nodes.append(
                        helper.make_node("Clip", inputs=[f"{sel_prefix}_le_raw_{idx}_{jdx}", "ccc_zero_f", "ccc_one_f"], outputs=[f"{sel_prefix}_le_{idx}_{jdx}"])
                    )
                    base_cmp = f"{sel_prefix}_lt_{idx}_{jdx}" if jdx < idx else f"{sel_prefix}_le_{idx}_{jdx}"
                    nodes.append(
                        helper.make_node("Add", inputs=[base_cmp, f"{sel_prefix}_other_absent_{idx}_{jdx}"], outputs=[f"{sel_prefix}_cmp_raw_{idx}_{jdx}"])
                    )
                    nodes.append(
                        helper.make_node("Clip", inputs=[f"{sel_prefix}_cmp_raw_{idx}_{jdx}", "ccc_zero_f", "ccc_one_f"], outputs=[f"{sel_prefix}_cmp_{idx}_{jdx}"])
                    )
                    cmp_name = f"{sel_prefix}_cmp_{idx}_{jdx}"
                next_active = f"{sel_prefix}_sel_{idx}_{jdx}"
                nodes.append(helper.make_node("Mul", inputs=[active, cmp_name], outputs=[next_active]))
                active = next_active
            indicators.append(active)
        selector_name = f"{sel_prefix}_selector"
        nodes.append(helper.make_node("Concat", inputs=indicators, outputs=[selector_name], axis=1))
        return selector_name

    if mode == "unique_extreme":
        max_selector = build_selector("ccc_max", "max")
        min_selector = build_selector("ccc_min", "min")
        nodes.append(helper.make_node("ReduceMax", inputs=["ccc_nonbg_counts"], outputs=["ccc_max_count"], axes=[1], keepdims=1))
        nodes.append(helper.make_node("Equal", inputs=["ccc_nonbg_counts", "ccc_max_count"], outputs=["ccc_is_max_b"]))
        nodes.append(helper.make_node("Cast", inputs=["ccc_is_max_b"], outputs=["ccc_is_max"], to=TensorProto.FLOAT))
        reduce_sum("ccc_is_max", "ccc_num_max", [1], keepdims=1)
        nodes.append(helper.make_node("Equal", inputs=["ccc_num_max", "ccc_one_f"], outputs=["ccc_unique_max_b"]))
        nodes.append(helper.make_node("Cast", inputs=["ccc_unique_max_b"], outputs=["ccc_unique_max"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Sub", inputs=["ccc_one_f", "ccc_unique_max"], outputs=["ccc_not_unique_max"]))
        nodes.append(helper.make_node("Mul", inputs=[max_selector, "ccc_unique_max"], outputs=["ccc_max_choice"]))
        nodes.append(helper.make_node("Mul", inputs=[min_selector, "ccc_not_unique_max"], outputs=["ccc_min_choice"]))
        nodes.append(helper.make_node("Add", inputs=["ccc_max_choice", "ccc_min_choice"], outputs=["ccc_selector"]))
    else:
        selector = build_selector("ccc", mode)
        if selector != "ccc_selector":
            nodes.append(helper.make_node("Identity", inputs=[selector], outputs=["ccc_selector"]))
    nodes.append(helper.make_node("Reshape", inputs=["ccc_selector", "ccc_shape1911"], outputs=["ccc_selector_4d"]))

    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["input", "ccc_cstart", "ccc_cend", "ccc_caxis", "ccc_cstep"],
            outputs=["ccc_nonbg_input"],
        )
    )
    nodes.append(helper.make_node("Mul", inputs=["ccc_nonbg_input", "ccc_selector_4d"], outputs=["ccc_selected_all"]))
    reduce_max("ccc_selected_all", "ccc_selected_mask", [1], keepdims=1)

    reduce_max("ccc_selected_mask", "ccc_row_presence", [1, 3], keepdims=0)
    reduce_max("ccc_selected_mask", "ccc_col_presence", [1, 2], keepdims=0)

    inits += [
        _make_int64("ccc_row_axis", [1]),
        _t("ccc_ids_f", np.arange(CANVAS, dtype=np.float32).reshape(1, CANVAS)),
        _t("ccc_ids_p1_f", np.arange(1, CANVAS + 1, dtype=np.float32).reshape(1, CANVAS)),
        _make_int64("ccc_rev_s", [2**31 - 1]),
        _make_int64("ccc_rev_e", [-(2**31)]),
        _make_int64("ccc_rev_a", [1]),
        _make_int64("ccc_rev_st", [-1]),
    ]

    nodes.append(helper.make_node("CumSum", inputs=["ccc_row_presence", "ccc_row_axis"], outputs=["ccc_row_csum"]))
    nodes.append(helper.make_node("Equal", inputs=["ccc_row_csum", "ccc_one_f"], outputs=["ccc_row_first_b"]))
    nodes.append(helper.make_node("Cast", inputs=["ccc_row_first_b"], outputs=["ccc_row_first_eq"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", inputs=["ccc_row_first_eq", "ccc_row_presence"], outputs=["ccc_row_first"]))
    nodes.append(helper.make_node("Mul", inputs=["ccc_row_first", "ccc_ids_f"], outputs=["ccc_y0_weighted"]))
    reduce_sum("ccc_y0_weighted", "ccc_y0_sum_f", [1], keepdims=0)
    nodes.append(helper.make_node("Reshape", inputs=["ccc_y0_sum_f", "ccc_shape1"], outputs=["ccc_y0_scalar_f"]))
    nodes.append(helper.make_node("Cast", inputs=["ccc_y0_scalar_f"], outputs=["ccc_y0"], to=TensorProto.INT64))

    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["ccc_row_presence", "ccc_rev_s", "ccc_rev_e", "ccc_rev_a", "ccc_rev_st"],
            outputs=["ccc_row_rev"],
        )
    )
    nodes.append(helper.make_node("CumSum", inputs=["ccc_row_rev", "ccc_row_axis"], outputs=["ccc_row_rev_csum"]))
    nodes.append(helper.make_node("Equal", inputs=["ccc_row_rev_csum", "ccc_one_f"], outputs=["ccc_row_last_b"]))
    nodes.append(helper.make_node("Cast", inputs=["ccc_row_last_b"], outputs=["ccc_row_last_eq"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", inputs=["ccc_row_last_eq", "ccc_row_rev"], outputs=["ccc_row_last_rev"]))
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["ccc_row_last_rev", "ccc_rev_s", "ccc_rev_e", "ccc_rev_a", "ccc_rev_st"],
            outputs=["ccc_row_last"],
        )
    )
    nodes.append(helper.make_node("Mul", inputs=["ccc_row_last", "ccc_ids_p1_f"], outputs=["ccc_y1_weighted"]))
    reduce_sum("ccc_y1_weighted", "ccc_y1_sum_f", [1], keepdims=0)
    nodes.append(helper.make_node("Reshape", inputs=["ccc_y1_sum_f", "ccc_shape1"], outputs=["ccc_y1_scalar_f"]))
    nodes.append(helper.make_node("Cast", inputs=["ccc_y1_scalar_f"], outputs=["ccc_y1"], to=TensorProto.INT64))

    nodes.append(helper.make_node("CumSum", inputs=["ccc_col_presence", "ccc_row_axis"], outputs=["ccc_col_csum"]))
    nodes.append(helper.make_node("Equal", inputs=["ccc_col_csum", "ccc_one_f"], outputs=["ccc_col_first_b"]))
    nodes.append(helper.make_node("Cast", inputs=["ccc_col_first_b"], outputs=["ccc_col_first_eq"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", inputs=["ccc_col_first_eq", "ccc_col_presence"], outputs=["ccc_col_first"]))
    nodes.append(helper.make_node("Mul", inputs=["ccc_col_first", "ccc_ids_f"], outputs=["ccc_x0_weighted"]))
    reduce_sum("ccc_x0_weighted", "ccc_x0_sum_f", [1], keepdims=0)
    nodes.append(helper.make_node("Reshape", inputs=["ccc_x0_sum_f", "ccc_shape1"], outputs=["ccc_x0_scalar_f"]))
    nodes.append(helper.make_node("Cast", inputs=["ccc_x0_scalar_f"], outputs=["ccc_x0"], to=TensorProto.INT64))

    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["ccc_col_presence", "ccc_rev_s", "ccc_rev_e", "ccc_rev_a", "ccc_rev_st"],
            outputs=["ccc_col_rev"],
        )
    )
    nodes.append(helper.make_node("CumSum", inputs=["ccc_col_rev", "ccc_row_axis"], outputs=["ccc_col_rev_csum"]))
    nodes.append(helper.make_node("Equal", inputs=["ccc_col_rev_csum", "ccc_one_f"], outputs=["ccc_col_last_b"]))
    nodes.append(helper.make_node("Cast", inputs=["ccc_col_last_b"], outputs=["ccc_col_last_eq"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Mul", inputs=["ccc_col_last_eq", "ccc_col_rev"], outputs=["ccc_col_last_rev"]))
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["ccc_col_last_rev", "ccc_rev_s", "ccc_rev_e", "ccc_rev_a", "ccc_rev_st"],
            outputs=["ccc_col_last"],
        )
    )
    nodes.append(helper.make_node("Mul", inputs=["ccc_col_last", "ccc_ids_p1_f"], outputs=["ccc_x1_weighted"]))
    reduce_sum("ccc_x1_weighted", "ccc_x1_sum_f", [1], keepdims=0)
    nodes.append(helper.make_node("Reshape", inputs=["ccc_x1_sum_f", "ccc_shape1"], outputs=["ccc_x1_scalar_f"]))
    nodes.append(helper.make_node("Cast", inputs=["ccc_x1_scalar_f"], outputs=["ccc_x1"], to=TensorProto.INT64))

    sn, si = static_crop_shift_nodes(
        "ccc_selected_mask", "ccc_mask_padded",
        "ccc_y0", "ccc_y1", "ccc_x0", "ccc_x1",
        prefix="ccc_sc",
    )
    nodes += sn
    inits += si

    sn, si = static_crop_shift_nodes(
        "ccc_ones_canvas", "ccc_support_padded",
        "ccc_y0", "ccc_y1", "ccc_x0", "ccc_x1",
        prefix="ccc_sup",
    )
    nodes += sn
    inits += si

    nodes.append(helper.make_node("Sub", inputs=["ccc_support_padded", "ccc_mask_padded"], outputs=["ccc_bg"]))
    nodes.append(helper.make_node("Mul", inputs=["ccc_selector_4d", "ccc_mask_padded"], outputs=["ccc_nonbg_out"]))
    nodes.append(helper.make_node("Concat", inputs=["ccc_bg", "ccc_nonbg_out"], outputs=["output"], axis=1))

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
        return bool(analysis.get("color_count_crop")) and analysis.get("color_count_crop_mode") in {"min", "max", "unique_extreme"}

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        mode = analysis.get("color_count_crop_mode")
        if mode not in {"min", "max", "unique_extreme"}:
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
