from __future__ import annotations

from pathlib import Path
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_float, _make_int64, save


def _self_kron_mask_net(h: int, w: int) -> onnx.ModelProto:
    out_h = h * h
    out_w = w * w
    nodes = []
    inits = []

    # Crop the original input grid.
    inits += [
        _make_int64("skm_rs", [0]),
        _make_int64("skm_re", [h]),
        _make_int64("skm_ra", [2]),
        _make_int64("skm_rst", [1]),
        _make_int64("skm_cs", [0]),
        _make_int64("skm_ce", [w]),
        _make_int64("skm_ca", [3]),
        _make_int64("skm_cst", [1]),
    ]
    nodes.append(helper.make_node("Slice", inputs=["input", "skm_rs", "skm_re", "skm_ra", "skm_rst"], outputs=["skm_rows"]))
    nodes.append(helper.make_node("Slice", inputs=["skm_rows", "skm_cs", "skm_ce", "skm_ca", "skm_cst"], outputs=["skm_crop"]))

    # Tile the cropped input h x w times.
    row_tiles = []
    for i in range(w):
        row_tiles.append("skm_crop")
    nodes.append(helper.make_node("Concat", inputs=row_tiles, outputs=["skm_row_tile"], axis=3))
    col_tiles = []
    for i in range(h):
        col_tiles.append("skm_row_tile")
    nodes.append(helper.make_node("Concat", inputs=col_tiles, outputs=["skm_tiled"], axis=2))

    # Build the non-zero mask from channels 1..9 and resize it blockwise.
    inits += [
        _make_int64("skm_nzs", [1]),
        _make_int64("skm_nze", [C]),
        _make_int64("skm_nza", [1]),
        _make_int64("skm_nzst", [1]),
    ]
    nodes.append(helper.make_node("Slice", inputs=["input", "skm_nzs", "skm_nze", "skm_nza", "skm_nzst"], outputs=["skm_nonbg"]))
    nodes.append(helper.make_node("ReduceMax", inputs=["skm_nonbg"], outputs=["skm_mask_full"], axes=[1], keepdims=1))
    nodes.append(helper.make_node("Slice", inputs=["skm_mask_full", "skm_rs", "skm_re", "skm_ra", "skm_rst"], outputs=["skm_mask_rows"]))
    nodes.append(helper.make_node("Slice", inputs=["skm_mask_rows", "skm_cs", "skm_ce", "skm_ca", "skm_cst"], outputs=["skm_mask_crop"]))

    inits.append(_make_float("skm_roi", []))
    inits.append(_make_float("skm_scales", []))
    inits.append(_make_int64("skm_sizes", [1, 1, out_h, out_w]))
    nodes.append(
        helper.make_node(
            "Resize",
            inputs=["skm_mask_crop", "skm_roi", "skm_scales", "skm_sizes"],
            outputs=["skm_mask_big"],
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )
    )

    nodes.append(helper.make_node("Mul", inputs=["skm_tiled", "skm_mask_big"], outputs=["skm_out_crop"]))

    inits.append(_make_int64("skm_pad", [0, 0, 0, 0, 0, 0, CANVAS - out_h, CANVAS - out_w]))
    nodes.append(helper.make_node("Pad", inputs=["skm_out_crop", "skm_pad"], outputs=["output"], mode="constant"))

    graph = helper.make_graph(
        nodes,
        "self_kron_mask",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class SelfKronMaskSolver(BaseSolver):
    PRIORITY = 13

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("self_kron_mask"))

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        train_pairs = task.get("train", [])
        if not train_pairs:
            return None
        h = len(train_pairs[0]["input"])
        w = len(train_pairs[0]["input"][0])
        if h * h > CANVAS or w * w > CANVAS:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{task_id}.onnx"
        try:
            save(_self_kron_mask_net(h, w), str(path))
            return path
        except Exception as e:
            print(f"    SelfKronMaskSolver failed: {e}")
            return None
