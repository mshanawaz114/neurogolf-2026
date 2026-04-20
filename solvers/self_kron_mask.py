from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_float, _make_int64, _t, save


def _self_kron_mask_net(h: int, w: int) -> onnx.ModelProto:
    out_h = h * h
    out_w = w * w
    nodes = []
    inits = []
    tile_kernel = np.zeros((C, 1, out_h - h + 1, out_w - w + 1), dtype=np.float32)
    for c in range(C):
        for dy in range(h):
            for dx in range(w):
                tile_kernel[c, 0, dy * h, dx * w] = 1.0
    block_kernel = np.ones((1, 1, h, w), dtype=np.float32)

    inits += [
        _t("skm_tile_w", tile_kernel),
        _t("skm_block_w", block_kernel),
        _make_float("skm_one", [1.0]),
        _make_int64("skm_rs", [0]),
        _make_int64("skm_re", [h]),
        _make_int64("skm_ra", [2]),
        _make_int64("skm_rst", [1]),
        _make_int64("skm_cs", [0]),
        _make_int64("skm_ce", [w]),
        _make_int64("skm_ca", [3]),
        _make_int64("skm_cst", [1]),
        _make_int64("skm_bg_s", [0]),
        _make_int64("skm_bg_e", [1]),
        _make_int64("skm_bg_a", [1]),
        _make_int64("skm_bg_st", [1]),
    ]

    nodes.append(helper.make_node("Slice", inputs=["input", "skm_rs", "skm_re", "skm_ra", "skm_rst"], outputs=["skm_rows"]))
    nodes.append(helper.make_node("Slice", inputs=["skm_rows", "skm_cs", "skm_ce", "skm_ca", "skm_cst"], outputs=["skm_crop"]))

    # Tile the cropped input into a full 9x9 kron canvas.
    nodes.append(
        helper.make_node(
            "ConvTranspose",
            inputs=["skm_crop", "skm_tile_w"],
            outputs=["skm_tiled"],
            kernel_shape=[out_h - h + 1, out_w - w + 1],
            pads=[0, 0, 0, 0],
            group=C,
        )
    )

    # Build the non-zero mask from the background channel only.
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=["skm_crop", "skm_bg_s", "skm_bg_e", "skm_bg_a", "skm_bg_st"],
            outputs=["skm_bg_crop"],
        )
    )
    nodes.append(helper.make_node("Sub", inputs=["skm_one", "skm_bg_crop"], outputs=["skm_mask_crop"]))
    nodes.append(
        helper.make_node(
            "ConvTranspose",
            inputs=["skm_mask_crop", "skm_block_w"],
            outputs=["skm_mask_big"],
            kernel_shape=[h, w],
            pads=[0, 0, 0, 0],
            strides=[h, w],
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
            save(_self_kron_mask_net(h, w), str(path), try_simplify=False)
            return path
        except Exception as e:
            print(f"    SelfKronMaskSolver failed: {e}")
            return None
