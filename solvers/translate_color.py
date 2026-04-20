from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from solvers.base import BaseSolver
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_int64, save


def _color_weights(mapping: dict[int, int]) -> onnx.TensorProto:
    weights = np.zeros((C, C, 1, 1), dtype=np.float32)
    full_mapping = {c: c for c in range(C)}
    full_mapping.update(mapping)
    for src in range(C):
        weights[full_mapping[src], src, 0, 0] = 1.0
    return numpy_helper.from_array(weights, name="tc_W")


def _translate_color_net(dy: int, dx: int, mapping: dict[int, int]) -> onnx.ModelProto:
    nodes, inits = [], []

    row_start = max(0, -dy)
    row_end = CANVAS - max(0, dy)
    col_start = max(0, -dx)
    col_end = CANVAS - max(0, dx)

    inits += [
        _make_int64("tc_rs", [row_start]),
        _make_int64("tc_re", [row_end]),
        _make_int64("tc_ra", [2]),
        _make_int64("tc_rst", [1]),
    ]
    nodes.append(helper.make_node("Slice", inputs=["input", "tc_rs", "tc_re", "tc_ra", "tc_rst"], outputs=["tc_rows"]))

    inits += [
        _make_int64("tc_cs", [col_start]),
        _make_int64("tc_ce", [col_end]),
        _make_int64("tc_ca", [3]),
        _make_int64("tc_cst", [1]),
    ]
    nodes.append(helper.make_node("Slice", inputs=["tc_rows", "tc_cs", "tc_ce", "tc_ca", "tc_cst"], outputs=["tc_crop"]))

    pad_top = max(0, dy)
    pad_bottom = max(0, -dy)
    pad_left = max(0, dx)
    pad_right = max(0, -dx)
    inits += [
        _make_int64("tc_pad", [0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right]),
        _color_weights(mapping),
    ]
    nodes.append(helper.make_node("Pad", inputs=["tc_crop", "tc_pad"], outputs=["tc_shifted"], mode="constant"))
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=["tc_shifted", "tc_W"],
            outputs=["output"],
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
        )
    )

    graph = helper.make_graph(
        nodes,
        "translate_color",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class TranslateColorSolver(BaseSolver):
    PRIORITY = 12

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("translation_color")) and analysis.get("translation_color_delta") is not None

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        delta = analysis.get("translation_color_delta")
        mapping = analysis.get("translation_color_mapping")
        if delta is None or mapping is None:
            return None
        dy, dx = delta
        path = out_dir / f"{task_id}.onnx"
        save(_translate_color_net(dy, dx, mapping), str(path))
        return path
