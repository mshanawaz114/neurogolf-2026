from __future__ import annotations

from pathlib import Path
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_int64, save


def _translate_net(dy: int, dx: int) -> onnx.ModelProto:
    nodes, inits = [], []

    row_start = max(0, -dy)
    row_end = CANVAS - max(0, dy)
    col_start = max(0, -dx)
    col_end = CANVAS - max(0, dx)

    inits += [
        _make_int64("tr_rs", [row_start]),
        _make_int64("tr_re", [row_end]),
        _make_int64("tr_ra", [2]),
        _make_int64("tr_rst", [1]),
    ]
    nodes.append(
        helper.make_node(
            "Slice", inputs=["input", "tr_rs", "tr_re", "tr_ra", "tr_rst"], outputs=["tr_rows"]
        )
    )

    inits += [
        _make_int64("tr_cs", [col_start]),
        _make_int64("tr_ce", [col_end]),
        _make_int64("tr_ca", [3]),
        _make_int64("tr_cst", [1]),
    ]
    nodes.append(
        helper.make_node(
            "Slice", inputs=["tr_rows", "tr_cs", "tr_ce", "tr_ca", "tr_cst"], outputs=["tr_crop"]
        )
    )

    pad_top = max(0, dy)
    pad_bottom = max(0, -dy)
    pad_left = max(0, dx)
    pad_right = max(0, -dx)
    inits.append(_make_int64("tr_pad", [0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right]))
    nodes.append(
        helper.make_node("Pad", inputs=["tr_crop", "tr_pad"], outputs=["output"], mode="constant")
    )

    graph = helper.make_graph(
        nodes,
        "translate",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class TranslateSolver(BaseSolver):
    PRIORITY = 12

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("translation")) and analysis.get("translation_delta") is not None

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        delta = analysis.get("translation_delta")
        if delta is None:
            return None
        dy, dx = delta
        path = out_dir / f"{task_id}.onnx"
        save(_translate_net(dy, dx), str(path))
        return path
