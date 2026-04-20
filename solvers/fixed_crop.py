from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.arc_utils import detect_fixed_submatrix, grid_to_array
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_int64, _t, save


def _fixed_crop_net(y0: int, y1: int, x0: int, x1: int) -> onnx.ModelProto:
    out_h = y1 - y0
    out_w = x1 - x0
    row_proj = np.zeros((CANVAS, CANVAS), dtype=np.float32)
    col_proj = np.zeros((CANVAS, CANVAS), dtype=np.float32)
    for out_y in range(out_h):
        row_proj[out_y, y0 + out_y] = 1.0
    for out_x in range(out_w):
        col_proj[x0 + out_x, out_x] = 1.0

    nodes = [
        helper.make_node("Reshape", inputs=["input", "fc_shape_cnn"], outputs=["fc_in2d"]),
        helper.make_node("MatMul", inputs=["fc_row_proj", "fc_in2d"], outputs=["fc_rows"]),
        helper.make_node("MatMul", inputs=["fc_rows", "fc_col_proj"], outputs=["fc_shifted"]),
        helper.make_node("Reshape", inputs=["fc_shifted", "fc_shape_1cnn"], outputs=["output"]),
    ]
    inits = [
        _t("fc_row_proj", row_proj),
        _t("fc_col_proj", col_proj),
        _make_int64("fc_shape_cnn", [C, CANVAS, CANVAS]),
        _make_int64("fc_shape_1cnn", [1, C, CANVAS, CANVAS]),
    ]

    graph = helper.make_graph(
        nodes,
        "fixed_crop",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class FixedCropSolver(BaseSolver):
    PRIORITY = 14

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("fixed_submatrix")) and analysis.get("fixed_submatrix_rect") is not None

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        rect = analysis.get("fixed_submatrix_rect")
        if rect is None:
            return None

        # Reconfirm against all available splits before building.
        candidates = []
        for split in ["train", "test", "arc-gen"]:
            for pair in task.get(split, []):
                det = detect_fixed_submatrix(grid_to_array(pair["input"]), grid_to_array(pair["output"]))
                if det is None:
                    return None
                candidates.append(tuple(int(v) for v in det))

        if len(set(candidates)) != 1:
            return None

        y0, y1, x0, x1 = candidates[0]
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{task_id}.onnx"
        try:
            save(_fixed_crop_net(y0, y1, x0, x1), str(path))
            return path
        except Exception as e:
            print(f"    FixedCropSolver failed: {e}")
            return None
