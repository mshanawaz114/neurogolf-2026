from __future__ import annotations

from pathlib import Path
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.arc_utils import detect_fixed_submatrix, grid_to_array
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_int64, save


def _fixed_crop_net(y0: int, y1: int, x0: int, x1: int) -> onnx.ModelProto:
    out_h = y1 - y0
    out_w = x1 - x0
    nodes = []
    inits = [
        _make_int64("fc_rs", [y0]),
        _make_int64("fc_re", [y1]),
        _make_int64("fc_ra", [2]),
        _make_int64("fc_rst", [1]),
        _make_int64("fc_cs", [x0]),
        _make_int64("fc_ce", [x1]),
        _make_int64("fc_ca", [3]),
        _make_int64("fc_cst", [1]),
        _make_int64("fc_pad", [0, 0, 0, 0, 0, 0, CANVAS - out_h, CANVAS - out_w]),
    ]

    nodes.append(
        helper.make_node(
            "Slice", inputs=["input", "fc_rs", "fc_re", "fc_ra", "fc_rst"], outputs=["fc_rows"]
        )
    )
    nodes.append(
        helper.make_node(
            "Slice", inputs=["fc_rows", "fc_cs", "fc_ce", "fc_ca", "fc_cst"], outputs=["fc_crop"]
        )
    )
    nodes.append(
        helper.make_node("Pad", inputs=["fc_crop", "fc_pad"], outputs=["output"], mode="constant")
    )

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
