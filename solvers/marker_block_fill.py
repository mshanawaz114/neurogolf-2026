from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.arc_utils import detect_marker_block_fill, grid_to_array
from utils.onnx_builder import CANVAS, CHANNELS as C, _t, save


def _marker_block_fill_kernel() -> np.ndarray:
    # 1x1 conv that maps marker colors 1..4 to fill colors 6..9 and preserves 5.
    W = np.zeros((C, C, 1, 1), dtype=np.float32)
    for src in range(C):
        dst = src
        if 1 <= src <= 4:
            dst = src + 5
        W[dst, src, 0, 0] = 1.0
    return W


def _marker_block_fill_net() -> onnx.ModelProto:
    nodes = [
        helper.make_node(
            "Conv",
            inputs=["input", "mbf_W"],
            outputs=["mapped"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    ]

    # Kernel expands markers and separator color 5 locally.
    W = np.zeros((C, C, 3, 3), dtype=np.float32)
    for src in range(1, 5):
        W[src + 5, src, :, :] = 1.0
    W[5, 5, 1, 1] = 1.0

    zero = np.array([0.0], dtype=np.float32)
    nine = np.array([9.0], dtype=np.float32)
    inits = [_t("mbf_W", W), _t("mbf_lo", zero), _t("mbf_hi", nine)]
    nodes.append(helper.make_node("Clip", inputs=["mapped", "mbf_lo", "mbf_hi"], outputs=["output"]))

    graph = helper.make_graph(
        nodes,
        "marker_block_fill",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class MarkerBlockFillSolver(BaseSolver):
    PRIORITY = 16

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("marker_block_fill"))

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        for split in ["train", "test", "arc-gen"]:
            for pair in task.get(split, []):
                if not detect_marker_block_fill(grid_to_array(pair["input"]), grid_to_array(pair["output"])):
                    return None

        path = out_dir / f"{task_id}.onnx"
        save(_marker_block_fill_net(), str(path), try_simplify=False)
        return path
