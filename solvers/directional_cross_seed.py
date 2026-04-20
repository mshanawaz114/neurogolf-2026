from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.arc_utils import detect_directional_cross_seed, grid_to_array
from utils.onnx_builder import CANVAS, CHANNELS as C, _t, save


def _directional_cross_seed_net() -> onnx.ModelProto:
    W = np.zeros((C, C, 3, 3), dtype=np.float32)
    seed = 1
    W[1, seed, 1, 1] = 1.0
    W[2, seed, 2, 1] = 1.0   # source one row below -> output up
    W[7, seed, 1, 2] = 1.0   # source one col right -> output left
    W[6, seed, 1, 0] = 1.0   # source one col left -> output right
    W[8, seed, 0, 1] = 1.0   # source one row above -> output down

    nodes = [
        helper.make_node(
            "Conv",
            inputs=["input", "dcs_W"],
            outputs=["conv_out"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Clip", inputs=["conv_out", "dcs_lo", "dcs_hi"], outputs=["output"]),
    ]
    inits = [
        _t("dcs_W", W),
        _t("dcs_lo", np.array([0.0], dtype=np.float32)),
        _t("dcs_hi", np.array([1.0], dtype=np.float32)),
    ]

    graph = helper.make_graph(
        nodes,
        "directional_cross_seed",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class DirectionalCrossSeedSolver(BaseSolver):
    PRIORITY = 16

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("directional_cross_seed"))

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        for split in ["train", "test", "arc-gen"]:
            for pair in task.get(split, []):
                if not detect_directional_cross_seed(grid_to_array(pair["input"]), grid_to_array(pair["output"])):
                    return None
        path = out_dir / f"{task_id}.onnx"
        save(_directional_cross_seed_net(), str(path), try_simplify=False)
        return path
