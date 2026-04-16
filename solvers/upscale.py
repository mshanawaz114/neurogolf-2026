from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_float, _make_int64, save


def _upscale_net(sy: int, sx: int) -> onnx.ModelProto:
    scaled_h = CANVAS * sy
    scaled_w = CANVAS * sx
    nodes, inits = [], []

    inits.append(_make_float("up_roi", np.array([], dtype=np.float32)))
    inits.append(_make_float("up_scales", np.array([], dtype=np.float32)))
    inits.append(_make_int64("up_sizes", [1, C, scaled_h, scaled_w]))
    nodes.append(
        helper.make_node(
            "Resize",
            inputs=["input", "up_roi", "up_scales", "up_sizes"],
            outputs=["up_resized"],
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )
    )
    inits += [
        _make_int64("up_rs", [0]),
        _make_int64("up_re", [CANVAS]),
        _make_int64("up_ra", [2]),
        _make_int64("up_rst", [1]),
    ]
    nodes.append(
        helper.make_node(
            "Slice", inputs=["up_resized", "up_rs", "up_re", "up_ra", "up_rst"], outputs=["up_rows"]
        )
    )
    inits += [
        _make_int64("up_cs", [0]),
        _make_int64("up_ce", [CANVAS]),
        _make_int64("up_ca", [3]),
        _make_int64("up_cst", [1]),
    ]
    nodes.append(
        helper.make_node(
            "Slice", inputs=["up_rows", "up_cs", "up_ce", "up_ca", "up_cst"], outputs=["output"]
        )
    )

    graph = helper.make_graph(
        nodes,
        "upscale",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class UpscaleSolver(BaseSolver):
    PRIORITY = 13

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("upscale")) and analysis.get("upscale_factor") is not None

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        factor = analysis.get("upscale_factor")
        if factor is None:
            return None

        sy, sx = factor
        path = out_dir / f"{task_id}.onnx"
        try:
            save(_upscale_net(sy, sx), str(path))
            return path
        except Exception as e:
            print(f"    UpscaleSolver({sy}x{sx}) failed: {e}")
            return None
