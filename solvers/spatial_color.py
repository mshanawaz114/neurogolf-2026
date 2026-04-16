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
    return numpy_helper.from_array(weights, name="sc_W")


def _flip_h_color_net(grid_w: int, mapping: dict[int, int]) -> onnx.ModelProto:
    start = CANVAS - grid_w
    nodes = []
    inits = [
        _make_int64("fh_s", [2**31 - 1]),
        _make_int64("fh_e", [-(2**31)]),
        _make_int64("fh_ax", [3]),
        _make_int64("fh_st", [-1]),
        _make_int64("slc_s", [start]),
        _make_int64("slc_e", [CANVAS]),
        _make_int64("slc_ax", [3]),
        _make_int64("slc_st", [1]),
        _make_int64("pad_v", [0, 0, 0, 0, 0, 0, 0, CANVAS - grid_w]),
        _color_weights(mapping),
    ]
    nodes.append(
        helper.make_node("Slice", inputs=["input", "fh_s", "fh_e", "fh_ax", "fh_st"], outputs=["flipped"])
    )
    nodes.append(
        helper.make_node("Slice", inputs=["flipped", "slc_s", "slc_e", "slc_ax", "slc_st"], outputs=["sliced"])
    )
    nodes.append(helper.make_node("Pad", inputs=["sliced", "pad_v"], outputs=["spatial"], mode="constant"))
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=["spatial", "sc_W"],
            outputs=["output"],
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
        )
    )

    graph = helper.make_graph(
        nodes,
        "flip_h_color",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _flip_v_color_net(grid_h: int, mapping: dict[int, int]) -> onnx.ModelProto:
    start = CANVAS - grid_h
    nodes = []
    inits = [
        _make_int64("fv_s", [2**31 - 1]),
        _make_int64("fv_e", [-(2**31)]),
        _make_int64("fv_ax", [2]),
        _make_int64("fv_st", [-1]),
        _make_int64("sv_s", [start]),
        _make_int64("sv_e", [CANVAS]),
        _make_int64("sv_ax", [2]),
        _make_int64("sv_st", [1]),
        _make_int64("pad_v", [0, 0, 0, 0, 0, 0, CANVAS - grid_h, 0]),
        _color_weights(mapping),
    ]
    nodes.append(
        helper.make_node("Slice", inputs=["input", "fv_s", "fv_e", "fv_ax", "fv_st"], outputs=["flipped"])
    )
    nodes.append(
        helper.make_node("Slice", inputs=["flipped", "sv_s", "sv_e", "sv_ax", "sv_st"], outputs=["sliced"])
    )
    nodes.append(helper.make_node("Pad", inputs=["sliced", "pad_v"], outputs=["spatial"], mode="constant"))
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=["spatial", "sc_W"],
            outputs=["output"],
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
        )
    )

    graph = helper.make_graph(
        nodes,
        "flip_v_color",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class SpatialColorSolver(BaseSolver):
    PRIORITY = 11

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("spatial_color")) and analysis.get("spatial_color_transform") in {
            "flip_h",
            "flip_v",
        }

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        transform = analysis.get("spatial_color_transform")
        mapping = analysis.get("spatial_color_mapping")
        in_shapes = analysis.get("input_shapes", [])
        if mapping is None or not in_shapes:
            return None

        path = out_dir / f"{task_id}.onnx"
        for h, w in in_shapes:
            try:
                if transform == "flip_h":
                    save(_flip_h_color_net(w, mapping), str(path))
                    return path
                if transform == "flip_v":
                    save(_flip_v_color_net(h, mapping), str(path))
                    return path
            except Exception as e:
                print(f"    SpatialColorSolver({transform}) H={h} W={w} failed: {e}")
        return None
