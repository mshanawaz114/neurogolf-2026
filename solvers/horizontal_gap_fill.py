from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_float, _make_int64, _t, save


def _slice_channel(nodes: list, inits: list, source: str, idx: int, prefix: str) -> str:
    inits += [
        _make_int64(f"{prefix}_s", [idx]),
        _make_int64(f"{prefix}_e", [idx + 1]),
        _make_int64(f"{prefix}_a", [1]),
        _make_int64(f"{prefix}_st", [1]),
    ]
    out = f"{prefix}_slice"
    nodes.append(
        helper.make_node(
            "Slice",
            inputs=[source, f"{prefix}_s", f"{prefix}_e", f"{prefix}_a", f"{prefix}_st"],
            outputs=[out],
        )
    )
    return out


def _horizontal_gap_fill_net(boundary: int, fill: int) -> onnx.ModelProto:
    nodes = []
    inits = []

    channels = [_slice_channel(nodes, inits, "input", idx, f"hgf_ch{idx}") for idx in range(C)]
    boundary_ch = channels[boundary]

    kernel = np.array([[[[1.0, 0.0, 1.0]]]], dtype=np.float32)
    inits.append(_t("hgf_kernel", kernel))
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=[boundary_ch, "hgf_kernel"],
            outputs=["hgf_conv"],
            kernel_shape=[1, 3],
            pads=[0, 1, 0, 1],
        )
    )

    inits += [
        _make_float("hgf_one", [1.0]),
        _make_float("hgf_zero", [0.0]),
        _make_float("hgf_half", [0.5]),
    ]
    nodes.append(helper.make_node("Sub", inputs=["hgf_conv", "hgf_half"], outputs=["hgf_shift"]))
    nodes.append(helper.make_node("Clip", inputs=["hgf_shift", "hgf_zero", "hgf_one"], outputs=["hgf_pair"]))
    nodes.append(helper.make_node("Sub", inputs=["hgf_one", boundary_ch], outputs=["hgf_open"]))
    nodes.append(helper.make_node("Mul", inputs=["hgf_pair", "hgf_open"], outputs=["hgf_mask"]))

    nodes.append(helper.make_node("Sub", inputs=[channels[0], "hgf_mask"], outputs=["hgf_ch0_out"]))
    nodes.append(helper.make_node("Add", inputs=[channels[fill], "hgf_mask"], outputs=["hgf_fill_raw"]))
    nodes.append(helper.make_node("Clip", inputs=["hgf_fill_raw", "hgf_zero", "hgf_one"], outputs=["hgf_fill_out"]))

    outputs = []
    for idx in range(C):
        if idx == 0:
            outputs.append("hgf_ch0_out")
        elif idx == fill:
            outputs.append("hgf_fill_out")
        else:
            outputs.append(channels[idx])
    nodes.append(helper.make_node("Concat", inputs=outputs, outputs=["output"], axis=1))

    graph = helper.make_graph(
        nodes,
        "horizontal_gap_fill",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class HorizontalGapFillSolver(BaseSolver):
    PRIORITY = 13

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("horizontal_gap_fill"))

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        boundary = analysis.get("horizontal_gap_boundary")
        fill = analysis.get("horizontal_gap_fill_color")
        if boundary is None or fill is None:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{task_id}.onnx"
        try:
            save(_horizontal_gap_fill_net(int(boundary), int(fill)), str(path), try_simplify=False)
            return path
        except Exception as e:
            print(f"    HorizontalGapFillSolver failed: {e}")
            return None
