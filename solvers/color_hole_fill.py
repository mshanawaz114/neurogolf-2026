from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_float, _make_int64, _t, save


def _slice_channel(nodes: list, inits: list, idx: int, prefix: str) -> str:
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
            inputs=["input", f"{prefix}_s", f"{prefix}_e", f"{prefix}_a", f"{prefix}_st"],
            outputs=[out],
        )
    )
    return out


def _color_hole_fill_net(boundary: int, fill: int, steps: int = 26) -> onnx.ModelProto:
    nodes = []
    inits = []

    channels = []
    for idx in range(C):
        channels.append(_slice_channel(nodes, inits, idx, f"hf_ch{idx}"))

    inits.append(_make_float("hf_one", [1.0]))
    nodes.append(helper.make_node("Sub", inputs=["hf_one", channels[boundary]], outputs=["hf_open"]))

    border = np.zeros((1, 1, CANVAS, CANVAS), dtype=np.float32)
    border[:, :, 0, :] = 1.0
    border[:, :, -1, :] = 1.0
    border[:, :, :, 0] = 1.0
    border[:, :, :, -1] = 1.0
    inits.append(_t("hf_border", border))
    nodes.append(helper.make_node("Mul", inputs=["hf_open", "hf_border"], outputs=["hf_reach0_raw"]))
    inits.append(_make_float("hf_zero", [0.0]))
    nodes.append(helper.make_node("Clip", inputs=["hf_reach0_raw", "hf_zero", "hf_one"], outputs=["hf_reach0"]))

    kernel = np.array(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    ).reshape(1, 1, 3, 3)
    inits.append(_t("hf_kernel", kernel))

    current = "hf_reach0"
    for idx in range(steps):
        dil = f"hf_dil{idx}"
        masked = f"hf_masked{idx}"
        nxt = f"hf_reach{idx + 1}"
        nodes.append(
            helper.make_node(
                "Conv",
                inputs=[current, "hf_kernel"],
                outputs=[dil],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            )
        )
        nodes.append(helper.make_node("Mul", inputs=[dil, "hf_open"], outputs=[masked]))
        nodes.append(helper.make_node("Clip", inputs=[masked, "hf_zero", "hf_one"], outputs=[nxt]))
        current = nxt

    nodes.append(helper.make_node("Sub", inputs=["hf_open", current], outputs=["hf_holes_raw"]))
    nodes.append(helper.make_node("Clip", inputs=["hf_holes_raw", "hf_zero", "hf_one"], outputs=["hf_holes"]))

    nodes.append(helper.make_node("Sub", inputs=[channels[0], "hf_holes"], outputs=["hf_ch0_out"]))
    nodes.append(helper.make_node("Add", inputs=[channels[fill], "hf_holes"], outputs=[f"hf_ch{fill}_raw"]))
    nodes.append(
        helper.make_node("Clip", inputs=[f"hf_ch{fill}_raw", "hf_zero", "hf_one"], outputs=[f"hf_ch{fill}_out"])
    )

    outputs = []
    for idx in range(C):
        if idx == 0:
            outputs.append("hf_ch0_out")
        elif idx == fill:
            outputs.append(f"hf_ch{fill}_out")
        else:
            outputs.append(channels[idx])
    nodes.append(helper.make_node("Concat", inputs=outputs, outputs=["output"], axis=1))

    graph = helper.make_graph(
        nodes,
        "color_hole_fill",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class ColorHoleFillSolver(BaseSolver):
    PRIORITY = 13

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("color_hole_fill"))

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        boundary = analysis.get("color_hole_fill_boundary")
        fill = analysis.get("color_hole_fill_fill")
        if boundary is None or fill is None:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{task_id}.onnx"
        try:
            save(_color_hole_fill_net(int(boundary), int(fill)), str(path), try_simplify=False)
            return path
        except Exception as e:
            print(f"    ColorHoleFillSolver failed: {e}")
            return None
