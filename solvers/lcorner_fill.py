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


def _lcorner_fill_net(boundary: int, fill: int) -> onnx.ModelProto:
    nodes = []
    inits = []

    channels = [_slice_channel(nodes, inits, "input", idx, f"lcf_ch{idx}") for idx in range(C)]
    boundary_ch = channels[boundary]

    inits += [
        _make_float("lcf_one", [1.0]),
        _make_float("lcf_zero", [0.0]),
        _make_float("lcf_thresh", [2.5]),
        _make_float("lcf_two", [2.0]),
    ]
    nodes.append(helper.make_node("Sub", inputs=["lcf_one", boundary_ch], outputs=["lcf_open"]))

    patterns = [
        ("tl", np.array([[[[0.0, 1.0], [1.0, 1.0]]]], dtype=np.float32), [0, 0, 0, 0, 0, 0, 1, 1]),
        ("tr", np.array([[[[1.0, 0.0], [1.0, 1.0]]]], dtype=np.float32), [0, 0, 0, 1, 0, 0, 1, 0]),
        ("bl", np.array([[[[1.0, 1.0], [0.0, 1.0]]]], dtype=np.float32), [0, 0, 1, 0, 0, 0, 0, 1]),
        ("br", np.array([[[[1.0, 1.0], [1.0, 0.0]]]], dtype=np.float32), [0, 0, 1, 1, 0, 0, 0, 0]),
    ]

    branch_masks = []
    for name, kernel, pads in patterns:
        inits.append(_t(f"lcf_k_{name}", kernel))
        nodes.append(
            helper.make_node(
                "Conv",
                inputs=[boundary_ch, f"lcf_k_{name}"],
                outputs=[f"lcf_conv_{name}"],
                kernel_shape=[2, 2],
                pads=[0, 0, 0, 0],
            )
        )
        nodes.append(helper.make_node("Sub", inputs=[f"lcf_conv_{name}", "lcf_thresh"], outputs=[f"lcf_shift_{name}"]))
        nodes.append(helper.make_node("Clip", inputs=[f"lcf_shift_{name}", "lcf_zero", "lcf_one"], outputs=[f"lcf_hit_{name}"]))
        nodes.append(helper.make_node("Mul", inputs=[f"lcf_hit_{name}", "lcf_two"], outputs=[f"lcf_hit2_{name}"]))
        inits.append(_make_int64(f"lcf_pad_{name}", pads))
        nodes.append(
            helper.make_node("Pad", inputs=[f"lcf_hit2_{name}", f"lcf_pad_{name}"], outputs=[f"lcf_pos_{name}"], mode="constant")
        )
        nodes.append(helper.make_node("Mul", inputs=[f"lcf_pos_{name}", "lcf_open"], outputs=[f"lcf_mask_{name}"]))
        branch_masks.append(f"lcf_mask_{name}")

    current = branch_masks[0]
    for idx, name in enumerate(branch_masks[1:], start=1):
        out = f"lcf_sum_{idx}" if idx < len(branch_masks) - 1 else "lcf_sum"
        nodes.append(helper.make_node("Add", inputs=[current, name], outputs=[out]))
        current = out
    nodes.append(helper.make_node("Clip", inputs=[current, "lcf_zero", "lcf_one"], outputs=["lcf_mask"]))

    nodes.append(helper.make_node("Sub", inputs=[channels[0], "lcf_mask"], outputs=["lcf_ch0_out"]))
    nodes.append(helper.make_node("Add", inputs=[channels[fill], "lcf_mask"], outputs=["lcf_fill_raw"]))
    nodes.append(helper.make_node("Clip", inputs=["lcf_fill_raw", "lcf_zero", "lcf_one"], outputs=["lcf_fill_out"]))

    outputs = []
    for idx in range(C):
        if idx == 0:
            outputs.append("lcf_ch0_out")
        elif idx == fill:
            outputs.append("lcf_fill_out")
        else:
            outputs.append(channels[idx])
    nodes.append(helper.make_node("Concat", inputs=outputs, outputs=["output"], axis=1))

    graph = helper.make_graph(
        nodes,
        "lcorner_fill",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class LCornerFillSolver(BaseSolver):
    PRIORITY = 13

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("lcorner_fill"))

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        boundary = analysis.get("lcorner_boundary")
        fill = analysis.get("lcorner_fill_color")
        if boundary is None or fill is None:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{task_id}.onnx"
        try:
            save(_lcorner_fill_net(int(boundary), int(fill)), str(path), try_simplify=False)
            return path
        except Exception as e:
            print(f"    LCornerFillSolver failed: {e}")
            return None
