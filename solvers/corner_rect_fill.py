from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.arc_utils import detect_corner_rectangle_fill, grid_to_array
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


def _corner_rect_fill_net(boundary: int, fill: int, sizes: tuple[tuple[int, int], ...]) -> onnx.ModelProto:
    nodes = []
    inits = []

    channels = [_slice_channel(nodes, inits, "input", idx, f"crf_ch{idx}") for idx in range(C)]
    boundary_ch = channels[boundary]

    inits += [
        _make_float("crf_neg35", [-3.5]),
        _make_float("crf_zero", [0.0]),
        _make_float("crf_one", [1.0]),
        _make_float("crf_two", [2.0]),
    ]

    padded_branches = []
    for idx, (dy, dx) in enumerate(sizes):
        det_kernel = np.zeros((1, 1, dy + 1, dx + 1), dtype=np.float32)
        det_kernel[0, 0, 0, 0] = 1.0
        det_kernel[0, 0, 0, dx] = 1.0
        det_kernel[0, 0, dy, 0] = 1.0
        det_kernel[0, 0, dy, dx] = 1.0
        inits.append(_t(f"crf_detk_{idx}", det_kernel))
        nodes.append(
            helper.make_node(
                "Conv",
                inputs=[boundary_ch, f"crf_detk_{idx}"],
                outputs=[f"crf_det_{idx}"],
                kernel_shape=[dy + 1, dx + 1],
                pads=[0, 0, 0, 0],
            )
        )
        nodes.append(helper.make_node("Add", inputs=[f"crf_det_{idx}", "crf_neg35"], outputs=[f"crf_sub_{idx}"]))
        nodes.append(helper.make_node("Clip", inputs=[f"crf_sub_{idx}", "crf_zero", "crf_one"], outputs=[f"crf_clip_{idx}"]))
        nodes.append(helper.make_node("Mul", inputs=[f"crf_clip_{idx}", "crf_two"], outputs=[f"crf_mark_{idx}"]))
        offset_masks = []
        for oy in range(1, dy):
            for ox in range(1, dx):
                pad_name = f"crf_pad_{idx}_{oy}_{ox}"
                out_name = f"crf_shift_{idx}_{oy}_{ox}"
                inits.append(_make_int64(pad_name, [0, 0, oy, ox, 0, 0, dy - oy, dx - ox]))
                nodes.append(
                    helper.make_node(
                        "Pad",
                        inputs=[f"crf_mark_{idx}", pad_name],
                        outputs=[out_name],
                        mode="constant",
                    )
                )
                offset_masks.append(out_name)

        current_fill = offset_masks[0]
        for fill_idx, name in enumerate(offset_masks[1:], start=1):
            out = f"crf_fillsum_{idx}_{fill_idx}" if fill_idx < len(offset_masks) - 1 else f"crf_fill_{idx}"
            nodes.append(helper.make_node("Add", inputs=[current_fill, name], outputs=[out]))
            current_fill = out
        padded_branches.append(current_fill)

    current = padded_branches[0]
    for idx, name in enumerate(padded_branches[1:], start=1):
        out = f"crf_sum_{idx}" if idx < len(padded_branches) - 1 else "crf_sum"
        nodes.append(helper.make_node("Add", inputs=[current, name], outputs=[out]))
        current = out

    nodes.append(helper.make_node("Clip", inputs=[current, "crf_zero", "crf_one"], outputs=["crf_mask"]))
    nodes.append(helper.make_node("Sub", inputs=[channels[0], "crf_mask"], outputs=["crf_ch0_out"]))
    nodes.append(helper.make_node("Add", inputs=[channels[fill], "crf_mask"], outputs=["crf_fill_raw"]))
    nodes.append(helper.make_node("Clip", inputs=["crf_fill_raw", "crf_zero", "crf_one"], outputs=["crf_fill_out"]))

    outputs = []
    for idx in range(C):
        if idx == 0:
            outputs.append("crf_ch0_out")
        elif idx == fill:
            outputs.append("crf_fill_out")
        else:
            outputs.append(channels[idx])
    nodes.append(helper.make_node("Concat", inputs=outputs, outputs=["output"], axis=1))

    graph = helper.make_graph(
        nodes,
        "corner_rect_fill",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class CornerRectFillSolver(BaseSolver):
    PRIORITY = 13

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("corner_rect_fill")) and bool(analysis.get("corner_rect_sizes"))

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        boundary = analysis.get("corner_rect_boundary")
        fill = analysis.get("corner_rect_fill_color")
        if boundary is None or fill is None:
            return None
        sizes: set[tuple[int, int]] = set()
        for split in ["train", "test", "arc-gen"]:
            for pair in task.get(split, []):
                det = detect_corner_rectangle_fill(grid_to_array(pair["input"]), grid_to_array(pair["output"]))
                if det is None or det[0] != boundary or det[1] != fill:
                    return None
                sizes.update(tuple(map(int, s)) for s in det[2])
        if not sizes:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{task_id}.onnx"
        try:
            save(_corner_rect_fill_net(int(boundary), int(fill), tuple(sorted(sizes))), str(path), try_simplify=False)
            return path
        except Exception as e:
            print(f"    CornerRectFillSolver failed: {e}")
            return None
