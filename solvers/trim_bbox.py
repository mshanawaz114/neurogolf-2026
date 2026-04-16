from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.arc_utils import detect_trim_bbox, grid_to_array
from utils.onnx_builder import CANVAS, CHANNELS as C, _make_int64, _t, save


def _pick_1d(vec_name: str, idx: int, prefix: str):
    inits = [
        _make_int64(f"{prefix}_s", [idx]),
        _make_int64(f"{prefix}_e", [idx + 1]),
        _make_int64(f"{prefix}_a", [1]),
        _make_int64(f"{prefix}_st", [1]),
    ]
    node = helper.make_node(
        "Slice",
        inputs=[vec_name, f"{prefix}_s", f"{prefix}_e", f"{prefix}_a", f"{prefix}_st"],
        outputs=[f"{prefix}_out"],
    )
    return node, inits, f"{prefix}_out"


def _candidate_selector(prefix: str, bg: int, y0: int, y1: int, x0: int, x1: int):
    nodes = []
    inits = []

    mask_w = np.ones((1, C, 1, 1), dtype=np.float32)
    mask_w[0, bg, 0, 0] = 0.0
    inits.append(_t(f"{prefix}_mask_w", mask_w))
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=["input", f"{prefix}_mask_w"],
            outputs=[f"{prefix}_mask"],
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
        )
    )
    nodes.append(
        helper.make_node("ReduceMax", inputs=[f"{prefix}_mask"], outputs=[f"{prefix}_cmax"], axes=[1], keepdims=0)
    )
    nodes.append(
        helper.make_node("ReduceMax", inputs=[f"{prefix}_cmax"], outputs=[f"{prefix}_col_presence"], axes=[1], keepdims=0)
    )
    nodes.append(
        helper.make_node("ReduceMax", inputs=[f"{prefix}_cmax"], outputs=[f"{prefix}_row_presence"], axes=[2], keepdims=0)
    )

    checks = []

    def require_present(name: str, vec: str, idx: int):
        node, local_inits, out = _pick_1d(vec, idx, f"{prefix}_{name}")
        nodes.append(node)
        inits.extend(local_inits)
        checks.append(out)

    def require_absent(name: str, vec: str, idx: int):
        node, local_inits, out = _pick_1d(vec, idx, f"{prefix}_{name}")
        nodes.append(node)
        inits.extend(local_inits)
        if not any(t.name == f"{prefix}_one" for t in inits):
            inits.append(_t(f"{prefix}_one", np.array([1.0], dtype=np.float32)))
        nodes.append(helper.make_node("Sub", inputs=[f"{prefix}_one", out], outputs=[f"{prefix}_{name}_abs"]))
        checks.append(f"{prefix}_{name}_abs")

    require_present("row_start", f"{prefix}_row_presence", y0)
    require_present("row_end", f"{prefix}_row_presence", y1 - 1)
    require_present("col_start", f"{prefix}_col_presence", x0)
    require_present("col_end", f"{prefix}_col_presence", x1 - 1)
    if y0 > 0:
        require_absent("row_before", f"{prefix}_row_presence", y0 - 1)
    if y1 < CANVAS:
        require_absent("row_after", f"{prefix}_row_presence", y1)
    if x0 > 0:
        require_absent("col_before", f"{prefix}_col_presence", x0 - 1)
    if x1 < CANVAS:
        require_absent("col_after", f"{prefix}_col_presence", x1)

    current = checks[0]
    for idx, check in enumerate(checks[1:], start=1):
        out = f"{prefix}_sel" if idx == len(checks) - 1 else f"{prefix}_sel_{idx}"
        nodes.append(helper.make_node("Mul", inputs=[current, check], outputs=[out]))
        current = out

    return nodes, inits, current


def _trim_bbox_net(bg: int, candidates: list[tuple[int, int, int, int]]) -> onnx.ModelProto:
    nodes = []
    inits = []
    weighted = []

    for idx, (y0, y1, x0, x1) in enumerate(candidates):
        prefix = f"tb{idx}"
        sel_nodes, sel_inits, sel = _candidate_selector(prefix, bg, y0, y1, x0, x1)
        nodes += sel_nodes
        inits += sel_inits

        inits += [
            _make_int64(f"{prefix}_rs", [y0]),
            _make_int64(f"{prefix}_re", [y1]),
            _make_int64(f"{prefix}_ra", [2]),
            _make_int64(f"{prefix}_rst", [1]),
        ]
        nodes.append(
            helper.make_node(
                "Slice",
                inputs=["input", f"{prefix}_rs", f"{prefix}_re", f"{prefix}_ra", f"{prefix}_rst"],
                outputs=[f"{prefix}_rows"],
            )
        )
        inits += [
            _make_int64(f"{prefix}_cs", [x0]),
            _make_int64(f"{prefix}_ce", [x1]),
            _make_int64(f"{prefix}_ca", [3]),
            _make_int64(f"{prefix}_cst", [1]),
        ]
        nodes.append(
            helper.make_node(
                "Slice",
                inputs=[f"{prefix}_rows", f"{prefix}_cs", f"{prefix}_ce", f"{prefix}_ca", f"{prefix}_cst"],
                outputs=[f"{prefix}_crop"],
            )
        )
        inits.append(_make_int64(f"{prefix}_pad", [0, 0, 0, 0, 0, 0, CANVAS - (y1 - y0), CANVAS - (x1 - x0)]))
        nodes.append(
            helper.make_node(
                "Pad", inputs=[f"{prefix}_crop", f"{prefix}_pad"], outputs=[f"{prefix}_padded"], mode="constant"
            )
        )
        nodes.append(helper.make_node("Mul", inputs=[f"{prefix}_padded", sel], outputs=[f"{prefix}_weighted"]))
        weighted.append(f"{prefix}_weighted")

    current = weighted[0]
    for idx, name in enumerate(weighted[1:], start=1):
        out = "output" if idx == len(weighted) - 1 else f"tb_add_{idx}"
        nodes.append(helper.make_node("Add", inputs=[current, name], outputs=[out]))
        current = out

    graph = helper.make_graph(
        nodes,
        "trim_bbox",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class TrimBBoxSolver(BaseSolver):
    PRIORITY = 14

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("trim_bbox")) and bool(analysis.get("trim_bbox_candidates"))

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        candidates = []
        bg = analysis.get("trim_bbox_bg")
        for split in ["train", "test", "arc-gen"]:
            for pair in task.get(split, []):
                det = detect_trim_bbox(grid_to_array(pair["input"]), grid_to_array(pair["output"]))
                if det is None:
                    return None
                if bg is None:
                    bg = det[0]
                if det[0] != bg:
                    return None
                candidates.append(det[1:])
        if not candidates:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        candidates = sorted({tuple(int(v) for v in candidate) for candidate in candidates})
        path = out_dir / f"{task_id}.onnx"
        try:
            save(_trim_bbox_net(int(bg), candidates), str(path))
            return path
        except Exception as e:
            print(f"    TrimBBoxSolver failed: {e}")
            return None
