from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.arc_utils import detect_bounce_seed, grid_to_array
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


def _width_selector(prefix: str, width: int):
    nodes = []
    inits = []

    nodes.append(helper.make_node("ReduceMax", inputs=["input"], outputs=[f"{prefix}_cmax"], axes=[1], keepdims=0))
    nodes.append(helper.make_node("ReduceMax", inputs=[f"{prefix}_cmax"], outputs=[f"{prefix}_col_presence"], axes=[1], keepdims=0))
    nodes.append(helper.make_node("ReduceMax", inputs=[f"{prefix}_cmax"], outputs=[f"{prefix}_row_presence"], axes=[2], keepdims=0))

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

    # height fixed at 10, width chosen dynamically
    require_present("row0", f"{prefix}_row_presence", 0)
    require_present("row9", f"{prefix}_row_presence", 9)
    require_absent("row10", f"{prefix}_row_presence", 10)
    require_present("col0", f"{prefix}_col_presence", 0)
    require_present("col_last", f"{prefix}_col_presence", width - 1)
    require_absent("col_after", f"{prefix}_col_presence", width)

    current = checks[0]
    for idx, check in enumerate(checks[1:], start=1):
        out = f"{prefix}_sel" if idx == len(checks) - 1 else f"{prefix}_sel_{idx}"
        nodes.append(helper.make_node("Mul", inputs=[current, check], outputs=[out]))
        current = out

    return nodes, inits, current


def _pattern_tensor(seed: int, height: int, width: int) -> np.ndarray:
    arr = np.zeros((1, C, CANVAS, CANVAS), dtype=np.float32)
    period = max(1, 2 * (width - 1))
    for r in range(height):
        t = (height - 1) - r
        if width == 1:
            c = 0
        else:
            m = t % period
            c = m if m <= width - 1 else period - m
        arr[0, seed, r, c] = 1.0
    return arr


def _bounce_seed_net(seed: int, height: int, widths: tuple[int, ...]) -> onnx.ModelProto:
    nodes = []
    inits = []
    weighted = []

    for idx, width in enumerate(widths):
        prefix = f"bs{idx}"
        sel_nodes, sel_inits, sel = _width_selector(prefix, width)
        nodes += sel_nodes
        inits += sel_inits
        inits.append(_t(f"{prefix}_pat", _pattern_tensor(seed, height, width)))
        nodes.append(helper.make_node("Mul", inputs=[f"{prefix}_pat", sel], outputs=[f"{prefix}_weighted"]))
        weighted.append(f"{prefix}_weighted")

    current = weighted[0]
    for idx, name in enumerate(weighted[1:], start=1):
        out = "output" if idx == len(weighted) - 1 else f"bs_add_{idx}"
        nodes.append(helper.make_node("Add", inputs=[current, name], outputs=[out]))
        current = out

    graph = helper.make_graph(
        nodes,
        "bounce_seed",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class BounceSeedSolver(BaseSolver):
    PRIORITY = 13

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("bounce_seed")) and bool(analysis.get("bounce_widths"))

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        seed = analysis.get("bounce_seed_color")
        height = analysis.get("bounce_height")
        widths = set(analysis.get("bounce_widths") or [])
        if seed is None or height is None:
            return None
        for split in ["train", "test", "arc-gen"]:
            for pair in task.get(split, []):
                det = detect_bounce_seed(grid_to_array(pair["input"]), grid_to_array(pair["output"]))
                if det is None or det[0] != seed or det[1] != height:
                    return None
                widths.add(det[2])
        if not widths:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{task_id}.onnx"
        try:
            save(_bounce_seed_net(int(seed), int(height), tuple(sorted(int(w) for w in widths))), str(path), try_simplify=False)
            return path
        except Exception as e:
            print(f"    BounceSeedSolver failed: {e}")
            return None
