from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.arc_utils import grid_to_array
from utils.onnx_builder import CANVAS, CHANNELS as C, _t, save


def _detect_seed_halo_pair(inp: np.ndarray, out: np.ndarray) -> tuple[int, int] | None:
    if inp.shape != out.shape:
        return None
    colors = [int(v) for v in np.unique(inp) if int(v) != 0]
    if len(colors) != 1:
        return None
    seed = colors[0]

    out_colors = [int(v) for v in np.unique(out) if int(v) != 0 and int(v) != seed]
    if len(out_colors) != 1:
        return None
    fill = out_colors[0]

    H, W = inp.shape
    pred = np.zeros_like(inp)
    ys, xs = np.where(inp == seed)
    for y, x in zip(ys, xs):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W:
                    pred[yy, xx] = fill
        pred[y, x] = seed
    if np.array_equal(pred, out):
        return seed, fill
    return None


def _seed_halo_net(seed: int, fill: int) -> onnx.ModelProto:
    W = np.zeros((C, C, 3, 3), dtype=np.float32)
    W[fill, seed, :, :] = 1.0
    W[fill, seed, 1, 1] = 0.0
    W[seed, seed, 1, 1] = 1.0

    nodes = [
        helper.make_node(
            "Conv",
            inputs=["input", "sh_W"],
            outputs=["sh_conv"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    ]
    inits = [_t("sh_W", W)]

    # Clamp the filled halo channel back to one-hot values.
    zero = np.array([0.0], dtype=np.float32)
    one = np.array([1.0], dtype=np.float32)
    inits.extend([_t("sh_lo", zero), _t("sh_hi", one)])
    nodes.append(helper.make_node("Clip", inputs=["sh_conv", "sh_lo", "sh_hi"], outputs=["output"]))

    graph = helper.make_graph(
        nodes,
        "seed_halo",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, CANVAS, CANVAS])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class SeedHaloSolver(BaseSolver):
    PRIORITY = 16

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("same_io_shape"))

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        detected: tuple[int, int] | None = None
        for split in ["train", "test", "arc-gen"]:
            for pair in task.get(split, []):
                cur = _detect_seed_halo_pair(grid_to_array(pair["input"]), grid_to_array(pair["output"]))
                if cur is None:
                    return None
                if detected is None:
                    detected = cur
                elif detected != cur:
                    return None
        if detected is None:
            return None

        seed, fill = detected
        path = out_dir / f"{task_id}.onnx"
        save(_seed_halo_net(seed, fill), str(path), try_simplify=False)
        return path
