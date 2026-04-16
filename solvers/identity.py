from __future__ import annotations

"""
identity.py — Solver for identity tasks (output == input).

Cost: 0 params, ~0 bytes, 0 MACs → score ≈ 25
Uses a single 1×1 conv with identity weight matrix.
"""

import numpy as np
from pathlib import Path
import onnx

from solvers.base import BaseSolver
from utils.onnx_builder import conv1x1, build_graph, save


class IdentitySolver(BaseSolver):
    PRIORITY = 1   # Try first — cheapest possible

    def can_solve(self, analysis: dict) -> bool:
        return analysis.get("identity", False)

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        # Identity: 10×10 identity matrix as 1×1 conv
        W = np.eye(10, dtype=np.float32)
        model = build_graph([conv1x1(W, name="id")])
        path = out_dir / f"{task_id}.onnx"
        save(model, str(path))
        return path
