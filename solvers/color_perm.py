from __future__ import annotations

"""
color_perm.py — Solver for colour-permutation tasks.

These are tasks where every pixel's position is unchanged but its colour
is remapped according to some consistent rule.

ONNX: single 1×1 conv with a 10×10 permutation-like weight matrix.
Cost: 100 params + 400 B + 90,000 MACs → score ≈ 13.6

This solver handles:
- Direct colour remaps (e.g. red→blue, blue→red)
- Colour inversion, colour swap, any consistent per-colour mapping
- Zero-hot pixels (outside the grid) are preserved as all-zeros correctly
  because the conv is linear and 0→0.
"""

from pathlib import Path
import numpy as np

from solvers.base import BaseSolver
from utils.arc_utils import analyse_task
from utils.onnx_builder import color_perm_net, save


class ColorPermSolver(BaseSolver):
    PRIORITY = 10   # Very high priority — cheap and common

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("color_permutation"))

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        mapping = analysis.get("color_mapping")
        if mapping is None:
            return None

        # Fill in identity for any missing colour mappings
        full_mapping = {c: c for c in range(10)}
        full_mapping.update(mapping)

        model = color_perm_net(full_mapping)
        path = out_dir / f"{task_id}.onnx"
        save(model, str(path))
        return path
