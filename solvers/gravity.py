from __future__ import annotations

"""
gravity.py — Solver for gravity tasks (non-zero pixels fall to bottom or top).

These are tasks where each column's non-zero pixels are compacted to one end.

Strategy: Sort-based simulation via an iterative ONNX graph.
Since ONNX doesn't have loops/scan, we implement this analytically:

Key insight: For gravity_down we can build an ONNX that repeatedly "sifts"
non-zero values downward. Each sift-step checks if a cell is zero and its
neighbour below is non-zero, then swaps them.

However, the number of steps needed = grid height, which can be up to 30.
Building 30 chained swap-steps analytically is feasible.

Alternative (simpler): The gravity transformation is equivalent to ArgSort
on each column — but ONNX lacks per-column sort.

Practical approach: Use a trained tiny model (1×1 or 3×3 conv) as LearnedSolver
will handle these. We skip gravity in analytical solvers and rely on LearnedSolver.

Actually wait — let me think about a deterministic ONNX approach:

For gravity_down, each column independently sorts: zeros go up, non-zeros go down.
This IS computable analytically IF we know the exact heights.

For same-size tasks (most gravity tasks), we can do:
  - For each column: extract values, sort (zeros first), put back
  - In ONNX without loops: unroll with a sorting network (30 elements = O(30²) comparisons)

A sorting network for 30 elements has ~30*log²(30) ≈ 235 comparators.
Each comparator is: max_val = Max(a,b), min_val = Min(a,b).
This is feasible for a 1-column case, but we need all 10 colour channels...

Actually, the correct approach for gravity is:
  - Sum all non-zero channels to get a presence mask
  - Apply a differentiable sort
  - But we need to track which channel each pixel belongs to

For now, we rely on LearnedSolver for gravity tasks.
This file provides the GravitySolver stub that always returns None,
causing fallthrough to LearnedSolver.

TODO: Implement deterministic gravity via sorting network if needed for score.
"""

from pathlib import Path
from solvers.base import BaseSolver


class GravitySolver(BaseSolver):
    """Stub — gravity tasks fall through to LearnedSolver."""
    PRIORITY = 15   # Try before learned (but this always returns None)

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("gravity_down") or analysis.get("gravity_up"))

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        # Falls through to LearnedSolver
        return None
