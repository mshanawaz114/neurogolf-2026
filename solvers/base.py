from __future__ import annotations

"""
base.py — Abstract base class for all NeuroGolf solvers.

Each solver must implement:
  - can_solve(analysis) → bool    : check if this solver applies
  - build(task, analysis) → path  : build the ONNX file and return its path
  - PRIORITY: int                 : lower = tried first (prefer simpler/cheaper)
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseSolver(ABC):
    PRIORITY: int = 50       # lower = higher priority

    @abstractmethod
    def can_solve(self, analysis: dict) -> bool:
        """Return True if this solver can handle the task based on analysis."""
        ...

    @abstractmethod
    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        """
        Build the ONNX file for this task.
        Returns the path to the saved .onnx file, or None on failure.
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
