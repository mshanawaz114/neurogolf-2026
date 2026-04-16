"""
tests/test_utils.py — Basic sanity checks for utility functions.
Run with: pytest tests/
"""

import sys
import math
import numpy as np
import pytest

sys.path.insert(0, ".")
from utils.scoring import compute_score, count_parameters, memory_bytes


def test_score_formula():
    # At cost=1, score should be max(1, 25 - ln(1)) = max(1, 25) = 25
    assert compute_score(1) == 25.0

    # At very high cost, score should floor at 1
    assert compute_score(10 ** 20) == 1.0

    # Known value: cost=100 → 25 - ln(100) ≈ 20.395
    expected = 25.0 - math.log(100)
    assert abs(compute_score(100) - expected) < 1e-6


def test_score_never_below_one():
    for cost in [1, 100, 10_000, 1_000_000, 10 ** 15]:
        assert compute_score(cost) >= 1.0
