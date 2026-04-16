from __future__ import annotations

"""
arc_utils.py — Core ARC-AGI data utilities for NeuroGolf 2026.

CRITICAL FORMAT: Input/output are ALWAYS padded to [1, 10, 30, 30].
- Pixels inside the grid: exactly one channel = 1.0, rest = 0.0
- Pixels outside the grid: ALL channels = 0.0  ("zero-hot")
- Grids are placed in the TOP-LEFT corner of the 30x30 canvas.
"""

import json
import numpy as np
from pathlib import Path
from typing import Any

CANVAS = 30   # Fixed canvas size
C = 10        # Number of colour channels


# ── Grid I/O ──────────────────────────────────────────────────────────────────

def load_task(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def grid_to_tensor(grid: list[list[int]]) -> np.ndarray:
    """
    Convert ARC integer grid to the competition's fixed [1, 10, 30, 30] format.
    Grid placed in top-left; everything else is zero-hot (all channels = 0).
    """
    arr = np.array(grid, dtype=np.int64)
    H, W = arr.shape
    assert H <= CANVAS and W <= CANVAS, f"Grid {H}×{W} exceeds {CANVAS}×{CANVAS} canvas"
    out = np.zeros((1, C, CANVAS, CANVAS), dtype=np.float32)
    for c in range(C):
        out[0, c, :H, :W] = (arr == c).astype(np.float32)
    return out


def tensor_to_grid(tensor: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Extract an H×W integer grid from a [1, 10, 30, 30] output tensor.
    Uses argmax over channel dim within the H×W region.
    """
    return np.argmax(tensor[0, :, :H, :W], axis=0).astype(np.int64)


def infer_grid_size(tensor: np.ndarray) -> tuple[int, int]:
    """
    Infer the actual grid size from a [1, 10, 30, 30] tensor.
    A pixel is 'inside' if any channel is non-zero.
    """
    inside = tensor[0].max(axis=0) > 0   # [30, 30] bool
    rows = np.any(inside, axis=1)          # which rows have content
    cols = np.any(inside, axis=0)          # which cols have content
    H = int(rows.sum())
    W = int(cols.sum())
    return max(H, 1), max(W, 1)


def grid_to_array(grid: list[list[int]]) -> np.ndarray:
    return np.array(grid, dtype=np.int64)


# ── ONNX inference ────────────────────────────────────────────────────────────

def run_onnx(onnx_path: str, grid: list[list[int]],
             out_H: int | None = None, out_W: int | None = None) -> np.ndarray:
    """
    Run ONNX on a single ARC grid.
    out_H, out_W: expected output size (defaults to input size if not given).
    This must be set correctly for tasks where output size ≠ input size.
    """
    import onnxruntime as ort
    H, W = len(grid), len(grid[0])
    if out_H is None: out_H = H
    if out_W is None: out_W = W
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = grid_to_tensor(grid)
    out = session.run(None, {session.get_inputs()[0].name: inp})[0]
    return tensor_to_grid(out, out_H, out_W)


def validate_onnx(onnx_path: str, task: dict, splits: list[str] | None = None) -> bool:
    """Return True only if ALL pairs in all specified splits are exactly correct."""
    if splits is None:
        splits = ["train", "test", "arc-gen"]
    for split in splits:
        for pair in task.get(split, []):
            expected = grid_to_array(pair["output"])
            out_H, out_W = expected.shape
            pred = run_onnx(onnx_path, pair["input"], out_H=out_H, out_W=out_W)
            if not np.array_equal(pred, expected):
                return False
    return True


def validate_callable(fn, task: dict, splits: list[str] | None = None) -> bool:
    """Validate a Python function numpy_array→numpy_array instead of ONNX."""
    if splits is None:
        splits = ["train", "test", "arc-gen"]
    for split in splits:
        for pair in task.get(split, []):
            inp = grid_to_array(pair["input"])
            pred = fn(inp)
            expected = grid_to_array(pair["output"])
            if not np.array_equal(pred, expected):
                return False
    return True


# ── Grid analysis helpers ─────────────────────────────────────────────────────

def task_grid_sizes(task: dict) -> list[tuple[tuple[int,int], tuple[int,int]]]:
    """Return list of ((inH,inW),(outH,outW)) for all pairs across all splits."""
    sizes = []
    for split in ["train", "test", "arc-gen"]:
        for pair in task.get(split, []):
            inp = grid_to_array(pair["input"])
            out = grid_to_array(pair["output"])
            sizes.append((inp.shape, out.shape))
    return sizes


def consistent_size_io(task: dict) -> bool:
    """True if all pairs have the same input size and same output size."""
    sizes = task_grid_sizes(task)
    if not sizes:
        return False
    in_sizes  = [s[0] for s in sizes]
    out_sizes = [s[1] for s in sizes]
    return len(set(in_sizes)) == 1 and len(set(out_sizes)) == 1


def same_size_io(task: dict) -> bool:
    """True if for every pair, input and output have the same shape."""
    for split in ["train", "test", "arc-gen"]:
        for pair in task.get(split, []):
            if grid_to_array(pair["input"]).shape != grid_to_array(pair["output"]).shape:
                return False
    return True


def detect_color_mapping(inp: np.ndarray, out: np.ndarray) -> dict[int,int] | None:
    """
    If transformation is a pure per-pixel color remap (positions unchanged),
    return the consistent mapping. Else None.
    """
    if inp.shape != out.shape:
        return None
    mapping = {}
    for v_in, v_out in zip(inp.flat, out.flat):
        s, d = int(v_in), int(v_out)
        if s in mapping:
            if mapping[s] != d:
                return None
        else:
            mapping[s] = d
    return mapping


def detect_transform(inp: np.ndarray, out: np.ndarray) -> str | None:
    """
    Detect the spatial transform applied to inp that produces out.
    Returns a string key or None.
    """
    if np.array_equal(inp, out):
        return "identity"
    if inp.shape == out.shape:
        if np.array_equal(np.fliplr(inp), out):   return "flip_h"
        if np.array_equal(np.flipud(inp), out):   return "flip_v"
        if np.array_equal(np.rot90(inp, 2), out): return "rotate_180"
    if inp.shape == out.shape[::-1]:
        if np.array_equal(np.rot90(inp, 1), out): return "rotate_90"
        if np.array_equal(np.rot90(inp, 3), out): return "rotate_270"
        if np.array_equal(inp.T, out):            return "transpose"
        if np.array_equal(np.fliplr(inp.T), out): return "anti_transpose"
    return None


def detect_tiling(inp: np.ndarray, out: np.ndarray) -> tuple[int,int] | None:
    iH, iW = inp.shape
    oH, oW = out.shape
    if oH % iH != 0 or oW % iW != 0:
        return None
    n, m = oH // iH, oW // iW
    if np.array_equal(np.tile(inp, (n, m)), out):
        return (n, m)
    return None


def detect_gravity_down(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.shape != out.shape:
        return False
    for col in range(inp.shape[1]):
        nz = inp[:, col][inp[:, col] != 0]
        expected = np.zeros(inp.shape[0], dtype=inp.dtype)
        if len(nz) > 0:
            expected[-len(nz):] = nz
        if not np.array_equal(out[:, col], expected):
            return False
    return True


def detect_gravity_up(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.shape != out.shape:
        return False
    for col in range(inp.shape[1]):
        nz = inp[:, col][inp[:, col] != 0]
        expected = np.zeros(inp.shape[0], dtype=inp.dtype)
        if len(nz) > 0:
            expected[:len(nz)] = nz
        if not np.array_equal(out[:, col], expected):
            return False
    return True


# ── Full task analysis ────────────────────────────────────────────────────────

def analyse_task(task: dict) -> dict[str, Any]:
    """
    Detect transformation type using TRAIN pairs only (small, reliable set).
    The solve pipeline validates the built ONNX against ALL splits afterward.

    Key insight: arc-gen has 262 pairs and may vary grid sizes or color palettes,
    so running detection on all pairs kills most valid detections. Instead we
    detect on train, build, then let validate_onnx() be the final gatekeeper.
    """
    # Detection pairs: train only (most reliable)
    detect_pairs = [
        (grid_to_array(p["input"]), grid_to_array(p["output"]))
        for p in task.get("train", [])
    ]
    # All pairs: for size/shape metadata
    all_pairs = []
    for split in ["train", "test", "arc-gen"]:
        for p in task.get(split, []):
            all_pairs.append((grid_to_array(p["input"]), grid_to_array(p["output"])))

    if not detect_pairs:
        return {}

    # --- Color mapping (from train only) ---
    mappings = [detect_color_mapping(i, o) for i, o in detect_pairs]
    is_color_perm = all(m is not None for m in mappings)
    if is_color_perm:
        ref = mappings[0]
        is_color_perm = all(m == ref for m in mappings[1:])
    color_mapping = mappings[0] if is_color_perm else None

    # --- Spatial transform (from train only) ---
    transforms = [detect_transform(i, o) for i, o in detect_pairs]
    consistent_transform = (
        transforms[0]
        if transforms and len(set(str(t) for t in transforms)) == 1
        else None
    )

    # --- Tiling (from train only) ---
    tilings = [detect_tiling(i, o) for i, o in detect_pairs]
    is_tiling = all(t is not None for t in tilings)
    tiling_factor = None
    if is_tiling:
        ref = tilings[0]
        is_tiling = all(t == ref for t in tilings[1:])
        tiling_factor = ref if is_tiling else None

    # --- Gravity (from train only) ---
    is_gravity_down = all(detect_gravity_down(i, o) for i, o in detect_pairs)
    is_gravity_up   = all(detect_gravity_up(i, o)   for i, o in detect_pairs)

    # --- Size info (train pairs only, for building fixed-size networks) ---
    train_in_shapes  = list({i.shape for i, o in detect_pairs})
    train_out_shapes = list({o.shape for i, o in detect_pairs})

    return {
        "color_permutation":    is_color_perm,
        "color_mapping":        color_mapping,
        "spatial_transform":    consistent_transform,
        "tiling":               is_tiling,
        "tiling_factor":        tiling_factor,
        "gravity_down":         is_gravity_down,
        "gravity_up":           is_gravity_up,
        "same_io_shape":        all(i.shape == o.shape for i, o in detect_pairs),
        "input_shapes":         train_in_shapes,
        "output_shapes":        train_out_shapes,
        "n_train":              len(detect_pairs),
        "n_pairs":              len(all_pairs),
    }
