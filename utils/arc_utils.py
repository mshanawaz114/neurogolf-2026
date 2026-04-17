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
from collections import deque

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


def detect_translation(inp: np.ndarray, out: np.ndarray) -> tuple[int, int] | None:
    """Detect pure zero-filled translation on same-sized grids."""
    if inp.shape != out.shape:
        return None
    H, W = inp.shape
    for dy in range(-H + 1, H):
        for dx in range(-W + 1, W):
            shifted = np.zeros_like(inp)
            y0 = max(0, dy)
            y1 = min(H, H + dy)
            x0 = max(0, dx)
            x1 = min(W, W + dx)
            sy0 = max(0, -dy)
            sy1 = sy0 + (y1 - y0)
            sx0 = max(0, -dx)
            sx1 = sx0 + (x1 - x0)
            shifted[y0:y1, x0:x1] = inp[sy0:sy1, sx0:sx1]
            if np.array_equal(shifted, out):
                return (dy, dx)
    return None


def detect_upscale(inp: np.ndarray, out: np.ndarray, max_scale: int = 5) -> tuple[int, int] | None:
    """Detect integer nearest-neighbour upscaling."""
    for sy in range(2, max_scale + 1):
        for sx in range(2, max_scale + 1):
            if np.array_equal(np.repeat(np.repeat(inp, sy, axis=0), sx, axis=1), out):
                return (sy, sx)
    return None


def detect_trim_bbox(inp: np.ndarray, out: np.ndarray) -> tuple[int, int, int, int, int] | None:
    """Detect crop to the bounding box of all pixels not equal to a background colour."""
    for bg in range(10):
        ys, xs = np.where(inp != bg)
        if len(ys) == 0:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        if np.array_equal(inp[y0:y1, x0:x1], out):
            return bg, y0, y1, x0, x1
    return None


def detect_fixed_submatrix(inp: np.ndarray, out: np.ndarray) -> tuple[int, int, int, int] | None:
    """Detect whether output is a fixed rectangular crop of the input."""
    in_h, in_w = inp.shape
    out_h, out_w = out.shape
    if out_h > in_h or out_w > in_w:
        return None
    for y0 in range(in_h - out_h + 1):
        for x0 in range(in_w - out_w + 1):
            y1 = y0 + out_h
            x1 = x0 + out_w
            if np.array_equal(inp[y0:y1, x0:x1], out):
                return y0, y1, x0, x1
    return None


def detect_color_count_crop(inp: np.ndarray, out: np.ndarray, mode: str) -> bool:
    """
    Detect crop-to-bbox of a selected non-zero colour, preserving only that colour.
    mode:
      - "max": select the most frequent non-zero colour
      - "min": select the least frequent non-zero colour
    """
    vals, cnts = np.unique(inp[inp != 0], return_counts=True)
    if len(vals) == 0:
        return False

    counts = [(int(v), int(c)) for v, c in zip(vals.tolist(), cnts.tolist())]
    if mode == "max":
        color = max(counts, key=lambda kv: (kv[1], -kv[0]))[0]
    elif mode == "min":
        color = min(counts, key=lambda kv: (kv[1], kv[0]))[0]
    else:
        raise ValueError(f"unknown color-count mode: {mode}")

    ys, xs = np.where(inp == color)
    if len(ys) == 0:
        return False
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    crop = np.zeros((y1 - y0, x1 - x0), dtype=inp.dtype)
    sub = inp[y0:y1, x0:x1]
    crop[sub == color] = color
    return np.array_equal(crop, out)


def detect_color_bbox_crop(inp: np.ndarray, out: np.ndarray, mode: str) -> bool:
    """
    Detect crop-to-bbox of the selected non-zero colour, where selection is based
    on the colour's overall bounding-box area in the input.
    mode:
      - "min_bbox": smallest bbox area, then smallest colour on ties
      - "max_bbox": largest bbox area, then smallest colour on ties
    """
    colors = [int(v) for v in np.unique(inp) if int(v) != 0]
    if not colors:
        return False

    items = []
    for color in colors:
        ys, xs = np.where(inp == color)
        if len(ys) == 0:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        items.append((color, (y1 - y0) * (x1 - x0), y0, y1, x0, x1))
    if not items:
        return False

    if mode == "min_bbox":
        color, _, y0, y1, x0, x1 = min(items, key=lambda t: (t[1], t[0]))
    elif mode == "max_bbox":
        color, _, y0, y1, x0, x1 = max(items, key=lambda t: (t[1], -t[0]))
    else:
        raise ValueError(f"unknown color-bbox mode: {mode}")

    crop = np.zeros((y1 - y0, x1 - x0), dtype=inp.dtype)
    sub = inp[y0:y1, x0:x1]
    crop[sub == color] = color
    return np.array_equal(crop, out)


def detect_color_count_preserve_crop(inp: np.ndarray, out: np.ndarray, mode: str) -> bool:
    """
    Detect crop of the bounding box of a selected non-zero colour while preserving
    all colours inside that rectangle.
    mode:
      - "min": select the least frequent non-zero colour
      - "max": select the most frequent non-zero colour
    """
    vals, cnts = np.unique(inp[inp != 0], return_counts=True)
    if len(vals) == 0:
        return False

    counts = [(int(v), int(c)) for v, c in zip(vals.tolist(), cnts.tolist())]
    if mode == "min":
        color = min(counts, key=lambda kv: (kv[1], kv[0]))[0]
    elif mode == "max":
        color = max(counts, key=lambda kv: (kv[1], -kv[0]))[0]
    else:
        raise ValueError(f"unknown color-count preserve mode: {mode}")

    ys, xs = np.where(inp == color)
    if len(ys) == 0:
        return False
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return np.array_equal(inp[y0:y1, x0:x1], out)


def detect_color_bbox_preserve_flip(inp: np.ndarray, out: np.ndarray, mode: str) -> bool:
    """
    Detect crop of the bounding box of a selected non-zero colour while preserving
    all colours inside that rectangle, followed by a horizontal flip.
    mode:
      - "min_bbox": select the smallest overall colour bounding box
      - "max_bbox": select the largest overall colour bounding box
    """
    colors = [int(v) for v in np.unique(inp) if int(v) != 0]
    if not colors:
        return False

    items = []
    for color in colors:
        ys, xs = np.where(inp == color)
        if len(ys) == 0:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        items.append((color, (y1 - y0) * (x1 - x0), y0, y1, x0, x1))
    if not items:
        return False

    if mode == "min_bbox":
        _, _, y0, y1, x0, x1 = min(items, key=lambda t: (t[1], t[0]))
    elif mode == "max_bbox":
        _, _, y0, y1, x0, x1 = max(items, key=lambda t: (t[1], -t[0]))
    else:
        raise ValueError(f"unknown color-bbox preserve-flip mode: {mode}")

    return np.array_equal(np.fliplr(inp[y0:y1, x0:x1]), out)


def detect_self_kron_mask(inp: np.ndarray, out: np.ndarray) -> bool:
    """
    Detect output = kron(inp != 0, inp), i.e. tile the whole input into blocks
    selected by the input's own non-zero mask.
    """
    mask = (inp != 0).astype(inp.dtype)
    pred = np.kron(mask, inp)
    return pred.shape == out.shape and np.array_equal(pred, out)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in a binary object using 4-connected background flood-fill.
    This matches scipy.ndimage.binary_fill_holes(..., structure=cross).
    """
    h, w = mask.shape
    open_cells = ~mask
    reachable = np.zeros_like(mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if open_cells[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if open_cells[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and open_cells[rr, cc] and not reachable[rr, cc]:
                reachable[rr, cc] = True
                q.append((rr, cc))

    return open_cells & ~reachable


def detect_color_hole_fill(inp: np.ndarray, out: np.ndarray) -> tuple[int, int] | None:
    """
    Detect tasks that preserve a single non-zero boundary colour and fill the
    holes it encloses with a new colour.
    """
    if inp.shape != out.shape:
        return None

    boundary_colors = [int(v) for v in np.unique(inp) if int(v) != 0]
    if len(boundary_colors) != 1:
        return None
    boundary = boundary_colors[0]

    changed = inp != out
    if np.any(inp[changed] != 0):
        return None
    fill_colors = [int(v) for v in np.unique(out[changed])]
    if len(fill_colors) != 1 or fill_colors[0] == 0:
        return None
    fill = fill_colors[0]

    pred = inp.copy()
    pred[_fill_holes(inp == boundary)] = fill
    if np.array_equal(pred, out):
        return boundary, fill
    return None


def detect_corner_rectangle_fill(inp: np.ndarray, out: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]] | None:
    """
    Detect tasks where a single non-zero color marks the four corners of one or
    more axis-aligned rectangles, and the output fills each rectangle interior
    with a single new color.
    """
    if inp.shape != out.shape:
        return None

    boundary_colors = [int(v) for v in np.unique(inp) if int(v) != 0]
    if len(boundary_colors) != 1:
        return None
    boundary = boundary_colors[0]

    changed = inp != out
    if np.any(inp[changed] != 0):
        return None
    fill_colors = [int(v) for v in np.unique(out[changed])]
    if len(fill_colors) != 1 or fill_colors[0] == 0:
        return None
    fill = fill_colors[0]

    pts = set(map(tuple, np.argwhere(inp == boundary).tolist()))
    mask = np.zeros_like(inp, dtype=bool)
    sizes: set[tuple[int, int]] = set()
    for r1, c1 in pts:
        for r2, c2 in pts:
            if r2 <= r1 + 1 or c2 <= c1 + 1:
                continue
            if {(r1, c1), (r1, c2), (r2, c1), (r2, c2)}.issubset(pts):
                mask[r1 + 1 : r2, c1 + 1 : c2] = True
                sizes.add((r2 - r1, c2 - c1))

    if not sizes or not np.array_equal(mask, changed):
        return None

    pred = inp.copy()
    pred[mask] = fill
    if np.array_equal(pred, out):
        return boundary, fill, tuple(sorted(sizes))
    return None


def detect_horizontal_gap_fill(inp: np.ndarray, out: np.ndarray) -> tuple[int, int] | None:
    """
    Detect tasks where a single colour gains fill pixels exactly in one-cell
    horizontal gaps between two same-colour pixels on the same row.
    """
    if inp.shape != out.shape:
        return None

    boundary_colors = [int(v) for v in np.unique(inp) if int(v) != 0]
    if len(boundary_colors) != 1:
        return None
    boundary = boundary_colors[0]

    changed = inp != out
    if np.any(inp[changed] != 0):
        return None
    fill_colors = [int(v) for v in np.unique(out[changed])]
    if len(fill_colors) != 1 or fill_colors[0] == 0:
        return None
    fill = fill_colors[0]

    mask = inp == boundary
    pred_mask = np.zeros_like(mask, dtype=bool)
    for r in range(mask.shape[0]):
        cols = np.where(mask[r])[0]
        colset = set(cols.tolist())
        for c in cols:
            if c + 2 in colset and c + 1 not in colset:
                pred_mask[r, c + 1] = True

    if not np.array_equal(pred_mask, changed):
        return None

    pred = inp.copy()
    pred[pred_mask] = fill
    if np.array_equal(pred, out):
        return boundary, fill
    return None


def detect_l_corner_fill(inp: np.ndarray, out: np.ndarray) -> tuple[int, int] | None:
    """
    Detect tasks where a single colour forms 3 cells of a 2x2 block and the
    output fills the missing corner with a new colour.
    """
    if inp.shape != out.shape:
        return None

    boundary_colors = [int(v) for v in np.unique(inp) if int(v) != 0]
    if len(boundary_colors) != 1:
        return None
    boundary = boundary_colors[0]

    changed = inp != out
    if np.any(inp[changed] != 0):
        return None
    fill_colors = [int(v) for v in np.unique(out[changed])]
    if len(fill_colors) != 1 or fill_colors[0] == 0:
        return None
    fill = fill_colors[0]

    mask = inp == boundary
    pred_mask = np.zeros_like(mask, dtype=bool)
    for r in range(mask.shape[0] - 1):
        for c in range(mask.shape[1] - 1):
            sub = mask[r : r + 2, c : c + 2]
            if sub.sum() == 3:
                miss = np.argwhere(~sub)
                if len(miss) == 1:
                    rr, cc = miss[0]
                    pred_mask[r + rr, c + cc] = True

    if not np.array_equal(pred_mask, changed):
        return None

    pred = inp.copy()
    pred[pred_mask] = fill
    if np.array_equal(pred, out):
        return boundary, fill
    return None


def detect_bounce_seed(inp: np.ndarray, out: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Detect a single non-zero seed at bottom-left expanding into a triangular-wave
    path up the grid.
    Returns (seed_color, height, width, period).
    """
    if inp.shape != out.shape:
        return None
    h, w = inp.shape
    nz = np.argwhere(inp != 0)
    if len(nz) != 1:
        return None
    r0, c0 = map(int, nz[0])
    seed = int(inp[r0, c0])
    if (r0, c0) != (h - 1, 0):
        return None

    pred = np.zeros_like(inp)
    period = max(1, 2 * (w - 1))
    for r in range(h):
        t = (h - 1) - r
        if w == 1:
            c = 0
        else:
            m = t % period
            c = m if m <= w - 1 else period - m
        pred[r, c] = seed

    if np.array_equal(pred, out):
        return seed, h, w, period
    return None


def detect_spatial_color_transform(
    inp: np.ndarray, out: np.ndarray
) -> tuple[str, dict[int, int]] | None:
    """
    Detect a spatial transform followed by a consistent colour mapping.
    Pure colour remaps are handled separately by ColorPermSolver, so this is
    only used for non-identity spatial transforms.
    """
    for transform in [
        "flip_h",
        "flip_v",
        "rotate_180",
        "rotate_90",
        "rotate_270",
        "transpose",
        "anti_transpose",
    ]:
        transformed = {
            "flip_h": np.fliplr(inp),
            "flip_v": np.flipud(inp),
            "rotate_180": np.rot90(inp, 2),
            "rotate_90": np.rot90(inp, 1),
            "rotate_270": np.rot90(inp, 3),
            "transpose": inp.T,
            "anti_transpose": np.fliplr(inp.T),
        }[transform]
        mapping = detect_color_mapping(transformed, out)
        if mapping is not None and any(mapping.get(c, c) != c for c in range(10)):
            return transform, mapping
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

    # --- Translation (from train only) ---
    translations = [detect_translation(i, o) for i, o in detect_pairs]
    is_translation = all(t is not None for t in translations)
    translation = None
    if is_translation:
        ref = translations[0]
        is_translation = all(t == ref for t in translations[1:])
        translation = ref if is_translation else None

    # --- Integer upscaling (from train only) ---
    upscales = [detect_upscale(i, o) for i, o in detect_pairs]
    is_upscale = all(s is not None for s in upscales)
    upscale_factor = None
    if is_upscale:
        ref = upscales[0]
        is_upscale = all(s == ref for s in upscales[1:])
        upscale_factor = ref if is_upscale else None

    # --- Spatial transform + colour mapping (from train only) ---
    spatial_color = [detect_spatial_color_transform(i, o) for i, o in detect_pairs]
    is_spatial_color = all(sc is not None for sc in spatial_color)
    spatial_color_transform = None
    spatial_color_mapping = None
    if is_spatial_color:
        ref_t, ref_m = spatial_color[0]
        is_spatial_color = all(t == ref_t and m == ref_m for t, m in spatial_color[1:])
        if is_spatial_color:
            spatial_color_transform = ref_t
            spatial_color_mapping = ref_m

    # --- Crop to non-background bounding box (from train only) ---
    trim_bboxes = [detect_trim_bbox(i, o) for i, o in detect_pairs]
    is_trim_bbox = all(tb is not None for tb in trim_bboxes)
    trim_bbox_bg = None
    trim_bbox_candidates = None
    if is_trim_bbox:
        ref_bg = trim_bboxes[0][0]
        is_trim_bbox = all(tb[0] == ref_bg for tb in trim_bboxes[1:])
        if is_trim_bbox:
            trim_bbox_bg = ref_bg
            trim_bbox_candidates = sorted({tb[1:] for tb in trim_bboxes})

    # --- Fixed submatrix crop (from train only) ---
    fixed_submatrices = [detect_fixed_submatrix(i, o) for i, o in detect_pairs]
    is_fixed_submatrix = all(fs is not None for fs in fixed_submatrices)
    fixed_submatrix = None
    if is_fixed_submatrix:
        ref = fixed_submatrices[0]
        is_fixed_submatrix = all(fs == ref for fs in fixed_submatrices[1:])
        fixed_submatrix = ref if is_fixed_submatrix else None

    # --- Crop bbox of selected colour by count rule (from train only) ---
    min_count_color_crop = all(detect_color_count_crop(i, o, "min") for i, o in detect_pairs)
    max_count_color_crop = all(detect_color_count_crop(i, o, "max") for i, o in detect_pairs)
    color_count_crop_mode = None
    if min_count_color_crop ^ max_count_color_crop:
        color_count_crop_mode = "min" if min_count_color_crop else "max"

    # --- Crop bbox of selected colour by bbox-area rule (from train only) ---
    min_bbox_color_crop = all(detect_color_bbox_crop(i, o, "min_bbox") for i, o in detect_pairs)
    max_bbox_color_crop = all(detect_color_bbox_crop(i, o, "max_bbox") for i, o in detect_pairs)
    color_bbox_crop_mode = None
    if min_bbox_color_crop ^ max_bbox_color_crop:
        color_bbox_crop_mode = "min_bbox" if min_bbox_color_crop else "max_bbox"

    # --- Crop bbox of selected colour by count rule, preserving full subgrid ---
    min_count_preserve_crop = all(detect_color_count_preserve_crop(i, o, "min") for i, o in detect_pairs)
    max_count_preserve_crop = all(detect_color_count_preserve_crop(i, o, "max") for i, o in detect_pairs)
    color_count_preserve_crop_mode = None
    if min_count_preserve_crop ^ max_count_preserve_crop:
        color_count_preserve_crop_mode = "min" if min_count_preserve_crop else "max"

    # --- Crop bbox of selected colour by bbox rule, preserve subgrid, then flip_h ---
    min_bbox_preserve_flip = all(detect_color_bbox_preserve_flip(i, o, "min_bbox") for i, o in detect_pairs)
    max_bbox_preserve_flip = all(detect_color_bbox_preserve_flip(i, o, "max_bbox") for i, o in detect_pairs)
    color_bbox_preserve_flip_mode = None
    if min_bbox_preserve_flip ^ max_bbox_preserve_flip:
        color_bbox_preserve_flip_mode = "min_bbox" if min_bbox_preserve_flip else "max_bbox"

    # --- Self-mask Kronecker tiling ---
    self_kron_mask = all(detect_self_kron_mask(i, o) for i, o in detect_pairs)

    # --- Fill holes in a single-colour boundary mask ---
    hole_fills = [detect_color_hole_fill(i, o) for i, o in detect_pairs]
    is_hole_fill = all(hf is not None for hf in hole_fills)
    hole_fill_boundary = None
    hole_fill_color = None
    if is_hole_fill:
        ref_boundary, ref_fill = hole_fills[0]
        is_hole_fill = all(hf == (ref_boundary, ref_fill) for hf in hole_fills[1:])
        if is_hole_fill:
            hole_fill_boundary = ref_boundary
            hole_fill_color = ref_fill

    # --- Fill rectangle interiors implied by same-color corner markers ---
    corner_rects = [detect_corner_rectangle_fill(i, o) for i, o in detect_pairs]
    is_corner_rect_fill = all(cr is not None for cr in corner_rects)
    corner_rect_boundary = None
    corner_rect_fill = None
    corner_rect_sizes = None
    if is_corner_rect_fill:
        ref_boundary, ref_fill, ref_sizes = corner_rects[0]
        is_corner_rect_fill = all((cr[0], cr[1]) == (ref_boundary, ref_fill) for cr in corner_rects[1:])
        if is_corner_rect_fill:
            corner_rect_boundary = ref_boundary
            corner_rect_fill = ref_fill
            size_set = set(ref_sizes)
            for cr in corner_rects[1:]:
                size_set.update(cr[2])
            corner_rect_sizes = tuple(sorted(size_set))

    # --- Fill horizontal one-cell gaps between matching pixels ---
    hgap_fills = [detect_horizontal_gap_fill(i, o) for i, o in detect_pairs]
    is_hgap_fill = all(hg is not None for hg in hgap_fills)
    hgap_boundary = None
    hgap_fill = None
    if is_hgap_fill:
        ref_boundary, ref_fill = hgap_fills[0]
        is_hgap_fill = all(hg == (ref_boundary, ref_fill) for hg in hgap_fills[1:])
        if is_hgap_fill:
            hgap_boundary = ref_boundary
            hgap_fill = ref_fill

    # --- Complete missing corners of 2x2 L-shapes ---
    lcorner_fills = [detect_l_corner_fill(i, o) for i, o in detect_pairs]
    is_lcorner_fill = all(lc is not None for lc in lcorner_fills)
    lcorner_boundary = None
    lcorner_fill = None
    if is_lcorner_fill:
        ref_boundary, ref_fill = lcorner_fills[0]
        is_lcorner_fill = all(lc == (ref_boundary, ref_fill) for lc in lcorner_fills[1:])
        if is_lcorner_fill:
            lcorner_boundary = ref_boundary
            lcorner_fill = ref_fill

    # --- Single seed expands into a bounce / triangular-wave path ---
    bounce_seeds = [detect_bounce_seed(i, o) for i, o in detect_pairs]
    is_bounce_seed = all(bs is not None for bs in bounce_seeds)
    bounce_seed_color = None
    bounce_height = None
    bounce_widths = None
    if is_bounce_seed:
        ref_color, ref_h, _, _ = bounce_seeds[0]
        is_bounce_seed = all(bs[0] == ref_color and bs[1] == ref_h for bs in bounce_seeds[1:])
        if is_bounce_seed:
            bounce_seed_color = ref_color
            bounce_height = ref_h
            bounce_widths = tuple(sorted({bs[2] for bs in bounce_seeds}))

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
        "translation":          is_translation,
        "translation_delta":    translation,
        "upscale":              is_upscale,
        "upscale_factor":       upscale_factor,
        "trim_bbox":            is_trim_bbox,
        "trim_bbox_bg":         trim_bbox_bg,
        "trim_bbox_candidates": trim_bbox_candidates,
        "fixed_submatrix":      is_fixed_submatrix,
        "fixed_submatrix_rect": fixed_submatrix,
        "color_count_crop":     color_count_crop_mode is not None,
        "color_count_crop_mode": color_count_crop_mode,
        "color_bbox_crop":      color_bbox_crop_mode is not None,
        "color_bbox_crop_mode": color_bbox_crop_mode,
        "color_count_preserve_crop": color_count_preserve_crop_mode is not None,
        "color_count_preserve_crop_mode": color_count_preserve_crop_mode,
        "color_bbox_preserve_flip": color_bbox_preserve_flip_mode is not None,
        "color_bbox_preserve_flip_mode": color_bbox_preserve_flip_mode,
        "self_kron_mask":       self_kron_mask,
        "color_hole_fill":      is_hole_fill,
        "color_hole_fill_boundary": hole_fill_boundary,
        "color_hole_fill_fill": hole_fill_color,
        "corner_rect_fill":     is_corner_rect_fill,
        "corner_rect_boundary": corner_rect_boundary,
        "corner_rect_fill_color": corner_rect_fill,
        "corner_rect_sizes":    corner_rect_sizes,
        "horizontal_gap_fill":  is_hgap_fill,
        "horizontal_gap_boundary": hgap_boundary,
        "horizontal_gap_fill_color": hgap_fill,
        "lcorner_fill":         is_lcorner_fill,
        "lcorner_boundary":     lcorner_boundary,
        "lcorner_fill_color":   lcorner_fill,
        "bounce_seed":          is_bounce_seed,
        "bounce_seed_color":    bounce_seed_color,
        "bounce_height":        bounce_height,
        "bounce_widths":        bounce_widths,
        "spatial_color":        is_spatial_color,
        "spatial_color_transform": spatial_color_transform,
        "spatial_color_mapping": spatial_color_mapping,
        "same_io_shape":        all(i.shape == o.shape for i, o in detect_pairs),
        "input_shapes":         train_in_shapes,
        "output_shapes":        train_out_shapes,
        "n_train":              len(detect_pairs),
        "n_pairs":              len(all_pairs),
    }
