# NeuroGolf 2026 — Agent Specification

> **Competition:** The 2026 NeuroGolf Championship
> **Host:** Kaggle · IJCAI-ECAI 2026 Competitions Track
> **URL:** https://www.kaggle.com/competitions/neurogolf-2026
> **Repo:** https://github.com/mshanawaz114/neurogolf-2026
> **Last updated:** April 16, 2026

---

## Mission

Design the **smallest possible ONNX neural networks** that correctly solve ARC-AGI image transformation tasks. Correct + minimal wins. Every byte and every multiply-accumulate operation counts against your score.

---

## Competition Timeline

| Event | Date (11:59 PM UTC unless noted) |
|---|---|
| Start Date | April 15, 2026 |
| Entry & Team Merger Deadline | July 8, 2026 |
| Final Submission Deadline | July 15, 2026 |
| Longest Leader window | April 16 12:00 AM UTC → July 15 11:59 PM UTC |

- **5 submissions per day max**
- **Select up to 2 Final Submissions** for judging
- Team size max: 5 members

---

## Current Leaderboard State (Day 1)

| Rank | Team | Score | Entries |
|---|---|---|---|
| 1 | Kameron Kilchrist | 1339.96 | 2 |
| 2 | Ali | 725.80 | 5 |
| 3 | Sergey Kuznetsov | 688.47 | 2 |
| Best public notebook | — | 210.46 | — |
| Baseline notebook | — | 132.78 | — |

Maximum possible score: **400 tasks × 25 pts = 10,000 pts**

---

## Critical Data Format (Read This First)

### Input tensor: always `[1, 10, 30, 30]`

Before being passed into any network, every input grid is converted to a fixed-size tensor:

```
shape: [BATCH=1, CHANNELS=10, HEIGHT=30, WIDTH=30]
dtype: float32
encoding: one-hot per pixel
```

- **Inside the grid:** exactly one channel = `1.0`, all others = `0.0`
- **Outside the grid (padding):** ALL channels = `0.0` (zero-hot)
- **Grids are always in the top-left corner** of the 30×30 canvas
- Minimum grid size: 1×1. Maximum: 30×30.

### Output tensor: also `[1, 10, 30, 30]`

Output must match the same format. Cells outside the expected output grid must be zero-hot.

### Task file format

Each `taskNNN.json` contains:

```json
{
  "train":   [ {"input": [[...]], "output": [[...]]}, ... ],
  "test":    [ {"input": [[...]], "output": [[...]]}, ... ],
  "arc-gen": [ {"input": [[...]], "output": [[...]]}, ... ]
}
```

- `"train"` + `"test"`: original ARC-AGI-1 pairs
- `"arc-gen"`: additional generated pairs from ARC-GEN-100K (262 pairs per task)
- **Your network must be correct on ALL three splits**, plus a private test set

---

## Scoring Formula

```
cost  = total_parameters + memory_footprint_bytes + total_MACs
score = max(1, 25 - ln(cost))
```

Where `ln` is the natural logarithm.

### Cost breakdown per operation type (30×30 canvas)

| Operation | Params | Bytes | MACs | Score |
|---|---|---|---|---|
| Slice/Transpose (flip, rotate) | ~4 int64 consts | ~32 B | **0** | ~21.4 |
| 1×1 Conv (10→10) | 100 | 400 B | 90,000 | ~13.6 |
| 3×3 Conv (10→10) | 900 | 3,600 B | 810,000 | ~10.8 |
| 3×3 Conv (10→16→10, 2-layer) | ~2,500 | ~10,000 B | ~2,700,000 | ~9.1 |

**Key insight:** Purely structural transforms (flip, rotate) cost almost nothing. Colour transforms need at minimum a 1×1 conv. Training is always a last resort.

---

## Functional Correctness Requirements

A network must pass **all four** validation sets to earn points:

1. `"train"` pairs in the task JSON
2. `"test"` pairs in the task JSON
3. `"arc-gen"` pairs (262 per task)
4. Private benchmark (unseen — prevents overfitting)

**Networks must generalise.** The private benchmark will have the same transformation but potentially different grids. Hard-coding specific pixel values will fail.

---

## ONNX Constraints

| Rule | Detail |
|---|---|
| Input shape | Static `[1, 10, 30, 30]` |
| Output shape | Static `[1, 10, 30, 30]` |
| Max file size | 1.44 MB per `.onnx` file |
| All tensor shapes | Must be statically defined |
| Banned operators | `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function` |
| Opset | 17 (recommended) |

---

## Submission Format

```
submission.zip
├── task001.onnx
├── task002.onnx
├── ...
└── task400.onnx
```

Partial submissions are valid. Build: `make zip`

---

## Solver Architecture (This Repo)

Solvers are tried in priority order (lowest number = tried first). The first solver that produces a **validated-correct** ONNX wins for that task.

### Priority 5 — `SpatialSolver`
Handles identity, flip_h, flip_v, rotate_90/180/270, transpose.
Uses ONNX `Slice` + `Pad` + `Transpose` — **zero MACs, near-zero params**.
Score per task: ~**21+**

### Priority 8 — `TilingSolver`
Handles direct input tiling such as `tile(input, n x m)`.
Uses repeated `Concat` + `Pad` — **zero MACs**.
Score per task: ~**21+**

### Priority 10 — `ColorPermSolver`
Handles pure colour-permutation tasks (positions unchanged, colours remapped).
Uses a 1×1 conv with a 10×10 permutation weight matrix.
Score per task: ~**13.6**

### Priority 12 — `TranslateSolver`
Handles zero-filled translations on same-sized grids.
Uses `Slice` + `Pad` analytically — **zero MACs**.
Score per task: ~**21+**

### Priority 14 — `ColorCountCropSolver`
Handles tasks where the output is the bounding-box crop of the selected non-zero colour:
- `min`: least frequent non-zero colour
- `max`: most frequent non-zero colour

Uses runtime count selection (`ArgMin`/`ArgMax`), dynamic `Gather`, and dynamic
`Slice` + `Pad` to place the cropped shape in the top-left output region.
Score per task: usually **19–21+**

### Priority 14 — `ColorBBoxCropSolver`
Handles tasks where the output is the bounding-box crop of the selected non-zero colour:
- `min_bbox`: smallest overall colour bounding box
- `max_bbox`: largest overall colour bounding box

Uses runtime bbox-area selection over colour channels, then dynamic `Gather`
and dynamic `Slice` + `Pad` to place the cropped mask in the top-left output region.
Score per task: usually **19–21+**

### Priority 14 — `ColorBBoxPreserveFlipSolver`
Handles tasks where the output is the horizontally flipped bounding-box crop of the
selected non-zero colour's full subgrid:
- `min_bbox`: smallest overall colour bounding box
- `max_bbox`: largest overall colour bounding box

Uses runtime bbox-area selection, dynamic bbox extraction from the selected colour,
then a horizontal `Slice`-flip before top-left padding.
Score per task: usually **20–21+**

### Priority 14 — `ColorCountPreserveCropSolver`
Handles tasks where the output is the bounding-box crop of the selected non-zero colour,
but preserves every colour inside that cropped rectangle:
- `min`: least frequent non-zero colour
- `max`: most frequent non-zero colour

Uses runtime count selection, dynamic bbox extraction from the selected colour,
and dynamic `Slice` + `Pad` on the original full tensor.
Score per task: usually **20–21+**

### Priority 13 — `UpscaleSolver`
Handles integer nearest-neighbour upscaling such as 2x2 or 3x3 expansion.
Uses ONNX `Resize` plus fixed crop/pad.
Score per task: ~**20+**

### Priority 14 — `TrimBBoxSolver`
Handles tasks where the output is the bounding-box crop of all non-background pixels.
Uses multi-candidate analytical crop selection across observed splits.
Score per task: usually **18–20+** depending on selector graph size.

### Priority 14 — `FixedCropSolver`
Handles tasks where the output is a fixed rectangular crop of the input.
Uses `Slice` + `Pad` analytically after reconfirming the same rectangle across all splits.
Score per task: usually **21+**

### Priority 15 — `GravitySolver`
Handles column-wise gravity for background colour `0`:
- `gravity_down`: compact non-zero cells to the bottom of each column
- `gravity_up`: compact non-zero cells to the top of each column

Uses `CumSum` to compute per-column ranks, then analytical row-placement masks
to move each source cell into its target row.
Score per task: usually **19–20+**

### Priority 90 — `LearnedSolver`
Fallback: trains a PyTorch or NumPy conv net over a staged architecture search.
Now includes larger late-stage receptive fields and restart seeds so some harder tasks
have a better chance to convert from `0` into a real submission.
Score per task: ~**8–13** depending on architecture needed.

### Current Exact Validated Tasks

The current deterministic stack validates exactly on:

`task014`, `task031`, `task032`, `task036`, `task049`, `task053`,
`task078`, `task087`, `task135`, `task140`, `task150`, `task155`,
`task177`, `task179`, `task223`, `task241`, `task276`, `task300`,
`task307`, `task309`, `task310`, `task326`

Current exact solved count: **22**

This is still far from leaderboard-contending coverage, but it is the current
stable analytical base.

---

## Repository Structure

```
neurogolf-2026/
├── agent.md                  ← this file
├── README.md                 ← setup and usage guide
├── Makefile                  ← common commands
├── pyproject.toml            ← project config (black, ruff, pytest)
├── requirements.txt          ← Python dependencies
│
├── tasks/                    ← task001.json … task400.json (download separately)
├── onnx/                     ← generated .onnx output files
├── solutions/                ← hand-crafted solution scripts (per task)
│   └── template.py           ← PyTorch → ONNX template
│
├── solvers/                  ← auto-solver modules
│   ├── base.py               ← abstract BaseSolver
│   ├── spatial.py            ← SpatialSolver (flip/rotate/transpose)
│   ├── color_perm.py         ← ColorPermSolver (1×1 conv)
│   ├── tiling.py             ← TilingSolver
│   ├── translate.py          ← TranslateSolver
│   ├── color_bbox_crop.py    ← ColorBBoxCropSolver
│   ├── color_bbox_preserve_flip.py ← ColorBBoxPreserveFlipSolver
│   ├── color_count_crop.py   ← ColorCountCropSolver
│   ├── color_count_preserve_crop.py ← ColorCountPreserveCropSolver
│   ├── upscale.py            ← UpscaleSolver
│   ├── fixed_crop.py         ← FixedCropSolver
│   ├── trim_bbox.py          ← TrimBBoxSolver
│   ├── gravity.py            ← GravitySolver
│   └── learned.py            ← LearnedSolver (PyTorch training fallback)
│
├── utils/
│   ├── arc_utils.py          ← data loading, grid↔tensor, task analysis
│   ├── onnx_builder.py       ← analytical ONNX graph construction
│   ├── scoring.py            ← cost + score calculator
│   ├── validate.py           ← ONNX validation runner
│   └── visualize.py          ← grid visualisation
│
├── scripts/
│   ├── download_arc.py       ← download tasks via Kaggle CLI
│   └── solve_all.py          ← main pipeline: analyse → solve → validate → zip
│
├── notebooks/
│   └── explore_tasks.ipynb   ← task explorer + analysis notebook
│
└── tests/
    └── test_utils.py         ← pytest unit tests
```

---

## Agent Instructions: How to Solve a New Task

1. **Load and inspect** — `python3 utils/visualize.py --task tasks/taskNNN.json`
2. **Run analysis** — check what `analyse_task()` returns
   Look first for: spatial, tiling, translation, upscale, trim-bbox, colour remap
3. **Run auto-solver first** — `python3 scripts/solve_all.py --task taskNNN`
4. **If auto-solver fails**, inspect the task manually and write a custom solver in `solutions/taskNNN.py`
5. **Validate** — `make validate TASK=NNN`
6. **Score** — `make score TASK=NNN`
7. **Submit** — `make zip` then upload `submission.zip` to Kaggle (max 5/day)

### Git Workflow For Fast Detection

Use git as a quick scoreboard for the codebase itself:

```bash
git status --short
git diff
git diff -- solvers/learned.py
git diff -- utils/arc_utils.py
```

Recommended habit during solver work:

1. Run `git status --short` before changing anything
2. After each solver experiment, inspect `git diff -- <file>`
3. Keep a mental note of whether a change improved:
   exact validated task count, learned solve rate, or model cost
4. If a run created `results.csv`, read the top lines immediately to confirm whether the change actually moved solved coverage

### Design principles for manual solutions

- **Think analytically first** — can the transformation be expressed as a closed-form ONNX graph?
- Prefer `Slice`, `Pad`, `Transpose` (0 MACs) over `Conv` (many MACs)
- Prefer `Conv1×1` over `Conv3×3` when only colour matters
- The weight matrix in a 1×1 conv is a **10×10 linear map** — many colour tasks reduce to this
- For tasks where the output grid is smaller than the input: crop with `Slice`, then re-pad
- Be suspicious of train-only detections. If `arc-gen` changes sizes, the ONNX often needs runtime shape-selection branches.
- Avoid the banned ops: `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`
- Test against ALL splits (train + test + arc-gen) before submitting

### Current Climb Strategy

To move toward the top of the leaderboard, the repo should optimize in this order:

1. Add reusable exact solvers for structural families with real yield
2. Harden existing analytical solvers against variable-size `test` and `arc-gen` splits
3. Improve learned fallback only after the easy exact points are exhausted
4. Prefer one change that adds several validated tasks over many changes that only shrink a model already solving the same task

---

## Prizes

| Prize | Amount |
|---|---|
| First Place | $12,000 |
| Second Place | $10,000 |
| Third Place | $10,000 |
| Top Student Team (≥50% students) | $8,000 |
| Longest Leader | $10,000 |
| **Total** | **$50,000** |

---

## Key External Resources

| Resource | URL |
|---|---|
| Competition | https://www.kaggle.com/competitions/neurogolf-2026 |
| ARC Prize Foundation | https://arcprize.org |
| ONNX docs | https://onnx.ai |
| ONNX opset 17 reference | https://onnx.ai/onnx/operators/ |
| ARC-AGI-1 dataset | https://github.com/fchollet/ARC-AGI |

---

## Citation

```
Michael D. Moffitt, Walter Reade, Ashley Oldacre, and Addison Howard.
The 2026 NeuroGolf Championship.
https://kaggle.com/competitions/neurogolf-2026, 2026. Kaggle.
```
