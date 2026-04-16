# NeuroGolf 2026 — Agent Specification

> **Competition:** The 2026 NeuroGolf Championship
> **Host:** Kaggle · IJCAI-ECAI 2026 Competitions Track
> **URL:** https://www.kaggle.com/competitions/neurogolf-2026
> **Repo:** https://github.com/mshanawaz114/neurogolf-2026
> **Last updated:** April 15, 2026

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

### Priority 10 — `ColorPermSolver`
Handles pure colour-permutation tasks (positions unchanged, colours remapped).
Uses a 1×1 conv with a 10×10 permutation weight matrix.
Score per task: ~**13.6**

### Priority 90 — `LearnedSolver`
Fallback: trains a tiny PyTorch net (multiple architectures tried smallest-first),
then exports to ONNX. Used only when no analytical solver applies.
Score per task: ~**9–13** depending on architecture needed.

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
│   ├── identity.py           ← IdentitySolver
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
2. **Run analysis** — check what `analyse_task()` returns (colour perm? flip? rotate?)
3. **Run auto-solver first** — `python3 scripts/solve_all.py --task taskNNN`
4. **If auto-solver fails**, inspect the task manually and write a custom solver in `solutions/taskNNN.py`
5. **Validate** — `make validate TASK=NNN`
6. **Score** — `make score TASK=NNN`
7. **Submit** — `make zip` then upload `submission.zip` to Kaggle (max 5/day)

### Design principles for manual solutions

- **Think analytically first** — can the transformation be expressed as a closed-form ONNX graph?
- Prefer `Slice`, `Pad`, `Transpose` (0 MACs) over `Conv` (many MACs)
- Prefer `Conv1×1` over `Conv3×3` when only colour matters
- The weight matrix in a 1×1 conv is a **10×10 linear map** — many colour tasks reduce to this
- For tasks where the output grid is smaller than the input: crop with `Slice`, then re-pad
- Avoid the banned ops: `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`
- Test against ALL splits (train + test + arc-gen) before submitting

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
