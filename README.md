# NeuroGolf 2026

Minimal ONNX neural networks for ARC-AGI image transformations — [Kaggle NeuroGolf 2026 Championship](https://www.kaggle.com/competitions/neurogolf-2026)

Score formula: `max(1, 25 - ln(params + memory_bytes + MACs))` — smaller networks score higher.

---

## Setup (macOS / Linux)

### 1. Install Python 3.11+

**macOS (recommended via Homebrew):**
```bash
brew install python@3.11
```

Or download directly from [python.org](https://www.python.org/downloads/).

### 2. Create a virtual environment

```bash
cd neurogolf-2026
python3 -m venv .venv
source .venv/bin/activate
```

> Add `source .venv/bin/activate` to your shell session each time you open a new terminal, or add it to your `.zshrc`.

### 3. Install dependencies

```bash
pip3 install -r requirements.txt
```

### 4. Download task data via Kaggle CLI

```bash
# Install Kaggle CLI
pip3 install kaggle

# Get your API key: kaggle.com → Settings → API → Create New Token
# Move it into place:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download all 400 task files
python3 scripts/download_arc.py
```

---

## Solve and Submit

```bash
# Analytical solvers only (fast, ~seconds)
python3 scripts/solve_all.py --no-learned

# Full pipeline including learned fallback (slow, ~minutes per task)
python3 scripts/solve_all.py

# Solve a single task
python3 scripts/solve_all.py --task task001

# Package submission
make zip
# → submission.zip ready to upload to Kaggle
```

### Kaggle-Safe Recovery Build

When Kaggle rejects a submission with `Error processing one or more onnx networks`,
use the full safe builder instead of hand-editing the zip:

```bash
python3 scripts/build_safe_submission.py

# or
make safe-zip

# build the stronger static-shape candidate that keeps the two gravity tasks
python3 scripts/build_safe_submission.py --profile static23

# or
make static23-zip

# build the stronger selector-aware candidate that also keeps task300 + task310
python3 scripts/build_safe_submission.py --profile static25

# or
make static25-zip
```

That script:

- keeps only generated task ONNX files whose operator sets stay inside the current conservative allowlist
- fills every other task with a safe identity fallback
- writes a full `400`-file package to `submission_full_safe.zip`
- writes an audit trail to `safe_submission_report.csv`

Current conservative allowlist:
`Add`, `Clip`, `Concat`, `Conv`, `Mul`, `Pad`, `ReduceMax`, `Resize`, `Slice`, `Sub`, `Transpose`

Current `static23` allowlist:
the conservative set plus
`Cast`, `CumSum`, `Equal`, `ReduceSum`, `Unsqueeze`

Current `static25` allowlist:
the `static23` set plus
`Abs`, `MatMul`, `Relu`, `Reshape`

The builder also supports exact task-level ablations, which is useful when
multiple risky solver families share the same operator set:

```bash
python3 scripts/build_safe_submission.py --profile static23 \
  --force-keep-task task300 \
  --force-keep-task task310
```

That lets you test a specific task or small family on Kaggle without turning on
every model that uses the same risky operators.

Based on the repo's April 17, 2026 Kaggle runs, treat a full `400`-file zip as required.
That is an inference from observed behavior:

- partial zips returned `400 Bad Request`
- some full zips uploaded but later failed ONNX processing
- only full packages can currently be trusted as real submission candidates

Suggested recovery workflow:

```bash
# 1. Solve tasks locally
python3 scripts/solve_all.py --no-learned

# 2. Build a Kaggle-safe full package
python3 scripts/build_safe_submission.py

# or build the stronger static-shape candidate
python3 scripts/build_safe_submission.py --profile static23

# 3. Inspect what was kept vs downgraded
sed -n '1,40p' safe_submission_report.csv

# 4. Submit the safe full zip
./.venv/bin/kaggle competitions submit -c neurogolf-2026 \
  -f submission_full_safe.zip \
  -m "Full safe fallback: 400 files, conservative ops only"
```

### Fast Repo Check

Use this before and after experiments so it is obvious what changed:

```bash
# See tracked edits + untracked solver files quickly
git status --short

# See the actual code diff for the current optimization pass
git diff

# Review just one file when a solver starts working or regresses
git diff -- solvers/spatial.py

# Inspect the latest solve summary if a run completed
sed -n '1,40p' results.csv
```

---

## Per-task Workflow

```bash
# Visualise a task
python3 utils/visualize.py --task tasks/task001.json

# Validate an ONNX solution
make validate TASK=001

# Estimate score/cost
make score TASK=001

# Open ONNX graph visually
make view TASK=001
```

---

## Data Format

Every input/output is padded to a fixed `[1, 10, 30, 30]` float32 tensor:

- **Inside the grid:** one channel = `1.0`, rest = `0.0`
- **Outside the grid:** all channels = `0.0` (zero-hot)
- Grids are always placed in the **top-left corner**
- Task files include `"train"`, `"test"`, and `"arc-gen"` (262 extra pairs) splits
- Networks must be correct on **all splits** including a private benchmark

---

## Project Structure

```
neurogolf-2026/
├── agent.md            ← full competition spec + solver strategy
├── tasks/              ← task001.json … task400.json
├── onnx/               ← generated .onnx files
├── solvers/            ← analytical + learned solvers
├── utils/              ← data, ONNX builder, scoring, validation
├── scripts/            ← download + solve pipeline
├── solutions/          ← per-task hand-crafted solutions
└── notebooks/          ← Jupyter exploration notebook
```

See [agent.md](agent.md) for the full technical specification.

---

## Solver Strategy

| Solver | Priority | Approach | Score/task |
|---|---|---|---|
| `SpatialSolver` | 5 | Slice + Pad + Transpose (zero MACs) | ~20+ |
| `TilingSolver` | 8 | Concat + Pad (zero MACs) | ~21+ |
| `ColorPermSolver` | 10 | 1×1 Conv with 10×10 weight matrix | ~13.6 |
| `TranslateSolver` | 12 | Slice + Pad translation (zero MACs) | ~21+ |
| `SelfKronMaskSolver` | 13 | Tile input by its own non-zero mask (`kron(mask,input)`) | ~21+ |
| `ColorHoleFillSolver` | 13 | Fill holes inside a single-colour boundary mask via border flood fill | ~11.9 |
| `CornerRectFillSolver` | 13 | Fill rectangle interiors implied by same-colour corner markers | ~12.7 |
| `HorizontalGapFillSolver` | 13 | Fill one-cell horizontal gaps between matching pixels on the same row | ~17.0 |
| `LCornerFillSolver` | 13 | Complete the missing corner of 2x2 L-shaped triominoes | ~15.4 |
| `BounceSeedSolver` | 13 | Expand a bottom-left seed into a triangular-wave bounce path by width | ~12.1 |
| `ColorBBoxCropSolver` | 14 | Select non-zero colour by bbox size and crop its mask | ~19-21 |
| `ColorBBoxPreserveFlipSolver` | 14 | Select non-zero colour by bbox size, crop its subgrid, then flip | ~20-21 |
| `ColorCountCropSolver` | 14 | Select min/max-frequency non-zero colour and crop its bbox | ~19-21 |
| `ColorCountPreserveCropSolver` | 14 | Select min/max-frequency non-zero colour and crop its full subgrid | ~20-21 |
| `UpscaleSolver` | 13 | Nearest-neighbour resize + crop | ~20+ |
| `FixedCropSolver` | 14 | Fixed rectangular submatrix crop | ~21+ |
| `TrimBBoxSolver` | 14 | Crop to bounding box of non-background pixels | ~18-20 |
| `GravitySolver` | 15 | Column-wise gravity via `CumSum` + row placement masks | ~19-20 |
| `LearnedSolver` | 90 | Adam-trained conv search with larger late-stage models | ~8–13 |

### Current Exact Coverage

As of the current repo state, the deterministic stack validates exactly on:

`task001`, `task002`, `task014`, `task031`, `task032`, `task036`,
`task049`, `task053`, `task078`, `task081`, `task087`, `task135`,
`task140`, `task150`, `task155`, `task177`, `task179`, `task223`,
`task241`, `task248`, `task251`, `task258`, `task273`, `task276`,
`task300`, `task307`, `task309`, `task310`, `task326`

That is `29` exact solves before counting any learned-fallback wins, with a
current deterministic total of `501.01` points.

### Current Kaggle Reality

Local deterministic score and accepted Kaggle score are not the same thing.

- Current best local deterministic total: `501.01`
- Current accepted Kaggle score: `177.23`

The gap exists because several stronger ONNX models still fail Kaggle processing.
Until the submission pipeline is stable, optimize for accepted full packages first.

### What To Detect First

When inspecting a new task, check these in order:

1. Same-shape pure spatial transform:
   `flip`, `rotate`, `transpose`
2. Pure colour remap:
   positions unchanged, colours differ
3. Same-shape structural fill:
   hole fill inside a single-colour boundary, or self-mask Kronecker tiling
4. Size-changing but structural transform:
   tiling, translation, integer upscale, bbox crop
5. Variable-size spatial family:
   if train works but `arc-gen` changes size, assume you need a multi-shape analytical ONNX
6. Only after those fail, spend time on learned fallback or one-off task logic

### Current Bottleneck

The main limitation is no longer the obvious spatial/color tasks. The remaining climb is mostly:

- higher-yield structural families not yet detected
- stronger learned coverage on the large unsolved set
- avoiding train-only detections that break on `test` or `arc-gen`

### Running with PyTorch (recommended, much faster)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 scripts/solve_all.py   # Full run with learned solver
```

PyTorch is ~50x faster than the NumPy fallback for training. On CPU (MacBook), expect ~1-2 minutes per unsolvable task.

---

## Resources

- [Competition page](https://www.kaggle.com/competitions/neurogolf-2026)
- [ARC Prize Foundation](https://arcprize.org)
- [ONNX operator reference](https://onnx.ai/onnx/operators/)
- [ARC-AGI dataset (v1)](https://github.com/fchollet/ARC-AGI)
