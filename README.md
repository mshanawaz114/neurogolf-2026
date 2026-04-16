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
| `ColorPermSolver` | 10 | 1×1 Conv with 10×10 weight matrix | ~13.6 |
| `TilingSolver` | 8 | Concat + Pad (zero MACs) | ~21+ |
| `GravitySolver` | 15 | Stub — falls through to LearnedSolver | — |
| `LearnedSolver` | 90 | Adam-trained tiny conv net (PyTorch or NumPy) | ~9–13 |

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
