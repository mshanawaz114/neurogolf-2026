# NeuroGolf 2026 — Agent Specification

> **Competition:** The 2026 NeuroGolf Championship  
> **Host:** Kaggle (part of IJCAI-ECAI 2026 Competitions Track)  
> **URL:** https://www.kaggle.com/competitions/neurogolf-2026  
> **Repo:** https://github.com/mshanawaz114/neurogolf-2026

---

## Mission

Design the **smallest possible neural networks** (in ONNX format) that correctly solve ARC-AGI image transformation tasks. The goal is not just correctness — it is correctness at minimum computational cost.

---

## Competition Timeline

| Event | Date (11:59 PM UTC) |
|---|---|
| Start Date | April 15, 2026 |
| Entry & Team Merger Deadline | July 8, 2026 |
| Final Submission Deadline | July 15, 2026 |
| Longest Leader window | April 16, 2026 12:00 AM UTC → July 15, 2026 11:59 PM UTC |

---

## Dataset

- **Source:** ARC-AGI public training set v1 — 400 tasks total
- **Task format:** Each task = a series of input/output grid pairs illustrating a specific transformation
- **Grid structure:** 2D image grids with a channel depth of 10 (one channel per colour/value 0–9)
- **External resource:** [arcprize.org](https://arcprize.org)

---

## What to Build

For each task, produce a single ONNX neural network file (`taskNNN.onnx`) that:

1. Takes an ARC-AGI input grid as input
2. Produces the correctly transformed output grid
3. Is as small and computationally cheap as possible

### Example (Hypothetical Task #000)

A single-layer 3×3 convolutional network:

```python
def weight(channel_out, channel_in, kernel_coord):
    if kernel_coord == ( 0,  0) and channel_in == channel_out: return 1.0
    if kernel_coord == ( 0,  0) and channel_in != 5 and channel_out == 0: return -1.0
    if kernel_coord == (-1, -1) and channel_in != 5 and channel_out == 0: return 1.0
    if kernel_coord == (-1, -1) and channel_in != 5 and channel_out == 5: return -1.0
    return 0.0

network = neurogolf_utils.single_layer_conv2d_network(weight, kernel_size=3)
```

Applied to a 30×30 grid (10 channels):
- **Parameters:** 900
- **Memory footprint:** 39,600 bytes
- **MACs:** 810,000 multiply-accumulate operations

---

## Scoring

**Score per task** (only awarded for functionally correct networks):

```
score = max(1, 25 - ln(cost))
```

Where `cost` is the **sum** of:

| Component | Description |
|---|---|
| Parameter count | Total number of learnable parameters in the network |
| Memory footprint | Total memory in **bytes** used by the network |
| MACs | Total multiply-accumulate operations to execute the network |

- Higher score = better (lower cost = higher score)
- Minimum score per task: `1`
- Maximum theoretical score: `25` (at cost = 1)

---

## Functional Correctness

A network must pass **all three** validation sets to be eligible for points:

1. Original ARC-AGI public training benchmarks
2. ARC-GEN-100K dataset
3. A private benchmark suite (guards against overfitting)

---

## Submission Format

Submit a single `submission.zip` file containing one ONNX file per task:

```
submission.zip
├── task001.onnx
├── task002.onnx
├── ...
└── task400.onnx
```

- At most one ONNX file per task
- File size limit: **1.44 MB per ONNX file**
- Partial submissions are valid (submit only the tasks you've solved)

---

## ONNX Network Constraints

All ONNX files must satisfy the following:

| Rule | Detail |
|---|---|
| Static shapes | All tensors and parameters must have statically-defined shapes |
| Max file size | 1.44 MB per `.onnx` file |
| Banned operators | `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function` |

These constraints are checked automatically by the official network validator.

---

## Prizes

| Prize | Amount |
|---|---|
| First Place | $12,000 |
| Second Place | $10,000 |
| Third Place | $10,000 |
| Top Student Team (≥50% students) | $8,000 |
| Longest Leader (holds 1st place longest) | $10,000 |
| **Total** | **$50,000** |

**Longest Leader** is determined by cumulative time in 1st place on the leaderboard between April 16 and July 15, 2026.

---

## Key Resources

| Resource | URL |
|---|---|
| Competition page | https://www.kaggle.com/competitions/neurogolf-2026 |
| ARC Prize Foundation | https://arcprize.org |
| ONNX documentation | https://onnx.ai |
| ARC-AGI training data (v1) | https://github.com/fchollet/ARC-AGI |

---

## Repository Structure (Planned)

```
neurogolf-2026/
├── agent.md              # This file — competition spec and agent instructions
├── README.md             # Project overview and setup instructions
├── tasks/                # ARC-AGI task JSON files (400 tasks)
├── solutions/            # Python scripts used to generate each ONNX solution
│   ├── task001.py
│   └── ...
├── onnx/                 # Generated .onnx files ready for submission
│   ├── task001.onnx
│   └── ...
├── utils/
│   ├── visualize.py      # Grid visualization helpers
│   ├── validate.py       # Local ONNX validation runner
│   └── scoring.py        # Cost + score calculator
├── submission.zip        # Final submission archive
└── requirements.txt      # Python dependencies
```

---

## Agent Instructions

When working on this competition, follow this strategy:

1. **Load the task** — Read the JSON for the target task from the `tasks/` folder
2. **Understand the transformation** — Identify the pattern from input/output grid pairs
3. **Design the network analytically** — Prefer hand-crafted weights over gradient-based training where possible; trained weights are rarely minimal
4. **Minimise cost aggressively** — Fewer layers, fewer channels, smaller kernels
5. **Validate locally** — Run `utils/validate.py` before submission
6. **Check constraints** — Static shapes, no banned ops, file ≤ 1.44 MB
7. **Score estimate** — Use `utils/scoring.py` to estimate `max(1, 25 - ln(cost))`

### ONNX Tips

- Use `onnx` + `numpy` to construct networks programmatically
- Prefer `Conv` nodes over fully-connected layers for spatial tasks
- Avoid dynamic shapes — use fixed grid sizes per task
- Use `onnxruntime` locally for inference testing
- Validate with `onnx.checker.check_model()` before submitting

---

## Citation

```
Michael D. Moffitt, Walter Reade, Ashley Oldacre, and Addison Howard.
The 2026 NeuroGolf Championship.
https://kaggle.com/competitions/neurogolf-2026, 2026. Kaggle.
```
