# NeuroGolf 2026

Minimal ONNX neural networks for ARC-AGI image transformations — [Kaggle NeuroGolf 2026 Championship](https://www.kaggle.com/competitions/neurogolf-2026)

## Goal

Design the **smallest possible neural networks** (ONNX format) that correctly solve ARC-AGI image transformation tasks. Jointly minimize parameter count, memory footprint, and multiply-accumulate operations.

## Scoring

```
score = max(1, 25 - ln(cost))
```

where `cost = parameters + memory (bytes) + MACs`

## Setup

```bash
git clone https://github.com/mshanawaz114/neurogolf-2026.git
cd neurogolf-2026
pip install -r requirements.txt
```

## Project Structure

```
neurogolf-2026/
├── agent.md              # Full competition spec and agent instructions
├── tasks/                # ARC-AGI task JSON files (400 tasks)
├── solutions/            # Python scripts to generate each ONNX solution
├── onnx/                 # Generated .onnx files ready for submission
├── utils/
│   ├── visualize.py      # Grid visualization helpers
│   ├── validate.py       # Local ONNX validation runner
│   └── scoring.py        # Cost + score calculator
├── requirements.txt
└── submission.zip        # Final submission archive
```

## Usage

```bash
# Visualize a task
python utils/visualize.py --task tasks/task001.json

# Build an ONNX solution
python solutions/task001.py

# Validate a solution
python utils/validate.py --onnx onnx/task001.onnx --task tasks/task001.json

# Estimate score
python utils/scoring.py --onnx onnx/task001.onnx
```

## Resources

- [Competition page](https://www.kaggle.com/competitions/neurogolf-2026)
- [ARC Prize Foundation](https://arcprize.org)
- [ARC-AGI dataset (v1)](https://github.com/fchollet/ARC-AGI)
- [ONNX documentation](https://onnx.ai)

## Citation

```
Michael D. Moffitt, Walter Reade, Ashley Oldacre, and Addison Howard.
The 2026 NeuroGolf Championship.
https://kaggle.com/competitions/neurogolf-2026, 2026. Kaggle.
```
