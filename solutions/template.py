"""
template.py — PyTorch → ONNX solution template for NeuroGolf 2026.

Copy this file to solutions/taskNNN.py and fill in the model for your task.

Usage:
    python solutions/taskNNN.py
    make validate TASK=NNN
    make score TASK=NNN
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import onnx
from onnxsim import simplify

# ── Config ────────────────────────────────────────────────────────────────────
TASK_ID = "000"          # <- change this
CHANNELS = 10            # ARC uses 10 colour channels (0–9)
DEVICE = "cpu"

TASK_PATH = Path(f"tasks/task{TASK_ID}.json")
ONNX_PATH = Path(f"onnx/task{TASK_ID}.onnx")
ONNX_PATH.parent.mkdir(exist_ok=True)

# ── Load task ─────────────────────────────────────────────────────────────────
def load_task():
    with open(TASK_PATH) as f:
        return json.load(f)

def grid_to_tensor(grid: list) -> torch.Tensor:
    """(H, W) int grid → one-hot float32 tensor [1, 10, H, W]"""
    arr = np.array(grid, dtype=np.int64)
    H, W = arr.shape
    one_hot = np.zeros((1, CHANNELS, H, W), dtype=np.float32)
    for c in range(CHANNELS):
        one_hot[0, c] = (arr == c).astype(np.float32)
    return torch.from_numpy(one_hot)

def tensor_to_grid(t: torch.Tensor) -> np.ndarray:
    """[1, 10, H, W] → (H, W) int grid via argmax"""
    return t[0].argmax(dim=0).cpu().numpy().astype(np.int64)

# ── Model definition ──────────────────────────────────────────────────────────
# Replace this with the smallest network that solves your task.
# Prefer nn.Conv2d over nn.Linear for spatial transformations.

class SolutionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Example: single 1×1 conv (identity-like pass-through)
        self.conv = nn.Conv2d(
            in_channels=CHANNELS,
            out_channels=CHANNELS,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        # TODO: initialise weights analytically to solve the task
        nn.init.eye_(self.conv.weight.view(CHANNELS, CHANNELS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

# ── Validate model in PyTorch ─────────────────────────────────────────────────
def validate_torch(model: nn.Module, task: dict) -> bool:
    model.eval()
    all_correct = True
    with torch.no_grad():
        for i, pair in enumerate(task["train"]):
            inp = grid_to_tensor(pair["input"])
            expected = np.array(pair["output"], dtype=np.int64)
            out = model(inp)
            predicted = tensor_to_grid(out)
            ok = np.array_equal(predicted, expected)
            print(f"  [train] Pair {i+1}: {'✓' if ok else '✗'}")
            if not ok:
                all_correct = False
    return all_correct

# ── Export to ONNX ────────────────────────────────────────────────────────────
def export_onnx(model: nn.Module, task: dict):
    model.eval()
    # Use first training input to determine H, W for static shape export
    sample_grid = task["train"][0]["input"]
    H, W = len(sample_grid), len(sample_grid[0])
    dummy_input = torch.zeros(1, CHANNELS, H, W)

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes=None,   # static shapes required by competition rules
    )
    print(f"  Exported: {ONNX_PATH}")

    # Simplify — removes redundant nodes, fuses constants
    model_onnx = onnx.load(str(ONNX_PATH))
    simplified, check = simplify(model_onnx)
    if check:
        onnx.save(simplified, str(ONNX_PATH))
        print("  Simplified ✓")
    else:
        print("  Simplification failed — using original")

    # Validate ONNX schema
    onnx.checker.check_model(onnx.load(str(ONNX_PATH)))
    print("  ONNX schema valid ✓")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    task = load_task()
    model = SolutionNet().to(DEVICE)

    print(f"\nTask {TASK_ID} — PyTorch validation:")
    passed = validate_torch(model, task)

    if passed:
        print(f"\nExporting to {ONNX_PATH}...")
        export_onnx(model, task)
        print(f"\nDone. Run: make validate TASK={TASK_ID}")
    else:
        print("\nModel is not correct yet — fix the weights before exporting.")
