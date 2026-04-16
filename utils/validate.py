"""
validate.py — Run an ONNX network against ARC-AGI task pairs and check correctness.

Usage:
    python utils/validate.py --onnx onnx/task001.onnx --task tasks/task001.json
"""

import json
import argparse
import numpy as np
import onnx
import onnxruntime as ort


def load_task(task_path: str):
    with open(task_path) as f:
        return json.load(f)


def grid_to_tensor(grid: list) -> np.ndarray:
    """Convert an ARC grid (H x W list of ints) to a one-hot float32 tensor [1, 10, H, W]."""
    arr = np.array(grid, dtype=np.int64)
    H, W = arr.shape
    one_hot = np.zeros((1, 10, H, W), dtype=np.float32)
    for c in range(10):
        one_hot[0, c] = (arr == c).astype(np.float32)
    return one_hot


def tensor_to_grid(tensor: np.ndarray) -> np.ndarray:
    """Convert a [1, 10, H, W] output tensor back to an (H x W) int grid via argmax."""
    return np.argmax(tensor[0], axis=0).astype(np.int64)


def validate(onnx_path: str, task_path: str, split: str = "train") -> bool:
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    task = load_task(task_path)
    pairs = task.get(split, [])

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    all_correct = True
    for i, pair in enumerate(pairs):
        inp = grid_to_tensor(pair["input"])
        expected = np.array(pair["output"], dtype=np.int64)

        outputs = session.run(None, {input_name: inp})
        predicted = tensor_to_grid(outputs[0])

        correct = np.array_equal(predicted, expected)
        status = "✓ PASS" if correct else "✗ FAIL"
        print(f"  [{split}] Pair {i+1}: {status}")

        if not correct:
            all_correct = False
            print(f"    Expected shape: {expected.shape}, Got: {predicted.shape}")
            diff = (predicted != expected).sum()
            print(f"    Mismatched cells: {diff} / {expected.size}")

    return all_correct


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Path to .onnx file")
    parser.add_argument("--task", required=True, help="Path to task JSON file")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="Which split to validate against (default: train)")
    args = parser.parse_args()

    print(f"\nValidating {args.onnx} on {args.task} [{args.split}]...")
    passed = validate(args.onnx, args.task, args.split)
    print(f"\nResult: {'ALL PASSED ✓' if passed else 'FAILED ✗'}\n")
