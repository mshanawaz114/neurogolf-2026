from __future__ import annotations

"""
diagnose.py — Analyse all 400 tasks and print a breakdown of what each
              task looks like, so we know exactly which solvers to build next.

Usage:
    python3 scripts/diagnose.py
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.arc_utils import load_task, analyse_task, grid_to_array

TASKS_DIR = Path("tasks")


def main():
    task_files = sorted(TASKS_DIR.glob("task*.json"))
    if not task_files:
        print("No task files found. Run: python3 scripts/download_arc.py")
        return

    categories = Counter()
    size_varies = 0
    io_size_changes = 0
    pure_spatial = []
    pure_color   = []
    unknown      = []

    for tf in task_files:
        task = load_task(tf)
        a    = analyse_task(task)

        # Check if arc-gen sizes vary vs train
        train_pairs = task.get("train", [])
        arcgen_pairs = task.get("arc-gen", [])
        train_sizes  = {(len(p["input"]), len(p["input"][0])) for p in train_pairs}
        arcgen_sizes = {(len(p["input"]), len(p["input"][0])) for p in arcgen_pairs}
        varies = not arcgen_sizes.issubset(train_sizes)
        if varies:
            size_varies += 1

        in_shapes  = a.get("input_shapes",  [])
        out_shapes = a.get("output_shapes", [])
        io_changes = in_shapes != out_shapes or (
            in_shapes and out_shapes and in_shapes[0] != out_shapes[0]
        )
        if io_changes:
            io_size_changes += 1

        t  = a.get("spatial_transform")
        cp = a.get("color_permutation", False)

        if t and t != "identity":
            categories["spatial_" + t] += 1
            pure_spatial.append((tf.stem, t))
        elif t == "identity":
            categories["identity"] += 1
        elif cp:
            categories["color_permutation"] += 1
            pure_color.append(tf.stem)
        elif a.get("tiling"):
            categories["tiling"] += 1
        elif a.get("gravity_down"):
            categories["gravity_down"] += 1
        elif a.get("gravity_up"):
            categories["gravity_up"] += 1
        else:
            categories["unknown"] += 1
            unknown.append(tf.stem)

    total = len(task_files)
    print(f"\n{'='*55}")
    print(f"  Task Distribution ({total} tasks)")
    print(f"{'='*55}")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        bar = "█" * (count * 30 // total)
        print(f"  {cat:25s} {count:4d}  {bar}")

    print(f"\n  Arc-gen has DIFFERENT grid sizes: {size_varies}")
    print(f"  Input/output size changes:        {io_size_changes}")

    print(f"\n{'='*55}")
    print(f"  Spatial tasks detected ({len(pure_spatial)}):")
    for tid, t in pure_spatial:
        print(f"    {tid}  →  {t}")

    print(f"\n  Color permutation tasks ({len(pure_color)}):")
    for tid in pure_color[:20]:
        print(f"    {tid}")
    if len(pure_color) > 20:
        print(f"    ... and {len(pure_color)-20} more")

    print(f"\n  Unknown tasks (need learned solver): {len(unknown)}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
