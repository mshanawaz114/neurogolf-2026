from __future__ import annotations

"""
solve_all.py — Auto-solve all 400 ARC tasks and produce submission.zip.

Pipeline per task:
  1. Load task JSON
  2. Analyse transformation (colour perm? spatial? gravity? ...)
  3. Try each solver in priority order (cheapest first)
  4. Validate the ONNX against train + test + arc-gen
  5. Save ONNX to onnx/taskNNN.onnx

Usage:
  python scripts/solve_all.py [--tasks-dir tasks/] [--onnx-dir onnx/] [--no-learned]
"""

import argparse
import math
import os
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.arc_utils   import load_task, analyse_task, validate_onnx
from utils.scoring     import analyse as score_onnx

# Import all solvers
from solvers.spatial    import SpatialSolver
from solvers.spatial_color import SpatialColorSolver
from solvers.color_perm import ColorPermSolver
from solvers.tiling     import TilingSolver
from solvers.translate  import TranslateSolver
from solvers.color_bbox_crop import ColorBBoxCropSolver
from solvers.color_bbox_preserve_flip import ColorBBoxPreserveFlipSolver
from solvers.color_count_crop import ColorCountCropSolver
from solvers.color_count_preserve_crop import ColorCountPreserveCropSolver
from solvers.fixed_crop import FixedCropSolver
from solvers.trim_bbox  import TrimBBoxSolver
from solvers.upscale    import UpscaleSolver
from solvers.gravity    import GravitySolver
from solvers.learned    import LearnedSolver


ALL_SOLVERS = [
    SpatialSolver(),
    SpatialColorSolver(),
    ColorPermSolver(),
    TilingSolver(),
    TranslateSolver(),
    ColorBBoxCropSolver(),
    ColorBBoxPreserveFlipSolver(),
    ColorCountCropSolver(),
    ColorCountPreserveCropSolver(),
    FixedCropSolver(),
    TrimBBoxSolver(),
    UpscaleSolver(),
    GravitySolver(),
    LearnedSolver(),
]


def solve_task(task_id: str, task: dict, onnx_dir: Path,
               use_learned: bool = True) -> dict:
    analysis = analyse_task(task)
    solvers = [s for s in ALL_SOLVERS
               if use_learned or not isinstance(s, LearnedSolver)]
    solvers.sort(key=lambda s: s.PRIORITY)

    for solver in solvers:
        if not solver.can_solve(analysis):
            continue
        print(f"  [{solver.name}] trying...")
        t0 = time.time()
        try:
            path = solver.build(task_id, task, analysis, onnx_dir)
        except Exception as e:
            print(f"    → error: {e}")
            continue
        if path is None:
            continue
        elapsed = time.time() - t0

        # Validate against all available splits
        try:
            ok = validate_onnx(str(path), task, splits=["train", "test", "arc-gen"])
        except Exception as e:
            print(f"    → validation error: {e}")
            ok = False

        if ok:
            info = score_onnx(str(path))
            print(f"  ✓ {task_id}  solver={solver.name}  "
                  f"score={info['score']:.2f}  cost={info['cost']:,}  t={elapsed:.1f}s")
            return {
                "task_id": task_id,
                "solver":  solver.name,
                "score":   info["score"],
                "cost":    info["cost"],
                "elapsed": elapsed,
            }
        else:
            print(f"    → incorrect, removing")
            path.unlink(missing_ok=True)

    print(f"  ✗ {task_id}  no solver succeeded")
    return {"task_id": task_id, "solver": None, "score": 0, "cost": 0, "elapsed": 0}


def make_zip(onnx_dir: Path, out_path: str = "submission.zip"):
    files = sorted(onnx_dir.glob("task*.onnx"))
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.name)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nsubmission.zip: {len(files)} files, {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir",  default="tasks/")
    parser.add_argument("--onnx-dir",   default="onnx/")
    parser.add_argument("--no-learned", action="store_true",
                        help="Skip the slow LearnedSolver")
    parser.add_argument("--task",       default=None,
                        help="Solve a single task, e.g. --task task001")
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    onnx_dir  = Path(args.onnx_dir)
    onnx_dir.mkdir(exist_ok=True)

    task_files = sorted(tasks_dir.glob("task*.json"))
    if args.task:
        task_files = [f for f in task_files if f.stem == args.task]

    if not task_files:
        print(f"No task files found in {tasks_dir}")
        return

    results = []
    solved = 0
    total_score = 0.0

    print(f"\n{'='*60}")
    print(f"  NeuroGolf 2026 — Solving {len(task_files)} tasks")
    print(f"  Learned solver: {'OFF' if args.no_learned else 'ON'}")
    print(f"{'='*60}\n")

    for task_file in task_files:
        task_id = task_file.stem
        print(f"\n[{task_id}]")
        task = load_task(task_file)
        result = solve_task(task_id, task, onnx_dir, not args.no_learned)
        results.append(result)
        if result["solver"] is not None:
            solved += 1
            total_score += result["score"]

    # Summary
    print(f"\n{'='*60}")
    print(f"  Solved: {solved} / {len(task_files)}")
    print(f"  Total score: {total_score:.2f}")
    print(f"  Avg score per solved task: "
          f"{total_score/solved:.2f}" if solved else "  N/A")
    print(f"{'='*60}")

    # Solver breakdown
    from collections import Counter
    solver_counts = Counter(r["solver"] for r in results if r["solver"])
    print("\nSolver breakdown:")
    for name, count in solver_counts.most_common():
        score_sum = sum(r["score"] for r in results if r["solver"] == name)
        print(f"  {name:20s}  {count:3d} tasks  {score_sum:.1f} pts")

    # Build zip
    if not args.task:
        make_zip(onnx_dir)

    # Save results CSV
    import csv
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id","solver","score","cost","elapsed"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print("Results saved to results.csv")


if __name__ == "__main__":
    main()
