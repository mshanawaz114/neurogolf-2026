from __future__ import annotations

"""
build_safe_submission.py — Build a Kaggle-safe full submission bundle.

Strategy:
  1. Inspect every generated task ONNX in `onnx/`
  2. Keep only models whose operator set is within a conservative allowlist
  3. Fill every remaining task with a safe identity fallback
  4. Emit a full 400-file zip plus a CSV audit report

This gives us a reproducible recovery path after Kaggle rejects one or more
advanced ONNX operator families during submission processing.
"""

import argparse
import csv
import shutil
import sys
import zipfile
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.onnx_builder import identity_net


CONSERVATIVE_SAFE_OPS = (
    "Add",
    "Clip",
    "Concat",
    "Conv",
    "Mul",
    "Pad",
    "ReduceMax",
    "Resize",
    "Slice",
    "Sub",
    "Transpose",
)

STATIC23_OPS = CONSERVATIVE_SAFE_OPS + (
    "Cast",
    "CumSum",
    "Equal",
    "ReduceSum",
    "Unsqueeze",
)

STATIC25_OPS = STATIC23_OPS + (
    "Abs",
    "MatMul",
    "Relu",
    "Reshape",
)

SPATIAL_REPAIR_OPS = CONSERVATIVE_SAFE_OPS + (
    "MatMul",
    "Reshape",
)

SAFE_PROFILES = {
    "conservative": set(CONSERVATIVE_SAFE_OPS),
    "spatial_repair": set(SPATIAL_REPAIR_OPS),
    "static23": set(STATIC23_OPS),
    "static25": set(STATIC25_OPS),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", default="tasks/")
    parser.add_argument("--source-onnx-dir", default="onnx/")
    parser.add_argument(
        "--overlay-onnx-dir",
        default=None,
        help="Optional secondary ONNX directory. Use its models only for --overlay-task ids.",
    )
    parser.add_argument("--safe-onnx-dir", default="onnx_safe_full/")
    parser.add_argument("--results-csv", default="results.csv")
    parser.add_argument("--report-csv", default="safe_submission_report.csv")
    parser.add_argument("--zip-path", default="submission_full_safe.zip")
    parser.add_argument(
        "--profile",
        choices=sorted(SAFE_PROFILES),
        default="conservative",
        help="Named operator allowlist profile for submission recovery.",
    )
    parser.add_argument(
        "--safe-op",
        action="append",
        dest="safe_ops",
        help="Append one allowed ONNX op type. Can be repeated.",
    )
    parser.add_argument(
        "--force-keep-task",
        action="append",
        dest="force_keep_tasks",
        help="Keep a generated task ONNX even if its ops fall outside the current profile.",
    )
    parser.add_argument(
        "--force-fallback-task",
        action="append",
        dest="force_fallback_tasks",
        help="Force a task to use the identity fallback even if its ops are otherwise allowed.",
    )
    parser.add_argument(
        "--only-keep-task",
        action="append",
        dest="only_keep_tasks",
        help="If provided, keep exact models only for these task ids and fall back on every other task.",
    )
    parser.add_argument(
        "--overlay-task",
        action="append",
        dest="overlay_tasks",
        help="When --overlay-onnx-dir is set, take these task ids from the overlay directory instead of the primary source.",
    )
    return parser.parse_args()


def load_results(results_csv: Path) -> dict[str, dict[str, str]]:
    if not results_csv.exists():
        return {}
    with results_csv.open(newline="") as f:
        return {row["task_id"]: row for row in csv.DictReader(f)}


def model_ops(model_path: Path) -> list[str]:
    model = onnx.load(str(model_path))
    return sorted({node.op_type for node in model.graph.node})


def write_identity_model(out_path: Path) -> None:
    model = identity_net()
    onnx.save(model, str(out_path))


def reset_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("task*.onnx"):
        old_file.unlink()


def build_bundle(
    tasks_dir: Path,
    source_onnx_dir: Path,
    overlay_onnx_dir: Path | None,
    safe_onnx_dir: Path,
    results_csv: Path,
    report_csv: Path,
    zip_path: Path,
    safe_ops: set[str],
    force_keep_tasks: set[str],
    force_fallback_tasks: set[str],
    only_keep_tasks: set[str] | None,
    overlay_tasks: set[str],
) -> None:
    results = load_results(results_csv)
    task_ids = sorted(task_path.stem for task_path in tasks_dir.glob("task*.json"))

    reset_dir(safe_onnx_dir)

    kept_rows: list[dict[str, str]] = []
    kept = 0
    fallback = 0
    kept_score = 0.0

    for task_id in task_ids:
        src_dir = overlay_onnx_dir if overlay_onnx_dir is not None and task_id in overlay_tasks else source_onnx_dir
        src = src_dir / f"{task_id}.onnx"
        dst = safe_onnx_dir / f"{task_id}.onnx"
        result = results.get(task_id, {})

        if only_keep_tasks is not None and task_id not in only_keep_tasks:
            write_identity_model(dst)
            fallback += 1
            kept_rows.append(
                {
                    "task_id": task_id,
                    "mode": "identity_fallback",
                    "solver": result.get("solver", ""),
                    "local_score": result.get("score", ""),
                    "source_model": str(src) if src.exists() else "",
                    "op_types": "",
                    "reason": "not_in_only_keep_set",
                }
            )
            continue

        if task_id in force_fallback_tasks:
            write_identity_model(dst)
            fallback += 1
            kept_rows.append(
                {
                    "task_id": task_id,
                    "mode": "identity_fallback",
                    "solver": result.get("solver", ""),
                    "local_score": result.get("score", ""),
                    "source_model": str(src) if src.exists() else "",
                    "op_types": "",
                    "reason": "forced_fallback",
                }
            )
            continue

        if src.exists():
            ops = model_ops(src)
            if task_id in force_keep_tasks or set(ops).issubset(safe_ops):
                shutil.copyfile(src, dst)
                kept += 1
                try:
                    kept_score += float(result.get("score", "") or 0.0)
                except ValueError:
                    pass
                kept_rows.append(
                    {
                        "task_id": task_id,
                        "mode": "kept_exact",
                        "solver": result.get("solver", ""),
                        "local_score": result.get("score", ""),
                        "source_model": str(src),
                        "op_types": " ".join(ops),
                        "reason": (
                            "forced_keep"
                            if task_id in force_keep_tasks
                            else ("overlay_op_set_safe" if src_dir == overlay_onnx_dir else "op_set_safe")
                        ),
                    }
                )
                continue

            write_identity_model(dst)
            fallback += 1
            kept_rows.append(
                {
                    "task_id": task_id,
                    "mode": "identity_fallback",
                    "solver": result.get("solver", ""),
                    "local_score": result.get("score", ""),
                    "source_model": str(src),
                    "op_types": " ".join(ops),
                    "reason": "unsafe_ops_detected",
                }
            )
            continue

        write_identity_model(dst)
        fallback += 1
        kept_rows.append(
            {
                "task_id": task_id,
                "mode": "identity_fallback",
                "solver": "",
                "local_score": "",
                "source_model": "",
                "op_types": "",
                "reason": "no_source_model",
            }
        )

    with report_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "mode",
                "solver",
                "local_score",
                "source_model",
                "op_types",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(kept_rows)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for model_path in sorted(safe_onnx_dir.glob("task*.onnx")):
            zf.write(model_path, model_path.name)

    print(f"safe_ops={sorted(safe_ops)}")
    print(f"kept_exact={kept}")
    print(f"kept_local_score={kept_score:.2f}")
    print(f"identity_fallback={fallback}")
    print(f"total_files={len(task_ids)}")
    print(f"safe_onnx_dir={safe_onnx_dir}")
    print(f"report_csv={report_csv}")
    print(f"zip_path={zip_path}")


def main() -> None:
    args = parse_args()
    safe_ops = set(SAFE_PROFILES[args.profile])
    if args.safe_ops:
        safe_ops.update(args.safe_ops)
    force_keep_tasks = set(args.force_keep_tasks or [])
    force_fallback_tasks = set(args.force_fallback_tasks or [])
    only_keep_tasks = set(args.only_keep_tasks) if args.only_keep_tasks else None
    overlay_tasks = set(args.overlay_tasks or [])
    build_bundle(
        tasks_dir=Path(args.tasks_dir),
        source_onnx_dir=Path(args.source_onnx_dir),
        overlay_onnx_dir=Path(args.overlay_onnx_dir) if args.overlay_onnx_dir else None,
        safe_onnx_dir=Path(args.safe_onnx_dir),
        results_csv=Path(args.results_csv),
        report_csv=Path(args.report_csv),
        zip_path=Path(args.zip_path),
        safe_ops=safe_ops,
        force_keep_tasks=force_keep_tasks,
        force_fallback_tasks=force_fallback_tasks,
        only_keep_tasks=only_keep_tasks,
        overlay_tasks=overlay_tasks,
    )


if __name__ == "__main__":
    main()
