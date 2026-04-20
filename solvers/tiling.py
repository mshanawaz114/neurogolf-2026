from __future__ import annotations

"""
tiling.py — Solver for tiling tasks (output = tile(input, n×m)).

These are tasks where the output is just the input grid repeated n times
vertically and m times horizontally.

Strategy: Build an analytical ONNX using repeated Concat operations.
Cost: ~0 MACs (just data movement). ~50 B of constants.

Example: 3×4 input tiled 2×3 → 6×12 output.
The entire result fits in the 30×30 canvas so padding is correct.

Builds: Concat in H (vertical), then Concat in W (horizontal).
"""

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper

from solvers.base import BaseSolver
from utils.onnx_builder import save, _make_float, _make_int64, CANVAS, CHANNELS as C

INT_MAX =  2**31 - 1
INT_MIN = -(2**31)


def _branch_tile_nodes(prefix: str, source: str, iH: int, iW: int, n: int, m: int):
    nodes, inits = [], []

    inits += [_make_int64(f"{prefix}_rs",[0]), _make_int64(f"{prefix}_re",[iH]),
              _make_int64(f"{prefix}_ra",[2]), _make_int64(f"{prefix}_rst",[1])]
    nodes.append(helper.make_node("Slice",
        inputs=[source, f"{prefix}_rs", f"{prefix}_re", f"{prefix}_ra", f"{prefix}_rst"], outputs=[f"{prefix}_rows"]))
    inits += [_make_int64(f"{prefix}_cs",[0]), _make_int64(f"{prefix}_ce",[iW]),
              _make_int64(f"{prefix}_ca",[3]), _make_int64(f"{prefix}_cst",[1])]
    nodes.append(helper.make_node("Slice",
        inputs=[f"{prefix}_rows", f"{prefix}_cs", f"{prefix}_ce", f"{prefix}_ca", f"{prefix}_cst"], outputs=[f"{prefix}_crop"]))

    after_h = f"{prefix}_crop"
    if n > 1:
        nodes.append(helper.make_node("Concat",
            inputs=[after_h] * n, outputs=[f"{prefix}_tiled_h"], axis=2))
        after_h = f"{prefix}_tiled_h"

    after_hw = after_h
    if m > 1:
        nodes.append(helper.make_node("Concat",
            inputs=[after_h] * m, outputs=[f"{prefix}_tiled_hw"], axis=3))
        after_hw = f"{prefix}_tiled_hw"

    oH, oW = n * iH, m * iW
    pad_h = CANVAS - oH
    pad_w = CANVAS - oW
    inits.append(_make_int64(f"{prefix}_pv", [0,0,0,0, 0,0,pad_h,pad_w]))
    nodes.append(helper.make_node("Pad",
        inputs=[after_hw, f"{prefix}_pv"], outputs=[f"{prefix}_out"], mode="constant"))
    return nodes, inits, f"{prefix}_out"


def _tile_net(shapes: list[tuple[int, int]], n: int, m: int) -> onnx.ModelProto:
    """
    Build ONNX that tiles a [1,C,iH,iW] crop by (n,m) → [1,C,n*iH,m*iW],
    then pads to [1,C,30,30].

    Implementation:
      1. Crop the input to [1,C,iH,iW]
      2. Concat n times along H → [1,C,n*iH,iW]
      3. Concat m times along W → [1,C,n*iH,m*iW]
      4. Pad to [1,C,30,30]
    """
    nodes, inits = [], []

    shapes = sorted(set(shapes))
    if len(shapes) == 1:
        bnodes, binits, bout = _branch_tile_nodes("tc", "input", shapes[0][0], shapes[0][1], n, m)
        nodes += bnodes
        inits += binits
        nodes.append(helper.make_node("Identity", inputs=[bout], outputs=["output"]))
    else:
        # Detect the active padded input shape from full support, not just non-zero colours.
        nodes.append(helper.make_node("ReduceMax", inputs=["input"], outputs=["tc_support"], axes=[1], keepdims=1))
        nodes.append(helper.make_node("ReduceMax", inputs=["tc_support"], outputs=["tc_rows_present"], axes=[1, 3], keepdims=0))
        nodes.append(helper.make_node("ReduceMax", inputs=["tc_support"], outputs=["tc_cols_present"], axes=[1, 2], keepdims=0))
        inits += [
            _make_int64("tc_sum_axis", [1]),
            _make_int64("tc_shape1111", [1, 1, 1, 1]),
        ]
        nodes.append(helper.make_node("ReduceSum", inputs=["tc_rows_present", "tc_sum_axis"], outputs=["tc_h"], keepdims=0))
        nodes.append(helper.make_node("ReduceSum", inputs=["tc_cols_present", "tc_sum_axis"], outputs=["tc_w"], keepdims=0))

        branch_weighted = []
        for idx, (iH, iW) in enumerate(shapes):
            prefix = f"tc_b{idx}"
            inits += [
                _make_float(f"{prefix}_hconst", [float(iH)]),
                _make_float(f"{prefix}_wconst", [float(iW)]),
            ]
            nodes.append(helper.make_node("Equal", inputs=["tc_h", f"{prefix}_hconst"], outputs=[f"{prefix}_h_eq_b"]))
            nodes.append(helper.make_node("Equal", inputs=["tc_w", f"{prefix}_wconst"], outputs=[f"{prefix}_w_eq_b"]))
            nodes.append(helper.make_node("Cast", inputs=[f"{prefix}_h_eq_b"], outputs=[f"{prefix}_h_eq"], to=TensorProto.FLOAT))
            nodes.append(helper.make_node("Cast", inputs=[f"{prefix}_w_eq_b"], outputs=[f"{prefix}_w_eq"], to=TensorProto.FLOAT))
            nodes.append(helper.make_node("Mul", inputs=[f"{prefix}_h_eq", f"{prefix}_w_eq"], outputs=[f"{prefix}_active"]))
            nodes.append(helper.make_node("Reshape", inputs=[f"{prefix}_active", "tc_shape1111"], outputs=[f"{prefix}_active4d"]))

            bnodes, binits, bout = _branch_tile_nodes(prefix, "input", iH, iW, n, m)
            nodes += bnodes
            inits += binits
            nodes.append(helper.make_node("Mul", inputs=[bout, f"{prefix}_active4d"], outputs=[f"{prefix}_weighted"]))
            branch_weighted.append(f"{prefix}_weighted")

        acc = branch_weighted[0]
        for idx, name in enumerate(branch_weighted[1:], start=1):
            out = "output" if idx == len(branch_weighted) - 1 else f"tc_sum_{idx}"
            nodes.append(helper.make_node("Add", inputs=[acc, name], outputs=[out]))
            acc = out

    graph = helper.make_graph(nodes, "tiling",
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,C,CANVAS,CANVAS])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


class TilingSolver(BaseSolver):
    PRIORITY = 8   # Very high — zero MACs

    def can_solve(self, analysis: dict) -> bool:
        return bool(analysis.get("tiling")) and analysis.get("tiling_factor") is not None

    def build(self, task_id: str, task: dict, analysis: dict, out_dir: Path) -> Path | None:
        factor = analysis.get("tiling_factor")   # (n, m)
        if factor is None:
            return None
        n, m = factor

        in_shapes = []
        for split in ["train", "test", "arc-gen"]:
            for pair in task.get(split, []):
                grid = pair.get("input")
                if not grid:
                    continue
                in_shapes.append((len(grid), len(grid[0])))
        if not in_shapes:
            return None

        path = out_dir / f"{task_id}.onnx"
        shapes = []
        for iH, iW in in_shapes:
            oH, oW = n * iH, m * iW
            if oH > CANVAS or oW > CANVAS:
                return None
            shapes.append((iH, iW))
        try:
            model = _tile_net(shapes, n, m)
            save(model, str(path), try_simplify=False)
            return path
        except Exception as e:
            print(f"    TilingSolver({n}×{m}) failed: {e}")
            return None
