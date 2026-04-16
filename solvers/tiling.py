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
from utils.onnx_builder import save, _make_int64, CANVAS, CHANNELS as C

INT_MAX =  2**31 - 1
INT_MIN = -(2**31)


def _tile_net(iH: int, iW: int, n: int, m: int) -> onnx.ModelProto:
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

    # Step 1: Crop [1,C,iH,iW] from input
    inits += [_make_int64("tc_rs",[0]), _make_int64("tc_re",[iH]),
              _make_int64("tc_ra",[2]), _make_int64("tc_rst",[1])]
    nodes.append(helper.make_node("Slice",
        inputs=["input","tc_rs","tc_re","tc_ra","tc_rst"], outputs=["tc_rows"]))
    inits += [_make_int64("tc_cs",[0]), _make_int64("tc_ce",[iW]),
              _make_int64("tc_ca",[3]), _make_int64("tc_cst",[1])]
    nodes.append(helper.make_node("Slice",
        inputs=["tc_rows","tc_cs","tc_ce","tc_ca","tc_cst"], outputs=["tc_crop"]))

    # Step 2: Concat along H (axis=2) n times
    if n > 1:
        h_inputs = ["tc_crop"] * n
        nodes.append(helper.make_node("Concat",
            inputs=h_inputs, outputs=["tc_tiled_h"], axis=2))
        after_h = "tc_tiled_h"
    else:
        after_h = "tc_crop"

    # Step 3: Concat along W (axis=3) m times
    if m > 1:
        w_inputs = [after_h] * m
        nodes.append(helper.make_node("Concat",
            inputs=w_inputs, outputs=["tc_tiled_hw"], axis=3))
        after_hw = "tc_tiled_hw"
    else:
        after_hw = after_h

    # Step 4: Pad to 30×30
    oH, oW = n * iH, m * iW
    pad_h = CANVAS - oH
    pad_w = CANVAS - oW
    inits.append(_make_int64("tc_pv", [0,0,0,0, 0,0,pad_h,pad_w]))
    nodes.append(helper.make_node("Pad",
        inputs=[after_hw, "tc_pv"], outputs=["output"], mode="constant"))

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

        in_shapes = analysis.get("input_shapes", [])
        if not in_shapes:
            return None

        path = out_dir / f"{task_id}.onnx"
        for shape in in_shapes:
            iH, iW = shape
            oH, oW = n * iH, m * iW
            if oH > CANVAS or oW > CANVAS:
                continue   # Output doesn't fit in 30×30 canvas — skip
            try:
                model = _tile_net(iH, iW, n, m)
                save(model, str(path))
                return path
            except Exception as e:
                print(f"    TilingSolver({n}×{m}) iH={iH} iW={iW} failed: {e}")
                continue

        return None
