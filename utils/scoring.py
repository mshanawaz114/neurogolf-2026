"""
scoring.py — Estimate NeuroGolf cost and score for an ONNX network.

Competition formula:
    cost  = total_parameters + memory_bytes + total_MACs
    score = max(1, 25 - ln(cost))

Usage:
    python utils/scoring.py --onnx onnx/task001.onnx
"""

import argparse
import math
import os
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto


def count_parameters(model: onnx.ModelProto) -> int:
    return sum(numpy_helper.to_array(i).size for i in model.graph.initializer)


def memory_bytes(model: onnx.ModelProto) -> int:
    dtype_bytes = {
        TensorProto.FLOAT:   4, TensorProto.DOUBLE: 8,
        TensorProto.INT32:   4, TensorProto.INT64:  8,
        TensorProto.FLOAT16: 2, TensorProto.INT8:   1,
        TensorProto.UINT8:   1, TensorProto.BOOL:   1,
    }
    total = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        total += arr.size * dtype_bytes.get(init.data_type, 4)
    return total


def count_macs(model: onnx.ModelProto) -> int:
    """
    Count multiply-accumulate ops. Only Conv and Gemm/MatMul have MACs.
    Uses static shapes from initializers + value_info.
    """
    shape_map = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        dims = [d.dim_value if d.dim_value > 0 else 1
                for d in vi.type.tensor_type.shape.dim]
        shape_map[vi.name] = dims
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        shape_map[init.name] = list(arr.shape)

    total = 0
    for node in model.graph.node:
        if node.op_type == "Conv":
            if len(node.input) >= 2 and node.input[1] in shape_map:
                w = shape_map[node.input[1]]
                o = shape_map.get(node.output[0], [1,1,1,1])
                if len(w) == 4 and len(o) >= 4:
                    c_out, c_in_g, kH, kW = w
                    H_out = o[2] if o[2] > 0 else 1
                    W_out = o[3] if o[3] > 0 else 1
                    total += int(c_out * c_in_g * kH * kW * H_out * W_out)
        elif node.op_type in ("Gemm", "MatMul"):
            a = shape_map.get(node.input[0] if node.input else "", [])
            b = shape_map.get(node.input[1] if len(node.input)>1 else "", [])
            if len(a) >= 2 and len(b) >= 2:
                total += int((a[-2] or 1) * (a[-1] or 1) * (b[-1] or 1))
    return total


def compute_score(cost: int) -> float:
    return max(1.0, 25.0 - math.log(cost)) if cost > 0 else 25.0


def analyse(onnx_path: str) -> dict:
    """Return dict with params, mem, macs, cost, score."""
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    # Run shape inference to populate value_info
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    params = count_parameters(model)
    mem    = memory_bytes(model)
    macs   = count_macs(model)
    cost   = params + mem + macs
    score  = compute_score(cost)
    size_b = os.path.getsize(onnx_path)
    return {"params": params, "mem": mem, "macs": macs,
            "cost": cost, "score": score, "file_bytes": size_b}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    args = parser.parse_args()
    info = analyse(args.onnx)
    print(f"\n{'='*44}")
    print(f"  File:       {args.onnx}")
    print(f"  File size:  {info['file_bytes']:,} B")
    print(f"{'='*44}")
    print(f"  Parameters: {info['params']:,}")
    print(f"  Memory:     {info['mem']:,} B")
    print(f"  MACs:       {info['macs']:,}")
    print(f"  Total cost: {info['cost']:,}")
    print(f"  Score:      {info['score']:.4f}")
    print(f"{'='*44}\n")
