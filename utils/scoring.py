"""
scoring.py — Estimate NeuroGolf cost and score for an ONNX network.

Usage:
    python utils/scoring.py --onnx onnx/task001.onnx

Score formula:
    cost  = total_parameters + memory_bytes + total_MACs
    score = max(1, 25 - ln(cost))
"""

import argparse
import math
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto


def count_parameters(model: onnx.ModelProto) -> int:
    total = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        total += arr.size
    return total


def memory_bytes(model: onnx.ModelProto) -> int:
    dtype_sizes = {
        TensorProto.FLOAT: 4,
        TensorProto.DOUBLE: 8,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
        TensorProto.FLOAT16: 2,
        TensorProto.INT8: 1,
        TensorProto.UINT8: 1,
        TensorProto.BOOL: 1,
    }
    total = 0
    for init in model.graph.initializer:
        size = dtype_sizes.get(init.data_type, 4)
        arr = numpy_helper.to_array(init)
        total += arr.size * size
    return total


def estimate_macs(model: onnx.ModelProto) -> int:
    """
    Rough MAC estimate based on Conv and Gemm nodes with static shapes.
    For a precise count, use tools like onnx-opcounter or netron.
    """
    shape_map = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else 1)
        shape_map[vi.name] = dims

    init_shapes = {i.name: list(numpy_helper.to_array(i).shape) for i in model.graph.initializer}
    shape_map.update(init_shapes)

    total_macs = 0
    for node in model.graph.node:
        if node.op_type == "Conv":
            # output: [N, C_out, H_out, W_out], weight: [C_out, C_in/groups, kH, kW]
            if len(node.input) >= 2 and node.input[1] in shape_map:
                w = shape_map[node.input[1]]
                out_shape = shape_map.get(node.output[0], [1, 1, 1, 1])
                if len(w) == 4 and len(out_shape) == 4:
                    c_out, c_in_per_group, kH, kW = w
                    _, _, H_out, W_out = [d if d > 0 else 1 for d in out_shape]
                    macs = c_out * c_in_per_group * kH * kW * H_out * W_out
                    total_macs += macs

        elif node.op_type == "Gemm" or node.op_type == "MatMul":
            if len(node.input) >= 2:
                a = shape_map.get(node.input[0], [])
                b = shape_map.get(node.input[1], [])
                if len(a) >= 2 and len(b) >= 2:
                    m = a[-2] if a[-2] > 0 else 1
                    k = a[-1] if a[-1] > 0 else 1
                    n = b[-1] if b[-1] > 0 else 1
                    total_macs += m * k * n

    return total_macs


def compute_score(cost: int) -> float:
    return max(1.0, 25.0 - math.log(cost)) if cost > 0 else 25.0


def analyse(onnx_path: str):
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    params = count_parameters(model)
    mem = memory_bytes(model)
    macs = estimate_macs(model)
    cost = params + mem + macs
    score = compute_score(cost)

    print(f"\n{'='*40}")
    print(f"  ONNX file:   {onnx_path}")
    print(f"{'='*40}")
    print(f"  Parameters:  {params:,}")
    print(f"  Memory:      {mem:,} bytes")
    print(f"  MACs (est.): {macs:,}")
    print(f"  Total cost:  {cost:,}")
    print(f"  Score:       {score:.4f}")
    print(f"{'='*40}\n")
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Path to .onnx file")
    args = parser.parse_args()
    analyse(args.onnx)
