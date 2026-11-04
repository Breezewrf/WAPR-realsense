# Adapted for WAPR-realsense; SPDX-License-Identifier: LGPL-2.1-only
"""TensorRT builder settings shared by the pose engine exporters."""

import math
import os
import platform
import re


def network_flags(trt):
    """TensorRT 8.5 needs explicit batch; TensorRT 10 always uses it."""
    version = tuple(int(part) for part in trt.__version__.split(".")[:2])
    if version[0] == 10:
        return 0
    if version[0] == 8 and version >= (8, 5):
        return 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    raise RuntimeError("Pose engine export requires TensorRT 8.5+ in the 8.x series or 10.x")


def workspace_bytes():
    """Use a smaller build workspace on ARM64; permit an explicit override."""
    default = "1" if platform.machine().lower() in ("aarch64", "arm64") else "8"
    value = float(os.environ.get("WAPR_TRT_WORKSPACE_GB", default))
    if not math.isfinite(value) or value <= 0 or int(value * 1024 ** 3) < 1:
        raise ValueError("WAPR_TRT_WORKSPACE_GB must be a positive finite size in GiB")
    return int(value * 1024 ** 3)


def onnx_opset_limit(tensorrt_version=None):
    """TRT 8.5 needs LayerNorm decomposed into the pre-opset-17 operations."""
    if tensorrt_version is not None:
        version = tuple(int(part) for part in tensorrt_version.split(".")[:2])
        if version < (8, 6):
            return 16
    return 17


def parser_error_message(parser, onnx_path):
    """Include the actual graph node so a missing plugin can be diagnosed."""
    import onnx
    model = onnx.load(str(onnx_path), load_external_data=False)
    lines = []
    for index in range(parser.num_errors):
        error = parser.get_error(index)
        lines.append(str(error))
        match = re.search(r"In node (\d+)", str(error))
        node_index = int(match.group(1)) if match else -1
        if 0 <= node_index < len(model.graph.node):
            node = model.graph.node[node_index]
            lines.append("ONNX_NODE index=%d op_type=%s domain=%s name=%s" % (
                node_index, node.op_type, node.domain or "ai.onnx", node.name))
    return "\n".join(lines)
