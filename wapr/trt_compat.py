# Adapted for WAPR-realsense; SPDX-License-Identifier: LGPL-2.1-only
"""TensorRT builder settings shared by the pose engine exporters."""

import math
import os
import platform


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
