# Adapted for WAPR-realsense; SPDX-License-Identifier: LGPL-2.1-only
"""Pose row budgets shared by export and runtime validation."""

from wapr import recipe
from wapr.pose_groups import wbps_group_sizes


def pose_profile(object_mode):
    if object_mode not in ("single", "multi"):
        raise ValueError("object_mode must be single or multi")
    group = int(recipe.n_view * recipe.n_inplane)
    if group not in wbps_group_sizes:
        raise ValueError("Unsupported recipe pose group: %d" % group)
    maximum = max(group, 6) if object_mode == "single" else max(wbps_group_sizes)
    return min(wbps_group_sizes), group, maximum


def validate_pose_profile(actual, object_mode):
    expected = pose_profile(object_mode)
    # Existing wide engines remain usable for a single object. This permits
    # gradual replacement without forcing a rebuild of all four networks.
    allowed_maximum = (expected[2], max(wbps_group_sizes)) if object_mode == "single" else (expected[2],)
    if tuple(actual[:2]) != expected[:2] or actual[2] not in allowed_maximum:
        raise RuntimeError("Engine profile %s is incompatible with %s mode; expected %s. "
                           "Rebuild with python -m wapr.export_engines --object-mode %s"
                           % (tuple(actual), object_mode, expected, object_mode))
