# Adapted for WAPR-realsense; SPDX-License-Identifier: LGPL-2.1-only
"""Tracking-only presets; pose initialization continues to use recipe.py."""

from collections import namedtuple

TRACKING_PRESETS = ("full", "balanced", "fast", "small-motion")
TrackingSettings = namedtuple("TrackingSettings", "wapr_iters sapr_iters hypotheses wapr_rows sapr_rows")
FULL_HYPOTHESES = ("previous", "depth_center", "wapr_center", "sapr_center",
                   "sapr_wapr_center", "sapr_wapr_previous")


def tracking_settings(preset):
    from wapr import recipe
    if preset == "full":
        return TrackingSettings(recipe.wapr_iters, recipe.sapr_iters, FULL_HYPOTHESES, 2, 3)
    if preset in ("balanced", "fast"):
        return TrackingSettings(2 if preset == "balanced" else 1, 1,
                                ("previous", "wapr_center", "sapr_wapr_center"), 1, 1)
    if preset == "small-motion":
        return TrackingSettings(0, 1, ("previous", "sapr_center"), 0, 1)
    raise ValueError("Unknown tracking preset: " + str(preset))
