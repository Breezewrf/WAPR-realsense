"""One category and one instance, with manual or optional automatic initialization."""

import time
from functools import partial
import numpy as np
from wapr.tracking_config import tracking_settings

from .geometry import check_bbox, valid_pose


class PoseSession:
    """Reuse WAPR initialization with configurable tracking-only workloads.

    Model initialization is separate so callers may supply the already-prepared
    estimator and mesh. The optional initializer owns automatic detection.
    """

    def __init__(self, estimator, mesh, diameter_m, minimum_score=0.5, track_function=None,
                 initializer=None, detect_interval=1.0, tracking_preset="full", reuse_tracking=True):
        tracking_settings(tracking_preset)
        if not np.isfinite(detect_interval) or detect_interval <= 0:
            raise ValueError("detect_interval must be finite and positive")
        if track_function is None:
            from wapr.tracking import track_many_categories_many_instances
            track_function = partial(track_many_categories_many_instances,
                                     tracking_preset=tracking_preset, reuse_tracking=reuse_tracking)
        self.estimator, self.mesh, self.diameter = estimator, mesh, diameter_m
        self.minimum_score = minimum_score
        self.track_function = track_function
        self.initializer = initializer
        self.detect_interval = detect_interval
        self.next_detection = 0.0
        self.pose = None

    def reset(self):
        self.pose = None
        self.next_detection = 0.0

    def step(self, frame, now=None, bbox=None):
        if bbox is not None:
            bbox = check_bbox(bbox, frame.depth_m)
            row = self.estimator.estimate_one_category_one_instance(
                frame.rgb, frame.depth_m, frame.K, self.mesh, self.diameter, bbox_xywh=bbox)
            status = "initialized_roi"
        elif self.pose is None:
            if self.initializer is None:
                return "waiting_roi", None
            current = time.monotonic() if now is None else now
            if current < self.next_detection:
                return "searching", None
            row = self.initializer(frame)
            # In production, count the interval from completion of the slow search.
            self.next_detection = (time.monotonic() if now is None else now) + self.detect_interval
            if not valid_pose(row, self.minimum_score):
                return "searching", None
            status = "initialized_auto"
        else:
            rows = self.track_function(self.estimator, frame.rgb, frame.depth_m, frame.K, [{
                "track_id": 1, "obj_id": 1, "mesh": self.mesh,
                "diameter_m": self.diameter, "pose_4x4": self.pose,
            }])
            row = rows[0] if rows else None
            status = "tracking"
        if not valid_pose(row, self.minimum_score):
            self.reset()
            return "lost", None
        self.pose = np.asarray(row["pose_4x4"], dtype=np.float32).copy()
        return status, row
