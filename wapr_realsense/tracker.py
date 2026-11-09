"""One category and one instance, with manual or optional automatic initialization."""

import time
from functools import partial
import numpy as np
from wapr.tracking_config import tracking_settings

from .geometry import check_bbox, valid_pose
from .kalman import PoseKalman
from .recovery import RecoverySettings
from wapr.timing import print_timing


class PoseSession:
    """Reuse WAPR initialization with configurable tracking-only workloads.

    Model initialization is separate so callers may supply the already-prepared
    estimator and mesh. The optional initializer owns automatic detection.
    """

    def __init__(self, estimator, mesh, diameter_m, minimum_score=0.5, track_function=None,
                 initializer=None, detect_interval=1.0, tracking_preset="full", reuse_tracking=True,
                 kalman_settings=None, recovery_settings=None):
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
        self.motion = PoseKalman(kalman_settings) if kalman_settings and kalman_settings.activate else None
        self.clock_domain = None
        self.recovery = recovery_settings or RecoverySettings()
        self.recovery_started = None
        self.recovery_hits = 0
        self.recovery_frame = None

    def reset(self):
        self.pose = None
        self.next_detection = 0.0
        if self.motion is not None:
            self.motion.reset()
        self.clock_domain = None
        self.recovery_started = None
        self.recovery_hits = 0
        self.recovery_frame = None

    @staticmethod
    def _refined(row):
        return row is not None and row.get("hypothesis", "").startswith(("wapr_", "sapr_"))

    def _unreliable(self, current):
        if self.recovery.activate and self.pose is not None:
            if self.recovery_started is None:
                self.recovery_started = current
                print_timing("RECOVERY", "started")
            self.recovery_hits = 0
            return "recovering", None
        self.reset()
        return "lost", None

    def step(self, frame, now=None, bbox=None):
        current = time.monotonic() if now is None else now
        if bbox is not None:
            self.reset()
        if self.recovery_started is not None:
            if current - self.recovery_started >= self.recovery.timeout_s:
                print_timing("RECOVERY", "timeout | result=lost")
                self.reset()
                return "lost", None
            key = (frame.timestamp_domain, frame.number, frame.timestamp_ms)
            old = self.recovery_frame
            if old is not None and key[0] == old[0] and (key[1] == old[1] or key[2] == old[2]):
                return "recovering", None
            self.recovery_frame = key
        if self.motion is not None:
            domain, timestamp = frame.timestamp_domain, frame.timestamp_ms / 1000.0
            if not np.isfinite(timestamp):
                domain, timestamp = "host_monotonic", frame.received_monotonic
        if bbox is not None:
            bbox = check_bbox(bbox, frame.depth_m)
            row = self.estimator.estimate_one_category_one_instance(
                frame.rgb, frame.depth_m, frame.K, self.mesh, self.diameter, bbox_xywh=bbox)
            status = "initialized_roi"
        elif self.pose is None:
            if self.initializer is None:
                return "waiting_roi", None
            if current < self.next_detection:
                return "searching", None
            row = self.initializer(frame)
            # In production, count the interval from completion of the slow search.
            self.next_detection = (time.monotonic() if now is None else now) + self.detect_interval
            if not valid_pose(row, self.minimum_score):
                return "searching", None
            status = "initialized_auto"
        else:
            prior = self.pose
            if self.motion is not None:
                if (timestamp <= self.motion.request_timestamp or
                        timestamp - self.motion.request_timestamp > self.motion.settings.max_dt_s):
                    self.recovery_hits = 0
                if self.clock_domain != domain:
                    self.motion.initiate(self.pose, timestamp)
                    self.recovery_hits = 0
                prior = self.motion.predict(timestamp, limit_horizon=True)
                if not np.isfinite(prior).all() or prior[2, 3] <= 0:
                    self.motion.initiate(self.pose, timestamp)
                    prior = self.pose
                self.clock_domain = domain
            rows = self.track_function(self.estimator, frame.rgb, frame.depth_m, frame.K, [{
                "track_id": 1, "obj_id": 1, "mesh": self.mesh,
                "diameter_m": self.diameter, "pose_4x4": prior,
            }])
            row = rows[0] if rows else None
            status = "tracking"
            current = time.monotonic() if now is None else now
            if self.recovery_started is not None and current - self.recovery_started >= self.recovery.timeout_s:
                print_timing("RECOVERY", "timeout | result=lost")
                self.reset()
                return "lost", None
        if not valid_pose(row, self.minimum_score):
            if status == "tracking":
                if self.recovery_started is None:
                    self.recovery_frame = (frame.timestamp_domain, frame.number, frame.timestamp_ms)
                return self._unreliable(current)
            self.reset()
            return "lost", None
        if self.recovery_started is not None:
            if not self._refined(row):
                self.recovery_hits = 0
                return "recovering", None
            self.recovery_hits += 1
            if self.recovery_hits < self.recovery.confirm_frames:
                return "recovering", None
            print_timing("RECOVERY", "succeeded | elapsed_s=%.3f" % (current - self.recovery_started))
            self.recovery_started = None
            self.recovery_hits = 0
            self.recovery_frame = None
        self.pose = np.asarray(row["pose_4x4"], dtype=np.float32).copy()
        if self.motion is not None:
            if status != "tracking":
                self.motion.initiate(self.pose, timestamp)
            elif self._refined(row):
                self.motion.update(self.pose)
            self.clock_domain = domain
        return status, row
