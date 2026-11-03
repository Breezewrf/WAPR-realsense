"""One category and one instance, with manual ROI initialization and recovery."""

import numpy as np

from .geometry import check_bbox, valid_pose


class PoseSession:
    """Reuse WAPR's unchanged initialization and six-candidate tracking calls.

    Model initialization is separate so callers may supply the already-prepared
    estimator and mesh. No automatic detector, segmentation or DINO is loaded.
    """

    def __init__(self, estimator, mesh, diameter_m, minimum_score=0.5, track_function=None):
        if track_function is None:
            from wapr.tracking import track_many_categories_many_instances
            track_function = track_many_categories_many_instances
        self.estimator, self.mesh, self.diameter = estimator, mesh, diameter_m
        self.minimum_score = minimum_score
        self.track_function = track_function
        self.pose = None

    def reset(self):
        self.pose = None

    def step(self, frame, now=None, bbox=None):
        # now remains accepted for compatibility with the first live prototype.
        if bbox is not None:
            bbox = check_bbox(bbox, frame.depth_m)
            row = self.estimator.estimate_one_category_one_instance(
                frame.rgb, frame.depth_m, frame.K, self.mesh, self.diameter, bbox_xywh=bbox)
            status = "initialized_roi"
        elif self.pose is None:
            return "waiting_roi", None
        else:
            rows = self.track_function(self.estimator, frame.rgb, frame.depth_m, frame.K, [{
                "track_id": 1, "obj_id": 1, "mesh": self.mesh,
                "diameter_m": self.diameter, "pose_4x4": self.pose,
            }])
            row = rows[0] if rows else None
            status = "tracking"
        if not valid_pose(row, self.minimum_score):
            self.pose = None
            return "lost", None
        self.pose = np.asarray(row["pose_4x4"], dtype=np.float32).copy()
        return status, row
