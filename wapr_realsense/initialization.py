"""Optional CAD-template detection with heavy dependencies loaded only in auto mode."""

from wapr.timing import ComponentTimer, TimingCollector, print_timing

from .geometry import valid_pose


class AutoInitializer:
    """Use WAPR's GroundingDINO/SAM2/DINOv2 frame pipeline for one mesh."""

    def __init__(self, estimator, detector, mesh, diameter_m, score_2d_min,
                 minimum_score, estimate_frame=None):
        if estimate_frame is None:
            from wapr.frame import estimate_frame_many_categories_many_instances
            estimate_frame = estimate_frame_many_categories_many_instances
        self.estimator, self.detector = estimator, detector
        self.meshes = {1: (mesh, diameter_m)}
        self.score_2d_min, self.minimum_score = score_2d_min, minimum_score
        self.estimate_frame = estimate_frame
        self.attempts = 0

    def __call__(self, frame):
        part = "INIT" if self.attempts == 0 else "REINIT"
        self.attempts += 1
        print_timing(part, f"frame={frame.number} | attempt={self.attempts} | started")
        with TimingCollector() as components, ComponentTimer(None, self.estimator.device) as total:
            poses, timing, _ = self.estimate_frame(
                self.estimator, self.detector, frame.rgb, frame.depth_m, frame.K,
                self.meshes, obj_ids=[1], score_2d_min=self.score_2d_min,
                score_6d_min=self.minimum_score,
            )
            accepted = [row for row in poses if valid_pose(row, self.minimum_score)]
            selected = max(accepted, key=lambda row: row["score_6d"], default=None)
        if "pose_hot_seconds" in timing:
            components.record("WAPR initialization", timing["pose_hot_seconds"] * 1000)
        print_timing(part, " | ".join(components.format(name) for name in
                     ("GroundingDINO", "SAM2", "DINOv2", "WAPR initialization")))
        detection = timing.get("total_ms")
        detection_text = "n/a" if detection is None else f"{detection:.2f} ms"
        print_timing(part, f"total={total.elapsed_ms:.2f} ms | detection total={detection_text} | "
                     f"masks={timing.get('candidates', 'n/a')} | pose candidates={timing.get('pose_candidates', 'n/a')} | "
                     f"accepted={len(accepted)} | result={'found' if selected is not None else 'searching'}")
        return selected


def build_initializer(args, estimator, mesh, prepared_mesh, diameter_m):
    """Keep ROI mode free of detector imports, downloads and template rendering."""
    if args.init != "auto":
        return None
    from wapr.det2d import WAPRDet2D, onboard_meshes

    template = onboard_meshes(
        {1: mesh}, cache_path="" if args.template_cache is None else str(args.template_cache),
        device=estimator.device, dino=args.detect_dino, cache_policy="version",
    )
    detector = WAPRDet2D(template, device=estimator.device,
                         backend=args.detect_backend, dino=args.detect_dino,
                         grounding=args.detect_grounding, use_sam=args.detect_use_sam)
    return AutoInitializer(estimator, detector, prepared_mesh, diameter_m,
                           args.detect_score, args.min_score)
