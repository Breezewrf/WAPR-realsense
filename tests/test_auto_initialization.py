"""CPU integration contracts for the optional detector and tracking lifecycle."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from wapr_realsense import app
from wapr_realsense.initialization import AutoInitializer, build_initializer
from wapr_realsense.tracker import PoseSession
from wapr.timing import ComponentTimer, TimingCollector
from test_realsense_pose import frame, result


class AutoInitializationTests(unittest.TestCase):
    def test_search_retry_tracking_loss_reacquisition_and_reset(self):
        initialize = Mock(side_effect=[None, result(0.1), result(), result(), result()])
        track = Mock(return_value=[result()])
        session = PoseSession(Mock(), object(), 0.5, initializer=initialize,
                              track_function=track, detect_interval=2)
        f = frame()
        self.assertEqual(session.step(f, 0), ("searching", None))
        self.assertEqual(session.step(f, 1), ("searching", None))
        self.assertEqual(initialize.call_count, 1)
        self.assertEqual(session.step(f, 2), ("searching", None))
        self.assertEqual(session.step(f, 4)[0], "initialized_auto")
        self.assertEqual(session.step(f, 5)[0], "tracking")
        self.assertEqual(initialize.call_count, 3)
        track.return_value = []
        self.assertEqual(session.step(f, 6), ("lost", None))
        self.assertIsNone(session.pose)
        self.assertEqual(session.step(f, 6.1)[0], "initialized_auto")
        session.reset()
        self.assertEqual(session.step(f, 6.2)[0], "initialized_auto")

    def test_retry_interval_starts_after_detection_completes(self):
        initialize = Mock(return_value=None)
        session = PoseSession(Mock(), object(), 0.5, initializer=initialize, track_function=Mock())
        f = frame()
        with patch("wapr_realsense.tracker.time.monotonic", side_effect=[10, 20, 20.5, 21, 25]):
            session.step(f)
            session.step(f)
            self.assertEqual(initialize.call_count, 1)
            session.step(f)
        self.assertEqual(initialize.call_count, 2)

    def test_mask_pipeline_restricts_target_and_selects_highest_valid_pose(self):
        best = result(0.9)
        invalid = result(1.0)
        invalid["pose_4x4"][2, 3] = -1
        pipeline = Mock(return_value=([result(0.7), invalid, result(0.2), best], {}, []))
        estimator, detector, mesh = Mock(device=None), Mock(), object()
        initialize = AutoInitializer(estimator, detector, mesh, 0.5, 0.2, 0.5, estimate_frame=pipeline)
        self.assertIs(initialize(frame()), best)
        self.assertEqual(pipeline.call_args.args[5], {1: (mesh, 0.5)})
        self.assertEqual(pipeline.call_args.kwargs,
                         {"obj_ids": [1], "score_2d_min": 0.2, "score_6d_min": 0.5})
        pipeline.return_value = ([], {}, [])
        self.assertIsNone(initialize(frame()))

    def test_initialization_logs_include_pose_total_and_retries_without_mixing_warmup(self):
        def estimate(*args, **kwargs):
            with TimingCollector() as warmup:
                with ComponentTimer("GroundingDINO"):
                    pass
            warmup.report_warmup(500)
            with ComponentTimer("GroundingDINO"):
                pass
            with ComponentTimer("SAM2"):
                pass
            with ComponentTimer("DINOv2"):
                pass
            return [], {"pose_hot_seconds": 3.0, "total_ms": 1000, "pose_candidates": 72}, []

        initialize = AutoInitializer(Mock(device=None), Mock(), object(), 0.5, 0.1, 0.5, estimate_frame=estimate)
        output = io.StringIO()
        with redirect_stdout(output):
            initialize(frame())
            initialize(frame())
        text = output.getvalue()
        self.assertIn("[INIT] frame=7 | attempt=1 | started", text)
        self.assertIn("[REINIT] frame=7 | attempt=2 | started", text)
        self.assertIn("WAPR initialization=3000.00 ms", text)
        self.assertIn("detection total=1000.00 ms", text)
        self.assertIn("pose candidates=72", text)
        self.assertIn("result=searching", text)
        self.assertNotIn("[TIMING]", text)

    def test_real_frame_pipeline_reports_each_warmup_once_and_keeps_hot_samples_separate(self):
        from wapr.frame import estimate_frame_many_categories_many_instances

        f = frame()
        mesh = object()
        pose = result()
        pose.update(R=pose["pose_4x4"][:3, :3], t_m=pose["pose_4x4"][:3, 3])
        estimator = SimpleNamespace(device=None, last_mesh_setup_seconds=0,
                                    _pose_meshes=Mock(side_effect=lambda meshes: meshes),
                                    warmup_pose=Mock(return_value=0.9),
                                    estimate_many_categories_many_instances=Mock(return_value=[pose]))

        def detect(*args, **kwargs):
            for name in ("GroundingDINO", "SAM2", "DINOv2"):
                with ComponentTimer(name):
                    pass
            instance = {"obj_id": 1, "score_2d": 0.9, "bbox": [0, 0, 16, 12], "mask": {}}
            return [instance], {"total_ms": 1, "candidates": 1, "outputs": 1}

        detector = SimpleNamespace(detect_many_categories_many_instances=Mock(side_effect=detect))
        masks = ModuleType("pycocotools.mask")
        masks.decode = Mock(return_value=np.ones(f.depth_m.shape, dtype=np.uint8))
        package = ModuleType("pycocotools")
        package.mask = masks
        output = io.StringIO()
        with patch.dict(sys.modules, {"pycocotools": package, "pycocotools.mask": masks}), \
             patch("torch.cuda.synchronize"), redirect_stdout(output):
            for _ in range(2):
                with TimingCollector() as hot:
                    _, timing, _ = estimate_frame_many_categories_many_instances(
                        estimator, detector, f.rgb, f.depth_m, f.K, {1: (mesh, 0.5)}, obj_ids=[1])
                self.assertEqual(hot.samples["GroundingDINO"][1], 1)
                self.assertEqual(timing["pose_candidates"], 1)
        text = output.getvalue()
        self.assertEqual(text.count("[WARMUP] GroundingDINO:"), 1)
        self.assertEqual(text.count("[WARMUP] SAM2:"), 1)
        self.assertEqual(text.count("[WARMUP] DINOv2:"), 1)
        self.assertEqual(text.count("[WARMUP] WAPR initialization:"), 1)
        self.assertIn("total=900.00 ms | avg=300.00 ms | calls=3", text)
        self.assertNotIn("[TIMING]", text)
        self.assertEqual(detector.detect_many_categories_many_instances.call_count, 5)
        self.assertEqual(estimator.warmup_pose.call_count, 1)
        self.assertEqual(estimator.warmup_pose.call_args.kwargs["calls"], 3)

    def test_builder_shares_encoder_choice_and_preserves_original_mesh_for_templates(self):
        det2d = ModuleType("wapr.det2d")
        det2d.onboard_meshes = Mock(return_value="templates.pt")
        det2d.WAPRDet2D = Mock()
        frame_module = ModuleType("wapr.frame")
        frame_module.estimate_frame_many_categories_many_instances = Mock()
        args = app.parse_args(["--init", "auto", "--detect-dino", "vits14", "--detect-grounding", "swint",
                               "--detect-backend", "torch", "--template-cache", "/tmp/templates.pt"])
        estimator, mesh, prepared = Mock(device="cuda:0"), object(), object()
        with patch.dict(sys.modules, {"wapr.det2d": det2d, "wapr.frame": frame_module}):
            initialize = build_initializer(args, estimator, mesh, prepared, 0.5)
        det2d.onboard_meshes.assert_called_once_with(
            {1: mesh}, cache_path="/tmp/templates.pt", device="cuda:0", dino="vits14", cache_policy="version")
        det2d.WAPRDet2D.assert_called_once_with(
            "templates.pt", device="cuda:0", backend="torch", dino="vits14", grounding="swint", use_sam=True)
        self.assertIs(initialize.meshes[1][0], prepared)
        args.detect_use_sam = False
        with patch.dict(sys.modules, {"wapr.det2d": det2d, "wapr.frame": frame_module}):
            build_initializer(args, estimator, mesh, prepared, 0.5)
        self.assertFalse(det2d.WAPRDet2D.call_args.kwargs["use_sam"])

    def test_bbox_pipeline_passes_bbox_without_mask_and_filters_depth(self):
        from wapr.frame import estimate_frame_many_categories_many_instances

        f, mesh = frame(), object()
        f.depth_m[:] = 0
        f.depth_m[3, 3] = 1
        pose = result()
        pose.update(R=pose["pose_4x4"][:3, :3], t_m=pose["pose_4x4"][:3, 3])
        estimator = SimpleNamespace(device=None, last_mesh_setup_seconds=0,
                                    _pose_meshes=Mock(side_effect=lambda meshes: meshes),
                                    warmup_pose=Mock(return_value=0),
                                    estimate_many_categories_many_instances=Mock(return_value=[pose]))
        detections = [{"obj_id": 1, "score_2d": .9, "bbox": box}
                      for box in ([2, 2, 4, 4], [8, 8, 3, 3], [100, 100, 3, 3])]
        detector = SimpleNamespace(_frame_detector_warmed=True,
                                   detect_many_categories_many_instances=Mock(return_value=(detections, {})))
        with patch("torch.cuda.synchronize"), patch.dict(sys.modules, {"pycocotools": None}), \
             redirect_stdout(io.StringIO()):
            poses, timing, _ = estimate_frame_many_categories_many_instances(
                estimator, detector, f.rgb, f.depth_m, f.K, {1: (mesh, 0.5)}, obj_ids=[1])
        items = estimator.estimate_many_categories_many_instances.call_args.args[3]
        self.assertEqual(items, [{"mesh": mesh, "diameter_m": 0.5, "bbox_xywh": [2., 2., 4., 4.]}])
        self.assertEqual(estimator.warmup_pose.call_args.args[3], items)
        self.assertEqual(len(poses), 1)
        self.assertEqual(timing["pose_candidates"], 1)

    def test_roi_builder_does_not_import_detection_or_torch(self):
        command = ("import sys; from wapr_realsense.app import parse_args; "
                   "from wapr_realsense.initialization import build_initializer; "
                   "assert build_initializer(parse_args([]), None, None, None, 1) is None; "
                   "assert 'wapr.det2d' not in sys.modules; assert 'torch' not in sys.modules")
        subprocess.run([sys.executable, "-B", "-c", command], check=True)

    def test_auto_replay_needs_no_bbox_or_mask_and_publishes_status(self):
        import wapr

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            f = frame()
            cv2.imwrite(str(folder / "rgb.png"), f.rgb)
            np.save(folder / "depth.npy", f.depth_m)
            (folder / "camera.json").write_text(json.dumps({"fx": 20, "fy": 20, "cx": 8, "cy": 6}))
            cfg = folder / "auto.json"
            cfg.write_text(json.dumps({"init": "auto", "no_display": True, "replay": ".",
                                       "max_frames": 2, "output": "out"}))
            estimator = Mock()
            mesh = Mock(extents=np.array([0.2, 0.3, 0.4]))
            estimator.prepare_meshes.return_value = [mesh]
            track_module = ModuleType("wapr.tracking")
            track_module.track_many_categories_many_instances = Mock(return_value=[result()])
            with patch.dict("os.environ"), \
                 patch.dict(wapr.__dict__, {"WAPREstimator": Mock(return_value=estimator)}), \
                 patch.dict(sys.modules, {"wapr.tracking": track_module}), \
                 patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.synchronize"), \
                 patch.object(app, "load_mesh", return_value=(mesh, 0.5)), \
                 patch.object(app, "build_initializer", return_value=Mock(return_value=result())) as builder, \
                 patch.object(app, "draw_pose", return_value=f.rgb), \
                 redirect_stdout(io.StringIO()) as terminal:
                app.main(["--config", str(cfg)])
            self.assertNotIn("POSE_JSON", terminal.getvalue())
            self.assertIn("[TRACKING] WAPR avg=", terminal.getvalue())
            self.assertEqual(builder.call_args.args[0].init, "auto")
            self.assertIs(builder.call_args.args[1], estimator)
            rows = [json.loads(line) for line in (folder / "out" / "poses.jsonl").read_text().splitlines()]
            self.assertEqual([row["status"] for row in rows], ["starting", "initialized_auto", "tracking", "stopped"])
            self.assertTrue(rows[-2]["valid"])
            self.assertFalse(rows[-1]["valid"])
            metadata = json.loads((folder / "out" / "session.json").read_text())
            self.assertEqual(metadata["automatic_detection"]["dino"], "vitl14")


if __name__ == "__main__":
    unittest.main()
