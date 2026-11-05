import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from wapr_realsense.app import parse_args
from wapr_realsense.output import read_latest_pose, transform_to_base

class ConfigurationTests(unittest.TestCase):
    def test_nested_yaml_detection_and_cli_precedence(self):
        self.assertTrue(parse_args([]).detect_use_sam)
        self.assertFalse(parse_args(["--no-detect-use-sam"]).detect_use_sam)
        with tempfile.TemporaryDirectory() as folder:
            cfg = Path(folder) / "target.yaml"
            cfg.write_text("""mesh: meshes/target.obj
groundingdino_sam2_dinov2_detect:
  activate: true
  use_sam2: false
  interval: 2.5
  score: 0.3
  backend: torch
  dino: vitb14
  grounding: swint
  template_cache: templates/target.pt
""")
            args = parse_args(["--config", str(cfg), "--no-display"])
            self.assertEqual(args.init, "auto")
            self.assertFalse(args.detect_use_sam)
            self.assertEqual(args.detect_interval, 2.5)
            self.assertEqual(args.detect_score, 0.3)
            self.assertEqual(args.detect_backend, "torch")
            self.assertEqual(args.detect_dino, "vitb14")
            self.assertEqual(args.detect_grounding, "swint")
            self.assertEqual(args.mesh, cfg.parent / "meshes/target.obj")
            self.assertEqual(args.template_cache, cfg.parent / "templates/target.pt")
            override = parse_args(["--config", str(cfg), "--init", "roi", "--detect-use-sam"])
            self.assertEqual(override.init, "roi")
            self.assertTrue(override.detect_use_sam)
            cfg.write_text("groundingdino_sam2_dinov2_detect:\n  activate: false\n")
            self.assertEqual(parse_args(["--config", str(cfg)]).init, "roi")
            self.assertEqual(parse_args(["--config", str(cfg), "--init", "auto"]).init, "auto")

    def test_invalid_nested_yaml_is_rejected(self):
        cases = ["", "[1, 2]", "mesh: [", "!!python/object:builtins.object {}",
                 "groundingdino_sam2_dinov2_detect: null",
                 "groundingdino_sam2_dinov2_detect: {activate: 'false'}",
                 "groundingdino_sam2_dinov2_detect: {activate: 1}",
                 "groundingdino_sam2_dinov2_detect: {use_sam2: 'false'}",
                 "groundingdino_sam2_dinov2_detect: {use_sam2: null}",
                 "init: auto", "detect_use_sam: false", "detect_score: 0.1",
                 "groundingdino_sam2_dinov2_detect: {unknown: true}",
                 "groundingdino_sam2_dinov2_detect: {score: 2}",
                 "init: auto\ngroundingdino_sam2_dinov2_detect: {activate: false}",
                 "detect_use_sam: true\ngroundingdino_sam2_dinov2_detect: {use_sam2: false}"]
        with tempfile.TemporaryDirectory() as folder:
            cfg = Path(folder) / "target.yml"
            for content in cases:
                with self.subTest(content=content), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cfg.write_text(content)
                    parse_args(["--config", str(cfg)])
            legacy = Path(folder) / "target.json"
            legacy.write_text("{}")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--config", str(legacy)])

    def test_timing_options_and_validation(self):
        args = parse_args([])
        self.assertEqual(args.timing_interval, 5.0)
        self.assertFalse(args.log_poses)
        self.assertFalse(args.profile_tracking)
        self.assertTrue(parse_args(["--profile-tracking"]).profile_tracking)
        self.assertEqual(args.tracking_preset, "full")
        self.assertTrue(args.reuse_tracking)
        self.assertEqual(parse_args(["--tracking-preset", "balanced"]).tracking_preset, "balanced")
        self.assertFalse(parse_args(["--no-reuse-tracking"]).reuse_tracking)
        args = parse_args(["--timing-interval", "2", "--log-poses"])
        self.assertEqual(args.timing_interval, 2.0)
        self.assertTrue(args.log_poses)
        for interval in ("0", "-1", "nan", "inf"):
            with self.subTest(interval=interval), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--timing-interval", interval])

    def test_relative_config_paths_and_explicit_cli_override(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = Path(folder) / "config.yaml"
            cfg.write_text(yaml.safe_dump({"mesh": "models/target.obj", "output": "out", "fps": 15}))
            args = parse_args(["--config", str(cfg), "--fps", "20"])
            self.assertEqual(args.mesh, cfg.parent / "models/target.obj")
            self.assertEqual(args.output, cfg.parent / "out")
            self.assertEqual(args.fps, 20)
            self.assertEqual(args.init, "roi")

    def test_invalid_config_and_headless_without_roi_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = Path(folder) / "config.yaml"
            for values in [{"unknown_key": True}, {"fps": "15"}, {"bbox": [0, 0, 100]}, {"backend": "auto"}]:
                cfg.write_text(yaml.safe_dump(values))
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(["--config", str(cfg)])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--no-display"])

    def test_auto_config_headless_paths_and_cli_override(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = Path(folder) / "auto.yaml"
            cfg.write_text(yaml.safe_dump({"groundingdino_sam2_dinov2_detect": {"activate": True,
                                       "template_cache": "templates/target.pt", "score": 0.3}, "no_display": True}))
            args = parse_args(["--config", str(cfg), "--detect-score", "0.4"])
            self.assertEqual(args.init, "auto")
            self.assertEqual(args.template_cache, cfg.parent / "templates/target.pt")
            self.assertEqual(args.detect_score, 0.4)

    def test_auto_invalid_options(self):
        cases = [["--detect-interval", "0"], ["--detect-interval", "nan"],
                 ["--detect-score", "inf"], ["--detect-score", "1.1"],
                 ["--init", "auto", "--bbox", "1", "1", "4", "4"]]
        for case in cases:
            with self.subTest(case=case), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(case)


class RoboticsOutputTests(unittest.TestCase):
    def record(self):
        return {"valid": True, "host_time_ns": 1_000_000_000, "frame_age_ms": 40.0,
                "frame_number": 1, "camera_frame": "color_optical_frame", "object_frame": "box",
                "pose_4x4": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]}

    def test_freshness_includes_source_processing_age(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "latest.json"
            path.write_text(json.dumps(self.record()))
            self.assertIsNotNone(read_latest_pose(path, 100, now_ns=1_050_000_000))
            self.assertIsNone(read_latest_pose(path, 100, now_ns=1_070_000_000))
            self.assertIsNone(read_latest_pose(path, 100, now_ns=900_000_000))
            record = self.record()
            record["valid"] = False
            path.write_text(json.dumps(record))
            self.assertIsNone(read_latest_pose(path, 100, now_ns=1_050_000_000))

    def test_fixed_camera_transform_direction(self):
        calibration = {"calibrated": True, "camera_frame": "color_optical_frame", "parent_frame": "base",
                       "T_base_camera": [[1, 0, 0, 0.1], [0, 1, 0, 0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]]}
        out = transform_to_base(self.record(), calibration)
        np.testing.assert_allclose(out["t_m"], [0.1, 0.2, 1.3])
        np.testing.assert_allclose(out["quaternion_xyzw"], [0, 0, 0, 1])
        self.assertEqual(out["parent_frame"], "base")
        calibration["calibrated"] = False
        with self.assertRaisesRegex(ValueError, "measured transform"):
            transform_to_base(self.record(), calibration)
        calibration["calibrated"] = True
        calibration["camera_frame"] = "different_camera"
        with self.assertRaisesRegex(ValueError, "camera_frame"):
            transform_to_base(self.record(), calibration)


if __name__ == "__main__":
    unittest.main()
