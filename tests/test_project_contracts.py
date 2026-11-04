import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wapr_realsense.app import parse_args
from wapr_realsense.output import read_latest_pose, transform_to_base

class ConfigurationTests(unittest.TestCase):
    def test_relative_config_paths_and_explicit_cli_override(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = Path(folder) / "config.json"
            cfg.write_text(json.dumps({"mesh": "models/target.obj", "output": "out", "fps": 15}))
            args = parse_args(["--config", str(cfg), "--fps", "20"])
            self.assertEqual(args.mesh, cfg.parent / "models/target.obj")
            self.assertEqual(args.output, cfg.parent / "out")
            self.assertEqual(args.fps, 20)
            self.assertEqual(args.init, "roi")

    def test_invalid_config_and_headless_without_roi_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = Path(folder) / "config.json"
            for values in [{"unknown_key": True}, {"fps": "15"}, {"bbox": [0, 0, 100]}, {"backend": "auto"}]:
                cfg.write_text(json.dumps(values))
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(["--config", str(cfg)])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--no-display"])


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
