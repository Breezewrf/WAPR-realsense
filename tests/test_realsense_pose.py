"""Run: .venv/bin/python -B -m unittest discover -s tests -v"""

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock

import numpy as np

from wapr_realsense.camera import RGBDFrame, convert_aligned_frame, rectification_maps
from wapr_realsense import app as live
from wapr_realsense.geometry import valid_pose

def frame():
    return RGBDFrame(np.zeros((12, 16, 3), np.uint8), np.ones((12, 16), np.float32),
                     np.array([[20, 0, 8], [0, 20, 6], [0, 0, 1]], np.float32),
                     7, 123.4, "hardware_clock", time.monotonic())


def result(score=0.8):
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = 1
    return {"pose_4x4": pose, "score_6d": score}


class CameraGeometryTests(unittest.TestCase):
    def test_sensor_scale_rgb_and_owned_buffers(self):
        bgr = np.full((2, 3, 3), [10, 20, 30], np.uint8)
        raw = np.array([[0, 4000, 1000], [2000, 100, 8000]], np.uint16)
        rgb, depth = convert_aligned_frame(bgr, raw, 0.00025)
        np.testing.assert_array_equal(rgb[0, 0], [30, 20, 10])
        np.testing.assert_allclose(depth, [[0, 1, 0.25], [0.5, 0.025, 2]])
        raw[:] = 0
        bgr[:] = 0
        self.assertEqual(depth[0, 1], 1)
        self.assertEqual(rgb[0, 0, 0], 30)

    def test_rectified_depth_uses_nearest_and_keeps_invalid_zero(self):
        bgr = np.zeros((1, 3, 3), np.uint8)
        raw = np.array([[1000, 0, 3000]], np.uint16)
        maps = (np.array([[0.1, 0.9, 2.1]], np.float32), np.zeros((1, 3), np.float32))
        _, depth = convert_aligned_frame(bgr, raw, 0.001, maps)
        np.testing.assert_allclose(depth, [[1, 0, 3]])

    def test_d455_inverse_brown_map_matches_sdk_deprojection(self):
        try:
            import pyrealsense2 as rs
        except ImportError:
            self.skipTest("Optional RealSense SDK is not installed")
        intr = rs.intrinsics()
        intr.width, intr.height = 16, 12
        intr.fx, intr.fy, intr.ppx, intr.ppy = 12, 12, 8, 6
        intr.model = rs.distortion.inverse_brown_conrady
        intr.coeffs = [0.08, -0.03, 0.001, -0.002, 0.005]
        K, maps = rectification_maps(intr, rs)
        for x, y in [(2, 3), (8, 6), (13, 9)]:
            observed_pixel = [float(maps[0][y, x]), float(maps[1][y, x])]
            ray = rs.rs2_deproject_pixel_to_point(intr, observed_pixel, 1.0)
            np.testing.assert_allclose(ray[:2], [(x-K[0, 2])/K[0, 0], (y-K[1, 2])/K[1, 1]], atol=2e-6)


class PoseLifecycleTests(unittest.TestCase):
    def make_session(self):
        est = Mock()
        est.estimate_one_category_one_instance.return_value = result()
        tracker = Mock(return_value=[result()])
        session = live.PoseSession(est, object(), 0.5, track_function=tracker)
        return session, est, tracker

    def test_roi_initialization_followed_by_tracking_and_loss(self):
        session, est, tracker = self.make_session()
        f = frame()
        self.assertEqual(session.step(f, 0)[0], "waiting_roi")
        self.assertEqual(session.step(f, 1, bbox=[2, 2, 8, 7])[0], "initialized_roi")
        self.assertEqual(session.step(f, 2)[0], "tracking")
        np.testing.assert_array_equal(tracker.call_args.args[4][0]["pose_4x4"], result()["pose_4x4"])
        tracker.return_value = [result(0.1)]
        status, row = session.step(f, 3)
        self.assertEqual(status, "lost")
        self.assertIsNone(row)
        self.assertIsNone(session.pose)
        self.assertEqual(session.step(f, 4)[0], "waiting_roi")
        self.assertEqual(est.estimate_one_category_one_instance.call_count, 1)

    def test_roi_without_depth_does_not_enter_estimation(self):
        session, est, _ = self.make_session()
        f = frame()
        f.depth_m[:] = 0
        with self.assertRaisesRegex(ValueError, "no valid depth"):
            session.step(f, 1, bbox=[2, 2, 8, 7])
        est.estimate_one_category_one_instance.assert_not_called()

    def test_reset_clears_prior_pose_before_manual_reinitialization(self):
        session, _, tracker = self.make_session()
        f = frame()
        session.step(f, bbox=[2, 2, 8, 7])
        session.reset()
        self.assertEqual(session.step(f), ("waiting_roi", None))
        tracker.assert_not_called()

    def test_invalid_pose_is_not_published(self):
        for value in [float("nan"), -1]:
            row = result()
            row["pose_4x4"][2, 3] = value
            self.assertFalse(valid_pose(row, 0.5))

    def test_latest_json_replaces_valid_pose_with_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = live.PoseWriter(directory)
            f = frame()
            first = live.pose_record(f, "tracking", result(), 12.5, "box.obj")
            writer.write(first)
            self.assertTrue(json.loads((Path(directory) / "latest.json").read_text())["valid"])
            lost = live.pose_record(f, "lost", None, 11, "box.obj")
            writer.write(lost)
            writer.close()
            latest = json.loads((Path(directory) / "latest.json").read_text())
            self.assertFalse(latest["valid"])
            self.assertIsNone(latest["pose_4x4"])
            self.assertEqual(latest["camera_timestamp_ms"], 123.4)
            lines = (Path(directory) / "poses.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(json.loads(lines[0])["status"], "starting")


if __name__ == "__main__":
    unittest.main()
