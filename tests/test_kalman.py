"""Prediction accuracy, SO(3) continuity, and the real tracking lifecycle."""

import contextlib
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

from wapr_realsense.app import parse_args
from wapr_realsense.camera import RGBDFrame
from wapr_realsense.kalman import KalmanSettings, PoseKalman, settings_from_args
from wapr_realsense.tracker import PoseSession


def pose_at(t):
    pose = np.eye(4)
    pose[:3, 3] = [0.2 * t, -0.1 * t, 1 + 0.05 * t]
    pose[:3, :3] = Rotation.from_rotvec([0, 0, np.deg2rad(179) + 0.4 * t]).as_matrix()
    return pose


def frame_at(t):
    return RGBDFrame(np.zeros((8, 8, 3), np.uint8), np.ones((8, 8), np.float32),
                     np.eye(3), round(t * 1000), t * 1000, "sensor", t)


class KalmanPredictionTests(unittest.TestCase):
    def test_irregular_constant_motion_predicts_better_than_previous_pose(self):
        motion = PoseKalman(KalmanSettings(activate=True))
        motion.initiate(pose_at(0), 0)
        errors, previous_errors = [], []
        t = 0
        previous = pose_at(0)
        for index in range(60):
            t += [0.04, 0.10, 0.07][index % 3]
            measurement = pose_at(t)
            predicted = motion.predict(t)
            if index > 10:
                def error(pose):
                    return [np.linalg.norm(pose[:3, 3] - measurement[:3, 3]),
                            Rotation.from_matrix(pose[:3, :3] @ measurement[:3, :3].T).magnitude()]
                errors.append(error(predicted))
                previous_errors.append(error(previous))
            motion.update(measurement)
            previous = measurement
        self.assertTrue(np.all(np.mean(errors, axis=0) < np.mean(previous_errors, axis=0) * 0.4))
        self.assertTrue(np.isfinite(motion.covariance).all())
        self.assertGreaterEqual(np.linalg.eigvalsh(motion.covariance).min(), -1e-12)

    def test_noncommuting_rotation_and_noisy_observations_stay_valid(self):
        motion = PoseKalman(KalmanSettings(activate=True))
        pose = pose_at(0)
        motion.initiate(pose, 0)
        rng = np.random.default_rng(3)
        for index in range(1, 90):
            pose = pose.copy()
            pose[:3, :3] = Rotation.from_rotvec([0.02, -0.01, 0.03]).as_matrix() @ pose[:3, :3]
            measurement = pose.copy()
            measurement[:3, :3] = Rotation.from_rotvec(rng.normal(0, 0.005, 3)).as_matrix() @ pose[:3, :3]
            measurement[:3, 3] += rng.normal(0, 0.002, 3)
            prediction = motion.predict(index / 15)
            np.testing.assert_allclose(prediction[:3, :3].T @ prediction[:3, :3], np.eye(3), atol=1e-6)
            motion.update(measurement)
        self.assertLess(Rotation.from_matrix(motion.pose[:3, :3] @ pose[:3, :3].T).magnitude(), 0.05)
        self.assertGreaterEqual(np.linalg.eigvalsh(motion.covariance).min(), -1e-12)

    def test_clock_discontinuity_and_long_gap_drop_stale_velocity(self):
        for timestamp in (0.1, 0.0, 2.0):
            motion = PoseKalman(KalmanSettings(activate=True))
            motion.initiate(pose_at(0), 0)
            motion.predict(0.1)
            motion.update(pose_at(0.1))
            prediction = motion.predict(timestamp)
            np.testing.assert_allclose(prediction, pose_at(0.1), atol=1e-7)
            np.testing.assert_array_equal(motion.velocity, 0)


class KalmanSessionTests(unittest.TestCase):
    def session(self, enabled=True):
        estimator = Mock()
        estimator.estimate_one_category_one_instance.return_value = {"pose_4x4": pose_at(0), "score_6d": 0.9}
        track = Mock()
        session = PoseSession(estimator, object(), 0.5, track_function=track,
                              kalman_settings=KalmanSettings(activate=enabled))
        session.step(frame_at(0), bbox=[0, 0, 4, 4])
        return session, track

    def test_predicts_tracking_input_and_publishes_unfiltered_wapr_result(self):
        session, track = self.session()
        for index in range(1, 10):
            row = {"pose_4x4": pose_at(index / 10), "score_6d": 0.9}
            track.return_value = [row]
            status, published = session.step(frame_at(index / 10))
            self.assertEqual(status, "tracking")
            self.assertIs(published, row)
            np.testing.assert_allclose(session.pose, row["pose_4x4"], atol=1e-7)
        prior = track.call_args.args[4][0]["pose_4x4"]
        self.assertGreater(prior[0, 3], pose_at(0.8)[0, 3])
        self.assertLess(abs(prior[0, 3] - pose_at(0.9)[0, 3]), 0.01)

    def test_disabled_uses_previous_pose(self):
        session, track = self.session(enabled=False)
        track.return_value = [{"pose_4x4": pose_at(0.1), "score_6d": 0.9}]
        session.step(frame_at(0.1))
        session.step(frame_at(0.2))
        np.testing.assert_allclose(track.call_args.args[4][0]["pose_4x4"], pose_at(0.1), atol=1e-7)

    def test_loss_reset_and_reinitialization_clear_motion(self):
        session, track = self.session()
        track.return_value = [{"pose_4x4": pose_at(0.1), "score_6d": 0.9}]
        session.step(frame_at(0.1))
        track.return_value = [{"pose_4x4": pose_at(10), "score_6d": 0.1}]
        self.assertEqual(session.step(frame_at(0.2)), ("lost", None))
        self.assertIsNone(session.motion.pose)
        self.assertIsNone(session.pose)
        session.step(frame_at(0.3), bbox=[0, 0, 4, 4])
        np.testing.assert_array_equal(session.motion.velocity, 0)
        session.reset()
        self.assertIsNone(session.motion.pose)

    def test_automatic_initialization_seeds_zero_velocity(self):
        initialize = Mock(return_value={"pose_4x4": pose_at(0), "score_6d": 0.9})
        track = Mock(return_value=[{"pose_4x4": pose_at(0.1), "score_6d": 0.9}])
        session = PoseSession(Mock(), object(), 0.5, initializer=initialize,
                              track_function=track, kalman_settings=KalmanSettings(activate=True))
        self.assertEqual(session.step(frame_at(0))[0], "initialized_auto")
        np.testing.assert_array_equal(session.motion.velocity, 0)
        self.assertEqual(session.step(frame_at(0.1))[0], "tracking")
        initialize.assert_called_once()

    def test_prediction_with_nonpositive_depth_falls_back_to_measurement(self):
        session, track = self.session()
        session.motion.velocity[2] = -20
        track.return_value = [{"pose_4x4": pose_at(0.1), "score_6d": 0.9}]
        session.step(frame_at(0.1))
        np.testing.assert_allclose(track.call_args.args[4][0]["pose_4x4"], pose_at(0), atol=1e-7)

    def test_clock_domain_change_and_invalid_sensor_time(self):
        session, track = self.session()
        track.return_value = [{"pose_4x4": pose_at(0.1), "score_6d": 0.9}]
        session.step(frame_at(0.1))
        session.step(replace(frame_at(0.2), timestamp_domain="new_sensor"))
        np.testing.assert_allclose(track.call_args.args[4][0]["pose_4x4"], pose_at(0.1), atol=1e-7)
        session.step(replace(frame_at(0.3), timestamp_ms=float("nan")))
        self.assertEqual(session.clock_domain, "host_monotonic")
        self.assertTrue(np.isfinite(session.motion.pose).all())


class KalmanConfigTests(unittest.TestCase):
    def test_yaml_cli_override_and_settings(self):
        self.assertFalse(parse_args([]).kalman_activate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.yaml"
            path.write_text("kalman_motion_prediction:\n  activate: true\n  max_dt_s: 0.25\n")
            args = parse_args(["--config", str(path)])
            self.assertTrue(settings_from_args(args).activate)
            self.assertEqual(settings_from_args(args).max_dt_s, 0.25)
            self.assertFalse(parse_args(["--config", str(path), "--no-kalman"]).kalman_activate)
        self.assertTrue(parse_args(["--kalman"]).kalman_activate)

    def test_bad_settings_are_rejected(self):
        cases = ["{activate: 'false'}", "{activate: null}", "{max_dt_s: 0}",
                 "{rotation_measurement_std_rad: .nan}", "{unknown: true}",
                 "{translation_measurement_std_m: true}", "null"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.yaml"
            for value in cases:
                path.write_text("kalman_motion_prediction: " + value)
                with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(["--config", str(path)])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--kalman-max-dt-s", "inf"])


if __name__ == "__main__":
    unittest.main()
