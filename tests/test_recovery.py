"""Bounded prediction-only recovery and public output contracts."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from wapr_realsense.app import parse_args
from wapr_realsense.kalman import KalmanSettings
from wapr_realsense.output import pose_record
from wapr_realsense.recovery import RecoverySettings, recovery_from_args
from wapr_realsense.tracker import PoseSession
from test_kalman import frame_at, pose_at


def row(t=0, score=0.9, hypothesis='sapr_center'):
    return dict(pose_4x4=pose_at(t), score_6d=score, hypothesis=hypothesis)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.estimator = Mock(device=None)
        self.estimator.estimate_one_category_one_instance.return_value = row()
        self.track = Mock(return_value=[])
        self.session = PoseSession(self.estimator, object(), .5, track_function=self.track,
                                   kalman_settings=KalmanSettings(activate=True),
                                   recovery_settings=RecoverySettings(activate=True))
        self.session.step(frame_at(0), now=0, bbox=[0, 0, 4, 4])

    def step(self, t, value=None):
        self.track.return_value = [] if value is None else [value]
        return self.session.step(frame_at(t), now=t)

    def test_preserves_reliable_state_predicts_and_freezes_without_measurement_updates(self):
        self.session.motion.velocity[0] = .2
        for t in (.1, .2, .3, .4, .5, .6, .7):
            self.assertEqual(self.step(t, row(20, score=.1)), ('recovering', None))
            prior = self.track.call_args.args[4][0]['pose_4x4']
            self.assertAlmostEqual(prior[0, 3], .2 * min(t, .5), places=6)
            np.testing.assert_allclose(self.session.pose, pose_at(0), atol=1e-7)
            np.testing.assert_allclose(self.session.motion.last_measurement, pose_at(0))
        self.assertEqual(self.session.recovery_started, .1)
        status, output = self.step(.8, row(.8))
        self.assertEqual((status, output), ('recovering', None))
        status, output = self.step(.9, row(.9))
        self.assertEqual(status, 'tracking')
        self.assertIsNotNone(output)
        np.testing.assert_allclose(self.session.motion.last_measurement, pose_at(.9), atol=1e-7)
        self.assertIsNone(self.session.recovery_started)

    def test_unaccepted_candidates_and_duplicate_frames_do_not_confirm(self):
        self.step(.1)
        self.step(.2, row(.2))
        calls = self.track.call_count
        self.assertEqual(self.step(.2, row(.2)), ('recovering', None))
        self.assertEqual(self.track.call_count, calls)
        self.step(.3, row(.3, hypothesis='previous'))
        self.assertEqual(self.session.recovery_hits, 0)
        self.step(.4, row(.4))
        self.step(.5, row(.5, score=.1))
        self.assertEqual(self.session.recovery_hits, 0)
        self.assertEqual(self.step(.6, row(.6))[0], 'recovering')
        self.assertEqual(self.step(.7, row(.7))[0], 'tracking')

    def test_timeout_clears_state_and_branches_to_manual_or_auto(self):
        self.step(.1)
        calls = self.track.call_count
        self.assertEqual(self.step(1.1), ('lost', None))
        self.assertEqual(self.track.call_count, calls)
        self.assertIsNone(self.session.pose)
        self.assertIsNone(self.session.motion.pose)
        self.assertEqual(self.step(1.2)[0], 'waiting_roi')
        self.session.initializer = Mock(return_value=row())
        self.assertEqual(self.step(1.3)[0], 'initialized_auto')

    def test_failed_initialization_does_not_enter_recovery(self):
        self.estimator.estimate_one_category_one_instance.return_value = row(score=.1)
        self.assertEqual(self.session.step(frame_at(.1), now=.1, bbox=[0, 0, 4, 4]), ('lost', None))
        self.assertIsNone(self.session.recovery_started)

    def test_disabled_kalman_retries_from_reliable_pose(self):
        self.session.motion = None
        self.step(.1)
        self.step(.2, row(.2))
        np.testing.assert_allclose(self.track.call_args.args[4][0]['pose_4x4'], pose_at(0), atol=1e-7)
        self.assertEqual(self.step(.3, row(.3))[0], 'tracking')

    def test_unrefined_normal_prediction_does_not_update_filter(self):
        self.session.motion.velocity[0] = .2
        self.step(.1, row(.1, hypothesis='previous'))
        np.testing.assert_allclose(self.session.motion.last_measurement, pose_at(0))
        self.assertEqual(self.session.motion.measurement_timestamp, 0)

    def test_clock_change_invalid_prediction_and_reset(self):
        self.step(.1)
        self.step(.2, row(.2))
        self.track.return_value = [row(.3)]
        status, _ = self.session.step(replace(frame_at(.3), timestamp_domain='new'), now=.3)
        self.assertEqual(status, 'recovering')
        self.assertEqual(self.session.recovery_hits, 1)
        self.session.motion.velocity[2] = -100
        self.step(.4)
        self.assertGreater(self.track.call_args.args[4][0]['pose_4x4'][2, 3], 0)
        self.session.reset()
        self.assertIsNone(self.session.recovery_started)
        self.assertIsNone(self.session.motion.pose)
        self.assertEqual(self.session.recovery_hits, 0)

    def test_recovering_output_is_invalid(self):
        status, output = self.step(.1)
        record = pose_record(frame_at(.1), status, output, 10, 'mesh.obj')
        self.assertEqual(record['status'], 'recovering')
        self.assertFalse(record['valid'])
        self.assertIsNone(record['pose_4x4'])


class RecoveryConfigTests(unittest.TestCase):
    def test_config_cli_and_validation(self):
        self.assertFalse(recovery_from_args(parse_args([])).activate)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'target.yaml'
            path.write_text('tracking_recovery: {activate: true, timeout_s: 2, confirm_frames: 3}')
            args = parse_args(['--config', str(path)])
            self.assertEqual(recovery_from_args(args), RecoverySettings(True, 2, 3))
            self.assertFalse(parse_args(['--config', str(path), '--no-tracking-recovery']).recovery_activate)
            for bad in ('null', '{activate: 1}', '{timeout_s: .nan}', '{timeout_s: 0}',
                        '{confirm_frames: 0}', '{confirm_frames: true}', '{unknown: 2}'):
                path.write_text('tracking_recovery: ' + bad)
                with self.subTest(bad=bad), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(['--config', str(path)])
        self.assertTrue(parse_args(['--tracking-recovery']).recovery_activate)
