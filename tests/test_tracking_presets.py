"""Candidate reuse/preset contracts on CPU; no engine or camera required."""

from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import tempfile
from pathlib import Path

from wapr import recipe
from wapr.tracking import _build_tracking_candidates, _tracking_score_engine
from wapr.tracking_config import tracking_settings
from wapr_realsense.tracker import PoseSession
from test_realsense_pose import frame, result


class CandidateTests(unittest.TestCase):
    def refine(self, calls, name, poses, owners, iters):
        calls.append((name, len(poses), list(owners), iters))
        out = poses.clone()
        # Nonlinear, row-independent updates distinguish all candidate branches.
        shift = (0.03 if name == "sapr" else 0.1) * iters
        out[:, 0, 3] += shift * (1 + poses[:, 0, 3])
        out[:, 1, 3] += shift * (2 + torch.tensor(owners))
        return out

    def test_full_reuses_exact_branches_and_restores_all_six_scoring_rows(self):
        previous = torch.eye(4).repeat(3, 1, 1)
        previous[:, 2, 3] = 1
        center = previous.clone()
        center[1, 0, 3] = 0.2
        center[2, 0, 3] = 1e-7  # close is not identical: keep this branch
        same = [True, False, False]
        settings = tracking_settings("full")
        original_calls, reused_calls = [], []
        original, _, _ = _build_tracking_candidates(previous, center, same, settings, False,
                                                    partial(self.refine, original_calls))
        reused, rows, restore = _build_tracking_candidates(previous, center, same, settings, True,
                                                         partial(self.refine, reused_calls))
        torch.testing.assert_close(original, reused, rtol=0, atol=0)
        self.assertEqual([entry[1] for entry in original_calls], [6, 9])
        self.assertEqual([entry[1] for entry in reused_calls], [5, 8])
        self.assertEqual(len(rows), 16)
        restored = reused.reshape(-1, 4, 4)[rows][restore].reshape(3, 6, 4, 4)
        torch.testing.assert_close(restored, original, rtol=0, atol=0)
        # Multiplicity-sensitive scoring must see six rows, not unique poses.
        torch.testing.assert_close(torch.softmax(restored[:, :, 0, 3], dim=1),
                                   torch.softmax(original[:, :, 0, 3], dim=1), rtol=0, atol=0)

    def test_no_reuse_when_center_and_previous_differ(self):
        previous = torch.eye(4)[None]
        center = previous.clone()
        center[:, 0, 3] = 0.1
        calls = []
        _, rows, restore = _build_tracking_candidates(previous, center, [False], tracking_settings("full"),
                                                      True, partial(self.refine, calls))
        self.assertEqual([entry[1] for entry in calls], [2, 3])
        self.assertEqual(rows, list(range(6)))
        self.assertEqual(restore, list(range(6)))

    def test_presets_reduce_only_tracking_iterations_and_hypotheses(self):
        initial_iterations = recipe.wapr_iters, recipe.sapr_iters
        previous = torch.eye(4).repeat(2, 1, 1)
        center = previous.clone()
        center[:, 0, 3] = 0.2
        for preset, iterations, group in (("balanced", [2, 1], 3), ("fast", [1, 1], 3),
                                           ("small-motion", [1], 2)):
            with self.subTest(preset=preset):
                calls = []
                settings = tracking_settings(preset)
                candidates, rows, restore = _build_tracking_candidates(previous, center, [False, False],
                    settings, True, partial(self.refine, calls))
                self.assertEqual(candidates.shape, (2, group, 4, 4))
                self.assertEqual([entry[3] for entry in calls], iterations)
                self.assertTrue(all(entry[1] == 2 for entry in calls))
                self.assertEqual(rows, restore)
                torch.testing.assert_close(candidates[:, 0], previous)
                self.assertEqual(len(settings.hypotheses), group)
                if preset == "small-motion":
                    self.assertEqual(calls[0][0], "sapr")
        self.assertEqual((recipe.wapr_iters, recipe.sapr_iters), initial_iterations)
        with self.assertRaises(ValueError):
            tracking_settings("unknown")

    def test_reduced_group_does_not_use_fixed_six_candidate_engine_for_multiple_objects(self):
        engine = SimpleNamespace(max_batch=120)
        estimator = SimpleNamespace(nets={name: SimpleNamespace(engine=engine)
                                         for name in ("wapr_wo_mask", "sapr", "wbps")})
        with patch("wapr.tracking.Path.is_file", return_value=False):
            self.assertIsNone(_tracking_score_engine(estimator, 1, group=3, wapr_rows=1, sapr_rows=1)[0])
            with self.assertRaisesRegex(FileNotFoundError, "wbps_batch"):
                _tracking_score_engine(estimator, 2, group=3, wapr_rows=1, sapr_rows=1)


class SessionTests(unittest.TestCase):
    def test_yaml_preset_and_reuse_can_be_overridden_from_cli(self):
        from wapr_realsense.app import parse_args
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.yaml"
            path.write_text("tracking_preset: balanced\nreuse_tracking: false\n")
            configured = parse_args(["--config", str(path)])
            self.assertEqual(configured.tracking_preset, "balanced")
            self.assertFalse(configured.reuse_tracking)
            override = parse_args(["--config", str(path), "--tracking-preset", "fast", "--reuse-tracking"])
            self.assertEqual(override.tracking_preset, "fast")
            self.assertTrue(override.reuse_tracking)

    def test_runtime_forwards_preset_and_reuse_without_changing_initialization(self):
        estimator = Mock()
        estimator.estimate_one_category_one_instance.return_value = result()
        with patch("wapr.tracking.track_many_categories_many_instances", return_value=[result()]) as track:
            session = PoseSession(estimator, object(), 0.5, tracking_preset="fast", reuse_tracking=False)
            self.assertEqual(session.step(frame(), bbox=[2, 2, 8, 7])[0], "initialized_roi")
            estimator.estimate_one_category_one_instance.assert_called_once()
            self.assertEqual(session.step(frame())[0], "tracking")
        self.assertEqual(track.call_args.kwargs, {"tracking_preset": "fast", "reuse_tracking": False})
        self.assertNotIn("tracking_preset", estimator.estimate_one_category_one_instance.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
