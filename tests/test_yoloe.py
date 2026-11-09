"""YOLOE bbox conversion, routing, and recovery contracts without model downloads."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from wapr_realsense.app import parse_args
from wapr_realsense.initialization import build_initializer
from wapr_realsense.recovery import RecoverySettings
from wapr_realsense.tracker import PoseSession
from wapr_realsense.yoloe import YOLOEDetector, YOLOEInitializer, build_yoloe
from test_kalman import frame_at, pose_at


def pose(score=.9):
    return {'pose_4x4': pose_at(0), 'score_6d': score, 'hypothesis': 'sapr_center'}


class YOLOETests(unittest.TestCase):
    def test_masks_define_bbox_and_empty_or_depthless_regions_are_removed(self):
        detector = object.__new__(YOLOEDetector)
        detector.conf, detector.max_candidates = .05, 3
        masks = torch.zeros(4, 8, 8)
        masks[0, 2:5, 1:6] = 1
        masks[1, 0:2, 0:2] = 1
        masks[2, 3:7, 3:7] = 1
        result = SimpleNamespace(masks=SimpleNamespace(data=masks),
                                 boxes=SimpleNamespace(conf=torch.tensor([.8, .9, .01, .95])))
        f = frame_at(0)
        f.depth_m[:2, :2] = 0
        boxes = detector.boxes_from_result(result, f)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]['bbox_xywh'], [1, 2, 5, 3])
        result.masks = None
        self.assertEqual(detector.boxes_from_result(result, f), [])

    def test_mask_coordinates_must_match_rectified_frame(self):
        detector = object.__new__(YOLOEDetector)
        result = SimpleNamespace(masks=SimpleNamespace(data=torch.ones(1, 4, 8)),
                                 boxes=SimpleNamespace(conf=torch.ones(1)))
        with self.assertRaisesRegex(RuntimeError, 'retina masks'):
            detector.boxes_from_result(result, frame_at(0))

    def test_initialization_uses_bbox_only_and_selects_pose_score(self):
        estimator = Mock(device=None)
        estimator.estimate_one_category_one_instance.side_effect = [pose(.6), pose(.9)]
        detector = Mock()
        detector.detect.return_value = [{'bbox_xywh': [1, 1, 3, 3], 'score_2d': .8},
                                       {'bbox_xywh': [2, 2, 4, 4], 'score_2d': .7}]
        init = YOLOEInitializer(estimator, detector, object(), .5, .5)
        selected = init(frame_at(0))
        self.assertEqual(selected['bbox_xywh'], [2, 2, 4, 4])
        for call in estimator.estimate_one_category_one_instance.call_args_list:
            self.assertEqual(set(call.kwargs), {'bbox_xywh'})
        detector.detect.return_value = []
        self.assertIsNone(init(frame_at(.1)))

    def test_recovery_recenters_without_identity_association(self):
        detector = object.__new__(YOLOEDetector)
        detector.detect = Mock(return_value=[{'bbox_xywh': [2, 2, 4, 4], 'score_2d': .8}])
        mesh = SimpleNamespace(vertices=np.array([[-.1, -.1, -.1], [.1, .1, .1]]), metadata={})
        prior = pose_at(0)
        shifted = detector.recovery_pose(frame_at(0), prior, mesh)
        np.testing.assert_array_equal(shifted[:3, :3], prior[:3, :3])
        np.testing.assert_array_equal(prior, pose_at(0))
        self.assertGreater(shifted[2, 3], 0)
        detector.detect.return_value = []
        self.assertIsNone(detector.recovery_pose(frame_at(0), prior, mesh))

    def test_detector_runs_only_after_tracking_enters_recovery(self):
        detector = Mock()
        center = pose_at(.2)
        detector.recovery_pose.return_value = center
        estimator, track = Mock(), Mock(return_value=[])
        estimator.estimate_one_category_one_instance.return_value = pose()
        session = PoseSession(estimator, object(), .5, track_function=track,
                              recovery_settings=RecoverySettings(True), recovery_detector=detector)
        session.step(frame_at(0), now=0, bbox=[0, 0, 4, 4])
        session.step(frame_at(.1), now=.1)
        detector.recovery_pose.assert_not_called()
        track.return_value = [pose()]
        self.assertEqual(session.step(frame_at(.2), now=.2)[0], 'recovering')
        np.testing.assert_array_equal(track.call_args.args[4][0]['center_pose_4x4'], center)
        detector.recovery_pose.return_value = None
        self.assertEqual(session.step(frame_at(.3), now=.3)[0], 'tracking')
        self.assertNotIn('center_pose_4x4', track.call_args.args[4][0])
        session.step(frame_at(.4), now=.4)
        self.assertEqual(detector.recovery_pose.call_count, 2)

    def test_backend_is_lazy_cached_and_bypasses_three_model_pipeline(self):
        args = SimpleNamespace(yoloe_activate=False, init='roi')
        self.assertIsNone(build_yoloe(args))
        args.yoloe_activate, args.init, args.min_score = True, 'auto', .5
        with patch('wapr_realsense.yoloe.YOLOEDetector') as factory:
            init = build_initializer(args, Mock(), object(), object(), .5)
            self.assertIsInstance(init, YOLOEInitializer)
            self.assertIs(build_yoloe(args), init.detector)
            factory.assert_called_once()

    def test_nested_config_paths_and_cli_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'target.yaml'
            path.write_text('yoloe_detection:\n  activate: true\n  model: ../models/target.pt\n'
                            '  prompt_embeddings: ../refs/box.npz\n  conf: 0.05\n')
            args = parse_args(['--config', str(path), '--yoloe-conf', '.1'])
            self.assertEqual(args.yoloe_model, str((path.parent / '../models/target.pt').resolve()))
            self.assertEqual(args.yoloe_prompt_embeddings, (path.parent / '../refs/box.npz').resolve())
            self.assertEqual(args.yoloe_conf, .1)
            self.assertFalse(parse_args(['--config', str(path), '--no-yoloe']).yoloe_activate)
            for content in ('{activate: true}', '{activate: true, prompt_embeddings: x, reference_image: y}',
                            '{reference_bbox: [0, 0, 10, 10]}', '{activate: true, reference_image: x}',
                            '{conf: .nan}', '{activate: "true"}', '{unknown: true}', '{max_candidates: 0}'):
                path.write_text('yoloe_detection: ' + content)
                with self.subTest(content=content), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(['--config', str(path)])

    def test_runtime_loads_npz_without_reference_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'target.npz'
            path.touch()
            args = SimpleNamespace(yoloe_prompt_embeddings=path, yoloe_model='target.pt',
                                   yoloe_device='cpu', yoloe_imgsz=640, yoloe_conf=.05, yoloe_max_candidates=3)
            model = Mock(names={0: 'target'})
            model.predict.return_value = [SimpleNamespace(masks=None, boxes=None)]
            factory = Mock(return_value=model)
            with patch.dict('sys.modules', ultralytics=SimpleNamespace(YOLOE=factory)):
                detector = YOLOEDetector(args)
                model.load_prompt_embeddings.assert_called_once_with(path)
                model.predict.assert_not_called()
                self.assertEqual(detector.detect(frame_at(0)), [])
                self.assertNotIn('refer_image', model.predict.call_args.kwargs)
                self.assertNotIn('visual_prompts', model.predict.call_args.kwargs)
                model.load_prompt_embeddings.assert_called_once()
                path.unlink()
                with self.assertRaises(FileNotFoundError):
                    YOLOEDetector(args)
                factory.assert_called_once()

    def test_removed_runtime_image_options_are_rejected(self):
        for option in ('--yoloe-reference-image', '--yoloe-reference-bbox'):
            with self.subTest(option=option), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args([option, 'unused'])
