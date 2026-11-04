"""SAM may return fewer masks than prompts; keep their original identities."""

from contextlib import nullcontext, redirect_stdout
import io
import json
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from wapr import det2d
from wapr.det2d import align_sam_proposals, bbox_regions, GroundingSAM, WAPRDet2D


class SAMProposalTests(unittest.TestCase):
    def setUp(self):
        self.boxes = torch.tensor([[0., 0., 3., 3.], [1., 1., 4., 4.], [2., 2., 5., 5.]])
        self.scores = torch.tensor([.9, .7, .8])

    def test_filtered_and_reordered_masks_keep_box_and_objectness(self):
        masks = torch.ones(2, 8, 8)
        result = SimpleNamespace(masks=SimpleNamespace(data=masks), boxes=SimpleNamespace(cls=torch.tensor([2., 0.])))
        boxes, actual_masks, scores = align_sam_proposals(result, self.boxes, self.scores, 8, 8)
        torch.testing.assert_close(boxes, self.boxes[[2, 0]])
        torch.testing.assert_close(scores, self.scores[[2, 0]])
        self.assertEqual(actual_masks.shape, (2, 8, 8))
        self.assertEqual(actual_masks.dtype, torch.bool)

    def test_no_masks_is_an_empty_detection(self):
        for masks in [None, SimpleNamespace(data=torch.empty(0, 8, 8))]:
            result = SimpleNamespace(masks=masks)
            boxes, masks, scores = align_sam_proposals(result, self.boxes, self.scores, 8, 8)
            self.assertEqual(boxes.shape, (0, 4))
            self.assertEqual(masks.shape, (0, 8, 8))
            self.assertEqual(scores.shape, (0,))

    def test_malformed_masks_and_prompt_ids_are_rejected(self):
        for ids in [[0., 0.], [0., 3.], [0., -1.], [.5, 1.], [0.], [0., float('nan')]]:
            result = SimpleNamespace(masks=SimpleNamespace(data=torch.ones(2, 8, 8)),
                                     boxes=SimpleNamespace(cls=torch.tensor(ids)))
            with self.subTest(ids=ids), self.assertRaisesRegex(RuntimeError, 'prompt indices'):
                align_sam_proposals(result, self.boxes, self.scores, 8, 8)
        result.masks.data = torch.ones(2, 4, 4)
        with self.assertRaisesRegex(RuntimeError, 'spatial'):
            align_sam_proposals(result, self.boxes, self.scores, 8, 8)


class BBoxDetectionTests(unittest.TestCase):
    def test_grounding_bbox_route_skips_sam_and_preserves_candidates(self):
        proposal = object.__new__(GroundingSAM)
        proposal.device, proposal.use_sam = torch.device('cpu'), False
        proposal.segmentor = Mock(side_effect=AssertionError('SAM must not run'))
        proposal.transform = Mock(return_value=(torch.zeros(3, 8, 8), None))
        proposal.model = Mock(return_value={'pred_logits': torch.tensor([[[10.]]]),
                                          'pred_boxes': torch.ones(1, 1, 4)})
        ops = ModuleType('torchvision.ops')
        ops.box_convert = Mock(return_value=torch.tensor([[.25, .25, .75, .75]]))
        ops.nms = Mock(return_value=torch.tensor([0]))
        package = ModuleType('torchvision')
        package.ops = ops
        with patch.dict(sys.modules, {'torchvision': package, 'torchvision.ops': ops,
                                      'ultralytics': None, 'ultralytics.cfg': None}), \
             patch('torch.cuda.synchronize'), redirect_stdout(io.StringIO()):
            boxes, regions, scores = proposal.predict(np.zeros((8, 8, 3), np.uint8))
        torch.testing.assert_close(boxes, torch.tensor([[2., 2., 6., 6.]]))
        self.assertEqual(regions.sum().item(), 16)
        self.assertEqual(regions.shape, (1, 8, 8))
        proposal.segmentor.assert_not_called()
        self.assertEqual(len(scores), 1)
        self.assertEqual(bbox_regions(boxes[:0], 8, 8).shape, (0, 8, 8))

    def test_bbox_detection_returns_no_segmentation_mask(self):
        detector = object.__new__(WAPRDet2D)
        detector.device, detector.use_sam = torch.device('cpu'), False
        boxes = torch.tensor([[1., 2., 5., 6.]])
        regions = bbox_regions(boxes, 8, 8)
        detector.proposals = Mock()
        detector.proposals.predict.return_value = (boxes, regions, torch.tensor([.9]))
        detector.encoder, detector.coordinates = Mock(), None
        detector.encoder.encode.return_value = {'cls': torch.zeros(1, 4)}
        detector.matcher = Mock(bank={'obj_ids': torch.tensor([1])})
        detector.matcher.score.return_value = torch.tensor([[.8]])
        with patch('torch.cuda.device', return_value=nullcontext()), \
             patch('torch.cuda.synchronize'), patch.dict(sys.modules, {'pycocotools': None}), \
             patch.object(det2d, 'square_crops', return_value=(torch.zeros(1, 3, 8, 8), regions)), \
             patch.object(det2d, 'lift_black_queries', side_effect=lambda crops, masks: crops), \
             patch.object(det2d, 'mask_nms_same_category', return_value=([0], torch.tensor([.8]), torch.tensor([1]))), \
             redirect_stdout(io.StringIO()):
            instances, _ = detector.detect_many_categories_many_instances(np.zeros((8, 8, 3), np.uint8))
        self.assertEqual(instances[0]['bbox'], [1., 2., 4., 4.])
        self.assertNotIn('mask', instances[0])
        self.assertNotIn('mask_bbox', instances[0])

    def test_bbox_weights_do_not_download_or_verify_sam(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(det2d, 'ensure_optional'), \
             patch.object(det2d, '_place_file') as place, patch.object(det2d, '_place_config'), \
             patch.object(det2d, 'verify_assets') as verify:
            det2d.prepare_det2d_weights(folder, use_sam=False)
            self.assertNotIn('sam2.1_l.pt', [call.args[1] for call in place.call_args_list])
            verify.assert_called_once_with(folder, use_sam=False)

        with open(det2d.det2d_manifest_path) as stream:
            artifacts = json.load(stream)['artifacts']
        paths = []

        def digest(path):
            paths.append(path)
            if path == det2d.det2d_manifest_path:
                return 'manifest'
            name = str(path).removeprefix('/weights/')
            return artifacts[name]['sha256']

        with patch.object(det2d, 'sha256', side_effect=digest):
            self.assertEqual(det2d.verify_assets('/weights', use_sam=False), 'manifest')
        self.assertFalse(any(path.endswith('sam2.1_l.pt') for path in paths))


if __name__ == '__main__':
    unittest.main()
