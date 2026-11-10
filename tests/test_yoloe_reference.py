"""Reference collection/export contracts; no camera, GUI or model downloads."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from examples.camera_yoloe_fuse import extract_sample, main, make_reference_mosaic
from wapr_realsense.yoloe_reference import encode_reference


class ReferenceTests(unittest.TestCase):
    def test_cropped_edge_and_multiple_views_keep_xyxy_coordinates(self):
        frame = np.zeros((90, 120, 3), np.uint8)
        frame[0:60, 0:50] = 255
        samples = [extract_sample(frame, (0, 0, 50, 60)),
                   extract_sample(frame, (70, 55, 120, 90))]
        self.assertEqual(samples[0][1], (0, 0, 50, 60))
        mosaic, boxes = make_reference_mosaic(samples)
        self.assertEqual(mosaic.shape, (320, 640, 3))
        self.assertTrue(np.all(boxes[:, 2:] > boxes[:, :2]))
        self.assertTrue(np.all(boxes >= 0))
        self.assertLessEqual(boxes[:, 2].max(), 640)
        self.assertLessEqual(boxes[:, 3].max(), 320)
        # The first ROI touches the top-left crop boundary, preserved in its tile.
        x, y = np.rint(boxes[0, :2]).astype(int)
        np.testing.assert_array_equal(mosaic[y+1, x+1], [255, 255, 255])

    def test_invalid_boxes_never_run_encoder_or_write_output(self):
        model = Mock()
        frame = np.zeros((10, 20, 3), np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'target.npz'
            for boxes in ([], [[0, 0, 30, 5]], [[1, 1, 0, 5]], [[0, 0, float('nan'), 5]]):
                with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                    encode_reference(model, frame, boxes, output)
            self.assertFalse(output.exists())
            model.predict.assert_not_called()
            model.save_prompt_embeddings.assert_not_called()

    def test_camera_f_fuses_exports_then_predicts_without_prompts(self):
        frame = np.zeros((80, 100, 3), np.uint8)
        model = Mock()
        model.predict.return_value = [SimpleNamespace(plot=Mock(return_value=frame))]
        def save_npz(path):
            np.savez(path, embeddings=np.ones((1, 1, 512), np.float32),
                     names=np.array(['target']), model=np.array('test'))
        model.save_prompt_embeddings.side_effect = save_npz
        cap = Mock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, frame)
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict('sys.modules', ultralytics=SimpleNamespace(YOLOE=Mock(return_value=model))), \
                patch('examples.camera_yoloe_fuse.cv2.VideoCapture', return_value=cap), \
                patch('examples.camera_yoloe_fuse.cv2.namedWindow'), \
                patch('examples.camera_yoloe_fuse.cv2.imshow'), \
                patch('examples.camera_yoloe_fuse.cv2.destroyWindow'), \
                patch('examples.camera_yoloe_fuse.cv2.destroyAllWindows') as close, \
                patch('examples.camera_yoloe_fuse.cv2.getWindowProperty', return_value=1), \
                patch('examples.camera_yoloe_fuse.cv2.selectROI', return_value=(5, 5, 30, 40)), \
                patch('examples.camera_yoloe_fuse.cv2.waitKey', side_effect=[ord('s'), ord('s'), ord('f'), -1, ord('q')]):
            main(['--camera', '8', '--device', 'cpu', '--output', folder])
            path = Path(folder)
            with np.load(path / 'prompt_embeddings.npz', allow_pickle=False) as data:
                self.assertEqual(data['embeddings'].shape, (1, 1, 512))
            self.assertTrue((path / 'reference_mosaic.png').is_file())
            boxes = yaml.safe_load((path / 'boxes.xyxy.yaml').read_text())
            self.assertEqual(len(boxes), 2)
            calls = model.predict.call_args_list
            self.assertEqual(len(calls), 3)  # One reference encoding, then two live frames.
            np.testing.assert_array_equal(calls[0].kwargs['visual_prompts']['cls'], [0, 0])
            np.testing.assert_allclose(calls[0].kwargs['visual_prompts']['bboxes'], boxes)
            for call in calls[1:]:
                self.assertNotIn('refer_image', call.kwargs)
                self.assertNotIn('visual_prompts', call.kwargs)
            cap.release.assert_called_once()
            close.assert_called_once()

    def test_camera_failure_releases_device(self):
        cap = Mock()
        cap.isOpened.return_value = False
        with patch.dict('sys.modules', ultralytics=SimpleNamespace(YOLOE=Mock())), \
                patch('examples.camera_yoloe_fuse.cv2.VideoCapture', return_value=cap), \
                patch('examples.camera_yoloe_fuse.cv2.destroyAllWindows'), \
                self.assertRaisesRegex(RuntimeError, 'Cannot open camera'):
            main(['--device', 'cpu'])
        cap.release.assert_called_once()
