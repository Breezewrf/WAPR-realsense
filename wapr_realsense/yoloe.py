"""Optional reference-guided YOLOE detection, consumed as bbox-only pose input."""
from pathlib import Path

import cv2
import numpy as np

from wapr.timing import ComponentTimer, print_timing
from .geometry import check_bbox, valid_pose


def build_yoloe(args):
    if not getattr(args, 'yoloe_activate', False):
        return None
    detector = getattr(args, '_yoloe_detector', None)
    if detector is None:
        detector = args._yoloe_detector = YOLOEDetector(args)
    return detector


class YOLOEDetector:
    def __init__(self, args):
        embeddings = Path(args.yoloe_prompt_embeddings)
        if not embeddings.is_file():
            raise FileNotFoundError('YOLOE prompt embeddings not found: %s' % embeddings)
        from ultralytics import YOLOE
        self.model = YOLOE(args.yoloe_model)
        self.device = args.yoloe_device
        self.imgsz, self.conf = args.yoloe_imgsz, args.yoloe_conf
        self.max_candidates = args.yoloe_max_candidates
        self.model.load_prompt_embeddings(embeddings)
        if len(self.model.names) != 1:
            raise ValueError('YOLOE references must describe exactly one target class')

    def detect(self, frame):
        # Ultralytics expects BGR NumPy images; RGB-D coordinates stay rectified.
        with ComponentTimer('YOLOE', ('cuda:' + self.device if self.device.isdecimal() else self.device)
                            if self.device.startswith('cuda') or self.device.isdecimal() else None):
            result = self.model.predict(source=cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR),
                                        device=self.device, imgsz=self.imgsz, conf=self.conf,
                                        retina_masks=True, verbose=False)[0]
            return self.boxes_from_result(result, frame)

    def warmup(self, frame):
        for _ in range(3):
            self.detect(frame)

    def boxes_from_result(self, result, frame):
        if result.masks is None or result.boxes is None:
            return []
        masks = result.masks.data.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        if len(masks) != len(scores) or masks.shape[1:] != frame.depth_m.shape:
            raise RuntimeError('YOLOE retina masks must match the original RGB-D frame and boxes')
        boxes = []
        for mask, score in zip(masks, scores):
            if not np.isfinite(score) or score < self.conf:
                continue
            foreground = np.asarray(mask > .5, np.uint8)
            if not np.any(foreground):
                continue
            bbox = list(cv2.boundingRect(foreground))
            try:
                bbox = check_bbox(bbox, frame.depth_m)
            except ValueError:
                continue
            boxes.append({'bbox_xywh': bbox, 'score_2d': float(score)})
        return sorted(boxes, key=lambda row: row['score_2d'], reverse=True)[:self.max_candidates]

    def recovery_pose(self, frame, prior, mesh):
        """Recenter using the highest-confidence depth-valid detection, without ID association."""
        boxes = self.detect(frame)
        if not boxes:
            return None
        x, y, w, h = boxes[0]['bbox_xywh']
        samples = frame.depth_m[y:y+h, x:x+w]
        samples = samples[np.isfinite(samples) & (samples > 0)]
        z = float(np.median(samples))
        u, v = x + (w - 1) / 2, y + (h - 1) / 2
        front = np.array([(u-frame.K[0, 2])*z/frame.K[0, 0],
                          (v-frame.K[1, 2])*z/frame.K[1, 1], z])
        # Prepared meshes are centered, whereas poses use the original mesh origin.
        from wapr.estimator import center_from_mesh
        vertices = np.asarray(mesh.vertices)[::max(1, len(mesh.vertices)//2048)] + center_from_mesh(mesh)
        relative = vertices @ prior[:3, :3].T
        offset = relative[relative[:, 2] <= np.percentile(relative[:, 2], 30)].mean(axis=0)
        pose = np.array(prior, copy=True)
        pose[:3, 3] = front - offset
        if not np.isfinite(pose).all() or pose[2, 3] <= 0:
            return None
        return pose


class YOLOEInitializer:
    def __init__(self, estimator, detector, mesh, diameter_m, minimum_score):
        self.estimator, self.detector, self.mesh = estimator, detector, mesh
        self.diameter, self.minimum_score = diameter_m, minimum_score
        self.attempts = 0

    def warmup(self, frame):
        from .initialization import warmup_pose_inputs
        self.detector.warmup(frame)
        warmup_pose_inputs(self.estimator, self.mesh, self.diameter, frame, use_mask=False)

    def __call__(self, frame):
        part = 'INIT' if self.attempts == 0 else 'REINIT'
        self.attempts += 1
        print_timing(part, 'YOLOE bbox initialization | frame=%s | started' % frame.number)
        with ComponentTimer(None, self.estimator.device) as timer:
            boxes = self.detector.detect(frame)
            accepted = []
            for box in boxes:
                row = self.estimator.estimate_one_category_one_instance(
                    frame.rgb, frame.depth_m, frame.K, self.mesh, self.diameter,
                    bbox_xywh=box['bbox_xywh'])
                if valid_pose(row, self.minimum_score):
                    accepted.append(dict(row, **box))
        selected = max(accepted, key=lambda row: row['score_6d'], default=None)
        print_timing(part, 'total=%.2f ms | candidates=%d | accepted=%d | result=%s' %
                     (timer.elapsed_ms, len(boxes), len(accepted), 'found' if selected else 'searching'))
        return selected
