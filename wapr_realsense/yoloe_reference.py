"""Offline single-target reference encoding shared by the camera and image tools."""
from pathlib import Path

import numpy as np


def encode_reference(model, image, boxes, output, device='cpu', imgsz=960, conf=.05):
    """Encode one target class offline and export the model-bound runtime NPZ."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) == 0 or not np.isfinite(boxes).all():
        raise ValueError('Reference boxes must be a nonempty list of finite xyxy coordinates')
    if (np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2:] <= boxes[:, :2]) or
            np.any(boxes[:, 2] > image.shape[1]) or np.any(boxes[:, 3] > image.shape[0])):
        raise ValueError('Reference boxes must lie inside the image')
    output = Path(output)
    if output.suffix.lower() != '.npz':
        raise ValueError('Prompt embeddings output must use the .npz suffix')
    model.predict(source=image, refer_image=image,
                  visual_prompts={'bboxes': boxes, 'cls': np.zeros(len(boxes), np.int32)},
                  imgsz=imgsz, device=device, conf=conf, verbose=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_prompt_embeddings(output)
    return output


