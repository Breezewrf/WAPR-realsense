"""Metric mesh loading, ROI validation and inexpensive pose overlays."""

from pathlib import Path

import cv2
import numpy as np
import trimesh

def load_mesh(path, unit="m", scale=1.0):
    """Retain the file's original object frame and texture; only scale units."""
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Expected one nonempty triangle mesh: %s" % path)
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(mesh.vertices).all():
        raise ValueError("Mesh coordinates and scale must be finite; scale must be positive")
    mesh.apply_scale(scale * {"m": 1.0, "mm": 0.001, "cm": 0.01}[unit])
    diameter = float(np.linalg.norm(mesh.extents))
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("Mesh has zero or invalid extent")
    mesh.metadata["name"] = Path(path).stem
    return mesh, diameter


def valid_pose(row, minimum_score):
    if row is None:
        return False
    pose = np.asarray(row["pose_4x4"])
    score = float(row["score_6d"])
    return (pose.shape == (4, 4) and np.isfinite(pose).all() and pose[2, 3] > 0
            and np.allclose(pose[3], [0, 0, 0, 1]) and np.isfinite(score)
            and score >= minimum_score)


def check_bbox(bbox, depth_m):
    x, y, w, h = [int(v) for v in bbox]
    height, width = depth_m.shape
    if x < 0 or y < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        raise ValueError("bbox must be x y width height inside the color image")
    region = depth_m[y:y+h, x:x+w]
    if not np.any(np.isfinite(region) & (region > 0)):
        raise ValueError("Selected region has no valid depth; move the object or camera")
    return [x, y, w, h]


def draw_pose(frame, mesh, row, status, processing_ms, show_text=True):
    """Draw only the box and axes; avoid a CPU loop over 166k triangles per frame."""
    bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    if row is not None:
        pose = np.asarray(row["pose_4x4"])
        low, high = mesh.bounds
        corners = np.array([[high[j] if i & (1 << j) else low[j] for j in range(3)]
                            for i in range(8)])
        camera = corners @ pose[:3, :3].T + pose[:3, 3]
        if np.all(camera[:, 2] > 1e-4):
            projected = camera @ frame.K.T
            pixels = projected[:, :2] / projected[:, 2:3]
            # Reject extreme projections before conversion to OpenCV integer pixels.
            if np.isfinite(pixels).all() and np.abs(pixels).max() < 1e6:
                pixels = np.rint(pixels).astype(np.int32)
                for i in range(8):
                    for bit in range(3):
                        j = i ^ (1 << bit)
                        if j > i:
                            cv2.line(bgr, tuple(pixels[i]), tuple(pixels[j]), (40, 40, 230), 2)
        axis_length = float(np.linalg.norm(mesh.extents)) * 0.2
        axes = np.vstack((np.zeros((1, 3)), np.eye(3) * axis_length))
        points = axes @ pose[:3, :3].T + pose[:3, 3]
        if np.all(points[:, 2] > 1e-4):
            uv = points @ frame.K.T
            uv = uv[:, :2] / uv[:, 2:3]
            if np.isfinite(uv).all() and np.abs(uv).max() < 1e6:
                uv = np.rint(uv).astype(np.int32)
                for endpoint, color in zip(uv[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                    cv2.line(bgr, tuple(uv[0]), tuple(endpoint), color, 2)
        if show_text:
            text = "t [m]: " + " ".join("%.3f" % x for x in pose[:3, 3])
            cv2.putText(bgr, text, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 240), 1)
    if show_text:
        cv2.putText(bgr, "%s | %.1f ms | r: reset, s: snapshot, q: quit" % (status, processing_ms),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 240), 1)
    return bgr

