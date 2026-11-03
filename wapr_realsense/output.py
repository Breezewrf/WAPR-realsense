"""Timestamped robotics records, atomic latest-pose files and snapshots."""

import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

def pose_record(frame, status, row, processing_ms, mesh_path,
                camera_frame="color_optical_frame", object_frame="target_object"):
    """JSON uses the original (scaled) mesh frame, not its internal centered copy."""
    pose = None if row is None else np.asarray(row["pose_4x4"], dtype=np.float64)
    return {"status": status, "valid": pose is not None, "object_id": 1,
            "mesh": str(Path(mesh_path).resolve()), "host_time_ns": time.time_ns(),
            "frame_number": frame.number, "camera_timestamp_ms": frame.timestamp_ms,
            "camera_timestamp_domain": frame.timestamp_domain,
            "frame_age_ms": (time.monotonic() - frame.received_monotonic) * 1000,
            "processing_ms": processing_ms, "camera_frame": camera_frame, "object_frame": object_frame,
            "pose_convention": "object_to_camera", "translation_unit": "m",
            "pose_4x4": None if pose is None else pose.tolist(),
            "R": None if pose is None else pose[:3, :3].tolist(),
            "t_m": None if pose is None else pose[:3, 3].tolist(),
            "quaternion_xyzw": None if pose is None else Rotation.from_matrix(pose[:3, :3]).as_quat().tolist(),
            "score_6d": None if row is None else float(row["score_6d"])}


class PoseWriter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = (self.directory / "poses.jsonl").open("a", encoding="utf-8")
        self.write({"status": "starting", "valid": False, "host_time_ns": time.time_ns(),
                    "frame_age_ms": 0.0, "pose_4x4": None, "R": None, "t_m": None,
                    "quaternion_xyzw": None})

    def write(self, record):
        line = json.dumps(record, allow_nan=False, separators=(",", ":"))
        self.stream.write(line + "\n")
        self.stream.flush()
        temporary = self.directory / "latest.json.tmp"
        temporary.write_text(line + "\n", encoding="utf-8")
        os.replace(temporary, self.directory / "latest.json")

    def close(self):
        self.stream.close()


def save_snapshot(directory, frame, preview, record):
    folder = Path(directory) / ("frame_%d_%d" % (frame.number, time.time_ns()))
    folder.mkdir()
    if not cv2.imwrite(str(folder / "rgb.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)):
        raise OSError("Could not save RGB snapshot")
    if not cv2.imwrite(str(folder / "pose.jpg"), preview):
        raise OSError("Could not save pose preview")
    np.save(folder / "depth.npy", frame.depth_m)
    K = frame.K
    (folder / "camera.json").write_text(json.dumps({"fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]), "depth_unit": "m", "rectified": True}, indent=2))
    (folder / "pose.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    print("SNAPSHOT", folder, flush=True)


def read_latest_pose(path, max_age_ms=500.0, now_ns=None):
    """Return a fresh accepted record, or None; suitable for a polling consumer.

    Age includes the producer's receive-to-record delay. This host-clock check is
    intended for a local reader; it does not synchronize clocks between machines.
    """
    if not np.isfinite(max_age_ms) or max_age_ms <= 0:
        raise ValueError("max_age_ms must be finite and positive")
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("valid") is not True:
            return None
        now_ns = time.time_ns() if now_ns is None else now_ns
        since_write = (now_ns - int(record["host_time_ns"])) / 1e6
        source_age = float(record["frame_age_ms"])
        if not np.isfinite(source_age) or source_age < 0 or since_write < 0 or since_write + source_age > max_age_ms:
            return None
        pose = np.asarray(record["pose_4x4"], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            return None
        return record
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return None


def transform_to_base(record, calibration):
    """Apply a caller-supplied calibrated T_base_camera; no calibration is inferred."""
    if calibration.get("calibrated") is not True:
        raise ValueError("Replace the calibration template with a measured transform and set calibrated=true")
    if calibration.get("camera_frame") != record.get("camera_frame"):
        raise ValueError("Calibration camera_frame does not match the pose's camera_frame")
    transform = np.asarray(calibration["T_base_camera"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("T_base_camera must be a finite 4x4 matrix")
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)):
        raise ValueError("T_base_camera must be a rigid transform")
    pose = transform @ np.asarray(record["pose_4x4"], dtype=np.float64)
    return {"valid": True, "parent_frame": calibration["parent_frame"],
            "object_frame": record["object_frame"], "frame_number": record["frame_number"],
            "host_time_ns": record["host_time_ns"], "translation_unit": "m",
            "pose_4x4": pose.tolist(), "t_m": pose[:3, 3].tolist(),
            "quaternion_xyzw": Rotation.from_matrix(pose[:3, :3]).as_quat().tolist()}

