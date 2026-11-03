"""RealSense RGB-D acquisition for WAPR; pyrealsense2 is an optional dependency.

深度对齐到彩色相机、换算为米，并转换成 WAPR 使用的无畸变针孔图像。
Only the newest frame is retained. GPU inference stays on the caller's thread.
"""

from dataclasses import dataclass
import threading
import time

import cv2
import numpy as np


@dataclass
class RGBDFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    number: int
    timestamp_ms: float
    timestamp_domain: str
    received_monotonic: float


def rectification_maps(intrinsics, rs):
    """Map pinhole output pixels into the original color image using the SDK.

    RGB and already-aligned depth use the same map. Nearest-neighbor depth
    resampling preserves invalid zeros and does not blend foreground/background.
    """
    K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                  [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]], dtype=np.float32)
    if not np.isfinite(K).all() or min(intrinsics.fx, intrinsics.fy) <= 0:
        raise ValueError("Invalid RealSense color intrinsics")
    coefficients = np.asarray(intrinsics.coeffs)
    if intrinsics.model == rs.distortion.none or not np.any(coefficients):
        return K, None
    if intrinsics.model == rs.distortion.brown_conrady:
        maps = cv2.initUndistortRectifyMap(
            K, coefficients, None, K, (intrinsics.width, intrinsics.height), cv2.CV_32FC1)
    elif intrinsics.model in (rs.distortion.modified_brown_conrady,
                              rs.distortion.inverse_brown_conrady):
        # Use the SDK's forward model rather than assuming it equals OpenCV Brown.
        print("REALSENSE_RECTIFY building color projection map", flush=True)
        yy, xx = np.indices((intrinsics.height, intrinsics.width), dtype=np.float32)
        rays = np.stack(((xx - intrinsics.ppx) / intrinsics.fx,
                         (yy - intrinsics.ppy) / intrinsics.fy, np.ones_like(xx)), axis=-1)
        pixels = np.asarray([rs.rs2_project_point_to_pixel(intrinsics, ray.tolist())
                             for ray in rays.reshape(-1, 3)], dtype=np.float32)
        pixels = pixels.reshape(intrinsics.height, intrinsics.width, 2)
        maps = (pixels[..., 0].copy(), pixels[..., 1].copy())
    else:
        raise ValueError("Unsupported color distortion model: %s. Use a rectified color profile."
                         % intrinsics.model)
    return K, maps


def convert_aligned_frame(bgr, raw_depth, depth_scale_m, maps=None):
    """Convert SDK aligned buffers to owned RGB uint8 and float32 meter depth."""
    if bgr.shape[:2] != raw_depth.shape or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Aligned depth and color dimensions differ")
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
        raise ValueError("depth_scale_m must be positive")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth_m = raw_depth.astype(np.float32) * np.float32(depth_scale_m)
    if maps is not None:
        rgb = cv2.remap(rgb, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        depth_m = cv2.remap(depth_m, *maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(depth_m)


class RealSenseCamera:
    """A D455-compatible synchronized source with one replaceable latest frame."""

    def __init__(self, width=640, height=480, fps=15, serial=None):
        self.width, self.height, self.fps, self.serial = width, height, fps, serial
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest = None
        self._error = None
        self._thread = None
        self._pipeline = None
        self.info = {}

    def __enter__(self):
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError("Install the camera SDK: python -m pip install pyrealsense2") from error
        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        profile = pipeline.start(config)
        self._pipeline = pipeline
        try:
            device = profile.get_device()
            scale = float(device.first_depth_sensor().get_depth_scale())
            intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            K, maps = rectification_maps(intrinsics, rs)
            self.info = {"name": device.get_info(rs.camera_info.name),
                         "serial": device.get_info(rs.camera_info.serial_number),
                         "depth_scale_m": scale, "width": self.width, "height": self.height,
                         "fps": self.fps, "alignment": "depth_to_color",
                         "original_distortion": str(intrinsics.model),
                         "original_distortion_coeffs": list(intrinsics.coeffs),
                         "rectified": True, "K": K.tolist(),
                         "pose_frame": "color_optical_frame", "depth_unit": "m"}
            align = rs.align(rs.stream.color)

            def acquire():
                try:
                    while not self._stop.is_set():
                        frames = pipeline.wait_for_frames(5000)
                        received = time.monotonic()
                        frames = align.process(frames)
                        color, depth = frames.get_color_frame(), frames.get_depth_frame()
                        if not color or not depth:
                            continue
                        rgb, depth_m = convert_aligned_frame(
                            np.asanyarray(color.get_data()), np.asanyarray(depth.get_data()), scale, maps)
                        frame = RGBDFrame(rgb, depth_m, K, int(color.get_frame_number()),
                                          float(color.get_timestamp()), str(color.get_frame_timestamp_domain()),
                                          received)
                        with self._condition:
                            self._latest = frame
                            self._condition.notify_all()
                except Exception as error:
                    if not self._stop.is_set():
                        with self._condition:
                            self._error = error
                            self._condition.notify_all()

            self._thread = threading.Thread(target=acquire, name="wapr-realsense", daemon=True)
            self._thread.start()
            return self
        except BaseException:
            self.close()
            raise

    def read(self, after_number=None, timeout=10.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError("RealSense acquisition failed: %s" % self._error) from self._error
                if self._stop.is_set():
                    raise RuntimeError("RealSense camera is closed")
                if self._latest is not None and self._latest.number != after_number:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No new RealSense RGB-D frame; check USB 3, device access and camera ownership")
                self._condition.wait(remaining)

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None
                if self._thread is not None:
                    self._thread.join(timeout=6)

    def __exit__(self, *_args):
        self.close()
