"""Compatibility import; the camera adapter lives in wapr_realsense.camera."""

from wapr_realsense.camera import RGBDFrame, RealSenseCamera, convert_aligned_frame, rectification_maps

__all__ = ["RGBDFrame", "RealSenseCamera", "convert_aligned_frame", "rectification_maps"]
