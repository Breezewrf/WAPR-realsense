"""Regression coverage for NVIDIA build-version normalization and display checks."""

import os
import unittest
from unittest.mock import patch

from wapr.bootstrap import _check_torch_version
from wapr_realsense import app


class RuntimePreparationTests(unittest.TestCase):
    def test_normalized_nvidia_version_does_not_signal_replacement(self):
        with patch("importlib.metadata.version", return_value="2.1.0a0+41361538.nv23.6"):
            _check_torch_version("2.1.0a0+41361538.nv23.06")

    def test_real_replacement_including_a_different_cuda_build_is_rejected(self):
        for loaded, installed in [("2.1.0a0+41361538.nv23.06", "2.1.0a0+41361538.nv23.7"),
                                  ("2.7.1+cu126", "2.7.1+cu128"),
                                  ("2.7.1+cu126", "2.8.0+cu126")]:
            with self.subTest(installed=installed), patch("importlib.metadata.version", return_value=installed):
                with self.assertRaisesRegex(RuntimeError, "Restart Python"):
                    _check_torch_version(loaded)

    def test_missing_display_fails_before_loading_models_or_camera(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(app.sys, "platform", "linux"), \
             patch.object(app, "load_mesh") as load_mesh, patch.object(app, "RealSenseCamera") as camera:
            with self.assertRaisesRegex(RuntimeError, "No display connection"):
                app.main(["--init", "roi"])
            load_mesh.assert_not_called()
            camera.assert_not_called()


if __name__ == "__main__":
    unittest.main()
