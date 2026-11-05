"""Export scheduling/profile contracts without building engines or using CUDA."""

import subprocess
import sys
import unittest
from unittest.mock import patch

from wapr.engine_profile import pose_profile, validate_pose_profile
from wapr.export_engines import ENGINE_NAMES, POSE_ENGINES, export_plan, main, _export_one
from wapr_realsense.app import parse_args


class ProfileTests(unittest.TestCase):
    def test_single_and_multi_profiles_and_runtime_compatibility(self):
        self.assertEqual(pose_profile("single"), (2, 12, 12))
        self.assertEqual(pose_profile("multi"), (2, 12, 120))
        for actual in ((2, 12, 12), (2, 12, 120)):
            validate_pose_profile(actual, "single")
        validate_pose_profile((2, 12, 120), "multi")
        for actual, mode in (((2, 12, 12), "multi"), ((2, 12, 24), "single"),
                             ((1, 12, 120), "multi"), ((2, 6, 12), "single")):
            with self.subTest(actual=actual, mode=mode), self.assertRaises(RuntimeError):
                validate_pose_profile(actual, mode)
        with self.assertRaises(ValueError):
            pose_profile("unknown")

    def test_application_defaults_to_single_and_accepts_multi(self):
        self.assertEqual(parse_args([]).object_mode, "single")
        self.assertEqual(parse_args(["--object-mode", "multi"]).object_mode, "multi")


class ExportSchedulingTests(unittest.TestCase):
    def test_export_set_and_independent_selection(self):
        self.assertEqual(export_plan("single"), POSE_ENGINES)
        self.assertEqual(export_plan("multi"), POSE_ENGINES + ("wbps_batch",))
        for name in ENGINE_NAMES:
            self.assertEqual(export_plan("multi", name), (name,))
        with self.assertRaises(ValueError):
            export_plan("single", "wbps_batch")

    def test_each_engine_runs_in_a_separate_waited_for_process(self):
        with patch("subprocess.run") as run:
            main(["--object-mode", "single"])
        self.assertEqual(run.call_count, 4)
        for call, name in zip(run.call_args_list, POSE_ENGINES):
            self.assertEqual(call.args[0], [sys.executable, "-u", "-m", "wapr.export_engines",
                "--object-mode", "single", "--engine", name, "--_worker"])
            self.assertEqual(call.kwargs, {"check": True})

    def test_independent_batch_build_does_not_rebuild_pose_engines(self):
        with patch("subprocess.run") as run:
            main(["--object-mode", "multi", "--engine", "wbps_batch"])
        run.assert_called_once()
        self.assertIn("wbps_batch", run.call_args.args[0])

    def test_failure_stops_before_starting_another_engine(self):
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "worker")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                main(["--object-mode", "multi"])
        run.assert_called_once()

    def test_worker_releases_export_model_memory_before_building(self):
        events = []
        with patch("wapr.export_engines._export_pose_onnx", side_effect=lambda *args:
                   (events.append("onnx") or ("net.onnx", "net.engine", ((2,), (12,), (12,))))), \
             patch("wapr.export_engines.release_export_memory", side_effect=lambda: events.append("release")), \
             patch("wapr.export_engines._build_fp16_engine", side_effect=lambda *args: events.append("build")):
            _export_one("sapr", "single")
        self.assertEqual(events, ["onnx", "release", "build"])


if __name__ == "__main__":
    unittest.main()
