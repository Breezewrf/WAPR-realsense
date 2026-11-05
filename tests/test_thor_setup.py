"""Thor installation checks without requiring Thor hardware."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_thor", ROOT / "scripts/check_thor.py")
thor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(thor)


class ThorSetupTests(unittest.TestCase):
    def test_native_build_targets_thor_with_cuda13(self):
        from wapr.bootstrap import native_build_options
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda _: (11, 0)),
                                version=SimpleNamespace(cuda="13.0"))
        with patch.dict("sys.modules", {"torch": torch}), \
             patch("wapr.bootstrap.ensure_cuda_build_stack", return_value="/usr/local/cuda/bin/nvcc"), \
             patch("wapr.bootstrap.cuda_device_index", return_value=0), \
             patch("wapr.bootstrap.subprocess.check_output", side_effect=["release 13.0", "sm_87 sm_110"]), \
             patch("importlib.metadata.version", return_value="3.0.4"):
            self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=110", native_build_options())

    def test_platform_and_version_guards(self):
        thor.validate_platform("Linux", "aarch64", (3, 12, 3))
        thor.validate_versions("13.0", (11, 0), "10.13.2.6")
        for args in [("Linux", "x86_64", (3, 12)), ("Linux", "aarch64", (3, 10))]:
            with self.assertRaises(RuntimeError):
                thor.validate_platform(*args)
        for args in [("12.6", (11, 0), "10.13"), ("13.0", (8, 7), "10.13"),
                     ("13.0", (11, 0), "11.0")]:
            with self.assertRaises(RuntimeError):
                thor.validate_versions(*args)

    def runtime(self, providers):
        session = Mock()
        session.get_providers.return_value = providers
        session.run.return_value = [np.full((2, 2), 2, np.float32)]
        return SimpleNamespace(
            __version__="1.24.0", get_available_providers=lambda: ["CUDAExecutionProvider"],
            SessionOptions=Mock(), InferenceSession=Mock(return_value=session))

    def test_cuda_session_disables_fallback_and_checks_model(self):
        runtime = self.runtime(["CUDAExecutionProvider"])
        with patch.dict("sys.modules", {"onnxruntime": runtime}):
            self.assertEqual(thor.check_onnxruntime_cuda()["cuda_matmul"], "passed")
        runtime.SessionOptions.return_value.add_session_config_entry.assert_called_once_with(
            "session.disable_cpu_ep_fallback", "1")
        runtime.InferenceSession.return_value.disable_fallback.assert_called_once()
        model = onnx.load_model_from_string(runtime.InferenceSession.call_args.args[0])
        onnx.checker.check_model(model)
        self.assertEqual(model.graph.node[0].op_type, "MatMul")
        self.assertEqual(runtime.InferenceSession.call_args.kwargs["providers"],
                         ["CUDAExecutionProvider"])

    def test_cpu_fallback_is_rejected(self):
        runtime = self.runtime(["CPUExecutionProvider"])
        with patch.dict("sys.modules", {"onnxruntime": runtime}):
            with self.assertRaisesRegex(RuntimeError, "failed to initialize CUDA"):
                thor.check_onnxruntime_cuda()
        runtime.InferenceSession.return_value.run.assert_not_called()

    def test_gpu_provider_must_be_available(self):
        runtime = self.runtime([])
        runtime.get_available_providers = lambda: ["CPUExecutionProvider"]
        with patch.dict("sys.modules", {"onnxruntime": runtime}):
            with self.assertRaisesRegex(RuntimeError, "does not expose"):
                thor.check_onnxruntime_cuda()
        runtime.InferenceSession.assert_not_called()


if __name__ == "__main__":
    unittest.main()
