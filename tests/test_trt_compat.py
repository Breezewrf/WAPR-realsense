"""Exercise version-specific engine APIs on CPU; no TensorRT install needed."""

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from wapr.nets import _Engine
from wapr.trt_compat import network_flags, workspace_bytes


class BuilderSettingsTests(unittest.TestCase):
    def test_explicit_batch_for_trt8_and_default_flags_for_trt10(self):
        flags = SimpleNamespace(EXPLICIT_BATCH=0)
        self.assertEqual(network_flags(SimpleNamespace(__version__="8.5.2.2",
                         NetworkDefinitionCreationFlag=flags)), 1)
        self.assertEqual(network_flags(SimpleNamespace(__version__="10.16.1.11")), 0)
        for version in ("8.4.0", "9.0.0", "11.0.0"):
            with self.assertRaises(RuntimeError):
                network_flags(SimpleNamespace(__version__=version))

    def test_workspace_defaults_override_and_invalid_values(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("wapr.trt_compat.platform.machine", return_value="aarch64"):
                self.assertEqual(workspace_bytes(), 1024 ** 3)
            with patch("wapr.trt_compat.platform.machine", return_value="x86_64"):
                self.assertEqual(workspace_bytes(), 8 * 1024 ** 3)
            os.environ["WAPR_TRT_WORKSPACE_GB"] = "0.5"
            self.assertEqual(workspace_bytes(), 512 * 1024 ** 2)
            for value in ("0", "-1", "nan", "inf", "invalid"):
                os.environ["WAPR_TRT_WORKSPACE_GB"] = value
                with self.assertRaises(ValueError):
                    workspace_bytes()


class EngineAPITests(unittest.TestCase):
    def load_engine(self, major):
        # Reorder binding indices to catch accidental dependence on feed/output order.
        self.indices = {"A": 2, "B": 0, "trans": 3, "rot": 1}
        shapes = ((1, 6, 2, 2), (2, 6, 2, 2), (12, 6, 2, 2))
        if major == 8:
            self.context = Mock(spec=["set_binding_shape", "get_binding_shape", "execute_async_v2"])
            self.context.set_binding_shape.return_value = True
            self.context.get_binding_shape.return_value = (2, 3)
            self.context.execute_async_v2.return_value = True
            self.engine = SimpleNamespace(
                create_execution_context=lambda: self.context, num_bindings=4,
                get_profile_shape=Mock(return_value=shapes),
                get_binding_index=self.indices.__getitem__)
        else:
            self.context = Mock(spec=["set_input_shape", "get_tensor_shape", "set_tensor_address", "execute_async_v3"])
            self.context.set_input_shape.return_value = True
            self.context.get_tensor_shape.return_value = (2, 3)
            self.context.execute_async_v3.return_value = True
            self.engine = SimpleNamespace(create_execution_context=lambda: self.context,
                                          get_tensor_profile_shape=Mock(return_value=shapes))
        logger = Mock()
        logger.ERROR = 0
        module = SimpleNamespace(__version__="%d.5.2" % major, Logger=logger,
                 Runtime=lambda _: SimpleNamespace(deserialize_cuda_engine=lambda _: self.engine))
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", tensorrt=module):
            path = Path(folder) / "test.engine"
            path.write_bytes(b"native-test-plan")
            return _Engine(path, ["trans", "rot"])

    def run_engine(self, runner):
        self.feeds = {name: torch.ones(2, 6, 2, 2) for name in ("A", "B")}
        with patch("torch.cuda.current_stream", return_value=SimpleNamespace(cuda_stream=123)):
            return runner.run(self.feeds)

    def test_trt85_sets_dynamic_binding_shapes_and_uses_binding_order(self):
        runner = self.load_engine(8)
        self.assertEqual((runner.min_batch, runner.opt_batch, runner.max_batch), (1, 2, 12))
        self.engine.get_profile_shape.assert_called_once_with(0, 2)
        outputs = self.run_engine(runner)
        self.context.set_binding_shape.assert_any_call(2, (2, 6, 2, 2))
        self.context.set_binding_shape.assert_any_call(0, (2, 6, 2, 2))
        pointers = dict(self.feeds, **outputs)
        addresses = [0] * 4
        for name, value in pointers.items():
            addresses[self.indices[name]] = value.data_ptr()
        self.context.execute_async_v2.assert_called_once_with(bindings=addresses, stream_handle=123)
        self.assertEqual(tuple(outputs["trans"].shape), (2, 3))

    def test_trt10_preserves_named_tensor_execution(self):
        runner = self.load_engine(10)
        self.engine.get_tensor_profile_shape.assert_called_once_with("A", 0)
        outputs = self.run_engine(runner)
        for name, value in dict(self.feeds, **outputs).items():
            self.context.set_tensor_address.assert_any_call(name, value.data_ptr())
        self.context.execute_async_v3.assert_called_once_with(123)

    def test_shape_and_enqueue_failures_are_reported(self):
        for major in (8, 10):
            with self.subTest(major=major):
                runner = self.load_engine(major)
                setter = self.context.set_binding_shape if major == 8 else self.context.set_input_shape
                setter.return_value = False
                with self.assertRaisesRegex(ValueError, "outside profile"):
                    self.run_engine(runner)
                setter.return_value = True
                execute = self.context.execute_async_v2 if major == 8 else self.context.execute_async_v3
                execute.return_value = False
                with self.assertRaisesRegex(RuntimeError, "execution failed"):
                    self.run_engine(runner)


if __name__ == "__main__":
    unittest.main()
