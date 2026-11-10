"""Exercise version-specific engine APIs on CPU; no TensorRT install needed."""

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from wapr.nets import _Engine, NetRunner, load_net
from wapr.trt_memory import SharedExecutionMemory
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
            self.assertEqual(workspace_bytes(), 4 * 1024 ** 3)
            os.environ["WAPR_TRT_WORKSPACE_GB"] = "0.5"
            self.assertEqual(workspace_bytes(), 512 * 1024 ** 2)
            for value in ("0", "-1", "nan", "inf", "invalid"):
                os.environ["WAPR_TRT_WORKSPACE_GB"] = value
                with self.assertRaises(ValueError):
                    workspace_bytes()


class EngineAPITests(unittest.TestCase):
    def load_engine(self, major, execution_memory=None):
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
        self.engine.create_execution_context = Mock(return_value=self.context)
        self.engine.create_execution_context_without_device_memory = Mock(return_value=self.context)
        self.engine.device_memory_size = 4096
        logger = Mock()
        logger.ERROR = 0
        module = SimpleNamespace(__version__="%d.5.2" % major, Logger=logger,
                 Runtime=lambda _: SimpleNamespace(deserialize_cuda_engine=lambda _: self.engine))
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", tensorrt=module):
            path = Path(folder) / "test.engine"
            path.write_bytes(b"native-test-plan")
            return _Engine(path, ["trans", "rot"], execution_memory=execution_memory)

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

    def test_shared_context_uses_external_memory_and_preserves_both_execution_apis(self):
        from contextlib import contextmanager
        for major in (8, 10):
            with self.subTest(major=major):
                memory = Mock()
                @contextmanager
                def use(context, device, stream):
                    context.external_memory_active = True
                    yield
                    context.external_memory_active = False
                memory.use.side_effect = use
                runner = self.load_engine(major, memory)
                self.engine.create_execution_context.assert_not_called()
                self.engine.create_execution_context_without_device_memory.assert_called_once()
                memory.reserve.assert_called_once_with(4096)
                execute = self.context.execute_async_v2 if major == 8 else self.context.execute_async_v3
                def enqueue(*args, **kwargs):
                    self.assertTrue(self.context.external_memory_active)
                    return True
                execute.side_effect = enqueue
                self.run_engine(runner)
                self.assertFalse(self.context.external_memory_active)


class LoadingTests(unittest.TestCase):
    def test_single_refinement_branch_is_padded_to_existing_engine_minimum(self):
        engine = SimpleNamespace(min_batch=2, max_batch=12, run=Mock(return_value={
            "trans": torch.ones(2, 3), "rot": torch.zeros(2, 3)}))
        runner = NetRunner("sapr", None, engine)
        output = runner(torch.zeros(1, 6, 2, 2), torch.zeros(1, 6, 2, 2))
        feeds = engine.run.call_args.args[0]
        self.assertEqual(feeds["A"].shape[0], 2)
        self.assertEqual(output["trans"].shape[0], 1)

    def test_trt_only_does_not_construct_or_load_a_torch_model(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wapr_wo_mask.engine"
            path.touch()
            memory = object()
            with patch("wapr.nets.RefineNet") as refine, patch("wapr.nets.WbpsNet") as wbps, \
                 patch("wapr.nets._load_weights") as weights, patch("wapr.nets._Engine") as engine:
                runner = load_net("wapr_wo_mask", "absent.pth", engine=path, execution_memory=memory)
                self.assertIsNone(runner.module)
                self.assertIs(runner.engine, engine.return_value)
                engine.assert_called_once_with(path, ("trans", "rot"), execution_memory=memory)
                refine.assert_not_called()
                wbps.assert_not_called()
                weights.assert_not_called()

    def test_torch_and_export_paths_still_load_checkpoints(self):
        for name, constructor in (("wapr_wo_mask", "RefineNet"), ("wbps", "WbpsNet")):
            with self.subTest(name=name), patch("wapr.nets." + constructor) as module, \
                 patch("wapr.nets._load_weights") as weights, patch("wapr.nets._Engine") as engine:
                runner = load_net(name, "test.pth", device="cpu")
                self.assertIs(runner.module, module.return_value)
                self.assertIsNone(runner.engine)
                weights.assert_called_once()
                module.return_value.to.assert_called_once_with("cpu")
                module.return_value.eval.assert_called_once()
                engine.assert_not_called()


class SharedMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = SharedExecutionMemory("cuda:0")
        self.stream = SimpleNamespace(cuda_stream=11)
        self.context = SimpleNamespace()
        self.buffer = Mock()
        self.buffer.data_ptr.return_value = 257  # deliberately unaligned
        self.device_patch = patch("torch.cuda.device")
        self.empty_patch = patch("torch.empty", return_value=self.buffer)
        self.event_patch = patch("torch.cuda.Event")
        self.device_patch.start()
        self.empty = self.empty_patch.start()
        self.event = self.event_patch.start().return_value
        self.addCleanup(self.device_patch.stop)
        self.addCleanup(self.empty_patch.stop)
        self.addCleanup(self.event_patch.stop)

    def test_largest_requirement_is_allocated_once_and_reused_across_contexts(self):
        for size in (1024, 4096, 2048, 512):
            self.memory.reserve(size)
        for context in (self.context, SimpleNamespace(), SimpleNamespace()):
            with self.memory.use(context, "cuda:0", self.stream):
                self.assertEqual(context.device_memory, 512)
        self.empty.assert_called_once_with(4096 + 255, dtype=torch.uint8, device=torch.device("cuda:0"))
        self.assertEqual(self.memory.allocated_bytes, 4096)
        self.event.synchronize.assert_not_called()
        self.assertEqual(self.event.record.call_count, 3)

    def test_stream_switch_and_growth_wait_for_pending_gpu_work(self):
        self.memory.reserve(1024)
        with self.memory.use(self.context, "cuda:0", self.stream):
            pass
        with self.memory.use(SimpleNamespace(), "cuda:0", SimpleNamespace(cuda_stream=22)):
            pass
        self.event.synchronize.assert_called_once()
        self.memory.reserve(8192)
        with self.memory.use(self.context, "cuda:0", SimpleNamespace(cuda_stream=22)):
            self.assertEqual(self.memory.allocated_bytes, 8192)
        self.assertEqual(self.event.synchronize.call_count, 2)
        self.assertEqual(self.empty.call_count, 2)

    def test_exception_records_completion_and_different_device_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "enqueue failed"):
            with self.memory.use(self.context, "cuda:0", self.stream):
                raise RuntimeError("enqueue failed")
        self.event.record.assert_called_once_with(self.stream)
        with self.assertRaisesRegex(ValueError, "same CUDA device"):
            with self.memory.use(self.context, "cuda:1", self.stream):
                self.fail("mismatched device was accepted")


if __name__ == "__main__":
    unittest.main()
