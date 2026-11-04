"""Check the actual LayerNorm export graph; this is not a TRT GPU test."""

import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from wapr.export_engines import onnx_export_options
from wapr.trt_compat import onnx_opset_limit, parser_error_message


class OpsetSelectionTests(unittest.TestCase):
    def test_layernorm_opset_is_limited_only_on_older_tensorrt(self):
        self.assertEqual(onnx_opset_limit("8.5.2.2"), 16)
        self.assertEqual(onnx_opset_limit("8.6.1"), 17)
        self.assertEqual(onnx_opset_limit("10.16.1.11"), 17)
        self.assertEqual(onnx_opset_limit(), 17)


@unittest.skipUnless(importlib.util.find_spec("onnx"), "ONNX export helpers are not installed")
class OnnxGraphTests(unittest.TestCase):
    def test_trt85_layernorm_export_has_basic_ops_and_preserves_dynamic_batch(self):
        import onnx
        from onnx.reference import ReferenceEvaluator

        torch.manual_seed(13)
        module = torch.nn.LayerNorm(8).eval()
        graph = io.BytesIO()
        torch.onnx.export(module, (torch.randn(2, 3, 8),), graph,
                          input_names=["x"], output_names=["y"],
                          dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
                          **onnx_export_options("8.5.2.2"))
        model = onnx.load_from_string(graph.getvalue())
        onnx.checker.check_model(model)
        self.assertEqual(model.opset_import[0].version, 16)
        operations = {node.op_type for node in model.graph.node}
        self.assertNotIn("LayerNormalization", operations)
        self.assertIn("ReduceMean", operations)
        evaluator = ReferenceEvaluator(model)
        for batch in (1, 3):
            inputs = torch.randn(batch, 3, 8)
            with torch.no_grad():
                expected = module(inputs).numpy()
            actual, = evaluator.run(None, {"x": inputs.numpy()})
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_missing_plugin_error_reports_actual_onnx_node(self):
        import onnx
        model = onnx.helper.make_model(onnx.helper.make_graph([
            onnx.helper.make_node("LayerNormalization", ["x", "w", "b"], ["y"], name="norm")
        ], "test", [], []))
        parser = SimpleNamespace(num_errors=1, get_error=Mock(
            return_value="In node 0 (importFallbackPluginImporter): UNSUPPORTED_NODE"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.onnx"
            onnx.save(model, path)
            message = parser_error_message(parser, path)
        self.assertIn("UNSUPPORTED_NODE", message)
        self.assertIn("ONNX_NODE index=0 op_type=LayerNormalization domain=ai.onnx name=norm", message)


if __name__ == "__main__":
    unittest.main()
