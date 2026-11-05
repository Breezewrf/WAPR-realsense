"""Single-object scoring and grouped-engine selection without CUDA execution."""

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from wapr.nets import NetRunner
from wapr.pose_groups import wbps_group_sizes
from wapr.tracking import _tracking_score_engine


class TrackingEngineTests(unittest.TestCase):
    def estimator(self):
        memory = object()
        return SimpleNamespace(nets={name: SimpleNamespace(engine=SimpleNamespace(
            max_batch=120, execution_memory=memory)) for name in ("wapr_wo_mask", "sapr", "wbps")})

    def test_single_object_needs_no_additional_engine_even_if_files_are_missing(self):
        estimator = self.estimator()
        with patch("wapr.tracking.recipe.engine_file", side_effect=AssertionError("Unexpected file lookup")), \
             patch("wapr.tracking._Engine") as load:
            engine, capacity = _tracking_score_engine(estimator, 1)
        self.assertIsNone(engine)
        self.assertEqual(capacity, 20)
        load.assert_not_called()

    def test_single_group_of_six_uses_the_existing_wbps_engine(self):
        engine = SimpleNamespace(min_batch=2, max_batch=120, run=Mock(return_value={
            "within_group": torch.zeros(1, 6), "between_group": torch.ones(1, 6)}))
        runner = NetRunner("wbps", None, engine)
        output = runner(torch.zeros(6, 6, 2, 2), torch.zeros(6, 6, 2, 2), L=6)
        engine.run.assert_called_once()
        self.assertIsNone(runner.batch_engine)
        self.assertEqual(tuple(output["within_group"].shape), (1, 6))

    def test_multiple_objects_still_require_an_independent_object_axis(self):
        estimator = self.estimator()
        with tempfile.TemporaryDirectory() as folder, \
             patch("wapr.tracking.recipe.engine_file", return_value=Path(folder) / "wbps.engine"):
            with self.assertRaisesRegex(FileNotFoundError, "export_tracking_engine"):
                _tracking_score_engine(estimator, 2)
            (Path(folder) / "wbps_batch.engine").touch()
            engine, capacity = _tracking_score_engine(estimator, 2)
            self.assertIsNone(engine)
            self.assertEqual(capacity, 20)

    def test_separate_tracking_engine_shares_memory_and_is_cached(self):
        estimator = self.estimator()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wbps_tracking.engine"
            path.touch()
            score = SimpleNamespace(min_batch=6, max_batch=max(wbps_group_sizes))
            with patch("wapr.tracking.recipe.engine_file", return_value=path.with_name("wbps.engine")), \
                 patch("wapr.tracking._Engine", return_value=score) as load:
                self.assertIs(_tracking_score_engine(estimator, 2)[0], score)
                self.assertIs(_tracking_score_engine(estimator, 2)[0], score)
                load.assert_called_once_with(path, ("within_group", "between_group"),
                    execution_memory=estimator.nets["wapr_wo_mask"].engine.execution_memory)

    def test_torch_path_does_not_load_engines(self):
        estimator = self.estimator()
        for net in estimator.nets.values():
            net.engine = None
        with patch("wapr.tracking._Engine") as load:
            engine, capacity = _tracking_score_engine(estimator, 1)
        self.assertIsNone(engine)
        self.assertEqual(capacity, max(wbps_group_sizes) // 6)
        load.assert_not_called()


class DiagnosticTests(unittest.TestCase):
    def test_reports_nan_and_infinity_for_every_output(self):
        path = Path(__file__).resolve().parents[1] / "scripts/check_trt_memory.py"
        spec = importlib.util.spec_from_file_location("wapr_memory_check", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        statistics = module.output_statistics({"trans": torch.tensor([[float("nan"), float("inf"), 0.]]),
                                               "rot": torch.ones(1, 3)})
        self.assertEqual(statistics["trans"], {"shape": [1, 3], "elements": 3, "nan": 1, "inf": 1})
        self.assertEqual(statistics["rot"]["nan"], 0)
        self.assertEqual(statistics["rot"]["inf"], 0)


if __name__ == "__main__":
    unittest.main()
