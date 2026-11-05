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
from wapr.timing import TimingCollector


class TrackingEngineTests(unittest.TestCase):
    def test_real_tracking_path_collects_refinement_and_scoring_without_changing_pose(self):
        import numpy as np
        from types import MethodType
        from wapr.estimator import WAPREstimator
        from wapr.tracking import track_many_categories_many_instances

        class Network:
            engine = None
            def __init__(self, name):
                self.name = name
            def __call__(self, A, B, L=None):
                if self.name == "wbps":
                    return {"within_group": torch.zeros(1, L), "between_group": torch.zeros(1, L)}
                return {"trans": torch.zeros(len(A), 3), "rot": torch.zeros(len(A), 3)}

        mesh = SimpleNamespace(bounds=np.array([[-0.1] * 3, [0.1] * 3]))
        estimator = SimpleNamespace(device="cpu", nets={name: Network(name)
            for name in ("wapr_wo_mask", "sapr", "wbps")}, _pose_meshes=lambda meshes: meshes,
            renderer=SimpleNamespace(cached_mesh_id=lambda mesh: 1))
        estimator._refine = MethodType(WAPREstimator._refine, estimator)
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = 1
        item = {"track_id": 1, "mesh": mesh, "diameter_m": 0.2, "pose_4x4": pose}
        crop_batches = []
        def crop(rgb, depth, mask, poses, *args, **kwargs):
            crop_batches.append(len(poses))
            A = torch.zeros(len(poses), 6, 2, 2)
            return A, A.clone(), poses.clone(), torch.full((len(poses),), 0.2)

        with patch("wapr.estimator.make_crop_pair", side_effect=crop), \
             patch("wapr.tracking.make_crop_pair", side_effect=crop), \
             patch("wapr.tracking.depth2xyzmap_batch", return_value=torch.zeros(1, 2, 2, 3)), \
             patch("torch.cuda.synchronize") as synchronize:
            inputs = (estimator, np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), np.float32), np.eye(3), [item])
            baseline = track_many_categories_many_instances(*inputs, reuse_tracking=False)
            self.assertEqual(crop_batches, [2, 2, 2, 3, 3, 6])
            crop_batches.clear()
            synchronize.assert_not_called()
            with TimingCollector(profile_pose=True) as components:
                profiled = track_many_categories_many_instances(*inputs)
            self.assertEqual(crop_batches, [1, 1, 1, 2, 2, 4])
            self.assertGreater(synchronize.call_count, 0)
            for preset, expected in (("balanced", [1, 1, 1, 3]), ("fast", [1, 1, 3]),
                                      ("small-motion", [1, 2])):
                crop_batches.clear()
                reduced = track_many_categories_many_instances(*inputs, tracking_preset=preset)
                self.assertEqual(crop_batches, expected)
                self.assertEqual(len(reduced), 1)
                self.assertTrue(np.isfinite(reduced[0]["pose_4x4"]).all())
        for name, count in (("WAPR", 1), ("WAPR crop", 3), ("WAPR network", 3),
                            ("SAPR", 1), ("SAPR crop", 2), ("SAPR network", 2),
                            ("WBPS", 1), ("WBPS crop", 1), ("WBPS network", 1)):
            self.assertEqual(components.samples[name][1], count, name)
        np.testing.assert_allclose(profiled[0]["pose_4x4"], baseline[0]["pose_4x4"])
        self.assertEqual(profiled[0]["score_6d"], baseline[0]["score_6d"])

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
