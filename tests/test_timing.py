"""Timing contracts without requiring a CUDA device or downloaded models."""

from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from wapr.timing import ComponentTimer, TimingCollector, TrackingTiming, print_timing, timing_phase, pose_component_timer
from wapr_realsense.app import timed_step


class ComponentTimingTests(unittest.TestCase):
    def test_pose_profiling_is_off_without_an_explicit_pose_collector(self):
        with patch("torch.cuda.synchronize") as synchronize:
            with pose_component_timer("WAPR", "cuda:0"):
                pass
            with TimingCollector() as detector:
                with pose_component_timer("WAPR", "cuda:0"):
                    pass
        self.assertEqual(detector.samples, {})
        synchronize.assert_not_called()

    def test_tracking_components_are_averaged_per_frame_and_reset_on_flush(self):
        components = TimingCollector(profile_pose=True)
        for name, value in (("WAPR", 60), ("WAPR crop", 15), ("WAPR network", 40),
                            ("SAPR", 30), ("SAPR crop", 10), ("SAPR network", 15),
                            ("WBPS", 20), ("WBPS crop", 5), ("WBPS network", 12)):
            components.record(name, value)
        tracking = TrackingTiming(interval=5, profile=True)
        output = io.StringIO()
        with redirect_stdout(output):
            tracking.add(120, now=0, components=components)
            tracking.add(140, now=5, components=components)
            tracking.add(90, now=6)
            tracking.flush()
            tracking.flush()
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], "[TRACKING] WAPR avg=130.00 ms | frames=2")
        self.assertIn("WAPR=60.00 ms (crop=15.00, network=40.00)", lines[1])
        self.assertIn("SAPR=30.00 ms (crop=10.00, network=15.00)", lines[1])
        self.assertIn("WBPS=20.00 ms (crop=5.00, network=12.00)", lines[1])
        self.assertIn("other=20.00 ms", lines[1])
        self.assertEqual(lines[2], "[TRACKING] WAPR avg=90.00 ms | frames=1")
        self.assertEqual(tracking.components.samples, {})

    def test_timed_step_collects_only_tracking_and_includes_the_lost_update(self):
        session = SimpleNamespace(pose=None, estimator=SimpleNamespace(device=None))
        def step(frame, bbox=None):
            with pose_component_timer("WAPR", None):
                pass
            if bbox is not None:
                session.pose = object()
                return "initialized_roi", {}
            session.pose = None
            return "lost", None
        session.step = step
        tracking = TrackingTiming(profile=True)
        output = io.StringIO()
        with redirect_stdout(output):
            timed_step(session, object(), bbox=[1, 1, 2, 2], tracking_timing=tracking)
            timed_step(session, object(), tracking_timing=tracking)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("[INIT]"))
        self.assertIn("frames=1", lines[1])
        self.assertIn("stages avg/frame:", lines[2])
        self.assertEqual(tracking.count, 0)

    def test_cuda_work_is_synchronized_before_reading_each_timestamp(self):
        events = []

        def timestamp():
            events.append("clock")
            return 1.0 if len(events) == 2 else 1.125

        output = io.StringIO()
        with patch("torch.cuda.synchronize", side_effect=lambda device: events.append(("sync", device))), \
             patch("wapr.timing.time.perf_counter", side_effect=timestamp), redirect_stdout(output):
            with ComponentTimer("DINOv2", "cuda:0") as timer:
                events.append("work")
        self.assertEqual(events, [("sync", "cuda:0"), "clock", "work", ("sync", "cuda:0"), "clock"])
        self.assertEqual(timer.elapsed_ms, 125.0)
        self.assertEqual(output.getvalue(), "[TIMING] DINOv2: 125.00 ms\n")

    def test_warmup_label_is_reset_even_after_an_exception(self):
        output = io.StringIO()
        with patch("wapr.timing.time.perf_counter", side_effect=[1, 1.125, 2, 2.25]), redirect_stdout(output):
            with self.assertRaisesRegex(ValueError, "failed"):
                with timing_phase("warmup"):
                    with ComponentTimer("SAM2"):
                        pass
                    raise ValueError("failed")
            with ComponentTimer("SAM2"):
                pass
        self.assertEqual(output.getvalue().splitlines(), [
            "[TIMING][warmup] SAM2: 125.00 ms", "[TIMING] SAM2: 250.00 ms"])

    def test_recursive_decorator_keeps_each_calls_start_time(self):
        @ComponentTimer("recursive")
        def run(depth):
            return run(depth - 1) + 1 if depth else 0

        output = io.StringIO()
        with patch("wapr.timing.time.perf_counter", side_effect=[1, 2, 3, 5]), redirect_stdout(output):
            self.assertEqual(run(1), 1)
        self.assertEqual(output.getvalue().splitlines(), [
            "[TIMING] recursive: 1000.00 ms", "[TIMING] recursive: 4000.00 ms"])

    def test_failed_component_propagates_exception_without_success_timing(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "inference failed"):
            with ComponentTimer("SAM2") as timer:
                raise RuntimeError("inference failed")
        self.assertIsNone(timer.elapsed_ms)
        self.assertEqual(output.getvalue(), "")

    def test_tracking_average_flushes_on_loss_and_excludes_initialization(self):
        session = SimpleNamespace(pose=object(), estimator=SimpleNamespace(device=None),
                                  step=Mock(return_value=("lost", None)))
        tracking = TrackingTiming()

        def lose_track(*args, **kwargs):
            session.pose = None
            return "lost", None

        session.step.side_effect = lose_track
        frame = object()
        output = io.StringIO()
        with patch("wapr.timing.time.perf_counter", side_effect=[1, 1.125, 2, 2.25, 3, 3.5]), \
             redirect_stdout(output):
            self.assertEqual(timed_step(session, frame, tracking_timing=tracking), ("lost", None, 125.0))
            session.step.side_effect = None
            session.step.return_value = ("searching", None)
            self.assertEqual(timed_step(session, frame, tracking_timing=tracking), ("searching", None, 250.0))
            session.pose = object()
            session.step.return_value = ("initialized_roi", None)
            self.assertEqual(timed_step(session, frame, bbox=[1, 1, 2, 2], tracking_timing=tracking),
                             ("initialized_roi", None, 500.0))
        self.assertEqual(output.getvalue().splitlines(), [
            "[TRACKING] WAPR avg=125.00 ms | frames=1",
            "[INIT] WAPR initialization=500.00 ms | result=initialized_roi"])
        self.assertEqual(tracking.count, 0)

    def test_tracking_uses_window_mean_and_flushes_remaining_samples_once(self):
        tracking = TrackingTiming(interval=5)
        output = io.StringIO()
        with redirect_stdout(output):
            tracking.add(10, now=0)
            tracking.add(20, now=2)
            self.assertEqual(output.getvalue(), "")
            tracking.add(30, now=5)
            tracking.add(100, now=6)
            tracking.flush()
            tracking.flush()
        self.assertEqual(output.getvalue().splitlines(), [
            "[TRACKING] WAPR avg=20.00 ms | frames=3",
            "[TRACKING] WAPR avg=100.00 ms | frames=1"])

    def test_nested_collectors_keep_warmup_out_of_initialization_timings(self):
        output = io.StringIO()
        with patch("wapr.timing.time.perf_counter", side_effect=[0, 1, 1, 3, 3, 3.5]), redirect_stdout(output):
            with TimingCollector() as initialization:
                with TimingCollector() as warmup:
                    with ComponentTimer("GroundingDINO"):
                        pass
                    with ComponentTimer("GroundingDINO"):
                        pass
                warmup.report_warmup(3100)
                with ComponentTimer("GroundingDINO"):
                    pass
        self.assertEqual(warmup.samples["GroundingDINO"], (3000.0, 2))
        self.assertEqual(initialization.samples["GroundingDINO"], (500.0, 1))
        self.assertIn("total=3000.00 ms | avg=1500.00 ms | calls=2", output.getvalue())
        self.assertIn("SAM2=skipped", output.getvalue())
        self.assertNotIn("[TIMING]", output.getvalue())

    def test_colors_are_distinct_and_disabled_for_redirection_and_no_color(self):
        output = io.StringIO()
        with patch("wapr.timing.os.environ", {"TERM": "xterm"}), \
             redirect_stdout(output), patch.object(output, "isatty", return_value=True):
            for part in ("WARMUP", "TRACKING", "REINIT"):
                print_timing(part, "sample")
        self.assertEqual(output.getvalue().splitlines(), [
            "\033[33m[WARMUP] sample\033[0m", "\033[32m[TRACKING] sample\033[0m",
            "\033[36m[REINIT] sample\033[0m"])
        output = io.StringIO()
        with redirect_stdout(output):
            print_timing("TRACKING", "redirected")
            with patch.object(output, "isatty", return_value=True), patch.dict("os.environ", {"NO_COLOR": ""}):
                print_timing("TRACKING", "no color")
        self.assertNotIn("\033[", output.getvalue())


if __name__ == "__main__":
    unittest.main()
