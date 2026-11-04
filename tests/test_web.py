"""Browser lifecycle and HTTP contracts; no GPU, models or camera required."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import http.client
import io
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from wapr.timing import print_timing, timing_listener
from wapr_realsense.app import parse_args
from wapr_realsense.initialization import AutoInitializer
from wapr_realsense.web import Dashboard, WebServer, track_camera, worker
from test_realsense_pose import frame, result


class DashboardTests(unittest.TestCase):
    def test_manual_bbox_uses_frozen_rgbd_and_rejects_invalid_or_replaced_selection(self):
        state = Dashboard(manual=True)
        state.set_status("INITIALIZED")
        first = frame()
        state.publish(first.rgb, frame=first)
        token = state.freeze_selection()["token"]
        second = frame()
        second.depth_m[:] = 0
        state.publish(second.rgb, frame=second)
        for invalid in ([0, 0, 1000, 1000], [0, 0, 0, 1], [True, 1, 2, 3]):
            with self.assertRaises(ValueError):
                state.request_detect(invalid, token)
        self.assertTrue(state.request_detect([1, 1, 5, 5], token))
        _, manual, _, detection = state.take_commands()
        self.assertTrue(manual)
        self.assertIs(detection[0], first)
        self.assertEqual(detection[1], [1, 1, 5, 5])
        state.set_status("INITIALIZED", notice="目标丢失，请重新手动框选")
        self.assertEqual(state.snapshot()["notice"], "目标丢失，请重新手动框选")
        with self.assertRaises(ValueError):
            state.request_detect([1, 1, 5, 5], token)
        state.publish(second.rgb, frame=second)
        empty_token = state.freeze_selection()["token"]
        with self.assertRaisesRegex(ValueError, "no valid depth"):
            state.request_detect([1, 1, 5, 5], empty_token)

    def test_mode_change_invalidates_old_results_and_lazy_auto_starts_warmup(self):
        state = Dashboard(manual=True, auto_ready=False)
        state.set_status("TRACKING")
        previous = state.generation
        self.assertTrue(state.request_mode(False))
        self.assertEqual(state.snapshot()["status"], "WARMUP")
        self.assertFalse(state.request_mode(True))
        state.set_status("TRACKING", generation=previous)
        self.assertEqual(state.snapshot()["status"], "WARMUP")
        self.assertTrue(state.take_commands()[2])

    def test_warmup_blanks_preview_and_only_one_concurrent_detect_is_accepted(self):
        state = Dashboard()
        state.publish(frame().rgb)
        self.assertIsNone(state.image)
        self.assertFalse(state.request_detect())
        state.set_status("INITIALIZED")
        state.publish(frame().rgb)
        self.assertIsNotNone(state.image)
        with ThreadPoolExecutor(max_workers=8) as pool:
            accepted = list(pool.map(lambda _: state.request_detect(), range(8)))
        self.assertEqual(sum(accepted), 1)
        self.assertEqual(state.snapshot()["status"], "Looking for object")
        self.assertIsNone(state.image)
        state.set_status("TRACKING")  # An in-flight result cannot undo Detect.
        self.assertEqual(state.snapshot()["status"], "Looking for object")
        state.detect.clear()
        state.set_status("TRACKING")
        self.assertTrue(state.snapshot()["can_detect"])

    def test_dashboard_receives_same_timing_and_listener_is_scoped(self):
        state = Dashboard()
        output = io.StringIO()
        with redirect_stdout(output):
            with timing_listener(state.record_timing):
                print_timing("WARMUP", "DINOv2: total=12 ms | calls=3")
                print_timing("INIT", "attempt=1 | started")
                print_timing("INIT", "result=found")
                print_timing("REINIT", "attempt=2 | started")
                print_timing("TRACKING", "WAPR avg=20 ms | frames=5")
                print_timing("TRACKING", "WAPR avg=30 ms | frames=6")
            print_timing("WARMUP", "outside listener")
        snapshot = state.snapshot()
        self.assertEqual(snapshot["search_part"], "REINIT")
        self.assertEqual(snapshot["timings"]["INIT"], ["attempt=2 | started"])
        self.assertEqual(snapshot["timings"]["TRACKING"], ["WAPR avg=30 ms | frames=6"])
        self.assertEqual(snapshot["timings"]["WARMUP"], ["DINOv2: total=12 ms | calls=3"])
        self.assertIn("[WARMUP] outside listener", output.getvalue())

    def test_explicit_warmup_discards_results_and_does_not_require_depth_or_target(self):
        f = frame()
        f.depth_m[:] = 0
        estimator = Mock(device=None)
        estimator.warmup_pose.return_value = 0.3
        detector = Mock(use_sam=True)
        mesh = object()
        pipeline = Mock()
        initializer = AutoInitializer(estimator, detector, mesh, 0.5, 0.1, 0.5, estimate_frame=pipeline)
        with redirect_stdout(io.StringIO()):
            initializer.warmup(f)
        self.assertEqual(detector.detect_many_categories_many_instances.call_count, 3)
        self.assertTrue(detector._frame_detector_warmed)
        self.assertEqual(estimator.warmup_pose.call_count, 2)
        for call in estimator.warmup_pose.call_args_list:
            self.assertTrue(np.all(call.args[1] == 1))
            self.assertIs(call.args[3][0]["mesh"], mesh)
        self.assertTrue(np.all(f.depth_m == 0))
        self.assertEqual(initializer.attempts, 0)
        pipeline.assert_not_called()

    def test_worker_failure_disables_controls_and_clears_preview(self):
        state = Dashboard()
        state.set_status("INITIALIZED")
        state.publish(frame().rgb)
        with patch("wapr_realsense.web.track_camera", side_effect=RuntimeError("Camera disconnected")), \
                redirect_stderr(io.StringIO()):
            worker(parse_args(["--web", "--init", "auto"]), state)
        self.assertEqual(state.snapshot()["status"], "ERROR")
        self.assertEqual(state.snapshot()["error"], "Camera disconnected")
        self.assertFalse(state.request_detect())
        self.assertIsNone(state.image)

    def test_real_runtime_waits_for_detect_and_automatically_recovers_after_loss(self):
        import wapr
        state = Dashboard()
        reads, statuses = [], []
        initialized = Mock(side_effect=[result(), result()])
        tracking_module = ModuleType("wapr.tracking")
        tracking_module.track_many_categories_many_instances = Mock(
            side_effect=[[result()], [result()], [result()], [], [result()]])
        initialized.warmup = Mock()
        mesh = Mock(extents=np.array([0.2, 0.2, 0.2]),
                    bounds=np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]]))
        estimator = Mock(device=None)
        estimator.prepare_meshes.return_value = [mesh]

        class Camera:
            info = {"name": "test"}

            def __init__(self, *args):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, **kwargs):
                index = len(reads)
                statuses.append(state.snapshot()["status"])
                if index == 2:
                    self_test.assertTrue(state.request_detect())
                f = frame()
                f.number = index
                reads.append(f)
                return f

        self_test = self
        with tempfile.TemporaryDirectory() as folder:
            args = parse_args(["--web", "--init", "auto", "--max-frames", "5", "--output", folder])
            with patch.dict(wapr.__dict__, {"WAPREstimator": Mock(return_value=estimator)}), \
                    patch.dict("sys.modules", {"wapr.tracking": tracking_module}), \
                    patch("torch.cuda.is_available", return_value=True), \
                    patch("wapr_realsense.camera.RealSenseCamera", Camera), \
                    patch("wapr_realsense.geometry.load_mesh", return_value=(mesh, 0.4)), \
                    patch("wapr_realsense.geometry.draw_pose", return_value=frame().rgb), \
                    patch("wapr_realsense.initialization.build_initializer", return_value=initialized), \
                    redirect_stdout(io.StringIO()):
                track_camera(args, state)
            records = [json.loads(line) for line in (Path(folder) / "poses.jsonl").read_text().splitlines()]
        self.assertEqual(statuses, ["WARMUP", "INITIALIZED", "INITIALIZED", "TRACKING",
                                    "Looking for object", "TRACKING"])
        self.assertEqual([row["status"] for row in records],
                         ["starting", "initialized", "initialized_auto", "lost", "initialized_auto", "tracking", "stopped"])
        self.assertFalse(records[1]["valid"])
        self.assertFalse(records[3]["valid"])
        initialized.warmup.assert_called_once()
        self.assertEqual(initialized.call_count, 2)

    def test_web_cli_validates_mode_and_never_needs_a_desktop_display(self):
        self.assertTrue(parse_args(["--web-demo", "--no-display"]).web)
        self.assertTrue(parse_args(["--web", "--init", "auto"]).web)
        self.assertEqual(parse_args(["--web"]).init, "roi")
        for flags in (["--web-port", "0"],
                      ["--web", "--init", "auto", "--replay", "."]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(flags)

    def test_manual_runtime_does_not_search_after_loss_and_uses_the_selected_frame(self):
        import wapr
        state = Dashboard(manual=True, auto_ready=False)
        frames, observed = [], []
        tracking_module = ModuleType("wapr.tracking")
        tracking_module.track_many_categories_many_instances = Mock(
            side_effect=[[result()], [result()], [result()], []])
        mesh = Mock(extents=np.array([0.2, 0.2, 0.2]),
                    bounds=np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]]))
        estimator = Mock(device=None)
        estimator.prepare_meshes.return_value = [mesh]
        estimator.warmup_pose.return_value = 0.1
        estimator.estimate_one_category_one_instance.return_value = result()
        automatic = Mock(return_value=result())
        case = self

        class Camera:
            info = {"name": "test"}

            def __init__(self, *args):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, **kwargs):
                index = len(frames)
                observed.append(state.snapshot())
                if index == 2:
                    selected = state.freeze_selection()
                    case.assertTrue(state.request_detect([1, 1, 5, 5], selected["token"]))
                if index == 6:
                    case.assertTrue(state.request_mode(False))
                    case.assertEqual(state.snapshot()["status"], "WARMUP")
                current = frame()
                current.number = index
                frames.append(current)
                return current

        with tempfile.TemporaryDirectory() as folder:
            args = parse_args(["--web", "--max-frames", "7", "--output", folder])
            with patch.dict(wapr.__dict__, {"WAPREstimator": Mock(return_value=estimator)}), \
                    patch.dict("sys.modules", {"wapr.tracking": tracking_module}), \
                    patch("torch.cuda.is_available", return_value=True), \
                    patch("wapr_realsense.camera.RealSenseCamera", Camera), \
                    patch("wapr_realsense.geometry.load_mesh", return_value=(mesh, 0.4)), \
                    patch("wapr_realsense.geometry.draw_pose", return_value=frame().rgb), \
                    patch("wapr_realsense.initialization.build_initializer", side_effect=[None, automatic]) as builder, \
                    redirect_stdout(io.StringIO()):
                track_camera(args, state)
            rows = [json.loads(line) for line in (Path(folder) / "poses.jsonl").read_text().splitlines()]
        self.assertEqual([row["status"] for row in rows],
                         ["starting", "waiting_roi", "initialized_roi", "lost", "waiting_roi", "waiting_roi",
                          "initialized", "initialized", "stopped"])
        self.assertIs(estimator.estimate_one_category_one_instance.call_args.args[0], frames[1].rgb)
        self.assertEqual(estimator.estimate_one_category_one_instance.call_count, 1)
        self.assertEqual(tracking_module.track_many_categories_many_instances.call_count, 4)
        self.assertEqual(observed[4]["notice"], "目标丢失，请重新手动框选")
        self.assertEqual(observed[5]["notice"], "目标丢失，请重新手动框选")
        self.assertEqual(builder.call_count, 2)
        self.assertEqual(builder.call_args.args[0].init, "auto")
        automatic.warmup.assert_called_once()
        automatic.assert_not_called()


class HTTPTests(unittest.TestCase):
    def test_simulated_cli_starts_blank_waits_for_detect_and_tracks(self):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        process = subprocess.Popen(
            [sys.executable, "-B", "-m", "wapr_realsense", "--web-demo", "--init", "auto", "--web-port", str(port)],
            cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def request(path, method="GET"):
            client = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            try:
                client.request(method, path, headers={"Content-Type": "application/json"})
                response = client.getresponse()
                return response.status, response.read()
            finally:
                client.close()

        def wait_for(status):
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                self.assertIsNone(process.poll(), "Web demo exited unexpectedly")
                try:
                    _, body = request("/api/status")
                    current = json.loads(body)
                    if current["status"] == status:
                        return current
                except OSError:
                    pass
                time.sleep(0.05)
            self.fail(f"Web demo did not reach {status}")

        try:
            self.assertFalse(wait_for("WARMUP")["can_detect"])
            self.assertEqual(request("/api/frame.jpg")[0], 204)
            self.assertEqual(request("/api/detect", "POST")[0], 409)
            self.assertTrue(wait_for("INITIALIZED")["demo"])
            self.assertEqual(request("/api/detect", "POST")[0], 202)
            self.assertFalse(wait_for("Looking for object")["can_detect"])
            status = wait_for("TRACKING")
            self.assertTrue(status["can_detect"])
            self.assertIn("SIMULATED", " ".join(status["timings"]["INIT"]))
            self.assertEqual(request("/api/frame.jpg")[0], 200)
        finally:
            process.send_signal(signal.SIGINT)
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, stderr.decode())
            self.assertIn(b"WAPR_WEB", stdout)

    def test_warmup_http_preview_detect_gating_and_assets(self):
        state = Dashboard()
        server = WebServer(("127.0.0.1", 0), state)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)

        def request(method, path, headers=None, body=None):
            client.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers or {})
            response = client.getresponse()
            return response.status, response.read()

        try:
            for path in ("/", "/app.js", "/style.css"):
                code, body = request("GET", path)
                self.assertEqual(code, 200)
                self.assertTrue(body)
            self.assertEqual(request("GET", "/api/frame.jpg")[0], 204)
            headers = {"Content-Type": "application/json"}
            self.assertEqual(request("POST", "/api/detect", headers)[0], 409)
            code, body = request("GET", "/api/status")
            self.assertEqual(json.loads(body)["status"], "WARMUP")
            state.set_status("INITIALIZED")
            state.publish(frame().rgb)
            code, image = request("GET", "/api/frame.jpg")
            self.assertEqual(code, 200)
            self.assertTrue(image.startswith(b"\xff\xd8"))
            self.assertEqual(request("POST", "/api/detect")[0], 415)
            self.assertEqual(request("POST", "/api/detect", headers)[0], 202)
            self.assertEqual(request("POST", "/api/detect", headers)[0], 409)
            state.take_commands()
            state.set_status("INITIALIZED")
            self.assertEqual(request("POST", "/api/mode", headers, {"manual": "yes"})[0], 400)
            self.assertEqual(request("POST", "/api/mode", headers, {"manual": True})[0], 202)
            state.take_commands()
            f = frame()
            state.publish(f.rgb, frame=f)
            code, body = request("POST", "/api/selection", headers)
            self.assertEqual(code, 200)
            token = json.loads(body)["token"]
            self.assertEqual(request("GET", f"/api/selection.jpg?token={token}")[0], 200)
            self.assertEqual(request("GET", f"/api/selection.jpg?token={token + 1}")[0], 409)
            self.assertEqual(request("POST", "/api/detect", headers, {"bbox": [1, 1, 5, 5], "token": token})[0], 202)
            self.assertEqual(request("GET", "/../web.py")[0], 404)
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
