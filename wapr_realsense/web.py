"""Local browser demo; HTTP stays responsive while one worker owns inference."""

from collections import deque
from copy import copy
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
import traceback
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from wapr.timing import ComponentTimer, TrackingTiming, print_timing, timing_listener


class Dashboard:
    """Small shared state; Detect is admitted atomically, never during warmup."""

    def __init__(self, demo=False, manual=False, auto_ready=True):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.detect = threading.Event()
        self.mode_changed = threading.Event()
        self.manual = manual
        self.auto_ready = auto_ready
        self.generation = 0
        self.pending_detection = (None, None)
        self.selection = None
        self.selection_id = 0
        self.preview_frame = None
        self.notice = None
        self.status = "WARMUP"
        self.error = None
        self.demo = demo
        self.image = None
        self.image_version = 0
        self.timings = {part: deque(maxlen=12) for part in ("WARMUP", "INIT", "TRACKING")}
        self.search_part = "INIT"

    def snapshot(self):
        with self.lock:
            return {"status": self.status, "error": self.error, "demo": self.demo,
                    "can_detect": self.status in ("INITIALIZED", "TRACKING"),
                    "manual": self.manual, "notice": self.notice,
                    "selection_token": self.selection[0] if self.selection else None,
                    "has_frame": self.image is not None,
                    "can_change_mode": self.status in ("INITIALIZED", "TRACKING"),
                    "image_version": self.image_version,
                    "timings": {part: list(lines) for part, lines in self.timings.items()},
                    "search_part": self.search_part}

    def set_status(self, status, error=None, notice=None, generation=None):
        with self.lock:
            if generation is not None and generation != self.generation:
                return
            if status == "TRACKING" and (self.detect.is_set() or self.mode_changed.is_set()):
                return  # A newer Detect command takes priority over this result.
            self.status, self.error = status, error
            self.notice = notice
            if status == "INITIALIZED" and notice is not None:
                self.selection = None
            if status in ("WARMUP", "ERROR", "STOPPED"):
                self.image = None
                self.image_version += 1
                self.selection = None
                self.preview_frame = None

    def request_mode(self, manual):
        with self.lock:
            if self.status not in ("INITIALIZED", "TRACKING"):
                return False
            if self.manual == manual:
                return True
            self.manual = manual
            self.generation += 1
            self.status = "WARMUP" if not manual and not self.auto_ready else "INITIALIZED"
            self.notice = None
            self.selection = None
            self.detect.clear()
            self.timings["INIT"].clear()
            self.timings["TRACKING"].clear()
            self.mode_changed.set()
            if self.status == "WARMUP":
                self.image = None
                self.image_version += 1
            return True

    def freeze_selection(self):
        with self.lock:
            if not self.manual or self.status not in ("INITIALIZED", "TRACKING"):
                raise ValueError("Manual selection is unavailable in the current state")
            if self.preview_frame is None or self.image is None:
                raise ValueError("Wait for a camera frame before drawing a box")
            self.selection_id += 1
            self.selection = (self.selection_id, self.preview_frame, self.image)
            height, width = self.preview_frame.depth_m.shape
            return {"token": self.selection_id, "width": width, "height": height}

    def request_detect(self, bbox=None, token=None):
        with self.lock:
            if self.status not in ("INITIALIZED", "TRACKING"):
                return False
            selected = None
            if self.manual:
                if self.selection is None or type(token) is not int or token != self.selection[0]:
                    raise ValueError("Freeze a new frame and draw a bounding box first")
                if not isinstance(bbox, list) or len(bbox) != 4 or any(type(v) is not int for v in bbox):
                    raise ValueError("bbox must contain four integer pixel coordinates [x, y, width, height]")
                from .geometry import check_bbox
                selected = self.selection[1]
                bbox = check_bbox(bbox, selected.depth_m)
            elif bbox is not None or token is not None:
                raise ValueError("Bounding boxes require manual initialization")
            self.pending_detection = (selected, bbox)
            self.selection = None
            self.generation += 1
            self.status = "Looking for object"
            self.notice = None
            self.timings["INIT"].clear()
            self.timings["TRACKING"].clear()
            self.image = None
            self.image_version += 1
            self.detect.set()
            return True

    def take_commands(self):
        with self.lock:
            changed, detect = self.mode_changed.is_set(), self.detect.is_set()
            self.mode_changed.clear()
            self.detect.clear()
            return self.generation, self.manual, changed, self.pending_detection if detect else None

    def record_timing(self, part, message):
        if part not in ("WARMUP", "INIT", "REINIT", "TRACKING"):
            return
        with self.lock:
            group = "INIT" if part == "REINIT" else part
            if group == "INIT":
                self.search_part = part
                if message.endswith(" | started"):
                    self.timings[group].clear()
            if group == "TRACKING" and not message.startswith("stages avg/frame:"):
                self.timings[group].clear()
            self.timings[group].append(message)

    def publish(self, bgr, frame=None, generation=None):
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError("Could not encode browser preview")
        with self.lock:
            if generation is not None and generation != self.generation:
                return
            if self.status not in ("WARMUP", "ERROR", "STOPPED") and not self.detect.is_set():
                self.image = encoded.tobytes()
                self.preview_frame = frame
                self.image_version += 1


class WebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, dashboard):
        self.dashboard = dashboard
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_content(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = urlsplit(self.path).path
        state = self.server.dashboard
        if path == "/api/status":
            return self.send_content(200, json.dumps(state.snapshot()).encode(), "application/json")
        if path == "/api/frame.jpg":
            with state.lock:
                image = state.image
            return self.send_content(204 if image is None else 200, image or b"", "image/jpeg")
        if path == "/api/selection.jpg":
            token = parse_qs(urlsplit(self.path).query).get("token", [None])[0]
            with state.lock:
                selection = state.selection
                image = selection[2] if selection and str(selection[0]) == token else None
            return self.send_content(409 if image is None else 200, image or b"Selection expired", "image/jpeg")
        static = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8")}
        if path in static:
            filename, content_type = static[path]
            return self.send_content(200, (Path(__file__).parent / "static" / filename).read_bytes(), content_type)
        self.send_content(404, b"Not found", "text/plain")

    def do_POST(self):
        path = urlsplit(self.path).path
        if path not in ("/api/detect", "/api/mode", "/api/selection"):
            return self.send_content(404, b"Not found", "text/plain")
        # A same-origin browser action; only fixed commands and small JSON payloads.
        # JSON content type prevents a cross-origin form from triggering Detect.
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.send_content(415, b"Expected application/json", "text/plain")
        state = self.server.dashboard
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 4096:
                raise ValueError("Request is too large")
            body = json.loads(self.rfile.read(length)) if length else {}
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            if path == "/api/selection":
                return self.send_content(200, json.dumps(state.freeze_selection()).encode(), "application/json")
            if path == "/api/mode":
                if type(body.get("manual")) is not bool:
                    raise ValueError("manual must be a boolean")
                accepted = state.request_mode(body["manual"])
            else:
                accepted = state.request_detect(body.get("bbox"), body.get("token"))
        except (ValueError, TypeError) as error:
            return self.send_content(400, json.dumps({"error": str(error)}).encode(), "application/json")
        self.send_content(202 if accepted else 409, json.dumps({"accepted": accepted}).encode(), "application/json")


def track_camera(args, state):
    """Load, warm up, then wait for Detect before accepting any real pose."""
    import torch
    from wapr import WAPREstimator
    from .app import timed_step
    from .camera import RealSenseCamera
    from .geometry import draw_pose, load_mesh
    from .initialization import build_initializer, warmup_pose_inputs
    from .output import PoseWriter, pose_record
    from .tracker import PoseSession

    if not torch.cuda.is_available():
        raise RuntimeError("WAPR requires working CUDA PyTorch")
    started = time.perf_counter()
    os.environ["WAPR_BACKEND"] = args.backend
    mesh, diameter = load_mesh(args.mesh, args.mesh_unit, args.mesh_scale)
    estimator = WAPREstimator(device="cuda:0", object_mode=args.object_mode)
    prepared = estimator.prepare_meshes([mesh])[0]
    initializer = build_initializer(args, estimator, mesh, prepared, diameter)
    output = args.output or Path.cwd() / "outputs" / "wapr_realsense" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    writer = PoseWriter(output)
    tracking = TrackingTiming(args.timing_interval, profile=args.profile_tracking)
    last_frame = None
    try:
        (output / "session.json").write_text(json.dumps({
            "mesh": str(args.mesh.resolve()), "mesh_unit": args.mesh_unit,
            "mesh_scale": args.mesh_scale, "diameter_m": diameter, "backend": args.backend,
            "object_mode": args.object_mode,
            "profile_tracking": args.profile_tracking,
            "init": args.init, "web": True, "minimum_score": args.min_score,
            "camera_frame": args.camera_frame, "object_frame": args.object_frame,
            "pose_convention": "original_scaled_mesh_to_color_optical_frame",
            "automatic_detection": {"interval_s": args.detect_interval, "score_2d_min": args.detect_score,
                                    "backend": args.detect_backend, "dino": args.detect_dino,
                                    "grounding": args.detect_grounding, "use_sam": args.detect_use_sam,
                                    "template_cache": str(args.template_cache) if args.template_cache else None}}, indent=2))
        print("LIVE_OUTPUT", output, flush=True)
        with RealSenseCamera(args.width, args.height, args.fps, args.serial) as camera:
            (output / "camera.json").write_text(json.dumps(dict(camera.info, pose_frame=args.camera_frame), indent=2))
            frame = camera.read()
            last_frame = frame
            if initializer is not None:
                initializer.warmup(frame)
            else:
                warmup_pose_inputs(estimator, prepared, diameter, frame)
            # Tracking has a distinct six-hypothesis workload and may lazily load
            # its engine. Exercise it now with a discarded synthetic pose.
            depth = np.ones_like(frame.depth_m)
            pose = np.eye(4, dtype=np.float32)
            pose[:3, 3] = np.array([0, 0, 1]) - np.asarray(prepared.bounds).mean(axis=0)
            with ComponentTimer(None, estimator.device) as timer:
                from wapr.tracking import track_many_categories_many_instances
                for _ in range(3):
                    track_many_categories_many_instances(estimator, frame.rgb, depth, frame.K, [{
                        "track_id": 1, "obj_id": 1, "mesh": prepared,
                        "diameter_m": diameter, "pose_4x4": pose}])
            print_timing("WARMUP", f"WAPR tracking: total={timer.elapsed_ms:.2f} ms | "
                         f"avg={timer.elapsed_ms / 3:.2f} ms | calls=3")
            print_timing("WARMUP", f"Startup total={(time.perf_counter() - started) * 1000:.2f} ms")
            session = PoseSession(estimator, prepared, diameter, minimum_score=args.min_score,
                                  initializer=initializer, detect_interval=args.detect_interval)
            state.set_status("INITIALIZED")
            active, number, count, manual_attempts = False, frame.number, 0, 0
            while not state.stop.is_set() and (not args.max_frames or count < args.max_frames):
                frame = camera.read(after_number=number)
                last_frame, number = frame, frame.number
                generation, manual, changed, detection = state.take_commands()
                if changed:
                    tracking.flush()
                    session.reset()
                    active = False
                    if not manual and initializer is None:
                        state.set_status("WARMUP", generation=generation)
                        automatic_args = copy(args)
                        automatic_args.init = "auto"
                        initializer = build_initializer(automatic_args, estimator, mesh, prepared, diameter)
                        initializer.warmup(frame)
                        with state.lock:
                            state.auto_ready = True
                        state.set_status("INITIALIZED", generation=generation)
                session.initializer = None if manual else initializer
                bbox = None
                if detection is not None:
                    tracking.flush()
                    session.reset()
                    selected_frame, bbox = detection
                    if selected_frame is not None:
                        frame = selected_frame
                    active = True
                if active:
                    # Notify before the synchronous search, not after it returns.
                    if session.pose is None:
                        state.set_status("Looking for object", generation=generation)
                        state.publish(draw_pose(frame, mesh, None, "searching", 0.0, show_text=False),
                                      frame=frame, generation=generation)
                    part = "INIT"
                    if bbox is not None:
                        manual_attempts += 1
                        part = "INIT" if manual_attempts == 1 else "REINIT"
                        print_timing(part, f"manual bbox={bbox} | frame={frame.number} | started")
                    status, row, ms = timed_step(session, frame, bbox=bbox, tracking_timing=tracking,
                                                initialization_part=part)
                    with state.lock:
                        superseded = generation != state.generation
                    if superseded:
                        continue
                    if row is not None:
                        state.set_status("TRACKING", generation=generation)
                    elif manual:
                        active = False
                        notice = "初始化失败，请重新手动框选" if bbox is not None else "目标丢失，请重新手动框选"
                        state.set_status("INITIALIZED", notice=notice, generation=generation)
                    else:
                        state.set_status("Looking for object", generation=generation)
                else:
                    status, row, ms = "waiting_roi" if manual else "initialized", None, 0.0
                record = pose_record(frame, status, row, ms, args.mesh, args.camera_frame, args.object_frame)
                record["initialization_mode"] = "roi" if manual else "auto"
                writer.write(record)
                if args.log_poses and count % args.print_every == 0:
                    print("POSE_JSON", json.dumps(record, allow_nan=False), flush=True)
                # One encoded frame is shared by every browser. Encoding is capped
                # by acquisition/processing; HTTP clients never queue RGB-D frames.
                state.publish(draw_pose(frame, mesh, row, status, ms, show_text=False),
                              frame=frame, generation=generation)
                count += 1
    finally:
        tracking.flush()
        try:
            if last_frame is not None:
                writer.write(pose_record(last_frame, "stopped", None, 0.0, args.mesh,
                                         args.camera_frame, args.object_frame))
        finally:
            writer.close()


def simulate(args, state):
    """Explicitly labelled demo to inspect the UI without CUDA or a camera."""
    from .camera import RGBDFrame

    started = time.monotonic()
    if state.stop.wait(2):
        return
    print_timing("WARMUP", "SIMULATED startup=2000.00 ms")
    state.set_status("INITIALIZED")
    active, search_started, tracking_started, last_report = False, None, None, None
    attempts = 0
    while not state.stop.wait(0.08):
        now = time.monotonic()
        generation, manual, changed, detection = state.take_commands()
        if changed:
            active, search_started, tracking_started = False, None, None
        if detection is not None:
            active, search_started, tracking_started = True, now, None
            attempts += 1
            part = "INIT" if attempts == 1 else "REINIT"
            print_timing(part, f"SIMULATED attempt={attempts} | started")
        # Periodic synthetic loss exercises automatic recovery without extra UI.
        if tracking_started is not None and now - tracking_started > 12:
            tracking_started = None
            if manual:
                active, search_started = False, None
                state.set_status("INITIALIZED", notice="目标丢失，请重新手动框选", generation=generation)
            else:
                search_started = now
                attempts += 1
                state.set_status("Looking for object", generation=generation)
                print_timing("REINIT", f"SIMULATED attempt={attempts} | started")
        if active and search_started is not None and now - search_started >= 2:
            state.set_status("TRACKING", generation=generation)
            print_timing("INIT" if attempts == 1 else "REINIT",
                         "SIMULATED search total=2000.00 ms | result=found")
            search_started, tracking_started, last_report = None, now, now
        if tracking_started is not None and now - last_report >= args.timing_interval:
            print_timing("TRACKING", "SIMULATED WAPR avg=24.00 ms | frames=60")
            last_report = now
        image = np.full((480, 640, 3), (25, 29, 34), dtype=np.uint8)
        for x in range(0, 640, 40):
            cv2.line(image, (x, 0), (x, 480), (36, 42, 48), 1)
        for y in range(0, 480, 40):
            cv2.line(image, (0, y), (640, y), (36, 42, 48), 1)
        if tracking_started is not None:
            x = int(250 + 70 * np.sin(now - started))
            cv2.rectangle(image, (x, 170), (x + 110, 290), (117, 216, 117), 2)
            cv2.line(image, (x, 290), (x + 50, 310), (80, 80, 240), 2)
            cv2.line(image, (x, 290), (x, 230), (80, 220, 80), 2)
            cv2.line(image, (x, 290), (x - 35, 315), (240, 130, 80), 2)
        cv2.putText(image, "SIMULATED INPUT", (20, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (135, 145, 155), 1)
        frame = RGBDFrame(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), np.ones((480, 640), np.float32),
                          np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], np.float32),
                          int((now - started) * 1000), 0.0, "simulated", now)
        state.publish(image, frame=frame, generation=generation)


def worker(args, state):
    with timing_listener(state.record_timing):
        try:
            (simulate if args.web_demo else track_camera)(args, state)
        except Exception as error:
            traceback.print_exc()
            state.set_status("ERROR", str(error))
        else:
            state.set_status("STOPPED")


def run(args):
    state = Dashboard(demo=args.web_demo, manual=args.init == "roi",
                      auto_ready=args.web_demo or args.init == "auto")
    server = WebServer((args.web_host, args.web_port), state)
    thread = threading.Thread(target=worker, args=(args, state), name="wapr-inference", daemon=True)
    thread.start()
    print(f"WAPR_WEB http://{args.web_host}:{server.server_port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        state.stop.set()
        server.server_close()
        thread.join(timeout=15)
