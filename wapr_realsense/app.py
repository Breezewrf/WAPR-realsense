"""RealSense RGB-D to a single object's 6D pose: initialize, then track."""

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml

from wapr.timing import ComponentTimer, TimingCollector, TrackingTiming, print_timing
from wapr.tracking_config import TRACKING_PRESETS

from .camera import RGBDFrame, RealSenseCamera
from .geometry import check_bbox, draw_pose, load_mesh
from .output import PoseWriter, pose_record, save_snapshot
from .tracker import PoseSession
from .initialization import build_initializer


def _flatten_detection_config(config):
    """Translate the detection section to the existing runtime/CLI options."""
    config = dict(config)
    section_name = "groundingdino_sam2_dinov2_detect"
    fields = {"activate": "init", "use_sam2": "detect_use_sam",
              "interval": "detect_interval", "score": "detect_score",
              "backend": "detect_backend", "dino": "detect_dino",
              "grounding": "detect_grounding", "template_cache": "template_cache"}
    for name in fields.values():
        if name in config:
            raise ValueError("Configure %s inside %s" % (name, section_name))
    if section_name not in config:
        return config
    section = config.pop(section_name)
    if not isinstance(section, dict):
        raise ValueError("%s must be a mapping" % section_name)
    for name, value in section.items():
        if name not in fields:
            raise ValueError("Unknown configuration field: %s.%s" % (section_name, name))
        target = fields[name]
        if name == "activate":
            if not isinstance(value, bool):
                raise ValueError("%s.activate must be a boolean" % section_name)
            value = "auto" if value else "roi"
        config[target] = value
    return config


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, help="YAML configuration (.yaml/.yml); CLI flags override its values")
    preliminary, _ = pre.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument("--mesh", type=Path, default=Path("assets/meshes/box/textured_mesh_modified.obj"))
    parser.add_argument("--mesh-unit", choices=["m", "mm", "cm"], default="m")
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--backend", choices=["trt", "torch"], default="trt")
    parser.add_argument("--object-mode", choices=["single", "multi"], default="single",
                        help="TensorRT engine capacity: compact single-object or full multi-object profiles")
    parser.add_argument("--tracking-preset", choices=TRACKING_PRESETS, default="full",
                        help="Tracking workload only: full, balanced (2/1 iterations), fast (1/1), small-motion (SAPR only)")
    reuse_options = parser.add_mutually_exclusive_group()
    reuse_options.add_argument("--reuse-tracking", action="store_true", default=True,
                              help="Reuse identical full-preset branches and crops (default)")
    reuse_options.add_argument("--no-reuse-tracking", dest="reuse_tracking", action="store_false",
                              help="Disable identical-branch/crop reuse for a full-preset baseline comparison")
    parser.add_argument("--init", choices=["roi", "auto"], default="roi",
                        help="Manual ROI (default) or GroundingDINO/SAM2/DINOv2 CAD detection")
    parser.add_argument("--detect-interval", type=float, default=1.0, help="Seconds between unsuccessful auto searches")
    parser.add_argument("--detect-score", type=float, default=0.1, help="2D template-match score gate (0 to 1)")
    parser.add_argument("--detect-backend", choices=["trt", "torch"], default="trt", help="DINOv2 encoder backend")
    parser.add_argument("--detect-dino", choices=["vits14", "vitb14", "vitl14"], default="vitl14")
    parser.add_argument("--detect-grounding", choices=["swint", "swinb"], default="swinb")
    sam_options = parser.add_mutually_exclusive_group()
    sam_options.add_argument("--detect-use-sam", action="store_true", default=True,
                        help="Use SAM2 masks; --no-detect-use-sam uses bounding boxes")
    sam_options.add_argument("--no-detect-use-sam", action="store_false", dest="detect_use_sam",
                             help="Use bounding boxes without SAM2")
    parser.add_argument("--template-cache", type=Path, help="Optional CAD template .pt cache; otherwise kept in GPU memory")
    parser.add_argument("--serial", help="Select a camera by serial number")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15, help="Camera capture rate, not pose output rate")
    parser.add_argument("--min-score", type=float, default=0.5, help="WBPS heuristic gate, not a success probability")
    parser.add_argument("--bbox", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--web", action="store_true", help="Serve the browser demo with automatic or manual initialization")
    parser.add_argument("--web-host", default="127.0.0.1", help="Use 0.0.0.0 for Jetson LAN access")
    parser.add_argument("--web-port", type=int, default=8000)
    parser.add_argument("--web-demo", action="store_true", help="Simulated browser lifecycle; no camera or models")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--log-poses", action="store_true", help="Also print verbose POSE_JSON records; files always contain every frame")
    parser.add_argument("--timing-interval", type=float, default=5.0, help="Seconds between tracking mean-time summaries")
    parser.add_argument("--profile-tracking", action="store_true",
                        help="Collect synchronized WAPR/SAPR/WBPS crop/network timings; adds profiling overhead")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--camera-frame", default="color_optical_frame")
    parser.add_argument("--object-frame", default="target_object")
    parser.add_argument("--check-camera", action="store_true", help="Read one aligned RGB-D frame, without pose models")
    parser.add_argument("--replay", type=Path, help="Offline RGB-D folder for interface validation")
    if preliminary.config is not None:
        try:
            if preliminary.config.suffix.lower() not in (".yaml", ".yml"):
                raise ValueError("Configuration file must use .yaml or .yml")
            content = preliminary.config.read_text(encoding="utf-8")
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Configuration must be a mapping")
            config = _flatten_detection_config(config)
        except (OSError, ValueError, yaml.YAMLError) as error:
            parser.error("Cannot read configuration: %s" % error)
        actions = {action.dest: action for action in parser._actions if action.dest not in ("help", "config")}
        for name, value in config.items():
            if name not in actions:
                parser.error("Unknown configuration field: %s" % name)
            action = actions[name]
            if value is None:
                if name not in ("serial", "bbox", "output", "replay", "template_cache"):
                    parser.error("Configuration field %s cannot be null" % name)
            elif action.type is Path:
                if not isinstance(value, str):
                    parser.error("Configuration path %s must be a string" % name)
                value = Path(value).expanduser()
                if not value.is_absolute():
                    value = (preliminary.config.resolve().parent / value).resolve()
            elif name == "bbox":
                if not isinstance(value, list) or len(value) != 4 or any(type(x) is not int for x in value):
                    parser.error("bbox must be four integers: [x, y, width, height]")
            elif action.type in (int, float):
                expected = (int,) if action.type is int else (int, float)
                if isinstance(value, bool) or not isinstance(value, expected):
                    parser.error("Invalid numeric configuration field: %s" % name)
            elif isinstance(action.default, bool):
                if not isinstance(value, bool):
                    parser.error("Configuration field %s must be a boolean" % name)
            elif not isinstance(value, str):
                parser.error("Configuration field %s must be a string" % name)
            if action.choices and value not in action.choices:
                parser.error("Invalid value for configuration field %s: %s" % (name, value))
            parser.set_defaults(**{name: value})
    args = parser.parse_args(argv)
    if args.web_demo:
        args.web = True
    if not 1 <= args.web_port <= 65535:
        parser.error("--web-port must be between 1 and 65535")
    if args.web:
        if args.replay or args.check_camera or args.bbox is not None:
            parser.error("--web cannot be combined with --replay, --check-camera or --bbox")
    for name in ("width", "height", "fps", "print_every"):
        if getattr(args, name) <= 0:
            parser.error("--%s must be positive" % name.replace("_", "-"))
    if not np.isfinite(args.mesh_scale) or args.mesh_scale <= 0 or not np.isfinite(args.min_score):
        parser.error("Mesh scale must be finite and positive; minimum score must be finite")
    if args.max_frames < 0:
        parser.error("--max-frames must be nonnegative")
    if not args.camera_frame.strip() or not args.object_frame.strip():
        parser.error("Frame names cannot be empty")
    if not np.isfinite(args.detect_interval) or args.detect_interval <= 0:
        parser.error("--detect-interval must be finite and positive")
    if not np.isfinite(args.detect_score) or not 0 <= args.detect_score <= 1:
        parser.error("--detect-score must be between 0 and 1")
    if not np.isfinite(args.timing_interval) or args.timing_interval <= 0:
        parser.error("--timing-interval must be finite and positive")
    if args.init == "auto" and args.bbox is not None:
        parser.error("--bbox is only supported with --init roi")
    if args.init == "roi" and args.no_display and args.bbox is None and not args.replay and not args.check_camera and not args.web:
        parser.error("Headless tracking requires --bbox X Y W H")
    return args

def timed_step(session, frame, bbox=None, tracking_timing=None, initialization_part="INIT"):
    """Measure one pose step, reporting tracking attempts even if they are rejected."""
    tracking = session.pose is not None and bbox is None
    if not tracking and tracking_timing is not None:
        tracking_timing.flush()
    profile = tracking and tracking_timing is not None and tracking_timing.profile
    components = TimingCollector(profile_pose=True) if profile else None
    collection = components if components is not None else nullcontext()
    with collection, ComponentTimer(None, session.estimator.device) as timer:
        status, row = session.step(frame, bbox=bbox)
    if tracking and tracking_timing is not None:
        tracking_timing.add(timer.elapsed_ms, components=components)
        if status == "lost":
            tracking_timing.flush()
    elif bbox is not None:
        print_timing(initialization_part, f"WAPR initialization={timer.elapsed_ms:.2f} ms | result={status}")
    return status, row, timer.elapsed_ms


def replay(args, session, mesh, output, writer, tracking_timing=None):
    """Exercise estimation and one real tracking update with a saved RGB-D frame."""
    folder = args.replay
    bgr = cv2.imread(str(folder / "rgb.png"))
    if bgr is None:
        raise FileNotFoundError(folder / "rgb.png")
    depth = np.load(folder / "depth.npy").astype(np.float32)
    meta_path = folder / "camera.json"
    if not meta_path.is_file():
        meta_path = folder / "meta.json"  # Also accept the existing pose_lmo sample.
    meta = json.loads(meta_path.read_text())
    K = np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]], dtype=np.float32)
    if depth.shape != bgr.shape[:2]:
        raise ValueError("Replay RGB and depth dimensions differ")
    frame = RGBDFrame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth, K, 0, 0.0, "offline", time.monotonic())
    bbox = args.bbox
    if args.init == "roi" and bbox is None:
        mask = cv2.imread(str(folder / "mask.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None or not np.any(mask):
            raise ValueError("ROI replay requires --bbox or a nonempty mask.png")
        bbox = list(cv2.boundingRect(mask))
    for index in range(args.max_frames or 2):
        frame.number = index
        frame.received_monotonic = time.monotonic()
        status, row, ms = timed_step(session, frame, bbox=bbox if index == 0 else None,
                                    tracking_timing=tracking_timing)
        record = pose_record(frame, status, row, ms, args.mesh, args.camera_frame, args.object_frame)
        writer.write(record)
        preview = draw_pose(frame, mesh, row, status, ms)
        if args.log_poses and index % args.print_every == 0:
            print("POSE_JSON", json.dumps(record, allow_nan=False), flush=True)
        if not cv2.imwrite(str(output / "replay_pose.jpg"), preview):
            raise OSError("Could not write replay preview")
    return frame



def main(argv=None):
    args = parse_args(argv)
    if args.web:
        from .web import run
        return run(args)
    if args.check_camera:
        with RealSenseCamera(args.width, args.height, args.fps, args.serial) as camera:
            frame = camera.read()
            depth = frame.depth_m[frame.depth_m > 0]
            print("REALSENSE_READY", json.dumps(dict(camera.info, rgb_shape=list(frame.rgb.shape),
                  valid_depth_fraction=float(np.mean(frame.depth_m > 0)),
                  median_depth_m=None if not depth.size else float(np.median(depth)))), flush=True)
            if args.output is not None:
                args.output.mkdir(parents=True, exist_ok=True)
                np.save(args.output / "depth.npy", frame.depth_m)
                K = frame.K
                metadata = dict(camera.info, fx=float(K[0, 0]), fy=float(K[1, 1]),
                                cx=float(K[0, 2]), cy=float(K[1, 2]))
                (args.output / "camera.json").write_text(json.dumps(metadata, indent=2))
                if not cv2.imwrite(str(args.output / "rgb.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)):
                    raise OSError("Could not save camera-check image")
        return
    if (sys.platform.startswith("linux") and not args.no_display and not args.replay
            and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError(
            "No display connection. Configure Docker/SSH display forwarding or use "
            "--no-display --bbox X Y W H for ROI tracking / 无显示连接，请配置显示转发，"
            "或用 --no-display --bbox X Y W H 无窗口运行")
    import torch
    from wapr import WAPREstimator
    if not torch.cuda.is_available():
        raise RuntimeError("WAPR requires working CUDA PyTorch")
    os.environ["WAPR_BACKEND"] = args.backend
    mesh, diameter = load_mesh(args.mesh, args.mesh_unit, args.mesh_scale)
    print("LIVE_MESH", {"file": str(args.mesh), "extents_m": mesh.extents.tolist(),
                        "diameter_m": diameter, "unit": args.mesh_unit, "scale": args.mesh_scale}, flush=True)
    estimator = WAPREstimator(device="cuda:0", object_mode=args.object_mode)
    prepared = estimator.prepare_meshes([mesh])[0]
    output = args.output or Path.cwd() / "outputs" / "wapr_realsense" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    writer = PoseWriter(output)
    (output / "session.json").write_text(json.dumps({"mesh": str(args.mesh.resolve()), "mesh_unit": args.mesh_unit,
        "mesh_scale": args.mesh_scale, "diameter_m": diameter, "backend": args.backend,
        "object_mode": args.object_mode, "init": args.init,
        "profile_tracking": args.profile_tracking,
        "tracking_preset": args.tracking_preset, "reuse_tracking": args.reuse_tracking,
        "minimum_score": args.min_score, "camera_frame": args.camera_frame, "object_frame": args.object_frame,
        "automatic_detection": None if args.init == "roi" else {
            "interval_s": args.detect_interval, "score_2d_min": args.detect_score,
            "dino": args.detect_dino, "grounding": args.detect_grounding, "backend": args.detect_backend,
            "use_sam": args.detect_use_sam,
            "template_cache": None if args.template_cache is None else str(args.template_cache.resolve())},
        "pose_convention": "original_scaled_mesh_to_color_optical_frame"}, indent=2))
    print("LIVE_OUTPUT", output, flush=True)
    last_frame = None
    tracking_timing = TrackingTiming(args.timing_interval, profile=args.profile_tracking)
    try:
        initializer = build_initializer(args, estimator, mesh, prepared, diameter)
        session = PoseSession(estimator, prepared, diameter, minimum_score=args.min_score,
                              initializer=initializer, detect_interval=args.detect_interval,
                              tracking_preset=args.tracking_preset, reuse_tracking=args.reuse_tracking)
        if args.replay:
            last_frame = replay(args, session, mesh, output, writer, tracking_timing)
            return
        with RealSenseCamera(args.width, args.height, args.fps, args.serial) as camera:
            (output / "camera.json").write_text(json.dumps(dict(camera.info, pose_frame=args.camera_frame), indent=2))
            print("REALSENSE_READY", json.dumps(camera.info), flush=True)
            count, number = 0, None
            pending_roi = args.init == "roi"
            if not args.no_display:
                cv2.namedWindow("WAPR-realsense", cv2.WINDOW_AUTOSIZE)
            while args.max_frames == 0 or count < args.max_frames:
                frame = camera.read(after_number=number)
                last_frame, number = frame, frame.number
                bbox = None
                if pending_roi:
                    if args.bbox is not None:
                        bbox = args.bbox
                    else:
                        frozen = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                        bbox = cv2.selectROI("Select object; Enter confirms, c cancels", frozen,
                                             showCrosshair=True, fromCenter=False)
                        cv2.destroyWindow("Select object; Enter confirms, c cancels")
                        if min(bbox[2:]) <= 0:
                            bbox = None
                    pending_roi = False
                    if bbox is not None:
                        try:
                            bbox = check_bbox(bbox, frame.depth_m)
                        except ValueError as error:
                            print("ROI_REJECTED", str(error), flush=True)
                            bbox = None
                status, row, processing_ms = timed_step(session, frame, bbox=bbox, tracking_timing=tracking_timing)
                record = pose_record(frame, status, row, processing_ms, args.mesh, args.camera_frame, args.object_frame)
                writer.write(record)
                if args.log_poses and count % args.print_every == 0:
                    print("POSE_JSON", json.dumps(record, allow_nan=False), flush=True)
                count += 1
                if not args.no_display:
                    preview = draw_pose(frame, mesh, row, status, processing_ms)
                    cv2.imshow("WAPR-realsense", preview)
                    key = cv2.waitKey(1) & 0xff
                    if key in (27, ord("q")) or cv2.getWindowProperty("WAPR-realsense", cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key == ord("r"):
                        session.reset()
                        pending_roi = args.init == "roi"
                    if key == ord("s"):
                        save_snapshot(output, frame, preview, record)
    finally:
        tracking_timing.flush()
        if last_frame is not None:
            writer.write(pose_record(last_frame, "stopped", None, 0.0, args.mesh, args.camera_frame, args.object_frame))
        writer.close()
        if not args.no_display and not args.replay:
            cv2.destroyAllWindows()



def entrypoint():
    try:
        main()
    except KeyboardInterrupt:
        pass
