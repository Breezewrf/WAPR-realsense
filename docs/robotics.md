# Robotics output contract / 机器人接口

## Transform and frames

`pose_4x4` is `T_camera_object`, mapping points from the original mesh frame (after
unit conversion and `mesh_scale`) to the rectified RealSense color optical frame.
WAPR's internal mesh centering is undone in its returned pose.

```text
p_camera = R @ p_object + t_m
T_base_object = T_base_camera @ T_camera_object
```

Camera X points right, Y down and Z forward. Rotation is a 3×3 matrix;
`quaternion_xyzw` is `[x, y, z, w]`. Translation and model coordinates are meters.
Set `camera_frame` and `object_frame` in the config to match the consuming system.
These labels name frames; they do not perform TF lookup or extrinsic calibration.

## Files and timestamps

| Field | Meaning |
| --- | --- |
| `valid` | Whether this frame has an accepted candidate, not a ground-truth correctness claim |
| `status` | `starting`, `waiting_roi`, `initialized_roi`, `tracking`, `lost` or `stopped` |
| `pose_4x4`, `R`, `t_m`, `quaternion_xyzw` | Pose representations, null when invalid |
| `camera_frame`, `object_frame` | Configured coordinate-frame labels |
| `score_6d` | WBPS score, not calibrated success probability |
| `frame_number` | SDK color-frame number corresponding to this result |
| `camera_timestamp_ms` | SDK color timestamp; interpret together with its clock domain |
| `camera_timestamp_domain` | SDK clock-domain name; not necessarily Unix time |
| `host_time_ns` | Host Unix time when forming the result, not exposure time |
| `processing_ms` | Synchronized estimation/tracking time, excluding capture wait, drawing and file writes |
| `frame_age_ms` | Host elapsed time from receiving the SDK frameset through result formation, including rectification and interaction |

The producer flushes one record per processed frame to `poses.jsonl`, and atomically
replaces `latest.json`. Startup replaces an earlier run's latest pose with an invalid record.
Normal/error exit attempts a final invalid `stopped` record. Forced termination cannot guarantee this.
No valid pose is carried forward after a rejected update. A high-scoring drifting pose can still be wrong;
the application has no automatic object-presence verification or lost-target recovery.

## Local reader

```bash
python examples/read_pose.py --input outputs/box/latest.json --max-age-ms 500
```

`read_latest_pose()` rejects invalid, malformed, future-dated or stale records.
Its freshness bound is `time_since_record_ms + frame_age_ms <= max_age_ms` on the local host.
This includes camera receive-to-record delay but does not measure exposure-to-host delay or synchronize
different computers. Publish across machines through a transport with a defined clock policy.
It is a freshness check, not a motion-quality or collision-safety filter.

The example only emits a new JSON result when the frame changes, and emits invalid status on loss/staleness.
There is no robot command output, ROS publisher or automatic actuation in this repository.

## Calibrated base-frame pose

Copy `configs/camera_to_base.example.json`, replace `T_base_camera` with your measured rigid transform,
check `camera_frame`, and set `calibrated=true`. The example template is deliberately marked uncalibrated.

```bash
python examples/read_pose.py --input outputs/box/latest.json \
  --camera-to-base /path/to/measured_camera_to_base.json
```

For a moving wrist camera, `T_base_camera` changes with robot joint state and must correspond to the
observation time. The static JSON example is appropriate only for a fixed camera, or one fixed robot pose.
The application does not infer hand-eye calibration, interpolate robot state or associate clocks for you.

## Python integration

```python
from wapr_realsense.output import read_latest_pose, transform_to_base

record = read_latest_pose("outputs/box/latest.json", max_age_ms=500)
if record is not None:
    T_camera_object = record["pose_4x4"]
    # Supply a measured fixed-camera calibration to transform_to_base when needed.
```

Camera adaptation is `wapr_realsense.camera.RealSenseCamera`; `PoseSession` reuses a prepared
WAPR estimator and mesh. Inference remains in the caller's thread because the renderer owns a GL context.
The acquisition thread replaces its latest frame, so slow processing drops intermediate camera frames
instead of building a queue of old observations.
