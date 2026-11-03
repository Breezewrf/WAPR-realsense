"""Read fresh accepted poses, optionally expressing them in a calibrated robot base."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wapr_realsense.output import read_latest_pose, transform_to_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/box/latest.json"))
    parser.add_argument("--max-age-ms", type=float, default=500)
    parser.add_argument("--camera-to-base", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    calibration = None if args.camera_to_base is None else json.loads(args.camera_to_base.read_text())
    last_key = object()
    while True:
        record = read_latest_pose(args.input, args.max_age_ms)
        key = None if record is None else (record["host_time_ns"], record["frame_number"])
        if args.once or key != last_key:
            payload = {"valid": False, "status": "unavailable_or_stale"} if record is None else record
            if record is not None and calibration is not None:
                payload = transform_to_base(record, calibration)
            print(json.dumps(payload, allow_nan=False), flush=True)
            last_key = key
        if args.once:
            break
        time.sleep(0.02)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
