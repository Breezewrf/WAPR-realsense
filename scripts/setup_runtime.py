"""Prepare only WAPR's pose runtime and weights; no detector or dataset downloads."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Inspect without installation, build or downloads")
    parser.add_argument("--sample", action="store_true", help="Also fetch the tiny LM-O frame for offline replay")
    args = parser.parse_args()
    from wapr.bootstrap import check_runtime, prepare_runtime
    if args.check:
        status = check_runtime()
        print(json.dumps(status, indent=2))
        return 1 if status.get("status") == "blocked" else 0
    prepare_runtime()
    from wapr.download_assets import check_and_fetch_pack
    check_and_fetch_pack("wapr_sapr_wbps")
    if args.sample:
        check_and_fetch_pack("pose_lmo")
    print("POSE_RUNTIME_READY; build engines with: python -m wapr.export_engines", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
