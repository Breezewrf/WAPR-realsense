"""Compatibility launcher. Preferred: python -m wapr_realsense --config configs/box.json."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wapr_realsense.app import entrypoint

if __name__ == "__main__":
    entrypoint()
