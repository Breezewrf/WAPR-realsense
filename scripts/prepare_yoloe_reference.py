"""Encode a reference image and xyxy boxes into a reusable YOLOE prompt NPZ."""
import argparse
from pathlib import Path
import sys

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wapr_realsense.yoloe_reference import encode_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='yoloe-26s-seg.pt')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--boxes', type=Path, required=True, help='YAML list of xyxy boxes in reference-image pixels')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--imgsz', type=int, default=960)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error('Cannot read reference image')
    from ultralytics import YOLOE
    model = YOLOE(args.model)
    try:
        encode_reference(model, image, yaml.safe_load(args.boxes.read_text()), args.output,
                         device=args.device, imgsz=args.imgsz)
    except ValueError as error:
        parser.error(str(error))
    print('YOLOE_REFERENCE_READY', args.output)


if __name__ == '__main__':
    main()
