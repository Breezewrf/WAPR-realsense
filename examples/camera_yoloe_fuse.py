"""Collect views of one target, fuse YOLOE visual prompts and export prompt_embeddings.npz.

Adapted from the operator's Ultralytics examples/camera_yoloe_fuse.py workflow.
S: select a reference ROI; F: encode, export and preview detection;
U: undo; R: clear collected views; M: toggle masks; Q/Esc: quit.
"""

import argparse
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wapr_realsense.yoloe_reference import encode_reference


def select_roi(frame):
    name = "Select reference - ENTER accept, C cancel"
    try:
        x, y, w, h = cv2.selectROI(name, frame, showCrosshair=True, fromCenter=False)
    finally:
        cv2.destroyWindow(name)
    if w <= 1 or h <= 1:
        return None
    return int(x), int(y), int(x + w), int(y + h)


def extract_sample(frame, bbox, context=.12):
    """Keep a little context and the ROI in crop-local xyxy coordinates."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    padx, pady = max(2, round((x2-x1)*context)), max(2, round((y2-y1)*context))
    left, top = max(0, x1-padx), max(0, y1-pady)
    right, bottom = min(w, x2+padx), min(h, y2+pady)
    return frame[top:bottom, left:right].copy(), (x1-left, y1-top, x2-left, y2-top)


def make_reference_mosaic(samples, tile_size=320):
    """Use the same 320 px tiles and 10 px margin as the original collection workflow."""
    if not samples:
        raise ValueError('Add at least one reference with S first')
    if tile_size < 32:
        raise ValueError('Tile size must be at least 32')
    cols = math.ceil(math.sqrt(len(samples)))
    rows = math.ceil(len(samples)/cols)
    canvas = np.full((rows*tile_size, cols*tile_size, 3), 114, np.uint8)
    boxes = []
    for i, (crop, local_bbox) in enumerate(samples):
        h, w = crop.shape[:2]
        scale = min((tile_size-20)/w, (tile_size-20)/h)
        nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        ox = (i % cols)*tile_size + (tile_size-nw)//2
        oy = (i // cols)*tile_size + (tile_size-nh)//2
        canvas[oy:oy+nh, ox:ox+nw] = resized
        x1, y1, x2, y2 = local_bbox
        boxes.append([ox+x1*scale, oy+y1*scale, ox+x2*scale, oy+y2*scale])
    return canvas, np.asarray(boxes, np.float32)


def export_references(model, samples, args):
    """F both encodes and exports; the tracker only needs the resulting NPZ."""
    mosaic, boxes = make_reference_mosaic(samples, args.tile_size)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / 'prompt_embeddings.npz'
    encode_reference(model, mosaic, boxes, output, device=args.device,
                     imgsz=args.prompt_imgsz, conf=args.conf)
    if not cv2.imwrite(str(args.output / 'reference_mosaic.png'), mosaic):
        raise OSError('Could not save reference mosaic')
    (args.output / 'boxes.xyxy.yaml').write_text(yaml.safe_dump(boxes.tolist()), encoding='utf-8')
    debug = mosaic.copy()
    for i, coords in enumerate(boxes):
        x1, y1, x2, y2 = np.rint(coords).astype(int)
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug, str(i+1), (x1, max(16, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
    if not cv2.imwrite(str(args.output / 'reference_boxes.jpg'), debug):
        raise OSError('Could not save reference visualization')
    print('YOLOE_REFERENCE_READY %s | views=%d' % (output.resolve(), len(samples)), flush=True)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', default='yoloe-26s-seg.pt')
    parser.add_argument('--conf', type=float, default=.05)
    parser.add_argument('--imgsz', type=int, default=640, help='Live detection size')
    parser.add_argument('--prompt-imgsz', type=int, default=960, help='Offline reference encoding size')
    parser.add_argument('--device', default='0', help='GPU index or cpu')
    parser.add_argument('--output', type=Path, default=Path('outputs/yoloe_reference'))
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--tile-size', type=int, default=320)
    parser.add_argument('--max-samples', type=int, default=9)
    parser.add_argument('--context', type=float, default=.12)
    args = parser.parse_args(argv)
    if not np.isfinite(args.conf) or not 0 <= args.conf <= 1:
        parser.error('--conf must be between 0 and 1')
    if not np.isfinite(args.context) or not 0 <= args.context <= 1:
        parser.error('--context must be between 0 and 1')
    if args.tile_size < 32 or min(args.max_samples, args.width, args.height, args.imgsz, args.prompt_imgsz) <= 0:
        parser.error('Sizes and sample count must be positive; --tile-size must be >= 32')
    if '-seg-pf' in args.model:
        parser.error('Use a visual-prompt *-seg.pt checkpoint, not *-seg-pf.pt')
    return args


def main(argv=None):
    args = parse_args(argv)
    from ultralytics import YOLOE
    print('Loading YOLOE...', flush=True)
    model = YOLOE(args.model)
    if not callable(getattr(model, 'save_prompt_embeddings', None)):
        raise RuntimeError('Install requirements-yoloe.txt: YOLOE NPZ export is required')
    cap = cv2.VideoCapture(args.camera)
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open camera %s' % args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        window = 'YOLOE references - S:add F:export U:undo R:clear M:masks Q:quit'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        samples, ready, masks = [], False, True
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera read failed')
            display = frame.copy()
            if ready:
                # Reference features are already cached; never re-encode them per frame.
                result = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf,
                                       retina_masks=True, device=args.device, verbose=False)[0]
                display = result.plot(img=display, boxes=True, masks=masks, labels=False, conf=True)
            status = ('DETECTING / NPZ EXPORTED' if ready else 'COLLECT / press F to encode and export')
            cv2.putText(display, '%s | views=%d/%d' % (status, len(samples), args.max_samples),
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('s'):
                if len(samples) >= args.max_samples:
                    print('Maximum references reached; U to undo or F to export', flush=True)
                    continue
                box = select_roi(frame.copy())
                if box is not None:
                    samples.append(extract_sample(frame, box, args.context))
                    ready = False
                    print('Added sample %d/%d: bbox=%s' % (len(samples), args.max_samples, box), flush=True)
            elif key == ord('f'):
                if not samples:
                    print('Press S to collect a reference first', flush=True)
                    continue
                ready = False
                export_references(model, samples, args)
                ready = True
            elif key == ord('u'):
                if samples:
                    samples.pop()
                    ready = False
                    print('Reference removed. Press F to re-encode and export.', flush=True)
            elif key == ord('r'):
                samples.clear()
                ready = False
                print('Collection cleared; existing exported NPZ is kept until F exports again.', flush=True)
            elif key == ord('m'):
                masks = not masks
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
