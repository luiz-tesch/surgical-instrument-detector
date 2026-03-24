"""
Run inference with a trained YOLOv8 surgical instrument detector.
Supports images, video files, and webcam/stream sources.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Detect surgical instruments")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained weights (best.pt)")
    parser.add_argument("--source", type=str, required=True,
                        help="Input: image path, video path, folder, or '0' for webcam")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save annotated output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
        project="runs/predict",
    )

    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes):
            print(f"{r.path}: {len(boxes)} detection(s)")
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                label = model.names[cls_id]
                print(f"  {label}: {conf:.2f}")
