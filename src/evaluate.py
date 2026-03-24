"""
Evaluate a trained YOLOv8 model on the test set.
Reports mAP50, mAP50-95, precision, recall per class.
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate surgical instrument detector")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights (e.g. runs/train/surgical_yolov8/weights/best.pt)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset YAML")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
    )

    print("\n=== Evaluation Results ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision:{metrics.box.mp:.4f}")
    print(f"Recall:   {metrics.box.mr:.4f}")
