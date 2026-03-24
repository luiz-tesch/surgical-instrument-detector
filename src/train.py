"""
Train YOLOv8 on surgical instrument dataset.
Windows: must run inside if __name__ == '__main__' (multiprocessing requirement).
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 surgical instrument detector")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset YAML (e.g. configs/data_cholec80.yaml)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Base model weights (yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size — reduce to 8 if VRAM OOM on GTX 1660 Ti")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: '0' for GPU, 'cpu' for CPU")
    parser.add_argument("--name", type=str, default="surgical_yolov8",
                        help="Run name (saved under runs/train/)")
    return parser.parse_args()


def train(args):
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,
        patience=20,
        project=str(Path.cwd() / "runs" / "train"),
        name=args.name,
        plots=True,
        save=True,
        exist_ok=True,
    )
    print(f"\nTraining complete. Best weights: runs/train/{args.name}/weights/best.pt")
    return results


if __name__ == "__main__":
    args = parse_args()
    train(args)
