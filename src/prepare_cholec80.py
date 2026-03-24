"""
Prepare Cholec80 dataset for YOLOv8 training.

Cholec80 provides:
  - 80 videos (.mp4) in videos/
  - Frame-level instrument presence labels in instrument_annotations/ (binary per-frame CSVs)
  - NO bounding boxes — instrument presence only

This script:
  1. Extracts frames from videos at a configurable FPS
  2. Converts binary presence labels to pseudo-labels (full-frame boxes) as a baseline
     NOTE: For proper detection, you need bounding box annotations.
     CholecT50 (subset of Cholec80) provides triplet annotations but still no boxes.
     Consider EndoVis 2017/2018 for box/segmentation annotations instead.

Usage:
    python src/prepare_cholec80.py --videos data/raw/cholec80/videos \
        --annotations data/raw/cholec80/instrument_annotations \
        --output data/cholec80 --fps 1
"""

import argparse
import csv
import cv2
import os
import shutil
from pathlib import Path


INSTRUMENT_COLS = [
    "Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"
]
CLASS_MAP = {name: i for i, name in enumerate(INSTRUMENT_COLS)}


def extract_frames(video_path: Path, output_dir: Path, fps: float = 1.0):
    """Extract frames from video at given FPS."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(video_fps / fps))
    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            fname = output_dir / f"{video_path.stem}_f{frame_idx:06d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved += 1
        frame_idx += 1
    cap.release()
    return saved, frame_idx


def annotation_to_yolo(row: dict, img_w: int, img_h: int) -> list[str]:
    """
    Convert presence labels to full-frame bounding boxes (baseline only).
    Returns list of YOLO label lines: 'class cx cy w h'
    """
    lines = []
    for col, cls_id in CLASS_MAP.items():
        if int(row.get(col, 0)) == 1:
            lines.append(f"{cls_id} 0.5 0.5 1.0 1.0")
    return lines


def process(videos_dir: Path, ann_dir: Path, output_dir: Path, fps: float):
    videos = sorted(videos_dir.glob("*.mp4"))
    print(f"Found {len(videos)} videos.")

    splits = {"train": videos[:64], "val": videos[64:72], "test": videos[72:]}

    for split, split_videos in splits.items():
        img_dir = output_dir / split / "images"
        lbl_dir = output_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for video in split_videos:
            ann_file = ann_dir / f"{video.stem}-tool.txt"
            if not ann_file.exists():
                print(f"  WARNING: annotation not found for {video.stem}, skipping.")
                continue

            # Load annotations indexed by frame number
            annotations = {}
            with open(ann_file, newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    frame = int(row["Frame"])
                    annotations[frame] = row

            # Extract frames and write labels
            cap = cv2.VideoCapture(str(video))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            interval = max(1, int(video_fps / fps))
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % interval == 0 and frame_idx in annotations:
                    stem = f"{video.stem}_f{frame_idx:06d}"
                    cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame)
                    label_lines = annotation_to_yolo(annotations[frame_idx], img_w, img_h)
                    with open(lbl_dir / f"{stem}.txt", "w") as lf:
                        lf.write("\n".join(label_lines))
                frame_idx += 1
            cap.release()

        print(f"  {split}: processed {len(split_videos)} videos")

    print(f"\nDone. Dataset saved to {output_dir}")
    print("NOTE: These are full-frame presence labels, NOT bounding boxes.")
    print("For proper detection training, use EndoVis 2017/2018 which provides box annotations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/cholec80"))
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second to extract (1.0 = 1 fps)")
    args = parser.parse_args()
    process(args.videos, args.annotations, args.output, args.fps)
