"""
Prepare EndoVis 2017 (Zenodo preprocessed) for YOLOv8 detection.

Dataset format (from endovis2017.zip):
  endovis2017/
    instrument_type_mapping.json   pixel_value → class_name
    train/
      image/  seq_N_frameXXX.bmp   RGB 512x512
      label/  seq_N_frameXXX.bmp   grayscale, pixel value = class ID (0=bg, 1-7=instrument)
    val1/ ... val10/
      image/
      label/

This script:
  1. Extracts the zip directly (no need to pre-extract)
  2. Converts segmentation masks → YOLO bounding boxes
  3. Splits val folds into val/test
  4. Writes JPGs + YOLO .txt labels to data/endovis2017/

Usage:
    python src/prepare_endovis.py --zip endovis2017.zip --output data/endovis2017

    # Use only named instruments (skip class 7 "Other"):
    python src/prepare_endovis.py --zip endovis2017.zip --output data/endovis2017 --skip-other
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# Pixel value in mask → YOLO class index (0-indexed, skipping background=0)
# Class 7 "Other" is excluded from YOLO by default (--skip-other flag)
MASK_TO_YOLO = {
    1: 0,  # Bipolar Forceps
    2: 1,  # Prograsp Forceps
    3: 2,  # Large Needle Driver
    4: 3,  # Vessel Sealer
    5: 4,  # Grasping Retractor
    6: 5,  # Monopolar Curved Scissors
    7: 6,  # Other (included unless --skip-other)
}


def mask_to_yolo_boxes(mask: np.ndarray, skip_other: bool) -> list[str]:
    """
    Convert a segmentation mask to YOLO bounding box lines.
    Each unique non-zero pixel value → one box per connected component.
    Returns list of 'cls cx cy w h' strings (normalized 0-1).
    """
    h, w = mask.shape
    lines = []
    for pixel_val, yolo_cls in MASK_TO_YOLO.items():
        if skip_other and pixel_val == 7:
            continue
        binary = (mask == pixel_val).astype(np.uint8)
        if binary.sum() == 0:
            continue
        # Find connected components so multi-instrument frames get separate boxes
        num_labels, labels = cv2.connectedComponents(binary)
        for comp_id in range(1, num_labels):
            comp_mask = (labels == comp_id).astype(np.uint8)
            ys, xs = np.where(comp_mask)
            if len(xs) < 10:  # skip tiny noise regions
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            cx = ((x_min + x_max) / 2) / w
            cy = ((y_min + y_max) / 2) / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def process(zip_path: Path, output_dir: Path, val_folds: list[int], test_folds: list[int],
            skip_other: bool):
    print(f"Reading {zip_path} ...")

    # Assign val folds to split
    fold_to_split = {}
    for i in range(1, 11):
        if i in test_folds:
            fold_to_split[f"val{i}"] = "test"
        elif i in val_folds:
            fold_to_split[f"val{i}"] = "val"
        else:
            fold_to_split[f"val{i}"] = "val"  # remaining folds → val

    # Create output dirs
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "test": 0, "no_instruments": 0}

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

        # Only process image files (not labels)
        image_files = [
            n for n in all_names
            if n.endswith(".bmp") and "/image/" in n
        ]
        print(f"Found {len(image_files)} image frames.")

        for img_path in image_files:
            # Determine split
            parts = img_path.split("/")  # e.g. ['endovis2017', 'train', 'image', 'seq_1_frame000.bmp']
            folder = parts[1]  # 'train', 'val1', ..., 'val10'
            filename = parts[-1]
            stem = Path(filename).stem

            if folder == "train":
                split = "train"
            else:
                split = fold_to_split.get(folder, "val")

            # Corresponding label path
            label_path = img_path.replace("/image/", "/label/")
            if label_path not in zf.namelist():
                print(f"  WARNING: no label for {img_path}, skipping.")
                continue

            # Load image
            img_bytes = zf.read(img_path)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is None:
                # Fallback: PIL
                img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # Load mask
            mask_bytes = zf.read(label_path)
            mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
            mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask_pil = Image.open(io.BytesIO(mask_bytes)).convert("L")
                mask = np.array(mask_pil)

            # Convert mask → YOLO boxes
            label_lines = mask_to_yolo_boxes(mask, skip_other)

            if not label_lines:
                stats["no_instruments"] += 1
                # Still save the image with empty label (negative sample)

            # Save image as JPG
            out_img = output_dir / split / "images" / f"{folder}_{stem}.jpg"
            cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Save label
            out_lbl = output_dir / split / "labels" / f"{folder}_{stem}.txt"
            with open(out_lbl, "w") as f:
                f.write("\n".join(label_lines))

            stats[split] += 1

    print("\n=== Done ===")
    print(f"  train: {stats['train']} frames")
    print(f"  val:   {stats['val']} frames")
    print(f"  test:  {stats['test']} frames")
    print(f"  frames with no instrument annotations: {stats['no_instruments']}")
    print(f"\nOutput: {output_dir}")
    print("Next: python src/train.py --data configs/data_endovis.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("endovis2017.zip"),
                        help="Path to endovis2017.zip")
    parser.add_argument("--output", type=Path, default=Path("data/endovis2017"),
                        help="Output directory for prepared dataset")
    parser.add_argument("--val-folds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8],
                        help="Val fold numbers to use for validation (default: 1-8)")
    parser.add_argument("--test-folds", type=int, nargs="+", default=[9, 10],
                        help="Val fold numbers to use for test (default: 9-10)")
    parser.add_argument("--skip-other", action="store_true", default=True,
                        help="Skip class 7 'Other' (default: True)")
    args = parser.parse_args()

    process(args.zip, args.output, args.val_folds, args.test_folds, args.skip_other)
