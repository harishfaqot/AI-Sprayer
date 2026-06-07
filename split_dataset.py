"""
YOLO Dataset Train/Val Splitter
Splits existing train images+labels into train and val sets.

Usage:
    python split_dataset.py --images path/to/images/train --labels path/to/labels/train
    python split_dataset.py --images path/to/images/train --labels path/to/labels/train --ratio 0.2 --seed 42
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def split_dataset(images_dir, labels_dir, val_ratio=0.2, seed=42):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    # Output dirs (sibling of train, named val)
    images_val_dir = images_dir.parent.parent / "images" / "val"
    labels_val_dir = labels_dir.parent.parent / "labels" / "val"
    images_train_dir = images_dir.parent.parent / "images" / "train"
    labels_train_dir = labels_dir.parent.parent / "labels" / "train"

    images_val_dir.mkdir(parents=True, exist_ok=True)
    labels_val_dir.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # Collect all label files
    label_files = sorted([f for f in labels_dir.iterdir() if f.suffix == ".txt"])

    if not label_files:
        print(f"[ERROR] No .txt label files found in: {labels_dir}")
        return

    # Match each label to its image
    paired = []
    missing_images = []

    for lbl in label_files:
        img_found = None
        for ext in IMG_EXTS:
            candidate = images_dir / (lbl.stem + ext)
            if candidate.exists():
                img_found = candidate
                break
        if img_found:
            paired.append((img_found, lbl))
        else:
            missing_images.append(lbl.stem)

    if missing_images:
        print(f"[WARNING] {len(missing_images)} labels have no matching image, skipping:")
        for name in missing_images[:10]:
            print(f"  - {name}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")

    if not paired:
        print("[ERROR] No valid image-label pairs found.")
        return

    # Shuffle and split
    random.seed(seed)
    random.shuffle(paired)

    val_count = max(1, int(len(paired) * val_ratio))
    val_pairs  = paired[:val_count]
    train_pairs = paired[val_count:]

    print(f"\nTotal pairs     : {len(paired)}")
    print(f"Train           : {len(train_pairs)}")
    print(f"Val             : {val_count} ({val_ratio*100:.0f}%)")
    print(f"\nMoving val files to:")
    print(f"  Images : {images_val_dir}")
    print(f"  Labels : {labels_val_dir}\n")

    # Move val files
    for img, lbl in val_pairs:
        shutil.move(str(img), images_val_dir / img.name)
        shutil.move(str(lbl), labels_val_dir / lbl.name)

    print(f"[DONE] Split complete.")
    print(f"  Train: {len(train_pairs)} samples in {images_train_dir}")
    print(f"  Val  : {len(val_pairs)} samples in {images_val_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split YOLO train dataset into train/val")
    parser.add_argument("--images", required=True, help="Path to images/train folder")
    parser.add_argument("--labels", required=True, help="Path to labels/train folder")
    parser.add_argument("--ratio",  type=float, default=0.2, help="Val ratio (default: 0.2)")
    parser.add_argument("--seed",   type=int,   default=42,  help="Random seed (default: 42)")
    args = parser.parse_args()

    split_dataset(args.images, args.labels, args.ratio, args.seed)