"""
YOLO Dataset Cleaner - Delete unlabeled images
Deletes images that don't have a matching label file.

Usage:
    python clean_unlabeled.py --images path/to/images --labels path/to/labels
    python clean_unlabeled.py --images path/to/images --labels path/to/labels --dry-run
    python clean_unlabeled.py --images path/to/images --labels path/to/labels --extensions .jpg .png
"""

import os
import argparse
from pathlib import Path


def clean_unlabeled_images(images_dir, labels_dir, dry_run=True, img_extensions=None):
    """
    Delete images that don't have corresponding label files.
    
    Args:
        images_dir: Path to images folder
        labels_dir: Path to labels folder
        dry_run: If True, only show what would be deleted without actually deleting
        img_extensions: Set of image extensions to consider (default: common YOLO formats)
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    # Supported image extensions
    if img_extensions is None:
        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    else:
        IMG_EXTS = set(ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                      for ext in img_extensions)
    
    # Check if directories exist
    if not images_dir.exists():
        print(f"[ERROR] Images directory not found: {images_dir}")
        return
    if not labels_dir.exists():
        print(f"[ERROR] Labels directory not found: {labels_dir}")
        return
    
    # Get all label files (without extension)
    label_stems = set()
    for label_file in labels_dir.iterdir():
        if label_file.suffix == ".txt":
            label_stems.add(label_file.stem)
    
    if not label_stems:
        print(f"[WARNING] No label files found in: {labels_dir}")
        print("Nothing to compare with. Exiting.")
        return
    
    print(f"\nFound {len(label_stems)} label files")
    print(f"Scanning {images_dir} for images...")
    
    # Find images without labels
    unlabeled_images = []
    total_images = 0
    
    for img_file in images_dir.iterdir():
        if img_file.suffix.lower() in IMG_EXTS:
            total_images += 1
            if img_file.stem not in label_stems:
                unlabeled_images.append(img_file)
    
    print(f"\nTotal images found: {total_images}")
    print(f"Images with labels: {total_images - len(unlabeled_images)}")
    print(f"Images WITHOUT labels: {len(unlabeled_images)}")
    
    if not unlabeled_images:
        print("\n[INFO] No unlabeled images found. Nothing to delete.")
        return
    
    # Show first few unlabeled images
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Unlabeled images to delete:")
    for img in unlabeled_images[:20]:
        print(f"  - {img.name}")
    if len(unlabeled_images) > 20:
        print(f"  ... and {len(unlabeled_images) - 20} more")
    
    # Delete or show deletion
    if dry_run:
        print(f"\n[DRY RUN] Would delete {len(unlabeled_images)} images")
        print("Run with --no-dry-run to actually delete")
    else:
        confirm = input(f"\n[CONFIRM] Delete {len(unlabeled_images)} unlabeled images? (y/N): ")
        if confirm.lower() == 'y':
            deleted_count = 0
            for img in unlabeled_images:
                try:
                    img.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"[ERROR] Failed to delete {img.name}: {e}")
            print(f"\n[DONE] Deleted {deleted_count} unlabeled images")
        else:
            print("Operation cancelled.")


def main():
    parser = argparse.ArgumentParser(
        description="Delete images that don't have corresponding label files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview only)
  python clean_unlabeled.py --images dataset/images/train --labels dataset/labels/train
  
  # Actually delete
  python clean_unlabeled.py --images dataset/images/train --labels dataset/labels/train --no-dry-run
  
  # Delete only JPG and PNG files
  python clean_unlabeled.py --images dataset/images --labels dataset/labels --extensions .jpg .png
  python clean_unlabeled.py --images Datasets/Grass_Segmentation/images/train --labels Datasets/Grass_Segmentation/labels/train --no-dry-run
        """
    )
    
    parser.add_argument("--images", required=True, 
                       help="Path to images folder")
    parser.add_argument("--labels", required=True, 
                       help="Path to labels folder")
    parser.add_argument("--no-dry-run", action="store_true", 
                       help="Actually delete files (default is dry-run)")
    parser.add_argument("--extensions", nargs="+", default=None,
                       help="Image extensions to consider (e.g., .jpg .png .jpeg)")
    
    args = parser.parse_args()
    
    # Dry run is True by default, set to False only if --no-dry-run is specified
    dry_run = not args.no_dry_run
    
    clean_unlabeled_images(
        images_dir=args.images,
        labels_dir=args.labels,
        dry_run=dry_run,
        img_extensions=args.extensions
    )


if __name__ == "__main__":
    main()