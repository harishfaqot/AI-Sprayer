"""
grass_segment.py — Real-time grass detection with pretrained SegFormer-B0
==========================================================================
Model   : SegFormer-B0 fine-tuned on ADE20K (150 classes, includes grass)
Source  : nvidia/segformer-b0-finetuned-ade-512-512 (HuggingFace, ~15 MB)
Hardware: Raspberry Pi 3  → ~1-2 FPS at 256px input
          Laptop CPU      → ~5-10 FPS

Install:
    pip install transformers torch opencv-python numpy

Run:
    python grass_segment.py                     # webcam
    python grass_segment.py --image photo.jpg   # single image
    python grass_segment.py --video clip.mp4    # video file
    python grass_segment.py --size 256          # network input size (smaller=faster)
    python grass_segment.py --headless          # no display (RPi without monitor)

Keys:  q = quit  |  s = save frame
"""

import argparse
import sys
import time

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ── Config ────────────────────────────────────────────────────
MODEL_ID    = "nvidia/segformer-b0-finetuned-ade-512-512"
GRASS_CLASS = 9      # ADE20K class index for "grass"
FIELD_CLASS = 29     # ADE20K "field" — also green ground, bonus
PLANT_CLASS = 17     # ADE20K "plant" — optional

# Which classes to highlight as "grass area"
GRASS_CLASSES = {GRASS_CLASS, FIELD_CLASS}

# Overlay color for grass mask (BGR)
GRASS_COLOR  = (0, 220, 60)    # bright green
OVERLAY_ALPHA = 0.45


# ── Load model (auto-downloads on first run, ~15 MB) ─────────
def load_model():
    print(f"Loading model: {MODEL_ID}")
    print("(Auto-downloads ~15 MB on first run...)")
    processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
    model     = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model.eval()
    print("Model ready.")
    return processor, model


# ── Inference → grass mask ────────────────────────────────────
def get_grass_mask(frame_bgr, processor, model, input_size):
    """
    Returns a binary mask (H, W) uint8 where 255 = grass detected.
    """
    h, w = frame_bgr.shape[:2]

    # Resize for faster inference
    small = cv2.resize(frame_bgr, (input_size, input_size))
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    inputs = processor(images=rgb, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Upsample logits to input_size
    logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=False,
    )
    seg_map = logits.argmax(dim=1)[0].numpy()  # (input_size, input_size)

    # Build grass mask
    grass_mask = np.zeros_like(seg_map, dtype=np.uint8)
    for cls in GRASS_CLASSES:
        grass_mask[seg_map == cls] = 255

    # Resize mask back to original frame size
    grass_mask = cv2.resize(grass_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return grass_mask


# ── Overlay mask on frame ─────────────────────────────────────
def overlay(frame, mask, color=GRASS_COLOR, alpha=OVERLAY_ALPHA):
    out = frame.copy()
    colored = np.zeros_like(frame)
    colored[:] = color
    where = mask > 0
    out[where] = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)[where]
    # Draw contour for crispness
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def grass_percent(mask):
    return 100.0 * np.count_nonzero(mask) / mask.size


# ── HUD ──────────────────────────────────────────────────────
def hud(img, fps, pct, size):
    def txt(text, pos, color=(255, 255, 255)):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color,   1, cv2.LINE_AA)
    color = (0, 220, 60) if pct > 5 else (255, 255, 255)
    txt(f"FPS: {fps:.1f}",           (8, 22))
    txt(f"Net: {size}px",            (8, 42))
    txt(f"Grass: {pct:.1f}%", (8, 62), color)
    txt("q:quit  s:save",            (8, img.shape[0] - 8))


# ── Image mode ───────────────────────────────────────────────
def run_image(args, processor, model):
    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"Cannot read: {args.image}")

    t0   = time.perf_counter()
    mask = get_grass_mask(frame, processor, model, args.size)
    ms   = (time.perf_counter() - t0) * 1000

    out  = overlay(frame, mask)
    pct  = grass_percent(mask)
    hud(out, 1000/ms, pct, args.size)

    print(f"Inference: {ms:.1f} ms | Grass: {pct:.1f}%")
    save_path = args.save or "grass_out.png"
    cv2.imwrite(save_path, out)
    print(f"Saved → {save_path}")

    if not args.headless:
        cv2.imshow("Grass Detection", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ── Camera / video mode ──────────────────────────────────────
def run_camera(args, processor, model):
    src = args.video if args.video else args.cam
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {src}")

    fps_hist = []
    save_n   = 0
    print("Running… press q to quit.")

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            if args.video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        mask = get_grass_mask(frame, processor, model, args.size)
        out  = overlay(frame, mask)
        pct  = grass_percent(mask)

        elapsed = time.perf_counter() - t0
        fps_hist.append(1.0 / max(elapsed, 1e-6))
        if len(fps_hist) > 10:
            fps_hist.pop(0)
        fps = sum(fps_hist) / len(fps_hist)

        if not args.headless:
            hud(out, fps, pct, args.size)
            cv2.imshow("Grass Detection", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f"grass_{save_n:04d}.png"
                cv2.imwrite(fname, out)
                print(f"Saved {fname}")
                save_n += 1
        else:
            print(f"\rFPS: {fps:.1f}  Grass: {pct:.1f}%", end="", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone.")


# ── Main ──────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cam",      type=int,   default=0)
    p.add_argument("--image",    type=str,   default=None)
    p.add_argument("--video",    type=str,   default=None)
    p.add_argument("--size",     type=int,   default=256,
                   help="Network input size (128=faster, 512=better quality)")
    p.add_argument("--save",     type=str,   default=None)
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()

    processor, model = load_model()

    if args.image:
        run_image(args, processor, model)
    else:
        run_camera(args, processor, model)


if __name__ == "__main__":
    main()