"""
depth.py — Monocular Depth Estimation (ready to run, no training needed)
=========================================================================
Model   : MiDaS v2.1 Small  (~14 MB ONNX, pretrained on 10 datasets)
Hardware: Raspberry Pi 3  →  ~1-2 FPS at 256x256
          Laptop CPU      →  ~15-25 FPS

Install:
    pip install onnxruntime opencv-python numpy requests

Run:
    python depth.py                      # webcam
    python depth.py --image photo.jpg    # single image
    python depth.py --video clip.mp4     # video file
    python depth.py --cam 0 --size 128   # smaller input, faster

Keys:  q = quit  |  s = save frame  |  c = cycle colormap
"""

import argparse
import os
import sys
import time
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

# ── Model config ─────────────────────────────────────────────
MODEL_URL  = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
MODEL_FILE = "midas_small.onnx"
INPUT_SIZE = 256          # MiDaS small default (square)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CMAPS      = [cv2.COLORMAP_INFERNO, cv2.COLORMAP_PLASMA,
              cv2.COLORMAP_MAGMA,   cv2.COLORMAP_JET]
CMAP_NAMES = ["INFERNO", "PLASMA", "MAGMA", "JET"]


# ── Auto-download model ───────────────────────────────────────
def download_model(path=MODEL_FILE):
    if os.path.exists(path):
        return
    print(f"Downloading MiDaS small (~14 MB) ...")
    urllib.request.urlretrieve(MODEL_URL, path,
        reporthook=lambda b, bs, total: print(
            f"  {min(b*bs, total)/1e6:.1f}/{total/1e6:.1f} MB", end="\r"))
    print(f"\nSaved → {path}")


# ── Pre/post processing ───────────────────────────────────────
def preprocess(frame_bgr, size):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x   = rgb.astype(np.float32) / 255.0
    x   = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[np.newaxis]       # (1, 3, H, W)


def colorize(depth_map, out_w, out_h, cmap):
    d = depth_map[0] if depth_map.ndim == 3 else depth_map
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)  # normalize to [0,1]
    d_u8 = (d * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(d_u8, cmap)
    return cv2.resize(coloured, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


# ── HUD ──────────────────────────────────────────────────────
def hud(img, fps, cmap_name, size):
    def txt(text, pos, color=(255,255,255)):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color,   1, cv2.LINE_AA)
    txt(f"FPS: {fps:.1f}",        (8, 22))
    txt(f"Net: {size}x{size}",    (8, 42))
    txt(f"Map: {cmap_name}",      (8, 62))
    txt("q:quit  s:save  c:cmap", (8, img.shape[0] - 8))


# ── Inference loop ────────────────────────────────────────────
def run(args):
    download_model(args.model)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4          # use all 4 RPi3 cores
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.model, opts,
                                providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    size = args.size

    # ── Image mode ──
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Cannot read: {args.image}")
        h, w = frame.shape[:2]
        t0    = time.perf_counter()
        depth = sess.run(None, {inp_name: preprocess(frame, size)})[0]
        ms    = (time.perf_counter() - t0) * 1000
        print(f"Inference: {ms:.1f} ms")
        coloured = colorize(depth, w, h, CMAPS[0])
        out = np.hstack([frame, coloured])
        hud(out, 1000/ms, CMAP_NAMES[0], size)
        save_path = args.save or "depth_out.png"
        cv2.imwrite(save_path, out)
        print(f"Saved → {save_path}")
        if not args.headless:
            cv2.imshow("Depth", out)
            cv2.waitKey(0)
        return

    # ── Camera / video mode ──
    src = args.video if args.video else args.cam
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Cannot open source: {src}")

    cmap_idx  = 0
    fps_hist  = []
    save_n    = 0
    writer    = None

    ok, sample = cap.read()
    if not ok:
        sys.exit("Cannot read from source.")
    h, w = sample.shape[:2]

    if args.save and args.video is None and args.image is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args.save, fourcc, 10, (w * 2, h))
        print(f"Saving → {args.save}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("Running… press q to quit.")

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            if args.video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        depth    = sess.run(None, {inp_name: preprocess(frame, size)})[0]
        coloured = colorize(depth, w, h, CMAPS[cmap_idx])
        canvas   = np.hstack([frame, coloured])

        elapsed = time.perf_counter() - t0
        fps_hist.append(1.0 / max(elapsed, 1e-6))
        if len(fps_hist) > 15:
            fps_hist.pop(0)
        fps = sum(fps_hist) / len(fps_hist)

        if not args.headless:
            hud(canvas, fps, CMAP_NAMES[cmap_idx], size)
            cv2.imshow("original | depth", canvas)
            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'): break
            elif key == ord('c'): cmap_idx = (cmap_idx + 1) % len(CMAPS)
            elif key == ord('s'):
                fname = f"depth_{save_n:04d}.png"
                cv2.imwrite(fname, canvas); save_n += 1
                print(f"Saved {fname}")
        else:
            print(f"\rFPS: {fps:.1f}", end="", flush=True)

        if writer:
            writer.write(canvas)

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print("\nDone.")


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default=MODEL_FILE, help="ONNX model path")
    p.add_argument("--cam",      type=int, default=0, help="Camera index")
    p.add_argument("--image",    help="Input image file")
    p.add_argument("--video",    help="Input video file")
    p.add_argument("--size",     type=int, default=INPUT_SIZE,
                   help="Network input size (default 256; use 128 for faster RPi)")
    p.add_argument("--save",     help="Save output file")
    p.add_argument("--headless", action="store_true", help="No display window")
    run(p.parse_args())