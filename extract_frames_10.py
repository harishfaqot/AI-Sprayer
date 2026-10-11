import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "video"
OUTPUT_DIR = VIDEO_DIR / "frames_all"
FRAME_SKIP = 10
VIDEO_NAMES = ["real1.mp4", "real2.mp4", "real3.mp4"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

saved_total = 0

for video_name in VIDEO_NAMES:
    video_path = VIDEO_DIR / video_name
    if not video_path.exists():
        print(f"Skipping missing file: {video_path}")
        continue

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        continue

    frame_index = 0
    saved_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % FRAME_SKIP == 0:
            frame_path = OUTPUT_DIR / f"{video_path.stem}_frame_{saved_frames:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_frames += 1
            saved_total += 1

        frame_index += 1

    cap.release()
    print(f"Saved {saved_frames} images from {video_name} to {OUTPUT_DIR}")

print(f"Done. Total saved: {saved_total} images in {OUTPUT_DIR}")
