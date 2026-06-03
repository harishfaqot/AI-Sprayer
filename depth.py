import cv2
import torch
import numpy as np
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation
)

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained(MODEL)

model = (
    AutoModelForDepthEstimation
    .from_pretrained(MODEL)
    .to(device)
)

model.eval()

if device == "cuda":
    model = model.half()

cap = cv2.VideoCapture("video/grass.mp4")

# LOW resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_count = 0
depth_vis = None

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Run depth every 5 frames
    if frame_count % 5 == 0:

        small = cv2.resize(
            frame,
            (256, 160)
        )

        rgb = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2RGB
        )

        inputs = processor(
            images=rgb,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        if device == "cuda":
            inputs = {
                k: v.half()
                for k, v in inputs.items()
            }

        with torch.no_grad():

            out = model(
                **inputs
            ).predicted_depth

        depth = torch.nn.functional.interpolate(
            out.unsqueeze(1),
            size=frame.shape[:2],
            mode="nearest"
        )

        depth = (
            depth
            .squeeze()
            .cpu()
            .numpy()
        )

        depth = cv2.normalize(
            depth,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        depth_vis = cv2.applyColorMap(
            depth,
            cv2.COLORMAP_INFERNO
        )

    if depth_vis is not None:
        cv2.imshow(
            "Depth",
            depth_vis
        )

    cv2.imshow(
        "Camera",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()