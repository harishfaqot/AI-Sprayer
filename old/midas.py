import cv2
import numpy as np
import onnxruntime as ort
import time

MODEL = "models/midas_v21_small_256.onnx"

# Load model
session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# Camera
# cap = cv2.VideoCapture("video/grass2.mp4")
cap = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

prev = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    # resize
    img = cv2.resize(frame, (256, 256))

    # BGR → RGB
    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # normalize
    img = img.astype(np.float32) / 255.0

    img = (
        img
        - np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )
    ) / np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    # NHWC → NCHW
    blob = img.transpose(2, 0, 1)

    blob = np.expand_dims(
        blob,
        axis=0
    ).astype(np.float32)

    # inference
    depth = session.run(
        None,
        {input_name: blob}
    )[0]

    depth = depth.squeeze()

    # resize back
    depth = cv2.resize(
        depth,
        (w, h)
    )

    # normalize
    depth = cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    depth_color = cv2.applyColorMap(
        depth,
        cv2.COLORMAP_INFERNO
    )

    now = time.time()

    fps = 1 / (now - prev)

    prev = now

    cv2.putText(
        depth_color,
        f"FPS {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Camera",
        frame
    )

    cv2.imshow(
        "Depth",
        depth_color
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()