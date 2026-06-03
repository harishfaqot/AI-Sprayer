from ultralytics import YOLO
import cv2

# Load model on GPU
model = YOLO("yolo26x-seg.pt")
# model.to("cuda")

cap = cv2.VideoCapture("people.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run on GPU
    results = model(
        frame,
        device="cpu"
    )

    annotated = results[0].plot()

    cv2.imshow("Segmentation", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()