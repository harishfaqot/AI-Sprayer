import cv2
import numpy as np

# =========================
# PARAMETERS
# =========================
ALPHA = 0.3  # smoothing

smooth_coverage = 0
smooth_spread = 0

# =========================
# VIDEO INPUT
# =========================
cap = cv2.VideoCapture("grass.mp4")

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # =========================
    # PREPROCESS (CRITICAL)
    # =========================
    frame = cv2.resize(frame, (320, 240))

    # 1. Blur (reduce noise)
    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    # 2. Convert to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # 3. Adaptive threshold based on brightness
    v_mean = np.mean(hsv[:, :, 2])

    # Adjust thresholds dynamically
    if v_mean < 80:  # dark
        lower = np.array([30, 30, 30])
        upper = np.array([90, 255, 255])
    elif v_mean > 180:  # very bright
        lower = np.array([40, 60, 60])
        upper = np.array([85, 255, 255])
    else:  # normal
        lower = np.array([35, 40, 40])
        upper = np.array([85, 255, 255])

    # =========================
    # SEGMENTATION
    # =========================
    mask = cv2.inRange(hsv, lower, upper)

    # =========================
    # MORPHOLOGY (CLEANING)
    # =========================
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # =========================
    # REMOVE SMALL NOISE (CONTOUR FILTER)
    # =========================
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean_mask = np.zeros_like(mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:  # tune this
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    mask = clean_mask

    # =========================
    # ROI (IGNORE SKY / TOP AREA)
    # =========================
    h, w = mask.shape
    roi = mask[int(h*0.3):h, :]  # ignore top 30%

    # =========================
    # COVERAGE
    # =========================
    total_pixels = roi.size
    grass_pixels = cv2.countNonZero(roi)

    coverage = grass_pixels / total_pixels

    # =========================
    # SPREAD
    # =========================
    coords = np.column_stack(np.where(roi > 0))

    if coords.size > 0:
        x_coords = coords[:, 1]
        spread_pixels = x_coords.max() - x_coords.min()
        spread_ratio = spread_pixels / w
    else:
        spread_ratio = 0

    # =========================
    # SMOOTHING
    # =========================
    smooth_coverage = ALPHA * coverage + (1 - ALPHA) * smooth_coverage
    smooth_spread   = ALPHA * spread_ratio + (1 - ALPHA) * smooth_spread

    # =========================
    # VISUALIZATION
    # =========================
    overlay = frame.copy()
    overlay[mask > 0] = (0, 255, 0)
    display = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Draw spread
    if coords.size > 0:
        x_min = int(x_coords.min())
        x_max = int(x_coords.max())
        y_mid = int(h * 0.7)
        cv2.line(display, (x_min, y_mid), (x_max, y_mid), (0, 0, 255), 2)

    # Text
    cv2.putText(display, f"Coverage: {smooth_coverage:.2f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(display, f"Spread: {smooth_spread:.2f}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # =========================
    # CONTROL OUTPUT
    # =========================
    pump_pwm = int(30 + 70 * smooth_coverage)
    nozzle_angle = int(30 + 60 * smooth_spread)

    cv2.putText(display, f"Pump: {pump_pwm}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(display, f"Nozzle: {nozzle_angle}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # =========================
    # SHOW
    # =========================
    cv2.imshow("Robust Detection", display)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()