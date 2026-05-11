import cv2
import numpy as np

# =========================
# PARAMETERS
# =========================
ALPHA = 0.3  # smoothing factor

smooth_coverage = 0
smooth_spread = 0

# =========================
# TRACKBAR SETUP
# =========================
def nothing(x):
    pass

cv2.namedWindow("Controls")

cv2.createTrackbar("H Min", "Controls", 35, 179, nothing)
cv2.createTrackbar("H Max", "Controls", 85, 179, nothing)
cv2.createTrackbar("S Min", "Controls", 40, 255, nothing)
cv2.createTrackbar("S Max", "Controls", 255, 255, nothing)
cv2.createTrackbar("V Min", "Controls", 40, 255, nothing)
cv2.createTrackbar("V Max", "Controls", 255, 255, nothing)

# =========================
# VIDEO INPUT
# =========================
cap = cv2.VideoCapture("video/grass.mp4")

if not cap.isOpened():
    print("Error: Cannot access video")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        # Restart video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # =========================
    # GET HSV VALUES FROM SLIDER
    # =========================
    h_min = cv2.getTrackbarPos("H Min", "Controls")
    h_max = cv2.getTrackbarPos("H Max", "Controls")
    s_min = cv2.getTrackbarPos("S Min", "Controls")
    s_max = cv2.getTrackbarPos("S Max", "Controls")
    v_min = cv2.getTrackbarPos("V Min", "Controls")
    v_max = cv2.getTrackbarPos("V Max", "Controls")

    LOWER_GREEN = np.array([h_min, s_min, v_min])
    UPPER_GREEN = np.array([h_max, s_max, v_max])

    # =========================
    # PROCESSING
    # =========================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # =========================
    # COVERAGE
    # =========================
    total_pixels = mask.size
    grass_pixels = cv2.countNonZero(mask)
    coverage = grass_pixels / total_pixels

    # =========================
    # SPREAD
    # =========================
    coords = np.column_stack(np.where(mask > 0))

    if coords.size > 0:
        x_coords = coords[:, 1]
        spread_pixels = x_coords.max() - x_coords.min()
        spread_ratio = spread_pixels / frame.shape[1]
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

    if coords.size > 0:
        x_min = int(x_coords.min())
        x_max = int(x_coords.max())
        y_mid = frame.shape[0] // 2
        cv2.line(display, (x_min, y_mid), (x_max, y_mid), (0, 0, 255), 2)

    cv2.putText(display, f"Coverage: {smooth_coverage:.2f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(display, f"Spread: {smooth_spread:.2f}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    pump_pwm = int(30 + 70 * smooth_coverage)
    nozzle_angle = int(30 + 60 * smooth_spread)

    cv2.putText(display, f"Pump PWM: {pump_pwm}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(display, f"Nozzle: {nozzle_angle}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # =========================
    # SHOW WINDOWS
    # =========================
    cv2.imshow("Original", frame)   # <-- REAL VIDEO (no overlay)
    cv2.imshow("Processed", display)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()