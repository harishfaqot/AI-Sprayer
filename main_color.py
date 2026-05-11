import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import os

# =========================
# VIDEO
# =========================
video_folder = "video"
video_list = [os.path.join(video_folder, f)
              for f in os.listdir(video_folder)
              if f.endswith((".mp4", ".avi", ".mov"))]
video_index = 0
cap = cv2.VideoCapture(video_list[video_index])

# =========================
# TKINTER SETUP
# =========================
root = tk.Tk()
root.title("Grass Detection Control Panel")
root.geometry("1000x700")

# =========================
# VARIABLES
# =========================
kernel_size = tk.IntVar(value=5)
min_area = tk.IntVar(value=300)
roi_start = tk.IntVar(value=30)
smooth_val = tk.IntVar(value=30)

smooth_coverage = 0
smooth_spread = 0

# =========================
# LAYOUT (2x2 GRID)
# =========================
main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

video_frame = tk.Frame(main_frame)
video_frame.pack(fill="both", expand=True)

# --- ORIGINAL ---
frame1 = tk.Frame(video_frame)
frame1.grid(row=0, column=0, sticky="nsew")

tk.Label(frame1, text="ORIGINAL", font=("Arial", 12, "bold")).pack()
label_original = tk.Label(frame1, bg="black")
label_original.pack(fill="both", expand=True)

# --- MASK ---
frame2 = tk.Frame(video_frame)
frame2.grid(row=0, column=1, sticky="nsew")

tk.Label(frame2, text="MASK", font=("Arial", 12, "bold")).pack()
label_mask = tk.Label(frame2, bg="black")
label_mask.pack(fill="both", expand=True)

# --- MASKED RGB ---
frame3 = tk.Frame(video_frame)
frame3.grid(row=1, column=0, sticky="nsew")

tk.Label(frame3, text="MASKED RGB", font=("Arial", 12, "bold")).pack()
label_masked_rgb = tk.Label(frame3, bg="black")
label_masked_rgb.pack(fill="both", expand=True)

# --- RESULT ---
frame4 = tk.Frame(video_frame)
frame4.grid(row=1, column=1, sticky="nsew")

tk.Label(frame4, text="RESULT", font=("Arial", 12, "bold")).pack()
label_overlay = tk.Label(frame4, bg="black")
label_overlay.pack(fill="both", expand=True)

# Grid scaling
for i in range(2):
    video_frame.columnconfigure(i, weight=1)
    video_frame.rowconfigure(i, weight=1)

# =========================
# SLIDERS
# =========================
slider_frame = tk.Frame(root)
slider_frame.pack(side="bottom", fill="x")

def add_slider(label, var, row, col, minv, maxv):
    tk.Label(slider_frame, text=label).grid(row=row, column=col*2, sticky="w")
    tk.Scale(slider_frame, from_=minv, to=maxv,
             orient="horizontal", variable=var,
             length=140).grid(row=row, column=col*2+1)

add_slider("Kernel", kernel_size, 0, 0, 1, 20)
add_slider("MinArea", min_area, 1, 0, 0, 5000)
add_slider("ROI %", roi_start, 2, 0, 0, 100)
add_slider("Smooth %", smooth_val, 3, 0, 0, 100)

# =========================
# RESIZE
# =========================
def resize_aspect(img, w, h):
    ih, iw = img.shape[:2]
    scale = min(w/iw, h/ih)
    return cv2.resize(img, (int(iw*scale), int(ih*scale)))

def show(label, img):
    w = label.winfo_width()
    h = label.winfo_height()

    if w > 10 and h > 10:
        img = resize_aspect(img, w, h)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(img)

    label.imgtk = imgtk
    label.configure(image=imgtk)

# =========================
# LOOP
# =========================
def update_frame():
    global smooth_coverage, smooth_spread, video_index, cap

    ret, frame = cap.read()
    if not ret:
        video_index += 1

        if video_index >= len(video_list):
            video_index = 0  # loop back to first

        cap.release()
        cap = cv2.VideoCapture(video_list[video_index])

        root.after(30, update_frame)
        return

    # =========================
    # PREPROCESS
    # =========================
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # =========================
    # ADAPTIVE HSV
    # =========================
    v_mean = np.mean(hsv[:, :, 2])

    if v_mean < 80:
        lower = np.array([30, 30, 30])
        upper = np.array([90, 255, 255])
    elif v_mean > 180:
        lower = np.array([40, 60, 60])
        upper = np.array([85, 255, 255])
    else:
        lower = np.array([35, 40, 40])
        upper = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # =========================
    # MORPHOLOGY
    # =========================
    k = max(1, kernel_size.get())
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # =========================
    # REMOVE NOISE
    # =========================
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)

    for c in contours:
        if cv2.contourArea(c) > min_area.get():
            cv2.drawContours(clean, [c], -1, 255, -1)

    mask = clean

    # =========================
    # MASKED RGB
    # =========================
    masked_rgb = cv2.bitwise_and(frame, frame, mask=mask)

    # =========================
    # ROI
    # =========================
    h, w = mask.shape
    roi_y = int(h * roi_start.get() / 100)
    roi = mask[roi_y:h, :]

    coverage = cv2.countNonZero(roi) / roi.size if roi.size else 0

    coords = np.column_stack(np.where(roi > 0))
    spread = (coords[:, 1].max() - coords[:, 1].min()) / w if coords.size else 0

    # =========================
    # SMOOTHING
    # =========================
    alpha = smooth_val.get() / 100
    smooth_coverage = alpha * coverage + (1 - alpha) * smooth_coverage
    smooth_spread   = alpha * spread + (1 - alpha) * smooth_spread

    pump = int(smooth_coverage * 100)
    nozzle = int(smooth_spread * 100)

    # =========================
    # VISUALIZATION
    # =========================
    overlay = frame.copy()
    overlay[mask > 0] = (0, 255, 0)
    masked = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # DARKEN IGNORED AREA (ROI VISUAL)
    dark = masked.copy()
    dark[:roi_y, :] = (30, 30, 30)
    masked = cv2.addWeighted(dark, 0.4, masked, 0.6, 0)

    # ROI LINE
    cv2.line(masked, (0, roi_y), (w, roi_y), (255, 0, 0), 3)
    cv2.putText(masked, "ROI START", (10, roi_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Nozzle
    cx, cy = w // 2, h // 2
    max_radius = min(w, h) // 2
    radius = int((nozzle / 100) * max_radius)

    color = (0, int(pump * 2.55), int(255 - pump * 2.55))

    overlay_circle = masked.copy()
    cv2.circle(overlay_circle, (cx, cy), radius, color, -1)

    masked = cv2.addWeighted(overlay_circle, pump / 100.0, masked, 1 - pump / 100.0, 0)
    cv2.circle(masked, (cx, cy), radius, color, 2)
    cv2.circle(masked, (cx, cy), 5, (0, 0, 255), -1)

    # Text
    lines = [
        f"Coverage: {smooth_coverage:.2f}",
        f"Spread  : {smooth_spread:.2f}",
        f"Pump    : {pump}%",
        f"Nozzle  : {nozzle}%"
    ]

    for i, t in enumerate(lines):
        cv2.putText(masked, t, (10, 40 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    cv2.putText(frame,
        f"Video: {os.path.basename(video_list[video_index])}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,1,
        (255, 255, 255),2)
    # =========================
    # DISPLAY
    # =========================
    show(label_original, frame)
    show(label_mask, mask_bgr)
    show(label_masked_rgb, masked_rgb)
    show(label_overlay, masked)

    root.after(10, update_frame)

# =========================
# RUN
# =========================
update_frame()
root.mainloop()
cap.release()