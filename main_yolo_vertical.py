"""
AI Sprayer - Grass Detection & Spray Control (TOP-DOWN VIEW)
=============================================================
Uses YOLO segmentation to detect grass from a top-down camera,
tracks the grass position relative to image center (crosshair + line),
computes grass area to set nozzle 0-100%, and controls pump ON/OFF.

CONFIG: Edit the variables below, then run.
"""

import cv2
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import time
import math

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_PATH   = r"runs\segment\train-8\weights\best.pt"
SOURCE       = "video/vgrass1.mp4"   # 0 = webcam, or "path/to/video.mp4"
DEVICE       = "cpu"                # "cpu" or "cuda" or 0
CONF         = 0.4
IMGSZ        = 640

# Grass area thresholds (total mask/bbox area in pixels, top-down view)
AREA_MIN     = 800      # below this total area → pump OFF
AREA_MAX     = 60000    # at/above this total area → nozzle 100%

# Scale conversion (provided): 100 px = 1 cm
PX_PER_CM    = 20.0
PX2_PER_CM2  = PX_PER_CM * PX_PER_CM

# Temporal smoothing / debounce (prevents flicker from brief dropouts/spikes)
DETECTION_HOLD_SEC = 0.18   # keep detection true for this long after a miss
AREA_SMOOTH_TAU    = 0.22   # seconds, EMA for area used in control decisions
NOZZLE_SMOOTH_TAU  = 0.18   # seconds, EMA for nozzle percentage display/output
# ────────────────────────────────────────────────────────────────────────────

# ── Colors (modern light) ────────────────────────────────────────────────────
BG          = "#FFFFFF"   # pure white
PANEL_BG    = "#F8F9FA"   # near-white panel
CARD_BG     = "#F1F3F5"   # light gray card
ACCENT_GRN  = "#16A34A"   # modern green
ACCENT_YLW  = "#D97706"   # amber
ACCENT_RED  = "#DC2626"   # clean red
ACCENT_BLU  = "#2563EB"   # vivid blue
TEXT_PRI    = "#111827"   # near-black
TEXT_SEC    = "#6B7280"   # gray-500
BORDER      = "#E5E7EB"   # gray-200


class AISprayer:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Sprayer — Grass Detection System (Top-Down)")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        # State
        self.running     = False
        self.cap         = None
        self.model       = None
        self.frame_count = 0
        self.fps_display = 0
        self.last_time   = time.time()

        # Detection state
        self.grass_detected  = False
        self.pump_on         = False
        self.nozzle_pct      = 0
        self.nozzle_pct_float = 0.0
        self.grass_area_px   = 0
        self.grass_area_cm2  = 0.0
        self.control_area_px = 0
        self.detect_count    = 0

        # Temporal filtering state
        self.last_loop_time     = time.time()
        self.last_detect_time   = 0.0
        self.smoothed_area_px   = 0.0

        # Tracking state (top-down position tracking)
        self.target_cx       = None   # target centroid x (pixels, in frame coords)
        self.target_cy       = None   # target centroid y (pixels, in frame coords)
        self.offset_x_px     = 0      # target_cx - frame center x
        self.offset_y_px     = 0      # target_cy - frame center y
        self.offset_dist_px  = 0      # euclidean distance from center to target

        # Source toggle
        self.source_var = tk.StringVar(value=SOURCE if isinstance(SOURCE, str) and not str(SOURCE).isdigit() else "webcam")

        self._build_ui()
        self._load_model()

    # ── UI BUILD ────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left: video feed
        left = tk.Frame(self.root, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        # Header
        hdr = tk.Frame(left, bg=BG)
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        tk.Label(hdr, text="⬡ AI SPRAYER", bg=BG, fg=ACCENT_GRN,
                 font=("Courier", 18, "bold")).pack(side="left")
        tk.Label(hdr, text="TOP-DOWN GRASS TRACKING", bg=BG, fg=TEXT_SEC,
                 font=("Courier", 9)).pack(side="left", padx=(10, 0), pady=(6, 0))

        self.fps_lbl = tk.Label(hdr, text="FPS: --", bg=BG, fg=TEXT_SEC,
                                font=("Courier", 9))
        self.fps_lbl.pack(side="right")

        # Video canvas
        self.canvas = tk.Canvas(left, bg="#000000", highlightthickness=1,
                                highlightbackground=BORDER)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas_text = self.canvas.create_text(
            320, 240, text="Loading model...", fill=TEXT_SEC,
            font=("Courier", 14))

        # Source controls
        ctrl = tk.Frame(left, bg=BG)
        ctrl.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        tk.Label(ctrl, text="SOURCE:", bg=BG, fg=TEXT_SEC,
                 font=("Courier", 8)).pack(side="left")

        self.src_entry = tk.Entry(ctrl, bg=CARD_BG, fg=TEXT_PRI,
                                  insertbackground=TEXT_PRI,
                                  font=("Courier", 10), bd=0,
                                  highlightthickness=1,
                                  highlightbackground=BORDER,
                                  highlightcolor=ACCENT_GRN, width=28)
        self.src_entry.insert(0, str(SOURCE))
        self.src_entry.pack(side="left", padx=8, ipady=4)

        self.btn_start = self._btn(ctrl, "▶  START", ACCENT_GRN, self._start)
        self.btn_start.pack(side="left", padx=4)

        self.btn_stop = self._btn(ctrl, "■  STOP", ACCENT_RED, self._stop)
        self.btn_stop.pack(side="left", padx=4)
        self.btn_stop.config(state="disabled")

        # Right: control panel
        right = tk.Frame(self.root, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        right.columnconfigure(0, weight=1)

        self._build_status_card(right, row=0)
        self._build_pump_card(right, row=1)
        self._build_nozzle_card(right, row=2)
        self._build_tracking_card(right, row=3)
        self._build_stats_card(right, row=4)
        self._build_log(right, row=5)

    def _btn(self, parent, text, color, cmd):
        return tk.Button(parent, text=text, command=cmd,
                         bg=color, fg="#000000",
                         font=("Courier", 9, "bold"),
                         bd=0, padx=12, pady=5, cursor="hand2",
                         activebackground=color)

    def _card(self, parent, title, row):
        f = tk.Frame(parent, bg=CARD_BG, padx=12, pady=10)
        f.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        f.columnconfigure(0, weight=1)
        tk.Label(f, text=title, bg=CARD_BG, fg=TEXT_SEC,
                 font=("Courier", 8)).grid(row=0, column=0, sticky="w")
        sep = tk.Frame(f, bg=BORDER, height=1)
        sep.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        return f

    def _build_status_card(self, parent, row):
        card = self._card(parent, "▸ DETECTION STATUS", row)

        self.status_dot = tk.Canvas(card, width=14, height=14, bg=CARD_BG,
                                    highlightthickness=0)
        self.status_dot.grid(row=2, column=0, sticky="w")
        self.status_circle = self.status_dot.create_oval(2, 2, 12, 12,
                                                          fill=TEXT_SEC, outline="")

        self.status_lbl = tk.Label(card, text="NO GRASS DETECTED",
                                   bg=CARD_BG, fg=TEXT_SEC,
                                   font=("Courier", 11, "bold"))
        self.status_lbl.grid(row=2, column=0, sticky="w", padx=20)

        self.detect_count_lbl = tk.Label(card, text="Detections: 0",
                                          bg=CARD_BG, fg=TEXT_SEC,
                                          font=("Courier", 8))
        self.detect_count_lbl.grid(row=3, column=0, sticky="w", pady=(4, 0))

    def _build_pump_card(self, parent, row):
        card = self._card(parent, "▸ PUMP CONTROL", row)

        self.pump_canvas = tk.Canvas(card, width=200, height=60,
                                     bg=CARD_BG, highlightthickness=0)
        self.pump_canvas.grid(row=2, column=0, sticky="ew")
        self._draw_pump(False)

    def _draw_pump(self, on):
        c = self.pump_canvas
        c.delete("all")
        color  = ACCENT_GRN if on else ACCENT_RED
        label  = "ON" if on else "OFF"
        bg_col = "#0d2918" if on else "#2d0f0f"

        c.create_rectangle(0, 0, 200, 60, fill=bg_col, outline="")
        c.create_rectangle(10, 10, 190, 50, fill=color, outline="")
        c.create_text(100, 30, text=f"PUMP  {label}",
                      fill="#000000", font=("Courier", 16, "bold"))

    def _build_nozzle_card(self, parent, row):
        card = self._card(parent, "▸ NOZZLE OPENING", row)

        # Gauge canvas
        self.gauge_canvas = tk.Canvas(card, width=200, height=120,
                                       bg=CARD_BG, highlightthickness=0)
        self.gauge_canvas.grid(row=2, column=0)
        self._draw_gauge(0)

        self.nozzle_lbl = tk.Label(card, text="0%", bg=CARD_BG,
                                    fg=TEXT_PRI, font=("Courier", 22, "bold"))
        self.nozzle_lbl.grid(row=3, column=0)

        self.nozzle_bar_frame = tk.Frame(card, bg=BORDER, height=8)
        self.nozzle_bar_frame.grid(row=4, column=0, sticky="ew", pady=(6, 0))

        self.nozzle_bar = tk.Frame(self.nozzle_bar_frame, bg=ACCENT_GRN, height=8)
        self.nozzle_bar.place(x=0, y=0, relheight=1, relwidth=0)

    def _draw_gauge(self, pct):
        c = self.gauge_canvas
        c.delete("all")
        cx, cy, r = 100, 100, 80

        # Background arc
        c.create_arc(cx-r, cy-r, cx+r, cy+r,
                     start=0, extent=180, style="arc",
                     outline=BORDER, width=10)

        # Value arc
        if pct > 0:
            extent = pct / 100 * 180
            color  = ACCENT_GRN if pct < 60 else ACCENT_YLW if pct < 85 else ACCENT_RED
            c.create_arc(cx-r, cy-r, cx+r, cy+r,
                         start=0, extent=extent, style="arc",
                         outline=color, width=10)

        # Needle
        angle = math.radians(180 - (pct / 100 * 180))
        nx = cx + (r - 14) * math.cos(angle)
        ny = cy - (r - 14) * math.sin(angle)
        c.create_line(cx, cy, nx, ny, fill=TEXT_PRI, width=2)
        c.create_oval(cx-4, cy-4, cx+4, cy+4, fill=TEXT_PRI, outline="")

        # Labels
        c.create_text(cx - r + 4, cy + 10, text="0", fill=TEXT_SEC,
                      font=("Courier", 8))
        c.create_text(cx + r - 4, cy + 10, text="100", fill=TEXT_SEC,
                      font=("Courier", 8))

    def _build_tracking_card(self, parent, row):
        """New: shows grass position offset relative to the center of the image
        (this is what drives the crosshair + tracking line drawn on the frame)."""
        card = self._card(parent, "▸ POSITION TRACKING", row)

        rows = [
            ("Offset X",  "0 px", "offx_lbl"),
            ("Offset Y",  "0 px", "offy_lbl"),
            ("Distance",  "0 px", "offdist_lbl"),
        ]
        for i, (label, default, attr) in enumerate(rows):
            tk.Label(card, text=label, bg=CARD_BG, fg=TEXT_SEC,
                     font=("Courier", 8)).grid(row=i+2, column=0, sticky="w", pady=1)
            lbl = tk.Label(card, text=default, bg=CARD_BG, fg=TEXT_PRI,
                           font=("Courier", 10, "bold"))
            lbl.grid(row=i+2, column=0, sticky="e", pady=1)
            setattr(self, attr, lbl)

    def _build_stats_card(self, parent, row):
        card = self._card(parent, "▸ MEASUREMENTS", row)

        rows = [
            ("Grass Area",   "0.00 cm²",  "area_lbl"),
            ("Instances",    "0",      "instance_lbl"),
        ]
        for i, (label, default, attr) in enumerate(rows):
            tk.Label(card, text=label, bg=CARD_BG, fg=TEXT_SEC,
                     font=("Courier", 8)).grid(row=i+2, column=0, sticky="w", pady=1)
            lbl = tk.Label(card, text=default, bg=CARD_BG, fg=TEXT_PRI,
                           font=("Courier", 10, "bold"))
            lbl.grid(row=i+2, column=0, sticky="e", pady=1)
            setattr(self, attr, lbl)

    def _build_log(self, parent, row):
        card = self._card(parent, "▸ SYSTEM LOG", row)
        parent.rowconfigure(row, weight=1)

        self.log_text = tk.Text(card, bg=BG, fg=ACCENT_GRN,
                                font=("Courier", 8),
                                bd=0, height=8, state="disabled",
                                wrap="word", highlightthickness=0)
        self.log_text.grid(row=2, column=0, sticky="nsew")
        card.rowconfigure(2, weight=1)

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    # ── MODEL LOAD ──────────────────────────────────────────────────────────

    def _load_model(self):
        def load():
            self._log("Loading YOLO model...")
            try:
                self.model = YOLO(MODEL_PATH)
                self._log(f"Model loaded: {MODEL_PATH}")
                self.root.after(0, lambda: self.canvas.itemconfig(
                    self.canvas_text, text="Model ready. Press START."))
            except Exception as e:
                self._log(f"ERROR: {e}")
        threading.Thread(target=load, daemon=True).start()

    # ── CAMERA LOOP ─────────────────────────────────────────────────────────

    def _start(self):
        src_input = self.src_entry.get().strip()
        source    = int(src_input) if src_input.isdigit() else src_input

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self._log(f"ERROR: Cannot open source '{source}'")
            return

        self.running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self._log(f"Started → source: {source}")

        threading.Thread(target=self._loop, daemon=True).start()

    def _stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self._log("Stopped.")

    def _loop(self):
        while self.running:
            loop_now = time.time()
            dt = max(0.001, loop_now - self.last_loop_time)
            self.last_loop_time = loop_now

            ret, frame = self.cap.read()
            if not ret:
                self._log("End of source or read error.")
                self.root.after(0, self._stop)
                break

            fh, fw = frame.shape[:2]
            frame_cx, frame_cy = fw / 2.0, fh / 2.0

            results = self.model(frame, device=DEVICE, conf=CONF,
                                 imgsz=IMGSZ, verbose=False)
            annotated = results[0].plot()

            # ── Compute detections (top-down: position + area) ─────────────
            boxes      = results[0].boxes
            masks      = results[0].masks
            n_detected = len(boxes) if boxes is not None else 0
            raw_detected = n_detected > 0

            total_area   = 0
            target_cx    = None
            target_cy    = None

            if n_detected > 0:
                # Per-instance bbox area (fallback if no masks) + pick the
                # largest instance's centroid as the tracking target.
                best_area  = -1
                best_cx    = None
                best_cy    = None

                for i, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = box.tolist()
                    w = x2 - x1
                    h = y2 - y1
                    bbox_area = w * h
                    total_area += bbox_area

                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    if bbox_area > best_area:
                        best_area = bbox_area
                        best_cx, best_cy = cx, cy

                if masks is not None:
                    # Use real mask pixel count for the nozzle-driving area
                    # (more accurate than bbox area for irregular grass patches).
                    total_area = int(masks.data.sum().item())

                    # Refine the tracking target using the mask centroid of the
                    # largest-area instance instead of its bbox center.
                    mask_data = masks.data  # (N, Hm, Wm) tensor, values 0/1
                    areas_per_instance = mask_data.sum(dim=(1, 2))
                    largest_idx = int(areas_per_instance.argmax().item())
                    m = mask_data[largest_idx]

                    ys, xs = m.nonzero(as_tuple=True)
                    if len(xs) > 0:
                        mh, mw = m.shape
                        # Mask coords → scale to original frame coords
                        scale_x = fw / mw
                        scale_y = fh / mh
                        best_cx = float(xs.float().mean().item()) * scale_x
                        best_cy = float(ys.float().mean().item()) * scale_y

                target_cx, target_cy = best_cx, best_cy

            self.grass_area_px   = int(total_area)
            self.grass_area_cm2  = self.grass_area_px / PX2_PER_CM2
            self.target_cx       = target_cx
            self.target_cy       = target_cy

            # Debounce detection to ignore short misses (e.g. 50 ms tracking loss).
            if raw_detected:
                self.last_detect_time = loop_now
            self.grass_detected = (loop_now - self.last_detect_time) <= DETECTION_HOLD_SEC

            # Smooth area to suppress short spikes (e.g. 100 ms oversized segment).
            area_alpha = 1.0 - math.exp(-dt / max(AREA_SMOOTH_TAU, 1e-3))
            target_area = float(total_area) if raw_detected else 0.0
            self.smoothed_area_px += area_alpha * (target_area - self.smoothed_area_px)
            self.control_area_px = int(max(0.0, self.smoothed_area_px))

            if target_cx is not None:
                self.offset_x_px    = target_cx - frame_cx
                self.offset_y_px    = target_cy - frame_cy
                self.offset_dist_px = math.hypot(self.offset_x_px, self.offset_y_px)
            else:
                self.offset_x_px    = 0
                self.offset_y_px    = 0
                self.offset_dist_px = 0

            # ── Pump/Nozzle control with temporal smoothing ────────────────
            self.pump_on = self.grass_detected and self.control_area_px >= AREA_MIN

            if self.pump_on:
                pct = (self.control_area_px - AREA_MIN) / max(AREA_MAX - AREA_MIN, 1)
                target_nozzle_pct = min(100.0, max(0.0, pct * 100.0))
            else:
                target_nozzle_pct = 0.0

            nozzle_alpha = 1.0 - math.exp(-dt / max(NOZZLE_SMOOTH_TAU, 1e-3))
            self.nozzle_pct_float += nozzle_alpha * (target_nozzle_pct - self.nozzle_pct_float)
            self.nozzle_pct = int(round(min(100.0, max(0.0, self.nozzle_pct_float))))

            if raw_detected:
                self.detect_count += 1

            # ── FPS ────────────────────────────────────────────────────────
            self.frame_count += 1
            now = time.time()
            if now - self.last_time >= 1.0:
                self.fps_display = self.frame_count
                self.frame_count = 0
                self.last_time   = now

            # ── Overlay on frame ───────────────────────────────────────────
            self._draw_overlay(annotated, frame_cx, frame_cy)

            # ── Send to UI ─────────────────────────────────────────────────
            self.root.after(0, self._update_ui,
                            annotated, n_detected)

            time.sleep(0.01)

    def _draw_overlay(self, frame, frame_cx, frame_cy):
        h, w = frame.shape[:2]
        center_pt = (int(frame_cx), int(frame_cy))

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 36), (13, 17, 23), -1)
        pump_txt   = "PUMP: ON" if self.pump_on else "PUMP: OFF"
        pump_col   = (57, 211, 83) if self.pump_on else (248, 81, 73)
        nozzle_txt = f"NOZZLE: {self.nozzle_pct}%"
        fps_txt    = f"FPS: {self.fps_display}"

        cv2.putText(frame, pump_txt,   (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, pump_col, 2)
        cv2.putText(frame, nozzle_txt, (160, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (88, 166, 255), 2)
        cv2.putText(frame, fps_txt,    (w - 100, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (139, 148, 158), 1)

        # ── Center crosshair (reference point, e.g. nozzle/spray position) ──
        cross_r = 14
        cv2.drawMarker(frame, center_pt, (88, 166, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=cross_r * 2,
                       thickness=2)
        cv2.circle(frame, center_pt, cross_r, (88, 166, 255), 1)

        # ── Tracking line: image center → grass target ─────────────────────
        if self.target_cx is not None and self.target_cy is not None:
            target_pt = (int(self.target_cx), int(self.target_cy))
            line_col  = (57, 211, 83) if self.pump_on else (227, 179, 65)

            cv2.line(frame, center_pt, target_pt, line_col, 2, cv2.LINE_AA)
            cv2.circle(frame, target_pt, 8, line_col, -1)
            cv2.circle(frame, target_pt, 12, (255, 255, 255), 1, cv2.LINE_AA)

            # Object annotation: small area label at the tracked object center.
            if self.target_cx is not None and self.target_cy is not None:
                area_txt = f"Area: {self.grass_area_cm2:.2f} cm^2"
                txt_pt = (int(self.target_cx) + 14, int(self.target_cy) - 10)
                cv2.putText(frame, area_txt, txt_pt, cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Nozzle/area bar (right side) — driven by grass area, not height
        bar_h  = min(h - 60, 200)
        bar_x  = w - 30
        bar_y1 = h - 20 - bar_h
        bar_y2 = h - 20
        fill   = int((self.nozzle_pct / 100) * bar_h)

        cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + 16, bar_y2),
                      (30, 36, 45), -1)
        if fill > 0:
            color = (57, 211, 83) if self.nozzle_pct < 60 else \
                    (227, 179, 65) if self.nozzle_pct < 85 else \
                    (248, 81, 73)
            cv2.rectangle(frame,
                          (bar_x, bar_y2 - fill), (bar_x + 16, bar_y2),
                          color, -1)
        cv2.putText(frame, "A", (bar_x + 2, bar_y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (139, 148, 158), 1)

    def _update_ui(self, frame, n_detected):
        # Video frame → canvas
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cw   = self.canvas.winfo_width()
        ch   = self.canvas.winfo_height()
        if cw > 1 and ch > 1:
            img   = Image.fromarray(rgb).resize((cw, ch), Image.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=photo)
            self.canvas._photo = photo  # prevent GC

        # Status
        if self.grass_detected:
            self.status_lbl.config(text="GRASS DETECTED", fg=ACCENT_GRN)
            self.status_dot.itemconfig(self.status_circle, fill=ACCENT_GRN)
        else:
            self.status_lbl.config(text="NO GRASS", fg=TEXT_SEC)
            self.status_dot.itemconfig(self.status_circle, fill=TEXT_SEC)

        self.detect_count_lbl.config(text=f"Detections: {self.detect_count}")

        # Pump
        self._draw_pump(self.pump_on)

        # Nozzle gauge + bar
        self._draw_gauge(self.nozzle_pct)
        self.nozzle_lbl.config(text=f"{self.nozzle_pct}%")
        bar_w = self.nozzle_bar_frame.winfo_width()
        if bar_w > 1:
            fill_w = int(bar_w * self.nozzle_pct / 100)
            color  = ACCENT_GRN if self.nozzle_pct < 60 else \
                     ACCENT_YLW if self.nozzle_pct < 85 else ACCENT_RED
            self.nozzle_bar.config(bg=color)
            self.nozzle_bar.place(x=0, y=0, width=fill_w, relheight=1)

        # Tracking
        self.offx_lbl.config(text=f"{int(self.offset_x_px)} px")
        self.offy_lbl.config(text=f"{int(self.offset_y_px)} px")
        self.offdist_lbl.config(text=f"{int(self.offset_dist_px)} px")

        # Stats
        self.area_lbl.config(text=f"{self.grass_area_cm2:.2f} cm²")
        self.instance_lbl.config(text=str(n_detected))

        # FPS
        self.fps_lbl.config(text=f"FPS: {self.fps_display}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1100x680")
    root.minsize(1600, 750)
    app = AISprayer(root)
    root.mainloop()