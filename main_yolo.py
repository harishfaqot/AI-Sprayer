"""
AI Sprayer - Grass Detection & Spray Control
=========================================
Uses YOLO segmentation to detect grass, estimate height from bounding box,
and control pump ON/OFF + nozzle 0-100% accordingly.

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
MODEL_PATH   = r"runs\segment\train-6\weights\best.pt"
SOURCE       = "video/grass4.mp4"   # 0 = webcam, or "path/to/video.mp4"
DEVICE       = "cpu"                # "cpu" or "cuda" or 0
CONF         = 0.4
IMGSZ        = 640

# Grass height thresholds (bbox height in pixels)
HEIGHT_MIN   = 30    # below this → pump OFF
HEIGHT_MAX   = 480   # at this → nozzle 100%
# ────────────────────────────────────────────────────────────────────────────

# # ── Colors (dark industrial theme) ──────────────────────────────────────────
# BG          = "#0d1117"
# PANEL_BG    = "#161b22"
# CARD_BG     = "#1c2333"
# ACCENT_GRN  = "#39d353"
# ACCENT_YLW  = "#e3b341"
# ACCENT_RED  = "#f85149"
# ACCENT_BLU  = "#58a6ff"
# TEXT_PRI    = "#f0f6fc"
# TEXT_SEC    = "#8b949e"
# BORDER      = "#30363d"

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
        self.root.title("AI Sprayer — Grass Detection System")
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
        self.grass_area_px   = 0
        self.grass_height_px = 0
        self.detect_count    = 0

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
        tk.Label(hdr, text="GRASS DETECTION SYSTEM", bg=BG, fg=TEXT_SEC,
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
        self._build_stats_card(right, row=3)
        self._build_log(right, row=4)

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

    def _build_stats_card(self, parent, row):
        card = self._card(parent, "▸ MEASUREMENTS", row)

        rows = [
            ("Grass Area",   "0 px²",  "area_lbl"),
            ("Grass Height", "0 px",   "height_lbl"),
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
            ret, frame = self.cap.read()
            if not ret:
                self._log("End of source or read error.")
                self.root.after(0, self._stop)
                break

            results = self.model(frame, device=DEVICE, conf=CONF,
                                 imgsz=IMGSZ, verbose=False)
            annotated = results[0].plot()

            # ── Compute detections ──────────────────────────────────────────
            boxes      = results[0].boxes
            masks      = results[0].masks
            n_detected = len(boxes) if boxes is not None else 0

            total_area   = 0
            max_height   = 0

            if n_detected > 0:
                for box in boxes.xyxy:
                    x1, y1, x2, y2 = box.tolist()
                    w = x2 - x1
                    h = y2 - y1
                    total_area += w * h
                    if h > max_height:
                        max_height = h

                if masks is not None:
                    # Use actual mask pixel count for area
                    total_area = int(masks.data.sum().item())

            self.grass_detected  = n_detected > 0
            self.grass_area_px   = int(total_area)
            self.grass_height_px = int(max_height)

            # ── Pump logic ─────────────────────────────────────────────────
            self.pump_on = self.grass_detected and max_height >= HEIGHT_MIN

            # ── Nozzle logic (linear scale) ─────────────────────────────────
            if self.pump_on:
                pct = (max_height - HEIGHT_MIN) / max(HEIGHT_MAX - HEIGHT_MIN, 1)
                self.nozzle_pct = min(100, max(0, int(pct * 100)))
            else:
                self.nozzle_pct = 0

            if self.grass_detected:
                self.detect_count += 1

            # ── FPS ────────────────────────────────────────────────────────
            self.frame_count += 1
            now = time.time()
            if now - self.last_time >= 1.0:
                self.fps_display = self.frame_count
                self.frame_count = 0
                self.last_time   = now

            # ── Overlay on frame ───────────────────────────────────────────
            self._draw_overlay(annotated)

            # ── Send to UI ─────────────────────────────────────────────────
            self.root.after(0, self._update_ui,
                            annotated, n_detected)

            time.sleep(0.01)

    def _draw_overlay(self, frame):
        h, w = frame.shape[:2]

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

        # Height bar (right side)
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
        cv2.putText(frame, "H", (bar_x + 2, bar_y1 - 6),
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

        # Stats
        self.area_lbl.config(text=f"{self.grass_area_px:,} px²")
        self.height_lbl.config(text=f"{self.grass_height_px} px")
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