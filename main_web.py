"""
Prototype Sprayer Weeder - Web Dashboard
==========================
Open http://<YOUR_PC_IP>:5000 from any device on the same network.

Stream:  /video_feed   → MJPEG (realtime annotated video)
Stats:   /stats_stream → SSE  (nozzle %, servo angle, pump state, FPS, etc.)
Control: /api/start    → POST
         /api/stop     → POST
PID:     /api/pid          → GET gains/metrics, POST {loop, kp, ki, kd}
         /api/pid/history  → GET ?since=t  (SP / PV / u samples)
         /api/pid/step     → POST start step-response test

CONFIG: Edit the variables below, then run.
"""

import cv2
import numpy as np
import threading
import time
import math
import json
import io
import socket
import os
import sys
import glob
from flask import Flask, Response, request, jsonify
from pid_control import ControlSystem
try:
    import psutil
except ImportError:
    psutil = None

# ── CONFIG ──────────────────────────────────────────────────────────────────
# Each value can be overridden per device with an environment variable (see deploy/sprayer-web.service)
MODEL_PATH   = os.environ.get("SPRAYER_MODEL", r"runs/segment/train-9/weights/best.pt")
SOURCE       = os.environ.get("SPRAYER_SOURCE", "video/real1.mp4")   # 0 = webcam, or "path/to/video.mp4"
DEVICE       = "cpu"                # "cpu" or "cuda"
REALTIME     = True                 # video file: skip frames to keep real-time speed
PROCESS_WIDTH = 1280                # resize frames to this width (0 = keep original)
CONF         = 0.4
IMGSZ        = int(os.environ.get("SPRAYER_IMGSZ", 640))   # a .onnx model only works at the size it was exported with
PORT         = 5000

# ── Spray ROI (fractions of frame) ──
# Horizontal band in the middle of the frame = area the nozzle can reach.
ROI_X1       = 0.10    # left edge  (0..1 of width)
ROI_X2       = 0.90    # right edge (0..1 of width)
ROI_Y_CENTER = 0.80    # band center (0..1 of height)
ROI_H        = 0.25    # band height (0..1 of height)

# ── Nozzle opening ──
MIN_SPRAY_PX  = 400    # min grass pixels inside ROI to turn pump on
COVERAGE_FULL = 0.30   # ROI coverage that gives 100% opening
NOZZLE_MIN    = 20     # opening % when pump just turns on

# ── Servo (fixed at center, pans nozzle left/right) ──
NOZZLE_HEIGHT = 1.0    # nozzle height above ground, in units of ROI half-width
SERVO_CENTER  = 90     # deg (nozzle pointing straight down)
SERVO_RANGE   = 60     # max +/- deg pan from center
SERVO_INVERT  = False  # True if servo turns the opposite way
# Servo / nozzle dynamics + PID gains → pid_control.py
DIR_DEADBAND  = 5      # |pan| deg below this = CENTER

# Colors (BGR for OpenCV overlay)
COL_GREEN    = (57, 211, 83)
COL_RED      = (73, 81, 248)
COL_BLUE     = (255, 166, 88)
COL_GRAY     = (158, 148, 139)
COL_YELLOW   = (0, 215, 255)
COL_CYAN     = (255, 220, 0)
COL_CLASS    = [(0, 140, 255), (200, 60, 220), (255, 140, 40)]   # Rumput Teki, Babandotan, Rumput Belulang
# ────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── Shared State ─────────────────────────────────────────────────────────────
state = {
    "running":       False,
    "grass_detected": False,
    "pump_on":        False,
    "nozzle_pct":     0,       # actual opening (PID PV)
    "nozzle_sp":      0,       # opening setpoint from vision
    "servo_sp":       0.0,     # pan setpoint from vision (deg)
    "grass_area_px":  0,
    "roi_grass_px":   0,
    "roi_coverage":   0.0,
    "servo_pan":      0.0,     # deg, - left / + right
    "hit_pos":        0.0,     # spray hit point on ground, -1 (left) .. +1 (right) of ROI
    "servo_angle":    SERVO_CENTER,
    "direction":      "TENGAH",
    "n_detected":     0,
    "detect_count":   0,
    "fps":            0,
    "infer_ms":       0,       # AI time per frame (ms)
    "source":         str(SOURCE),
}
state_lock   = threading.Lock()
frame_lock   = threading.Lock()
latest_frame = None   # JPEG bytes of latest annotated frame
model        = None
cap          = None
_worker      = None
control      = ControlSystem()   # PID loops run in their own thread

# ── SSE helpers ──────────────────────────────────────────────────────────────
sse_clients = []
sse_lock    = threading.Lock()

def push_sse(data: dict):
    msg = f"data: {json.dumps(data)}\n\n"
    with sse_lock:
        for q in list(sse_clients):
            try:
                q.put_nowait(msg)
            except Exception:
                pass

# ── Model load ────────────────────────────────────────────────────────────────
def load_model():
    global model
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH, task="segment")
        print(f"[Sprayer Weeder] Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"[Sprayer Weeder] Model load error: {e}")

threading.Thread(target=load_model, daemon=True).start()

# ── Detection loop ────────────────────────────────────────────────────────────
def roi_rect(w, h):
    x1 = int(ROI_X1 * w)
    x2 = int(ROI_X2 * w)
    y1 = int((ROI_Y_CENTER - ROI_H / 2) * h)
    y2 = int((ROI_Y_CENTER + ROI_H / 2) * h)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def grass_mask_from_result(result, shape):
    """Binary mask (frame size) of all detected grass."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if result.masks is not None:
        for poly in result.masks.xy:
            if len(poly) >= 3:
                cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    elif result.boxes is not None:
        for x1, y1, x2, y2 in result.boxes.xyxy.tolist():
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
    return mask


def label_chip(frame, text, x, y, color, k=1.0):
    """Readable label: white letters on a solid colour chip whose bottom edge sits at y.
    (Thin coloured text on top of green plants is unreadable.)"""
    fs, th, pad = 0.62 * k, max(1, int(round(1.7 * k))), int(5 * k)
    (tw, tdh), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    h, w = frame.shape[:2]
    x = max(0, min(int(x), w - tw - 2 * pad))
    y = max(tdh + base + 2 * pad, min(int(y), h))
    cv2.rectangle(frame, (x, y - tdh - base - 2 * pad), (x + tw + 2 * pad, y), color, -1)
    cv2.putText(frame, text, (x + pad, y - pad - base), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)


def draw_detections(frame, result):
    """Weed masks + labels, one colour per species, drawn in place. Far cheaper than result.plot() on the Pi's CPU."""
    k = frame.shape[1] / 1280
    boxes = result.boxes
    cls = [int(c) for c in boxes.cls.tolist()] if boxes is not None else []
    color = lambda i: COL_CLASS[(cls[i] if i < len(cls) else 0) % len(COL_CLASS)]
    if result.masks is not None:
        layer, outlines = frame.copy(), []
        for i, p in enumerate(result.masks.xy):
            if len(p) >= 3:
                poly = p.astype(np.int32)
                cv2.fillPoly(layer, [poly], color(i))
                outlines.append((poly, color(i)))
        if outlines:
            cv2.addWeighted(layer, 0.35, frame, 0.65, 0, dst=frame)
            for poly, col in outlines:
                cv2.polylines(frame, [poly], True, col, max(2, int(round(2 * k))), cv2.LINE_AA)
    if boxes is not None:
        for i, ((x1, y1, _, _), p) in enumerate(zip(boxes.xyxy.tolist(), boxes.conf.tolist())):
            label_chip(frame, f"{result.names[cls[i]]} {p:.0%}", x1, y1, color(i), k)
    return frame


def system_stats():
    """Board load shown on the dashboard: CPU %, RAM, temperature."""
    out = {"cpu": None, "ram": None, "ram_used_mb": None, "ram_total_mb": None, "temp": None}
    if psutil:
        vm = psutil.virtual_memory()
        out.update(cpu=psutil.cpu_percent(interval=0.2), ram=vm.percent,
                   ram_used_mb=int(vm.used / 1048576), ram_total_mb=int(vm.total / 1048576))
    temps = []
    for f in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(f) as fh:
                temps.append(int(fh.read()) / 1000)
        except (OSError, ValueError):
            pass
    if temps:
        out["temp"] = round(max(temps), 1)
    return out


def _draw_overlay(frame, s, roi, roi_mask, centroid_x):
    h, w = frame.shape[:2]
    rx1, ry1, rx2, ry2 = roi
    spraying = s["pump_on"]

    # ROI band + grass inside ROI highlighted
    layer = frame.copy()
    rec_color = COL_GREEN if spraying else COL_YELLOW
    cv2.rectangle(layer, (rx1, ry1), (rx2, ry2), rec_color, -1)
    frame[:] = cv2.addWeighted(layer, 0.12, frame, 0.88, 0)
    roi_view = frame[ry1:ry2, rx1:rx2]
    tint = np.zeros_like(roi_view)
    tint[:] = COL_RED
    sel = roi_mask > 0
    roi_view[sel] = cv2.addWeighted(roi_view, 0.5, tint, 0.5, 0)[sel]
    k  = w / 1280          # scale drawing to frame resolution
    t1 = max(1, int(round(1 * k)))
    t2 = max(2, int(round(2 * k)))
    roi_col = COL_GREEN if spraying else COL_YELLOW
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), roi_col, t2)
    label_chip(frame, "AREA SEMPROT", rx1 + int(6 * k), ry2 - int(6 * k), (60, 70, 62), k)

    # Grass centroid (servo target)
    if centroid_x is not None:
        cv2.drawMarker(frame, (int(centroid_x), (ry1 + ry2) // 2), COL_RED,
                       cv2.MARKER_CROSS, int(22 * k), t2)

    # Servo fixed at ROI center, nozzle pans toward hit point on the ground
    px     = (rx1 + rx2) // 2
    py     = max(int(50 * k), ry1 - int(40 * k))
    pan    = math.radians(s["servo_pan"])
    hit_x  = int(px + s["hit_pos"] * (rx2 - rx1) / 2)
    if spraying and s["nozzle_pct"] > 1:
        half_w = int((rx2 - rx1) * (0.03 + 0.10 * s["nozzle_pct"] / 100))
        spray = frame.copy()
        pts = np.array([[px, py], [hit_x - half_w, ry2], [hit_x + half_w, ry2]], np.int32)
        cv2.fillPoly(spray, [pts], COL_CYAN)
        frame[:] = cv2.addWeighted(spray, 0.30, frame, 0.70, 0)

    # Pan range arc + nozzle arm
    arm = int(45 * k)
    cv2.ellipse(frame, (px, py), (arm, arm), 90, -SERVO_RANGE, SERVO_RANGE, COL_GRAY, t1)
    sp_pan = math.radians(s["servo_sp"])
    sp_tip = (int(px + arm * 1.3 * math.sin(sp_pan)), int(py + arm * 1.3 * math.cos(sp_pan)))
    cv2.line(frame, (px, py), sp_tip, COL_RED, t1)              # setpoint (SP)
    tip = (int(px + arm * math.sin(pan)), int(py + arm * math.cos(pan)))   # actual (PV)
    cv2.line(frame, (px, py), tip, (255, 255, 255), int(7 * k))
    cv2.line(frame, (px, py), tip, COL_CYAN if spraying else COL_GRAY, int(4 * k))
    sw, sh = int(18 * k), int(12 * k)
    cv2.rectangle(frame, (px - sw, py - sh), (px + sw, py + sh), (40, 40, 40), -1)
    cv2.circle(frame, (px, py), int(6 * k), (255, 255, 255), -1)
    # Pump / nozzle / servo / FPS numbers are shown as sharp text in the web page, not burned into the video.

class FrameGrabber:
    """Reads the source in its own thread and keeps only the newest frame.
    A video file is paced to real time, so decoding runs beside the AI instead of in front of it."""

    def __init__(self, source_cap, realtime):
        self.cap, self.frame, self.seq, self.ended = source_cap, None, 0, False
        self._cv = threading.Condition()
        self._stop = False
        fps = source_cap.get(cv2.CAP_PROP_FPS) or 0
        is_file = realtime and source_cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 and fps > 0
        self._period = 1.0 / fps if is_file else 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        nxt = time.perf_counter()
        while not self._stop:
            ok, f = self.cap.read()
            with self._cv:
                if ok:
                    self.frame, self.seq = f, self.seq + 1
                else:
                    self.ended = True
                self._cv.notify_all()
            if not ok:
                return
            if self._period:
                nxt += self._period
                wait = nxt - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                else:
                    nxt = time.perf_counter()      # slower than real time: do not try to catch up

    def next(self, last_seq, timeout=1.0):
        """(frame, seq) of the newest frame after last_seq; (None, last_seq) on timeout;
        (None, -1) once the source has ended."""
        with self._cv:
            if not self._cv.wait_for(lambda: self.seq > last_seq or self.ended, timeout):
                return None, last_seq
            if self.seq > last_seq:
                return self.frame, self.seq
            return None, -1

    def stop(self):
        self._stop = True
        self._thread.join(timeout=2)


def detection_loop():
    global latest_frame
    my_cap      = cap            # this session's capture (a new START may replace the global)
    frame_count = 0
    last_time   = time.time()
    infer_ms    = 0.0
    grabber     = FrameGrabber(my_cap, REALTIME)
    last_seq    = 0

    while state["running"]:
        frame, last_seq = grabber.next(last_seq)
        if last_seq == -1:                      # source ended
            with state_lock:
                state["running"] = False
            push_sse({"event": "stopped", "reason": "end_of_source"})
            break
        if frame is None:                       # nothing new yet
            continue

        if PROCESS_WIDTH and frame.shape[1] > PROCESS_WIDTH:
            scale = PROCESS_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (PROCESS_WIDTH, int(frame.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)

        t_inf       = time.perf_counter()
        results     = model(frame, device=DEVICE, conf=CONF, imgsz=IMGSZ, verbose=False)
        result      = results[0]
        dt_ms       = (time.perf_counter() - t_inf) * 1000
        infer_ms    = dt_ms if not infer_ms else 0.8 * infer_ms + 0.2 * dt_ms   # smoothed
        annotated   = draw_detections(frame, result)
        n_detected  = len(result.boxes) if result.boxes is not None else 0

        # Grass inside spray ROI
        fh, fw     = frame.shape[:2]
        roi        = roi_rect(fw, fh)
        rx1, ry1, rx2, ry2 = roi
        grass_mask = grass_mask_from_result(result, frame.shape)
        roi_mask   = grass_mask[ry1:ry2, rx1:rx2]
        total_area = cv2.countNonZero(grass_mask)
        roi_px     = cv2.countNonZero(roi_mask)
        coverage   = roi_px / max(roi_mask.size, 1)

        grass_detected = n_detected > 0
        pump_on        = roi_px >= MIN_SPRAY_PX
        centroid_x     = None
        if pump_on:
            # Opening from grass area in ROI
            ratio      = min(1.0, coverage / COVERAGE_FULL)
            nozzle_pct = int(round(NOZZLE_MIN + (100 - NOZZLE_MIN) * ratio))
            # Aim servo at grass centroid inside ROI (→ PID setpoint)
            m          = cv2.moments(roi_mask, binaryImage=True)
            centroid_x = rx1 + m["m10"] / m["m00"]
            offset     = (centroid_x - (rx1 + rx2) / 2) / max((rx2 - rx1) / 2, 1)  # -1..1
            target     = math.degrees(math.atan2(offset, NOZZLE_HEIGHT))
            target     = max(-SERVO_RANGE, min(SERVO_RANGE, target))
        else:
            nozzle_pct = 0
            target     = 0.0   # point straight down

        # Setpoints → PID loops; read back actual (simulated) actuator state
        control.set_targets(target, nozzle_pct)
        pid         = control.snapshot()
        servo_pan   = pid["servo_pv"]
        nozzle_act  = int(round(pid["nozzle_pv"]))
        hit_pos     = math.tan(math.radians(servo_pan)) * NOZZLE_HEIGHT
        sign        = -1 if SERVO_INVERT else 1
        servo_angle = SERVO_CENTER + sign * servo_pan
        direction   = "KIRI" if servo_pan < -DIR_DEADBAND else "KANAN" if servo_pan > DIR_DEADBAND else "TENGAH"

        # FPS
        frame_count += 1
        now = time.time()
        fps = state["fps"]
        if now - last_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_time = now

        with state_lock:
            if grass_detected:
                state["detect_count"] += 1
            state.update({
                "grass_detected":  grass_detected,
                "pump_on":         pump_on,
                "nozzle_pct":      nozzle_act,
                "nozzle_sp":       nozzle_pct,
                "servo_sp":        round(pid["servo_sp"], 1),
                "grass_area_px":   int(total_area),
                "roi_grass_px":    int(roi_px),
                "roi_coverage":    round(coverage * 100, 1),
                "servo_pan":       round(servo_pan, 1),
                "hit_pos":         round(hit_pos, 3),
                "servo_angle":     round(servo_angle, 1),
                "direction":       direction,
                "n_detected":      n_detected,
                "fps":             fps,
                "infer_ms":        round(infer_ms),
            })
            s = dict(state)

        _draw_overlay(annotated, s, roi, roi_mask, centroid_x)

        # Encode to JPEG
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        with frame_lock:
            latest_frame = buf.tobytes()

        # Push stats via SSE
        push_sse({k: s[k] for k in (
            "pump_on", "nozzle_pct", "nozzle_sp", "servo_sp", "grass_detected", "grass_area_px",
            "roi_grass_px", "roi_coverage", "servo_pan", "hit_pos", "servo_angle",
            "direction", "n_detected", "detect_count", "fps", "infer_ms")})

        time.sleep(0.01)

    # Release cap when stopped, park actuators
    control.set_targets(0.0, 0.0)
    path = control.stop_log()
    if path:
        print(f"[Sprayer Weeder] PID log saved: {path}")
    grabber.stop()
    my_cap.release()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return Response(render_page(), mimetype="text/html")


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                frame = latest_frame
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)   # ~33 fps ceiling
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats_stream")
def stats_stream():
    import queue
    q = queue.Queue(maxsize=30)
    with sse_lock:
        sse_clients.append(q)

    def generate():
        try:
            # Send current state immediately on connect
            with state_lock:
                s = dict(state)
            yield f"data: {json.dumps(s)}\n\n"
            while True:
                try:
                    msg = q.get(timeout=5)
                    yield msg
                except Exception:
                    yield ": keepalive\n\n"
        finally:
            with sse_lock:
                if q in sse_clients:
                    sse_clients.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


CAMERA_W   = int(os.environ.get("SPRAYER_CAM_W", 1280))
CAMERA_H   = int(os.environ.get("SPRAYER_CAM_H", 720))
CAMERA_FPS = int(os.environ.get("SPRAYER_CAM_FPS", 30))


def open_source(src):
    """A file or URL is opened as is. A camera number is opened as MJPG at 1280x720: the camera's own default
    (often uncompressed 640x480) loads the CPU and the USB port much more."""
    if not isinstance(src, int):
        return cv2.VideoCapture(src)
    cam = cv2.VideoCapture(src, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(src)
    if cam.isOpened():
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
        cam.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # always the newest frame, no growing delay
        print(f"[Sprayer Weeder] Camera {src}: {int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cam.get(cv2.CAP_PROP_FPS):.0f} fps")
    return cam


@app.route("/api/start", methods=["POST"])
def api_start():
    global cap, _worker
    if state["running"]:
        return jsonify({"ok": False, "msg": "Sudah berjalan"})
    if model is None:
        return jsonify({"ok": False, "msg": "Model AI belum selesai dimuat, tunggu sebentar"})

    data   = request.get_json(silent=True) or {}
    source = data.get("source", state["source"])
    src    = int(source) if str(source).isdigit() else source

    cap = open_source(src)
    if not cap.isOpened():
        return jsonify({"ok": False, "msg": f"Tidak bisa membuka sumber video: {source}"})

    with state_lock:
        state["running"]      = True
        state["detect_count"] = 0
        state["source"]       = str(source)

    log_path = control.start_log("run")
    _worker = threading.Thread(target=detection_loop, daemon=True)
    _worker.start()
    push_sse({"event": "started", "source": str(source)})
    return jsonify({"ok": True, "log": log_path})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with state_lock:
        state["running"] = False
    push_sse({"event": "stopped", "reason": "user"})
    return jsonify({"ok": True})


@app.route("/api/state")
def api_state():
    with state_lock:
        return jsonify(dict(state))


@app.route("/api/pid", methods=["GET", "POST"])
def api_pid():
    if request.method == "POST":
        d = request.get_json(silent=True) or {}
        if d.get("loop") not in ("servo", "nozzle"):
            return jsonify({"ok": False, "msg": "loop harus 'servo' atau 'nozzle'"})
        try:
            control.set_gains(d["loop"], d.get("kp"), d.get("ki"), d.get("kd"))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "msg": "Nilai gain tidak valid"})
    return jsonify({"ok": True, "gains": control.gains(), "metrics": control.metrics,
                    "mode": control.mode, "log": control.log_path, **control.snapshot()})


@app.route("/api/pid/history")
def api_pid_history():
    since = request.args.get("since", -1.0, type=float)
    return jsonify({"now": control.snapshot()["t"], "samples": control.get_history(since)})


@app.route("/api/pid/step", methods=["POST"])
def api_pid_step():
    ok = control.start_step_test()
    return jsonify({"ok": ok, "msg": "" if ok else "Uji step sedang berjalan"})


@app.route("/api/system")
def api_system():
    return jsonify(system_stats())


# ── HTML Page ─────────────────────────────────────────────────────────────────
PAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")


def render_page():
    """dashboard.html with the few server-side values filled in (re-read on every request)."""
    with open(PAGE_FILE, encoding="utf-8") as f:
        return (f.read().replace("{{ source }}", str(SOURCE))
                .replace("{{ servo_invert }}", "true" if SERVO_INVERT else "false")
                .replace("{{ nozzle_height }}", str(NOZZLE_HEIGHT)))


@app.route("/api/sources")
def api_sources():
    """Video files the user can pick in the dashboard (0 = camera)."""
    files = {p.replace("\\", "/") for ext in ("mp4", "MP4", "avi") for p in glob.glob(f"video/*.{ext}")}
    return jsonify(["0"] + sorted(files))


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Print local IP for convenience
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    print("=" * 50)
    print(f"  Prototype Sprayer Weeder - Web Dashboard")
    print(f"  Local:   http://localhost:{PORT}")
    print(f"  Network: http://{local_ip}:{PORT}")
    print("=" * 50)

    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False)