"""
AI Sprayer - Web Dashboard
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
from flask import Flask, Response, render_template_string, request, jsonify
from pid_control import ControlSystem

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_PATH   = r"runs/segment/train-9/weights/best.pt"
SOURCE       = "video/real1.mp4"   # 0 = webcam, or "path/to/video.mp4"
DEVICE       = "cpu"                # "cpu" or "cuda"
REALTIME     = True                 # video file: skip frames to keep real-time speed
PROCESS_WIDTH = 1280                # resize frames to this width (0 = keep original)
CONF         = 0.4
IMGSZ        = 640
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
    "direction":      "CENTER",
    "n_detected":     0,
    "detect_count":   0,
    "fps":            0,
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
        model = YOLO(MODEL_PATH)
        print(f"[AI Sprayer] Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"[AI Sprayer] Model load error: {e}")

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
    cv2.putText(frame, "SPRAY AREA", (rx1 + int(6 * k), ry2 - int(8 * k)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, roi_col, t1)

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
    cv2.putText(frame, f"{s['servo_angle']:.0f}deg", (px + sw + int(6 * k), py + int(5 * k)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, (255, 255, 255), t1)

    # Top status bar
    fs = 0.65 * k
    ty = int(26 * k)
    cv2.rectangle(frame, (0, 0), (w, int(38 * k)), (23, 17, 13), -1)
    pump_txt  = "PUMP: ON" if spraying else "PUMP: OFF"
    pump_col  = COL_GREEN if spraying else COL_RED
    cv2.putText(frame, pump_txt, (int(10 * k), ty), cv2.FONT_HERSHEY_SIMPLEX, fs, pump_col, t2)
    cv2.putText(frame, f"NOZZLE: {s['nozzle_pct']}%", (int(150 * k), ty), cv2.FONT_HERSHEY_SIMPLEX, fs, COL_BLUE, t2)
    cv2.putText(frame, f"TARGET {s['nozzle_sp']}%  {s['servo_sp']:+.0f}deg", (int(640 * k), ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, COL_GRAY, t1)
    cv2.putText(frame, f"SERVO: {s['servo_angle']:.0f}deg {s['direction']}", (int(330 * k), ty),
                cv2.FONT_HERSHEY_SIMPLEX, fs, COL_CYAN, t2)
    cv2.putText(frame, f"FPS: {s['fps']}", (w - int(105 * k), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * k, COL_GRAY, t1)

def detection_loop():
    global latest_frame, cap
    frame_count = 0
    last_time   = time.time()

    # Video file → skip frames so playback follows wall-clock time
    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 0
    is_file   = REALTIME and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 and src_fps > 0
    start_t   = time.time()
    frame_idx = 0

    while state["running"]:
        if is_file:
            target_idx = int((time.time() - start_t) * src_fps)
            ok = True
            while frame_idx < target_idx and ok:
                ok = cap.grab()
                frame_idx += 1
        ret, frame = cap.read()
        frame_idx += 1
        if not ret:
            with state_lock:
                state["running"] = False
            push_sse({"event": "stopped", "reason": "end_of_source"})
            break

        if PROCESS_WIDTH and frame.shape[1] > PROCESS_WIDTH:
            scale = PROCESS_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (PROCESS_WIDTH, int(frame.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)

        results     = model(frame, device=DEVICE, conf=CONF, imgsz=IMGSZ, verbose=False)
        result      = results[0]
        annotated   = result.plot()
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
        direction   = "LEFT" if servo_pan < -DIR_DEADBAND else "RIGHT" if servo_pan > DIR_DEADBAND else "CENTER"

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
            })
            s = dict(state)

        _draw_overlay(annotated, s, roi, roi_mask, centroid_x)

        # Encode to JPEG
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with frame_lock:
            latest_frame = buf.tobytes()

        # Push stats via SSE
        push_sse({k: s[k] for k in (
            "pump_on", "nozzle_pct", "nozzle_sp", "servo_sp", "grass_detected", "grass_area_px",
            "roi_grass_px", "roi_coverage", "servo_pan", "hit_pos", "servo_angle",
            "direction", "n_detected", "detect_count", "fps")})

        time.sleep(0.01)

    # Release cap when stopped, park actuators
    control.set_targets(0.0, 0.0)
    path = control.stop_log()
    if path:
        print(f"[AI Sprayer] PID log saved: {path}")
    if cap:
        cap.release()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


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


@app.route("/api/start", methods=["POST"])
def api_start():
    global cap, _worker
    if state["running"]:
        return jsonify({"ok": False, "msg": "Already running"})
    if model is None:
        return jsonify({"ok": False, "msg": "Model not loaded yet"})

    data   = request.get_json(silent=True) or {}
    source = data.get("source", state["source"])
    src    = int(source) if str(source).isdigit() else source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return jsonify({"ok": False, "msg": f"Cannot open source: {source}"})

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
            return jsonify({"ok": False, "msg": "loop must be 'servo' or 'nozzle'"})
        try:
            control.set_gains(d["loop"], d.get("kp"), d.get("ki"), d.get("kd"))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "msg": "invalid gain value"})
    return jsonify({"ok": True, "gains": control.gains(), "metrics": control.metrics,
                    "mode": control.mode, "log": control.log_path, **control.snapshot()})


@app.route("/api/pid/history")
def api_pid_history():
    since = request.args.get("since", -1.0, type=float)
    return jsonify({"now": control.snapshot()["t"], "samples": control.get_history(since)})


@app.route("/api/pid/step", methods=["POST"])
def api_pid_step():
    ok = control.start_step_test()
    return jsonify({"ok": ok, "msg": "" if ok else "Step test already running"})


# ── HTML Page ─────────────────────────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Sprayer — Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap');

  :root {
    --bg:       #FAFAFA;
    --panel:    #FFFFFF;
    --card:     #F4F4F5;
    --border:   #E4E4E7;
    --grn:      #16A34A;
    --grn-dim:  #D1FAE5;
    --red:      #DC2626;
    --red-dim:  #FEE2E2;
    --ylw:      #D97706;
    --blu:      #2563EB;
    --txt:      #18181B;
    --muted:    #71717A;
    --radius:   10px;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'JetBrains Mono', monospace;
    background: var(--bg);
    color: var(--txt);
    min-height: 100vh;
    display: grid;
    grid-template-rows: 56px 1fr;
    grid-template-columns: 1fr;
  }

  /* ── Header ── */
  header {
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 20px;
    gap: 12px;
    position: sticky; top: 0; z-index: 10;
  }
  .logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem; font-weight: 800;
    color: var(--grn);
    letter-spacing: -0.02em;
  }
  .logo span { color: var(--muted); font-weight: 700; }
  .badge {
    font-size: 0.6rem; font-weight: 600;
    padding: 2px 8px; border-radius: 999px;
    background: var(--grn-dim); color: var(--grn);
    letter-spacing: 0.08em; text-transform: uppercase;
  }
  .badge.off { background: var(--red-dim); color: var(--red); }
  .hdr-right { margin-left: auto; display: flex; gap: 8px; align-items: center; }
  .fps-badge {
    font-size: 0.65rem; color: var(--muted);
    border: 1px solid var(--border); padding: 3px 10px; border-radius: 6px;
  }

  /* ── Main layout ── */
  main {
    display: grid;
    grid-template-columns: 1fr 300px;
    gap: 16px;
    padding: 16px;
    align-items: start;
  }
  @media (max-width: 900px) {
    main { grid-template-columns: 1fr; }
  }

  /* ── Video panel ── */
  .video-wrap {
    background: #000;
    border-radius: var(--radius);
    overflow: hidden;
    position: relative;
    aspect-ratio: 16/9;
    border: 1px solid var(--border);
  }
  .video-wrap img {
    width: 100%; height: 100%;
    object-fit: cover;
    display: block;
  }
  .video-placeholder {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    color: var(--muted); gap: 8px;
    font-size: 0.8rem;
  }
  .video-placeholder .icon { font-size: 2.5rem; opacity: 0.3; }

  /* Source bar */
  .src-bar {
    display: flex; gap: 8px; margin-top: 10px; align-items: center;
  }
  .src-bar label { font-size: 0.65rem; color: var(--muted); white-space: nowrap; }
  .src-bar input {
    flex: 1; font-family: inherit; font-size: 0.75rem;
    padding: 6px 10px; border: 1px solid var(--border);
    border-radius: 6px; background: var(--card); color: var(--txt);
    outline: none;
  }
  .src-bar input:focus { border-color: var(--grn); }

  /* Buttons */
  .btn {
    font-family: inherit; font-size: 0.7rem; font-weight: 700;
    padding: 6px 14px; border-radius: 6px; border: none; cursor: pointer;
    letter-spacing: 0.04em; transition: opacity 0.15s, transform 0.1s;
    white-space: nowrap;
  }
  .btn:active { transform: scale(0.97); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-grn { background: var(--grn); color: #fff; }
  .btn-red { background: var(--red); color: #fff; }

  /* ── Right panel ── */
  .right-panel { display: flex; flex-direction: column; gap: 12px; }

  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px;
  }
  .card-title {
    font-size: 0.6rem; font-weight: 700; color: var(--muted);
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 10px;
  }

  /* Status indicator */
  .status-row { display: flex; align-items: center; gap: 8px; }
  .dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--muted); flex-shrink: 0;
    transition: background 0.3s;
  }
  .dot.on { background: var(--grn); box-shadow: 0 0 0 3px var(--grn-dim); }
  .status-text { font-size: 0.85rem; font-weight: 700; color: var(--muted); }
  .status-text.on { color: var(--grn); }
  .detect-count { font-size: 0.65rem; color: var(--muted); margin-top: 6px; }

  /* Pump */
  .pump-box {
    border-radius: 7px; padding: 14px 10px;
    text-align: center; font-size: 1rem; font-weight: 700;
    letter-spacing: 0.08em; transition: background 0.3s, color 0.3s;
    background: var(--red-dim); color: var(--red);
  }
  .pump-box.on { background: var(--grn-dim); color: var(--grn); }

  /* Gauge */
  .gauge-wrap { text-align: center; }
  svg.gauge { display: block; margin: 0 auto; }
  .nozzle-pct { font-size: 1.6rem; font-weight: 700; margin-top: 2px; }
  .progress-track {
    height: 7px; background: var(--border); border-radius: 99px;
    overflow: hidden; margin-top: 8px;
  }
  .progress-fill {
    height: 100%; width: 0%; border-radius: 99px;
    background: var(--grn); transition: width 0.2s, background 0.2s;
  }

  /* Stats */
  .stat-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 4px 0; border-bottom: 1px solid var(--border);
    font-size: 0.72rem;
  }
  .stat-row:last-child { border-bottom: none; }
  .stat-key { color: var(--muted); }
  .stat-val { font-weight: 700; }

  /* Log */
  .log-box {
    background: var(--card); border-radius: 6px;
    padding: 8px; font-size: 0.65rem; color: var(--grn);
    height: 110px; overflow-y: auto; line-height: 1.6;
  }
  .log-box .log-line { color: var(--muted); }
  .log-box .log-line span { color: var(--txt); }

  /* PID panel */
  .pid-card { margin-top: 12px; }
  .pid-charts { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 700px) { .pid-charts { grid-template-columns: 1fr; } }
  .pid-chart canvas { width: 100%; height: 150px; display: block;
                      background: var(--card); border-radius: 6px; }
  .pid-chart-hdr { display: flex; justify-content: space-between; align-items: baseline;
                   font-size: 0.65rem; color: var(--muted); margin-bottom: 4px; }
  .pid-chart-hdr b { color: var(--txt); }
  .legend i { display: inline-block; width: 14px; height: 2px; vertical-align: middle; margin: 0 3px 0 8px; }
  .gains { display: flex; gap: 6px; margin-top: 8px; align-items: center; flex-wrap: wrap; }
  .gains label { font-size: 0.6rem; color: var(--muted); }
  .gains input { width: 66px; font-family: inherit; font-size: 0.7rem; padding: 4px 6px;
                 border: 1px solid var(--border); border-radius: 5px; background: var(--card); color: var(--txt); }
  .btn-gry { background: var(--txt); color: #fff; }
  .metrics { width: 100%; border-collapse: collapse; font-size: 0.65rem; margin-top: 8px; }
  .metrics th, .metrics td { padding: 3px 4px; text-align: right; border-bottom: 1px solid var(--border); }
  .metrics th { color: var(--muted); font-weight: 600; }
  .metrics td:first-child, .metrics th:first-child { text-align: left; }
  .pid-actions { display: flex; gap: 8px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
  .pid-note { font-size: 0.62rem; color: var(--muted); }
  .help { font-size: 0.68rem; color: var(--muted); line-height: 1.6; margin-bottom: 10px; }
  .help b { color: var(--txt); }
  .help ul { margin: 4px 0 0 16px; }
  details.help summary { cursor: pointer; color: var(--blu); font-weight: 600; }
  .vlegend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 10px;
             font-size: 0.65rem; color: var(--muted); }
  .vlegend span { display: inline-flex; align-items: center; gap: 5px; }
  .sw { width: 14px; height: 10px; border-radius: 2px; display: inline-block; }
  .sw.line { height: 3px; }

  /* Pulse animation for dot */
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
  .dot.on { animation: pulse 1.5s infinite; }
</style>
</head>
<body>

<header>
  <div class="logo">⬡ AI SPRAYER <span>/ WEB</span></div>
  <div id="runBadge" class="badge off">STOPPED</div>
  <div class="hdr-right">
    <div class="fps-badge" id="fpsBadge">FPS: --</div>
  </div>
</header>

<main>
  <!-- LEFT: video -->
  <div>
    <div class="video-wrap" id="videoWrap">
      <div class="video-placeholder" id="videoPlaceholder">
        <div class="icon">📷</div>
        <div>Press START to begin detection</div>
      </div>
      <img id="videoImg" src="" alt="stream" style="display:none"
           onerror="this.style.display='none'; document.getElementById('videoPlaceholder').style.display='flex'">
    </div>
    <div class="src-bar">
      <label>SOURCE</label>
      <input type="text" id="srcInput" value="{{ source }}" placeholder="0 = webcam | path/to/video.mp4">
      <button class="btn btn-grn" id="btnStart" onclick="startStream()">▶ START</button>
      <button class="btn btn-red" id="btnStop" onclick="stopStream()" disabled>■ STOP</button>
    </div>

    <!-- Video legend -->
    <div class="vlegend">
      <span><i class="sw" style="border:2px solid #FFD700"></i>Spray area (waiting)</span>
      <span><i class="sw" style="border:2px solid #53D339"></i>Spray area (spraying)</span>
      <span><i class="sw" style="background:#F85149;opacity:.6"></i>Grass inside spray area</span>
      <span><b style="color:#F85149">+</b>Grass center (where to aim)</span>
      <span><i class="sw line" style="background:#F85149"></i>Target angle</span>
      <span><i class="sw line" style="background:#00DCFF;height:5px"></i>Actual nozzle angle</span>
      <span><i class="sw" style="background:#00DCFF;opacity:.4"></i>Spray</span>
    </div>

    <!-- PID response -->
    <div class="card pid-card">
      <div class="card-title">▸ PID Control <span id="pidMode" style="float:right"></span></div>

      <details class="help">
        <summary>What is this? (click to read)</summary>
        <p style="margin-top:6px">The camera decides a <b>target</b> for two servos:
          the <b>angle servo</b> (aims the nozzle left/right at the grass) and the
          <b>nozzle servo</b> (opens the nozzle more when there is more grass).
          The target changes every frame and can jump. The <b>PID controller</b> moves each servo
          toward its target smoothly: fast when far, slowing down when close, without
          overshooting or shaking. It runs 200 times per second, in software only.</p>
        <ul>
          <li><b>Kp</b> (push): higher = reacts faster. Too high = overshoots / shakes.</li>
          <li><b>Ki</b> (fix leftover error): keep small. Too high = overshoots and goes slow to settle.</li>
          <li><b>Kd</b> (brake): higher = less overshoot. Too high = sluggish / jittery.</li>
        </ul>
        <p style="margin-top:4px"><b>STEP TEST</b> sends fixed targets (angle 0 → +30° → −30° → 0,
          nozzle 0 → 80% → 30% → 0) to measure how good the tuning is.
          Servos are simulated for now (no hardware connected).</p>
      </details>

      <div class="pid-charts">
        <div class="pid-chart">
          <div class="pid-chart-hdr">
            <span><b>Angle servo</b> (degree, − left / + right)</span>
            <span class="legend"><i style="background:#DC2626"></i>Target<i style="background:#2563EB"></i>Actual</span>
          </div>
          <canvas id="chartServo"></canvas>
          <div class="gains">
            <label title="Push: bigger = faster">Kp</label><input id="servo_kp" type="number" step="any">
            <label title="Fix leftover error: keep small">Ki</label><input id="servo_ki" type="number" step="any">
            <label title="Brake: bigger = less overshoot">Kd</label><input id="servo_kd" type="number" step="any">
            <button class="btn btn-gry" onclick="applyGains('servo')">APPLY</button>
          </div>
        </div>
        <div class="pid-chart">
          <div class="pid-chart-hdr">
            <span><b>Nozzle servo</b> (opening %)</span>
            <span class="legend"><i style="background:#DC2626"></i>Target<i style="background:#2563EB"></i>Actual</span>
          </div>
          <canvas id="chartNozzle"></canvas>
          <div class="gains">
            <label title="Push: bigger = faster">Kp</label><input id="nozzle_kp" type="number" step="any">
            <label title="Fix leftover error: keep small">Ki</label><input id="nozzle_ki" type="number" step="any">
            <label title="Brake: bigger = less overshoot">Kd</label><input id="nozzle_kd" type="number" step="any">
            <button class="btn btn-gry" onclick="applyGains('nozzle')">APPLY</button>
          </div>
        </div>
      </div>
      <div class="pid-note" style="margin-top:6px">Good tuning: the blue line follows the red dashed line quickly, without going past it.</div>
      <div class="pid-actions">
        <button class="btn btn-grn" id="btnStep" onclick="stepTest()">STEP TEST</button>
        <span class="pid-note" id="pidLog">CSV log: —</span>
      </div>
      <div class="help" id="metricsHelp" hidden style="margin:8px 0 0">
        <b>Rise</b> = time to get from 10% to 90% of the way (smaller = faster) ·
        <b>Overshoot</b> = how far it went past the target (smaller = better) ·
        <b>Settling</b> = time until it stays at the target ±2% (smaller = better) ·
        <b>Error</b> = what is left at the end (closer to 0 = better)
      </div>
      <table class="metrics" id="metricsTable" hidden>
        <thead><tr><th>Servo</th><th>Target change</th><th>Rise (s)</th><th>Overshoot</th><th>Settling (s)</th><th>Error</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <!-- RIGHT: control panel -->
  <div class="right-panel">

    <!-- Status -->
    <div class="card">
      <div class="card-title">▸ Detection Status</div>
      <div class="status-row">
        <div class="dot" id="statusDot"></div>
        <div class="status-text" id="statusText">NO GRASS</div>
      </div>
      <div class="detect-count" id="detectCount">Detections: 0</div>
    </div>

    <!-- Pump -->
    <div class="card">
      <div class="card-title">▸ Pump Control</div>
      <div class="pump-box" id="pumpBox">PUMP  OFF</div>
    </div>

    <!-- Nozzle -->
    <div class="card">
      <div class="card-title">▸ Nozzle Opening</div>
      <div class="gauge-wrap">
        <svg class="gauge" width="180" height="100" viewBox="0 0 180 100">
          <!-- track -->
          <path d="M 14 90 A 76 76 0 0 1 166 90" fill="none" stroke="#E4E4E7" stroke-width="10" stroke-linecap="round"/>
          <!-- fill -->
          <path id="gaugeFill" d="M 14 90 A 76 76 0 0 1 166 90" fill="none" stroke="#16A34A" stroke-width="10" stroke-linecap="round"
                stroke-dasharray="238.76" stroke-dashoffset="238.76" style="transition: stroke-dashoffset 0.2s, stroke 0.2s"/>
          <!-- needle -->
          <line id="gaugeNeedle" x1="90" y1="90" x2="90" y2="22" stroke="#18181B" stroke-width="2" stroke-linecap="round"
                style="transform-origin:90px 90px; transform:rotate(-90deg); transition:transform 0.2s"/>
          <circle cx="90" cy="90" r="4" fill="#18181B"/>
          <!-- labels -->
          <text x="10" y="100" font-family="JetBrains Mono" font-size="9" fill="#71717A">0</text>
          <text x="155" y="100" font-family="JetBrains Mono" font-size="9" fill="#71717A">100</text>
        </svg>
        <div class="nozzle-pct" id="nozzlePct">0%</div>
        <div class="detect-count" id="nozzleSp">Target: 0%</div>
        <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
      </div>
    </div>

    <!-- Servo -->
    <div class="card">
      <div class="card-title">▸ Angle Servo</div>
      <svg width="100%" height="90" viewBox="0 0 240 90">
        <!-- ground / ROI width -->
        <line x1="20" y1="80" x2="220" y2="80" stroke="#E4E4E7" stroke-width="6" stroke-linecap="round"/>
        <line x1="120" y1="74" x2="120" y2="86" stroke="#A1A1AA" stroke-width="1"/>
        <polygon id="servoSpray" points="120,14 120,80 120,80" fill="#0EA5E9" opacity="0.3"/>
        <!-- servo body + arm -->
        <line id="servoArm" x1="120" y1="14" x2="120" y2="40" stroke="#A1A1AA" stroke-width="5" stroke-linecap="round"/>
        <rect x="108" y="6" width="24" height="16" rx="3" fill="#27272A"/>
        <circle cx="120" cy="14" r="3" fill="#fff"/>
        <text x="8" y="84" font-family="JetBrains Mono" font-size="9" fill="#71717A">L</text>
        <text x="226" y="84" font-family="JetBrains Mono" font-size="9" fill="#71717A">R</text>
      </svg>
      <div class="stat-row"><span class="stat-key">Actual angle</span><span class="stat-val" id="servoAngle">90°</span></div>
      <div class="stat-row"><span class="stat-key">Target</span><span class="stat-val" id="servoSp">0°</span></div>
      <div class="stat-row"><span class="stat-key">Direction</span><span class="stat-val" id="servoDir">CENTER</span></div>
    </div>

    <!-- Stats -->
    <div class="card">
      <div class="card-title">▸ Measurements</div>
      <div class="stat-row"><span class="stat-key">Grass Area (frame)</span><span class="stat-val" id="statArea">0 px²</span></div>
      <div class="stat-row"><span class="stat-key">Grass Area (ROI)</span><span class="stat-val" id="statRoiArea">0 px²</span></div>
      <div class="stat-row"><span class="stat-key">ROI Coverage</span><span class="stat-val" id="statCoverage">0%</span></div>
      <div class="stat-row"><span class="stat-key">Instances</span><span class="stat-val" id="statInst">0</span></div>
    </div>

    <!-- Log -->
    <div class="card" style="flex:1">
      <div class="card-title">▸ System Log</div>
      <div class="log-box" id="logBox"></div>
    </div>

  </div>
</main>

<script>
// ── SSE ──────────────────────────────────────────────────────────────────────
const GAUGE_LEN = 238.76;
let es = null;

function connectSSE() {
  if (es) es.close();
  es = new EventSource("/stats_stream");
  es.onmessage = (e) => {
    const d = JSON.parse(e.data);
    if (d.event === "stopped") {
      onStopped(d.reason);
      return;
    }
    if (d.event === "started") { return; }
    updateUI(d);
  };
  es.onerror = () => {
    setTimeout(connectSSE, 3000);
  };
}

function updateUI(d) {
  // FPS
  document.getElementById("fpsBadge").textContent = `FPS: ${d.fps ?? "--"}`;

  // Status dot
  const dot  = document.getElementById("statusDot");
  const stxt = document.getElementById("statusText");
  if (d.grass_detected) {
    dot.className = "dot on";
    stxt.className = "status-text on";
    stxt.textContent = "GRASS DETECTED";
  } else {
    dot.className = "dot";
    stxt.className = "status-text";
    stxt.textContent = "NO GRASS";
  }
  document.getElementById("detectCount").textContent = `Detections: ${d.detect_count ?? 0}`;

  // Pump
  const pump = document.getElementById("pumpBox");
  pump.className = "pump-box" + (d.pump_on ? " on" : "");
  pump.textContent = d.pump_on ? "PUMP  ON" : "PUMP  OFF";

  // Stats
  document.getElementById("statArea").textContent   = `${(d.grass_area_px ?? 0).toLocaleString()} px²`;
  document.getElementById("statRoiArea").textContent  = `${(d.roi_grass_px ?? 0).toLocaleString()} px²`;
  document.getElementById("statCoverage").textContent = `${d.roi_coverage ?? 0}%`;

  lastPump = !!d.pump_on;
  document.getElementById("servoDir").textContent   = d.direction ?? "CENTER";
  document.getElementById("statInst").textContent   = d.n_detected ?? 0;
}

// ── Actuators (PID PV) ────────────────────────────────────────────────────────
const SERVO_INVERT = {{ servo_invert }};
const NOZZLE_HEIGHT = {{ nozzle_height }};
let lastPump = false;

function updateActuators(p) {
  // Nozzle gauge = actual opening
  const pct = Math.round(p.nozzle_pv);
  const fill = document.getElementById("gaugeFill");
  fill.setAttribute("stroke-dashoffset", GAUGE_LEN * (1 - pct / 100));
  fill.setAttribute("stroke", pct < 60 ? "#16A34A" : pct < 85 ? "#D97706" : "#DC2626");
  document.getElementById("gaugeNeedle").style.transform = `rotate(${-90 + pct * 1.8}deg)`;
  document.getElementById("nozzlePct").textContent = `${pct}%`;
  document.getElementById("nozzleSp").textContent  = `Target: ${Math.round(p.nozzle_sp)}%`;
  const pf = document.getElementById("progressFill");
  pf.style.width = `${pct}%`;
  pf.style.background = pct < 60 ? "var(--grn)" : pct < 85 ? "var(--ylw)" : "var(--red)";

  // Servo = actual pan angle
  const pan = p.servo_pv * Math.PI / 180;
  const hit = Math.max(-1, Math.min(1, Math.tan(pan) * NOZZLE_HEIGHT));
  const hx  = 120 + hit * 100;
  const arm = document.getElementById("servoArm");
  const spraying = (lastPump || p.mode === "step") && pct > 1;
  arm.setAttribute("x2", 120 + 26 * Math.sin(pan));
  arm.setAttribute("y2", 14 + 26 * Math.cos(pan));
  arm.setAttribute("stroke", spraying ? "#0EA5E9" : "#A1A1AA");
  const hw  = spraying ? 6 + 20 * pct / 100 : 0;
  document.getElementById("servoSpray").setAttribute("points", `120,14 ${hx - hw},80 ${hx + hw},80`);
  const ang = 90 + (SERVO_INVERT ? -1 : 1) * p.servo_pv;
  document.getElementById("servoAngle").textContent = `${ang.toFixed(1)}°`;
  document.getElementById("servoSp").textContent    = `${p.servo_sp >= 0 ? "+" : ""}${p.servo_sp.toFixed(1)}° pan`;
}

// ── PID charts ────────────────────────────────────────────────────────────────
const WINDOW_S = 10;
let pidHist = [], pidSince = -1, pidNow = 0, lastMode = "vision";

function drawChart(id, iSp, iPv, iU, yMin, yMax, uMin) {
  const cv = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const L = 30, R = 6, T = 6, B = 16;
  const t0 = pidNow - WINDOW_S;
  const X = t => L + (t - t0) / WINDOW_S * (W - L - R);
  const Y = v => T + (1 - (v - yMin) / (yMax - yMin)) * (H - T - B);
  const YU = u => T + (1 - (u - uMin) / (1 - uMin)) * (H - T - B);

  g.font = "9px JetBrains Mono, monospace"; g.fillStyle = "#71717A"; g.strokeStyle = "#E4E4E7"; g.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = yMin + (yMax - yMin) * k / 4, y = Y(v);
    g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke();
    g.fillText(Math.round(v), 2, y + 3);
  }
  for (let s = 0; s <= WINDOW_S; s += 2) g.fillText(`-${WINDOW_S - s}s`, X(t0 + s) - 8, H - 3);

  const pts = pidHist.filter(h => h[0] >= t0);
  const line = (fn, col, w, dash) => {
    g.strokeStyle = col; g.lineWidth = w; g.setLineDash(dash || []); g.beginPath();
    pts.forEach((h, k) => { const x = X(h[0]), y = fn(h); k ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.stroke(); g.setLineDash([]);
  };
  line(h => Y(h[iSp]), "#DC2626", 1.5, [4, 3]);
  line(h => Y(h[iPv]), "#2563EB", 2);
}

async function pollPid() {
  try {
    const r = await fetch(`/api/pid/history?since=${pidSince}`);
    const d = await r.json();
    if (d.samples.length) {
      if (d.samples[0][0] < pidSince) pidHist = [];
      pidHist.push(...d.samples);
      pidSince = d.samples[d.samples.length - 1][0];
      pidHist = pidHist.filter(h => h[0] >= pidSince - WINDOW_S - 1);
    }
    pidNow = d.now;
    drawChart("chartServo",  1, 2, 3, -60, 60, -1);
    drawChart("chartNozzle", 4, 5, 6,   0, 100, -1);

    const p = await (await fetch("/api/pid")).json();
    updateActuators(p);
    document.getElementById("pidMode").textContent = p.mode === "step" ? "STEP TEST…" : "";
    document.getElementById("btnStep").disabled = p.mode === "step";
    if (p.log) document.getElementById("pidLog").textContent = `CSV log: ${p.log}`;
    if (lastMode === "step" && p.mode !== "step") { showMetrics(p.metrics); log("Step test done."); }
    lastMode = p.mode;
  } catch (e) {}
  setTimeout(pollPid, 200);
}

function fillGains(g) {
  for (const loop of ["servo", "nozzle"])
    for (const k of ["kp", "ki", "kd"])
      document.getElementById(`${loop}_${k}`).value = g[loop][k];
}

async function applyGains(loop) {
  const v = k => parseFloat(document.getElementById(`${loop}_${k}`).value);
  const body = {loop, kp: v("kp"), ki: v("ki"), kd: v("kd")};
  const d = await (await fetch("/api/pid", {method: "POST",
    headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)})).json();
  if (d.ok) { fillGains(d.gains); log(`${loop} gains → Kp=${body.kp} Ki=${body.ki} Kd=${body.kd}`); }
  else log(`ERROR: ${d.msg}`);
}

async function stepTest() {
  const d = await (await fetch("/api/pid/step", {method: "POST"})).json();
  log(d.ok ? "Step test started." : `ERROR: ${d.msg}`);
}

function showMetrics(m) {
  const tb = document.querySelector("#metricsTable tbody");
  tb.innerHTML = "";
  for (const loop of ["servo", "nozzle"]) {
    const unit = loop === "servo" ? "°" : "%";
    for (const r of (m[loop] || [])) {
      tb.innerHTML += `<tr><td>${loop === "servo" ? "Angle" : "Nozzle"}</td><td>${r.from}→${r.to}${unit}</td>` +
        `<td>${r.rise_time ?? "—"}</td><td>${r.overshoot_pct}%</td>` +
        `<td>${r.settling_time}</td><td>${r.sse}${unit}</td></tr>`;
    }
  }
  document.getElementById("metricsTable").hidden = false;
  document.getElementById("metricsHelp").hidden  = false;
  if (m.log) document.getElementById("pidLog").textContent = `CSV log: ${m.log}`;
}

// ── Controls ──────────────────────────────────────────────────────────────────
function log(msg) {
  const box = document.getElementById("logBox");
  const ts  = new Date().toLocaleTimeString();
  box.innerHTML += `<div class="log-line">[${ts}] <span>${msg}</span></div>`;
  box.scrollTop = box.scrollHeight;
}

async function startStream() {
  const src = document.getElementById("srcInput").value.trim();
  log(`Starting → ${src}`);
  const res = await fetch("/api/start", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({source: src})
  });
  const data = await res.json();
  if (data.ok) {
    document.getElementById("btnStart").disabled = true;
    document.getElementById("btnStop").disabled  = false;
    document.getElementById("runBadge").textContent = "RUNNING";
    document.getElementById("runBadge").className   = "badge";
    // Show video
    const img = document.getElementById("videoImg");
    img.src = `/video_feed?t=${Date.now()}`;
    img.style.display = "block";
    document.getElementById("videoPlaceholder").style.display = "none";
    log("Stream started.");
  } else {
    log(`ERROR: ${data.msg}`);
  }
}

async function stopStream() {
  await fetch("/api/stop", {method: "POST"});
  onStopped("user");
}

function onStopped(reason) {
  document.getElementById("btnStart").disabled = false;
  document.getElementById("btnStop").disabled  = true;
  document.getElementById("runBadge").textContent = "STOPPED";
  document.getElementById("runBadge").className   = "badge off";
  document.getElementById("videoImg").style.display = "none";
  document.getElementById("videoPlaceholder").style.display = "flex";
  log(`Stopped. Reason: ${reason ?? "unknown"}`);
}

// ── Init ──────────────────────────────────────────────────────────────────────
connectSSE();
log("Dashboard connected.");

// Restore running state on page reload
fetch("/api/state").then(r => r.json()).then(d => {
  if (d.running) {
    document.getElementById("btnStart").disabled = true;
    document.getElementById("btnStop").disabled  = false;
    document.getElementById("runBadge").textContent = "RUNNING";
    document.getElementById("runBadge").className   = "badge";
    const img = document.getElementById("videoImg");
    img.src = `/video_feed?t=${Date.now()}`;
    img.style.display = "block";
    document.getElementById("videoPlaceholder").style.display = "none";
    log("Reconnected to active session.");
  }
  updateUI(d);
});

fetch("/api/pid").then(r => r.json()).then(p => {
  fillGains(p.gains);
  if (p.metrics && p.metrics.servo) showMetrics(p.metrics);
});
pollPid();
</script>
</body>
</html>
""".replace("{{ source }}", str(SOURCE)) \
 .replace("{{ servo_invert }}", "true" if SERVO_INVERT else "false") \
 .replace("{{ nozzle_height }}", str(NOZZLE_HEIGHT))


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
    print(f"  AI Sprayer Web Dashboard")
    print(f"  Local:   http://localhost:{PORT}")
    print(f"  Network: http://{local_ip}:{PORT}")
    print("=" * 50)

    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False)