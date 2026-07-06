"""
AI Sprayer - Web Dashboard
==========================
Open http://<YOUR_PC_IP>:5000 from any device on the same network.

Stream:  /video_feed   → MJPEG (realtime annotated video)
Stats:   /stats_stream → SSE  (nozzle %, pump state, FPS, etc.)
Control: /api/start    → POST
         /api/stop     → POST

CONFIG: Edit the variables below, then run.
"""

import cv2
import threading
import time
import math
import json
import io
import socket
from flask import Flask, Response, render_template_string, request, jsonify

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_PATH   = r"runs/segment/train-8/weights/best.pt"
SOURCE       = "video/grass4.mp4"   # 0 = webcam, or "path/to/video.mp4"
DEVICE       = "cpu"                # "cpu" or "cuda"
CONF         = 0.4
IMGSZ        = 640
PORT         = 5000

HEIGHT_MIN   = 30
HEIGHT_MAX   = 480

# Colors (BGR for OpenCV overlay)
COL_GREEN    = (57, 211, 83)
COL_RED      = (73, 81, 248)
COL_BLUE     = (255, 166, 88)
COL_GRAY     = (158, 148, 139)
# ────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── Shared State ─────────────────────────────────────────────────────────────
state = {
    "running":       False,
    "grass_detected": False,
    "pump_on":        False,
    "nozzle_pct":     0,
    "grass_area_px":  0,
    "grass_height_px":0,
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
def _draw_overlay(frame, s):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 38), (23, 17, 13), -1)
    pump_txt  = "PUMP: ON" if s["pump_on"] else "PUMP: OFF"
    pump_col  = COL_GREEN if s["pump_on"] else COL_RED
    cv2.putText(frame, pump_txt,          (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, pump_col, 2)
    cv2.putText(frame, f"NOZZLE: {s['nozzle_pct']}%", (170, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_BLUE, 2)
    cv2.putText(frame, f"FPS: {s['fps']}", (w - 105, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_GRAY, 1)

    # Height bar (right edge)
    bar_h  = min(h - 60, 200)
    bar_x  = w - 30
    bar_y1 = h - 20 - bar_h
    bar_y2 = h - 20
    fill   = int((s["nozzle_pct"] / 100) * bar_h)
    cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + 16, bar_y2), (45, 36, 30), -1)
    if fill > 0:
        col = COL_GREEN if s["nozzle_pct"] < 60 else (65, 179, 227) if s["nozzle_pct"] < 85 else COL_RED
        cv2.rectangle(frame, (bar_x, bar_y2 - fill), (bar_x + 16, bar_y2), col, -1)
    cv2.putText(frame, "H", (bar_x + 2, bar_y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_GRAY, 1)


def detection_loop():
    global latest_frame, cap
    frame_count = 0
    last_time   = time.time()

    while state["running"]:
        ret, frame = cap.read()
        if not ret:
            with state_lock:
                state["running"] = False
            push_sse({"event": "stopped", "reason": "end_of_source"})
            break

        results     = model(frame, device=DEVICE, conf=CONF, imgsz=IMGSZ, verbose=False)
        annotated   = results[0].plot()
        boxes       = results[0].boxes
        masks       = results[0].masks
        n_detected  = len(boxes) if boxes is not None else 0

        total_area = 0
        max_height = 0
        if n_detected > 0:
            for box in boxes.xyxy:
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1
                total_area += w * h
                if h > max_height:
                    max_height = h
            if masks is not None:
                total_area = int(masks.data.sum().item())

        grass_detected  = n_detected > 0
        pump_on         = grass_detected and max_height >= HEIGHT_MIN
        if pump_on:
            pct = (max_height - HEIGHT_MIN) / max(HEIGHT_MAX - HEIGHT_MIN, 1)
            nozzle_pct = min(100, max(0, int(pct * 100)))
        else:
            nozzle_pct = 0

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
                "nozzle_pct":      nozzle_pct,
                "grass_area_px":   int(total_area),
                "grass_height_px": int(max_height),
                "n_detected":      n_detected,
                "fps":             fps,
            })
            s = dict(state)

        _draw_overlay(annotated, s)

        # Encode to JPEG
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with frame_lock:
            latest_frame = buf.tobytes()

        # Push stats via SSE
        push_sse({
            "pump_on":         s["pump_on"],
            "nozzle_pct":      s["nozzle_pct"],
            "grass_detected":  s["grass_detected"],
            "grass_area_px":   s["grass_area_px"],
            "grass_height_px": s["grass_height_px"],
            "n_detected":      s["n_detected"],
            "detect_count":    s["detect_count"],
            "fps":             s["fps"],
        })

        time.sleep(0.01)

    # Release cap when stopped
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

    _worker = threading.Thread(target=detection_loop, daemon=True)
    _worker.start()
    push_sse({"event": "started", "source": str(source)})
    return jsonify({"ok": True})


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
        <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
      </div>
    </div>

    <!-- Stats -->
    <div class="card">
      <div class="card-title">▸ Measurements</div>
      <div class="stat-row"><span class="stat-key">Grass Area</span><span class="stat-val" id="statArea">0 px²</span></div>
      <div class="stat-row"><span class="stat-key">Grass Height</span><span class="stat-val" id="statHeight">0 px</span></div>
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

  // Nozzle gauge
  const pct = d.nozzle_pct ?? 0;
  const offset = GAUGE_LEN * (1 - pct / 100);
  const fill = document.getElementById("gaugeFill");
  fill.setAttribute("stroke-dashoffset", offset);
  fill.setAttribute("stroke", pct < 60 ? "#16A34A" : pct < 85 ? "#D97706" : "#DC2626");

  // Needle: -90deg (0%) to +90deg (100%)
  const deg = -90 + (pct / 100) * 180;
  document.getElementById("gaugeNeedle").style.transform = `rotate(${deg}deg)`;

  document.getElementById("nozzlePct").textContent = `${pct}%`;
  const pf = document.getElementById("progressFill");
  pf.style.width = `${pct}%`;
  pf.style.background = pct < 60 ? "var(--grn)" : pct < 85 ? "var(--ylw)" : "var(--red)";

  // Stats
  document.getElementById("statArea").textContent   = `${(d.grass_area_px ?? 0).toLocaleString()} px²`;
  document.getElementById("statHeight").textContent = `${d.grass_height_px ?? 0} px`;
  document.getElementById("statInst").textContent   = d.n_detected ?? 0;
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
</script>
</body>
</html>
""".replace("{{ source }}", str(SOURCE))


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