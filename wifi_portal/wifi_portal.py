#!/usr/bin/env python3
"""
WiFi manager for the Orange Pi (like WiFiManager on ESP32, but for NetworkManager).

  - Watches wlan0. If it has no WiFi for a while, it starts a hotspot + captive portal.
  - The user joins the hotspot, picks a network and types the password.
  - The Pi tries it. On failure the old settings are restored and the hotspot comes back
    with the error message. On success the hotspot goes away.
  - While in hotspot mode with nobody connected, it retries the saved networks every
    `retry_every` seconds (e.g. a phone hotspot that was switched off earlier).

Only the standard library. Runs as root under systemd (see wifi-portal.service).
Run with --dry-run to log decisions without touching the network.
"""
import html
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

CONF_PATH  = "/etc/wifi-portal.json"
STATE_DIR  = "/var/lib/wifi-portal"
FORCE_FILE = "/run/wifi-portal-force-ap"      # `touch` it to start the hotspot now (one-shot)
DNS_SNIPPET = "/etc/NetworkManager/dnsmasq-shared.d/90-wifi-portal.conf"
AP_CON     = "sprayer-ap"

DEFAULTS = dict(
    iface="wlan0",
    ap_ssid_prefix="AI-Sprayer-Setup",
    ap_password="sprayer1234",      # WPA2, min 8 chars. Open hotspot would send WiFi passwords in clear.
    ap_channel=6,                   # 2.4 GHz: every phone can see it
    ap_ip="10.42.0.1",
    boot_grace=45,                  # s NetworkManager gets to connect after the service starts
    drop_grace=30,                  # s without WiFi before the hotspot starts
    retry_every=180,                # s in hotspot, nobody connected → retry saved networks
    connect_timeout=35,             # s to wait for a new network
    eth_counts=True,                # wired link counts as "connected" (no hotspot)
    app_port=5000,                  # shown on the "connected" page
)

DRY = "--dry-run" in sys.argv

# What each OS expects from its "is there internet?" check. Sent only to phones whose user pressed
# "Open Dashboard": the phone then treats the hotspot as a normal network and keeps the dashboard reachable
# (otherwise Android sends traffic over mobile data and the local dashboard cannot be opened).
_HTML_OK = b"<HTML><HEAD><TITLE>Success</TITLE></HEAD><BODY>Success</BODY></HTML>"
PROBE_OK = {
    "/generate_204": (204, b"", "text/plain"),
    "/gen_204": (204, b"", "text/plain"),
    "/hotspot-detect.html": (200, _HTML_OK, "text/html"),
    "/library/test/success.html": (200, _HTML_OK, "text/html"),
    "/connecttest.txt": (200, b"Microsoft Connect Test", "text/plain"),
    "/ncsi.txt": (200, b"Microsoft NCSI", "text/plain"),
    "/success.txt": (200, b"success\n", "text/plain"),
}
ACK_TTL = 24 * 3600                 # s a phone stays "acknowledged" (per IP, kept in memory only)


LOG_FILE = os.path.join(STATE_DIR, "wifi-portal.log")   # survives reboots (journald here is RAM-only)


def log(*a):
    line = "[wifi-portal] " + " ".join(str(x) for x in a)
    print(line, flush=True)
    try:
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 200_000:
            with open(LOG_FILE, "rb") as f:
                f.seek(-100_000, 2)
                tail = f.read()
            with open(LOG_FILE, "wb") as f:
                f.write(tail)
        with open(LOG_FILE, "a") as f:
            f.write(f"{time.strftime('%m-%d %H:%M:%S')} up{int(time.monotonic()):>6}s {line}\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass


# ── pure helpers (unit tested) ───────────────────────────────────────────────
def split_terse(line):
    """Split one line of `nmcli -t` output on unescaped ':' and unescape the fields."""
    return [f.replace("\\:", ":").replace("\\\\", "\\") for f in re.split(r"(?<!\\):", line)]


def parse_scan(out):
    """`nmcli -t -f SSID,SIGNAL,SECURITY,CHAN` → unique networks, strongest first."""
    best = {}
    for line in out.splitlines():
        f = split_terse(line)
        if len(f) < 3 or not f[0]:
            continue
        ssid, sec = f[0], f[2].strip()
        try:
            sig = int(f[1])
        except ValueError:
            sig = 0
        if ssid not in best or sig > best[ssid]["signal"]:
            best[ssid] = dict(ssid=ssid, signal=sig, security=sec, open=sec in ("", "--"))
    return sorted(best.values(), key=lambda n: -n["signal"])


def key_mgmt_for(security):
    """'open' | 'wpa-psk' | 'sae' | None (not supported: WEP / enterprise)."""
    tok = set(security.replace("--", "").split())
    if not tok:
        return "open"
    if "802.1X" in tok or "WEP" in tok:
        return None
    if "WPA3" in tok and not tok & {"WPA1", "WPA2", "WPA"}:
        return "sae"
    return "wpa-psk"


def validate_credentials(ssid, password):
    """Returns an error text, or None if OK. Empty password = open network."""
    if not ssid or len(ssid.encode()) > 32 or any(ord(c) < 32 for c in ssid):
        return "Nama WiFi harus 1-32 karakter."
    if password and not (8 <= len(password) <= 63 or re.fullmatch(r"[0-9a-fA-F]{64}", password)):
        return "Password WiFi harus 8-63 karakter."
    if any(ord(c) < 32 for c in password):
        return "Password mengandung karakter yang tidak valid."
    return None


def explain_error(out):
    low = out.lower()
    if "secrets were required" in low or "4-way" in low or "wrong password" in low:
        return "Password salah."
    if "no network with ssid" in low:
        return "WiFi tidak ditemukan (terlalu jauh?)."
    if "timeout" in low or "timed out" in low:
        return "Waktu habis: password salah atau sinyal terlalu lemah."
    return "Gagal tersambung: " + (out.strip().splitlines() or ["penyebab tidak diketahui"])[-1][:160]


def sd_notify(msg):
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        s.sendto(msg.encode(), "\0" + addr[1:] if addr[0] == "@" else addr)
        s.close()
    except OSError:
        pass


# ── system calls ─────────────────────────────────────────────────────────────
MUTATING = {("con", "add"), ("con", "modify"), ("con", "delete"), ("con", "up"),
            ("con", "down"), ("dev", "connect"), ("dev", "disconnect"), ("radio", "wifi")}


def run(*args, timeout=20):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except OSError as e:
        return 127, str(e)


def nm(*args, timeout=20):
    words = [a for a in args if not a.startswith("-") and not a.isdigit()]
    if DRY and tuple(words[:2]) in MUTATING:
        log("DRY-RUN would run: nmcli", *("***" if "psk" in str(args[i - 1]) else a
                                         for i, a in enumerate(args)))
        return 0, ""
    return run("nmcli", *args, timeout=timeout)


def load_config():
    cfg = dict(DEFAULTS)
    try:
        with open(CONF_PATH) as f:
            cfg.update(json.load(f))
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as e:
        log("bad config, using defaults:", e)
    if len(str(cfg["ap_password"])) < 8:
        log("ap_password too short, using default")
        cfg["ap_password"] = DEFAULTS["ap_password"]
    return cfg


# ── controller ───────────────────────────────────────────────────────────────
class Controller:
    def __init__(self, cfg):
        self.cfg = cfg
        self.iface = cfg["iface"]
        self.t_start = time.monotonic()
        self.deadline = self.t_start + cfg["boot_grace"]   # when to give up and start the hotspot
        self.was_online = False
        self.server = None
        self.networks = []
        self.job = None
        self.job_lock = threading.Lock()
        self.last_client = time.monotonic()
        self.ap_failures = 0
        self.stale = 0
        self.stale_fixed = False
        self.beat = time.monotonic()
        self.last_trace = 0
        self.running = True
        self.last_result = self._load_result()
        self.ack = {}                                # client IP → monotonic time of "Open Dashboard"
        os.makedirs(STATE_DIR, exist_ok=True)
        log("boot state:", self.status(), "profiles:", self.wifi_profiles())

    # persistence of the last connect attempt (shown on the portal)
    def _load_result(self):
        try:
            with open(os.path.join(STATE_DIR, "last_result.json")) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def record(self, ok, ssid, msg):
        self.last_result = dict(ok=ok, ssid=ssid, msg=msg, ts=int(time.time()))
        try:
            with open(os.path.join(STATE_DIR, "last_result.json"), "w") as f:
                json.dump(self.last_result, f)
        except OSError as e:
            log("cannot save result:", e)

    # ── status ──
    def status(self):
        """None if NetworkManager did not answer (never decide anything on a failed read)."""
        rc, out = nm("-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "dev")
        if rc != 0:
            return None
        st = dict(wifi_ok=False, ap_up=False, eth_ok=False, conn=None)
        for line in out.splitlines():
            f = split_terse(line)
            if len(f) < 4:
                continue
            dev, typ, state, con = f[:4]
            if dev == self.iface and state.startswith("connected"):
                if con == AP_CON:
                    st["ap_up"] = True
                else:
                    st["wifi_ok"], st["conn"] = True, con
            elif typ == "ethernet" and state.startswith("connected"):
                st["eth_ok"] = True
        return st

    def wifi_profiles(self):
        rc, out = nm("-t", "-f", "NAME,TYPE", "con")
        return [f[0] for f in map(split_terse, out.splitlines())
                if rc == 0 and len(f) >= 2 and f[1] == "802-11-wireless" and f[0] != AP_CON]

    def find_profile(self, ssid):
        for name in self.wifi_profiles():
            rc, out = nm("-g", "802-11-wireless.ssid", "con", "show", name)
            if rc == 0 and out == ssid:
                return name
        return None

    def gateway_alive(self):
        """ARP check of the default gateway (works even when the router ignores ping)."""
        rc, out = run("ip", "-4", "route", "show", "default", "dev", self.iface)
        m = re.search(r"default via (\S+)", out)
        if not m:
            return False
        gw = m.group(1)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.sendto(b"x", (gw, 9))           # forces the kernel to resolve/refresh the ARP entry
            s.close()
        except OSError:
            return False
        time.sleep(1)
        rc, out = run("ip", "neigh", "show", "to", gw, "dev", self.iface)
        return bool(out) and not re.search(r"FAILED|INCOMPLETE", out)

    def clients(self):
        rc, out = run("iw", "dev", self.iface, "station", "dump")
        return len(re.findall(r"^Station ", out, re.M)) if rc == 0 else 0

    # ── hotspot ──
    def ap_ssid(self):
        try:
            with open(f"/sys/class/net/{self.iface}/address") as f:
                tail = f.read().strip().replace(":", "")[-4:].upper()
        except OSError:
            tail = "0000"
        return f"{self.cfg['ap_ssid_prefix']}-{tail}"

    def scan(self):
        nm("dev", "wifi", "rescan", "ifname", self.iface, timeout=15)
        time.sleep(3)
        rc, out = nm("-t", "-f", "SSID,SIGNAL,SECURITY,CHAN", "dev", "wifi", "list",
                     "ifname", self.iface, timeout=15)
        if rc == 0:
            self.networks = parse_scan(out)
        log(f"scan: {len(self.networks)} networks")

    def _write_dns_snippet(self):
        # every DNS name → us, so phones detect the captive portal
        want = f"address=/#/{self.cfg['ap_ip']}\n"
        try:
            os.makedirs(os.path.dirname(DNS_SNIPPET), exist_ok=True)
            if not os.path.exists(DNS_SNIPPET) or open(DNS_SNIPPET).read() != want:
                with open(DNS_SNIPPET, "w") as f:
                    f.write(want)
        except OSError as e:
            log("cannot write dns snippet:", e)

    def enter_ap(self):
        if DRY:
            log("DRY-RUN would start hotspot", self.ap_ssid())
            self.deadline = time.monotonic() + 60
            return False
        self.scan()                                 # radio is busy once the hotspot is up
        self._write_dns_snippet()
        c = self.cfg
        nm("con", "delete", AP_CON)
        rc, out = nm("con", "add", "type", "wifi", "ifname", self.iface, "con-name", AP_CON,
                     "autoconnect", "no", "ssid", self.ap_ssid(),
                     "802-11-wireless.mode", "ap", "802-11-wireless.band", "bg",
                     "802-11-wireless.channel", str(c["ap_channel"]),
                     "wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", c["ap_password"],
                     "wifi-sec.proto", "rsn", "wifi-sec.pairwise", "ccmp", "wifi-sec.group", "ccmp",
                     "ipv4.method", "shared", "ipv4.addresses", f"{c['ap_ip']}/24",
                     "ipv6.method", "ignore")
        if rc == 0:
            rc, out = nm("--wait", "30", "con", "up", AP_CON, "ifname", self.iface, timeout=45)
        if rc != 0:
            self.ap_failures += 1
            log(f"hotspot failed ({self.ap_failures}): {out}")
            if self.ap_failures % 3 == 0:           # radio stuck? power-cycle it
                run("rfkill", "unblock", "wifi")
                nm("radio", "wifi", "off")
                time.sleep(2)
                nm("radio", "wifi", "on")
            self.deadline = time.monotonic() + min(10 * self.ap_failures, 60)
            return False
        self.ap_failures = 0
        self.last_client = time.monotonic()
        self.stale = 0
        log("hotspot up:", self.ap_ssid())
        return self.start_server()

    def leave_ap(self):
        self.stop_server()
        nm("con", "down", AP_CON)

    def start_server(self):
        self.stop_server()
        ip = self.cfg["ap_ip"]
        for _ in range(20):                         # wait for NM to put the IP on the interface
            if ip in run("ip", "-4", "-o", "addr", "show", "dev", self.iface)[1]:
                break
            time.sleep(0.5)
        try:
            srv = ThreadingHTTPServer((ip, 80), make_handler(self))
        except OSError as e:
            log("portal cannot bind:", e)
            return False
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.server = srv
        log(f"portal on http://{ip}/")
        return True

    def stop_server(self):
        srv, self.server = self.server, None
        if srv:
            srv.shutdown()
            srv.server_close()

    # ── connect request from the portal ──
    def submit(self, ssid, password, manual):
        with self.job_lock:
            if self.job:
                return False
            # let the HTTP answer reach the phone before the hotspot goes down
            self.job = dict(ssid=ssid, password=password, manual=manual, at=time.monotonic() + 3)
            return True

    def precheck(self, ssid, pw):
        """→ (key_mgmt, error). Runs in the HTTP handler so the user sees errors at once."""
        err = validate_credentials(ssid, pw)
        if err:
            return None, err
        known = next((n for n in self.networks if n["ssid"] == ssid), None)
        kind = key_mgmt_for(known["security"]) if known else ("wpa-psk" if pw else "open")
        if kind is None:
            return None, "WiFi jenis WEP / enterprise (kantor) belum didukung."
        if kind != "open" and not pw:
            return None, "WiFi ini membutuhkan password."
        return kind, None

    def do_connect(self, job):
        ssid, pw = job["ssid"], job["password"]
        log("connecting to", ssid)
        kind, _ = self.precheck(ssid, pw)
        timeout = self.cfg["connect_timeout"]
        self.leave_ap()                             # frees the radio
        existing = self.find_profile(ssid)
        old_psk = None
        name = existing or ssid
        if existing:
            old_psk = nm("-s", "-g", "802-11-wireless-security.psk", "con", "show", existing)[1] or None
            if kind != "open":
                nm("con", "modify", existing, "wifi-sec.key-mgmt", kind, "wifi-sec.psk", pw)
        else:
            args = ["con", "add", "type", "wifi", "ifname", self.iface, "con-name", name, "ssid", ssid,
                    "connection.autoconnect", "yes", "connection.autoconnect-priority", "10",
                    "ipv4.method", "auto", "ipv6.method", "auto"]
            if job["manual"]:
                args += ["802-11-wireless.hidden", "yes"]
            if kind != "open":
                args += ["wifi-sec.key-mgmt", kind, "wifi-sec.psk", pw]
            nm(*args)
        rc, out = nm("--wait", str(timeout), "con", "up", name, "ifname", self.iface, timeout=timeout + 15)
        st = self.status()
        if rc == 0 and st and st["wifi_ok"]:
            log("connected to", ssid)
            self.record(True, ssid, "Berhasil tersambung.")
            self.deadline = None
            self.was_online = True
            return
        log(f"connect failed: {out}")
        if existing:                                # put the old password back
            if old_psk:
                nm("con", "modify", existing, "wifi-sec.psk", old_psk)
        else:
            nm("con", "delete", name)
        self.record(False, ssid, explain_error(out))

    # ── main loop ──
    def run_loop(self):
        while self.running:
            self.beat = time.monotonic()
            try:
                self.tick()
            except Exception as e:                  # never let one bad tick kill the service
                log("tick error:", repr(e))
                time.sleep(5)

    def tick(self):
        now = time.monotonic()
        with self.job_lock:
            job = self.job if self.job and now >= self.job["at"] else None
        if job:
            self.do_connect(job)
            with self.job_lock:
                self.job = None
            st = self.status()
            if st and not st["wifi_ok"] and not st["ap_up"]:
                self.enter_ap()                     # failed → hotspot with the error message
            return

        if os.path.exists(FORCE_FILE):
            os.remove(FORCE_FILE)
            log("force-ap requested")
            self.deadline = now
            self.was_online = False
            if not self.enter_ap():
                return

        st = self.status()
        if st is None:
            time.sleep(2)
            return

        if self.server and not st["ap_up"]:
            self.stop_server()
        if st["ap_up"] and not self.server:
            self.start_server()                     # e.g. service restarted while hotspot was up

        if st["wifi_ok"]:
            self.was_online, self.deadline = True, None
            self.check_gateway()
            time.sleep(5)
            return

        self.stale = 0
        if now - self.last_trace > 15:
            self.last_trace = now
            left = None if self.deadline is None else round(self.deadline - now)
            log(f"no wifi: state={st} profiles={self.wifi_profiles()} hotspot_in={left}s")
        if st["ap_up"]:
            self.ap_tick(now)
            time.sleep(2)
            return

        if self.cfg["eth_counts"] and st["eth_ok"]:
            self.deadline = None
            time.sleep(5)
            return

        if self.deadline is None:                   # WiFi just dropped
            self.deadline = now + self.cfg["drop_grace"]
            log(f"wifi lost, hotspot in {self.cfg['drop_grace']}s unless it comes back")
        if not self.wifi_profiles() and not self.was_online:
            self.deadline = min(self.deadline, self.t_start + 10)   # nothing saved: no point waiting
        if now >= self.deadline:
            self.enter_ap()
        else:
            time.sleep(2)

    def check_gateway(self):
        """Associated but the router is gone (zombie link): reconnect once, then give up."""
        if self.gateway_alive():
            self.stale = 0
            self.stale_fixed = False
            return
        self.stale += 1
        if self.stale == 12 and not self.stale_fixed:        # ~1 min of silence
            st = self.status()
            log("gateway unreachable, reconnecting", st and st["conn"])
            if st and st["conn"]:
                nm("--wait", "30", "con", "up", st["conn"], "ifname", self.iface, timeout=45)
            self.stale_fixed = True
            self.stale = 0
        elif self.stale_fixed and self.stale == 12:
            log("still unreachable, dropping to hotspot mode")
            self.deadline, self.was_online = time.monotonic(), False
            nm("dev", "disconnect", self.iface)

    def ap_tick(self, now):
        if self.clients() > 0:
            self.last_client = now
        elif now - self.last_client > self.cfg["retry_every"]:
            log("nobody on the hotspot, retrying saved networks")
            self.leave_ap()
            nm("dev", "wifi", "rescan", "ifname", self.iface, timeout=15)
            time.sleep(3)
            nm("--wait", "30", "dev", "connect", self.iface, timeout=45)
            st = self.status()
            if st and st["wifi_ok"]:
                log("reconnected to", st["conn"])
                self.deadline, self.was_online = None, True
            else:
                self.enter_ap()


# ── web portal ───────────────────────────────────────────────────────────────
PAGE = """<!doctype html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Prototype Sprayer Weeder - Pengaturan WiFi</title>
<style>
:root{--bg:#fafafa;--card:#fff;--txt:#18181b;--mut:#71717a;--bd:#e4e4e7;--acc:#16a34a;--err:#dc2626}
@media(prefers-color-scheme:dark){:root{--bg:#0e0e10;--card:#18181b;--txt:#f4f4f5;--mut:#a1a1aa;--bd:#3f3f46}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font:16px system-ui,sans-serif}
main{max-width:460px;margin:0 auto;padding:16px}h1{font-size:1.2rem;margin:8px 0 14px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px;margin-bottom:12px}
.msg{padding:10px 12px;border-radius:8px;margin-bottom:12px;font-size:.9rem}
.ok{background:#dcfce7;color:#14532d}.bad{background:#fee2e2;color:#7f1d1d}
label.net{display:flex;align-items:center;gap:10px;padding:11px 4px;border-bottom:1px solid var(--bd)}
label.net:last-of-type{border-bottom:0}label.net span.s{margin-left:auto;color:var(--mut);font-size:.8rem}
input[type=text],input[type=password]{width:100%;padding:11px;border:1px solid var(--bd);border-radius:8px;
background:var(--bg);color:var(--txt);font-size:1rem;margin-top:8px}
button{width:100%;padding:13px;border:0;border-radius:8px;background:var(--acc);color:#fff;font-size:1rem;
font-weight:600;margin-top:8px}small{color:var(--mut)}
a.btn{display:block;text-align:center;text-decoration:none;padding:13px;border-radius:8px;background:#2563eb;
color:#fff;font-weight:600;margin-top:10px}
</style></head><body><main><h1>Prototype Sprayer Weeder · Pengaturan WiFi</h1>{{result}}
<div class="card"><b>Tidak ada WiFi di sini?</b><br><small>Pakai alat langsung lewat hotspot ini. Tidak perlu internet.</small>
<a class="btn" href="/use-hotspot">Buka Dashboard</a></div>
<form method="post" action="/connect" autocomplete="off">
<div class="card">{{networks}}
<label class="net"><input type="radio" name="pick" value="__manual__"{{manual_checked}}> WiFi lain / WiFi tersembunyi…</label>
<input type="text" id="manual" name="manual" placeholder="Nama WiFi (SSID)" autocapitalize="off" autocorrect="off">
</div>
<div class="card"><b>Password</b> <small>(kosongkan jika WiFi tanpa password)</small>
<input type="password" id="pw" name="password" autocomplete="off">
<label><input type="checkbox" onclick="pw.type=this.checked?'text':'password'"> tampilkan password</label></div>
<button type="submit">Sambungkan</button></form>
<p><small>Setelah menekan Sambungkan, WiFi buatan alat ini mati sampai sekitar satu menit. Jika berhasil, WiFi itu tetap mati.
Jika gagal (misalnya password salah), WiFi itu muncul lagi dan alasannya tampil di halaman ini.</small></p>
<script>const m=document.getElementById('manual');
m.addEventListener('input',()=>{document.querySelector('input[value=__manual__]').checked=true})</script>
</main></body></html>"""

DONE = """<!doctype html><html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Menyambungkan…</title><style>body{font:16px system-ui,sans-serif;max-width:460px;margin:0 auto;padding:20px}</style></head>
<body><h2>Menyambungkan ke “{{ssid}}”…</h2>
<p>WiFi buatan alat ini akan mati sekarang, jadi HP Anda akan terputus.</p>
<ul><li><b>Berhasil:</b> WiFi buatan alat tetap mati. Sambungkan HP Anda ke WiFi yang sama, lalu buka alat di
<code>http://{{host}}.local</code></li>
<li><b>Gagal:</b> dalam sekitar satu menit WiFi <b>{{ap}}</b> muncul lagi. Sambungkan lagi untuk melihat penyebabnya.</li></ul>
</body></html>"""


def render_page(ctrl):
    saved = set(ctrl.wifi_profiles())
    rows = []
    for n in ctrl.networks[:25]:
        lock = "" if n["open"] else " 🔒"
        tag = "tersimpan · " if n["ssid"] in saved else ""
        rows.append('<label class="net"><input type="radio" name="pick" value="%s"> %s%s'
                    '<span class="s">%s%d%%</span></label>'
                    % (html.escape(n["ssid"], True), html.escape(n["ssid"]), lock, tag, n["signal"]))
    if not rows:
        rows.append("<small>Tidak ada WiFi ditemukan. Ketik namanya di bawah.</small>")
    r, res = ctrl.last_result, ""
    if r:
        cls = "ok" if r["ok"] else "bad"
        res = '<div class="msg %s">Percobaan terakhir: <b>%s</b> — %s</div>' % (cls, html.escape(r["ssid"]), html.escape(r["msg"]))
    return (PAGE.replace("{{result}}", res).replace("{{networks}}", "".join(rows))
            .replace("{{manual_checked}}", "" if ctrl.networks else " checked"))


def make_handler(ctrl):
    ap_ip = ctrl.cfg["ap_ip"]

    class H(BaseHTTPRequestHandler):
        timeout = 10
        server_version = "wifi-portal"

        def log_message(self, fmt, *a):
            pass

        def _send(self, code, body=b"", ctype="text/html; charset=utf-8", extra=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            if code != 204:
                self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _redirect(self):
            self._send(302, extra=[("Location", f"http://{ap_ip}/")])

        def do_GET(self):
            host = (self.headers.get("Host") or "").split(":")[0]
            path = self.path.split("?")[0]
            if path == "/api/status":
                return self._send(200, json.dumps(dict(last=ctrl.last_result, networks=ctrl.networks)).encode(),
                                  "application/json")
            ip = self.client_address[0]
            if path == "/use-hotspot":                # user chose to work without WiFi
                ctrl.ack[ip] = time.monotonic()
                return self._send(302, extra=[("Location", f"http://{ap_ip}:{ctrl.cfg['app_port']}/")])
            if path in PROBE_OK and time.monotonic() - ctrl.ack.get(ip, -ACK_TTL) < ACK_TTL:
                code, body, ctype = PROBE_OK[path]
                return self._send(code, body, ctype)
            if host != ap_ip or path != "/":          # captive-portal probes land here → show the portal
                return self._redirect()
            self._send(200, render_page(ctrl).encode())

        do_HEAD = do_GET

        def do_POST(self):
            if self.path != "/connect":
                return self._redirect()
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = 0
            if n <= 0 or n > 4096:
                return self._send(413, b"Permintaan tidak valid")
            f = parse_qs(self.rfile.read(n).decode("utf-8", "replace"), keep_blank_values=True)
            one = lambda k: (f.get(k) or [""])[0]
            manual = one("pick") == "__manual__"
            ssid = one("manual").strip() if manual else one("pick")
            pw = one("password")
            _, err = ctrl.precheck(ssid, pw)
            if err:
                ctrl.record(False, ssid or "-", err)
                return self._redirect()
            if not ctrl.submit(ssid, pw, manual):
                return self._send(409, b"Sedang menyambung, mohon tunggu.")
            self._send(200, DONE.replace("{{ssid}}", html.escape(ssid))
                       .replace("{{host}}", html.escape(socket.gethostname()))
                       .replace("{{ap}}", html.escape(ctrl.ap_ssid())).encode())

    return H


# ── entry point ──────────────────────────────────────────────────────────────
def main():
    ctrl = Controller(load_config())
    signal.signal(signal.SIGTERM, lambda *_: setattr(ctrl, "running", False))
    signal.signal(signal.SIGINT, lambda *_: setattr(ctrl, "running", False))

    def heartbeat():                                 # systemd restarts us if the main loop hangs
        while True:
            if time.monotonic() - ctrl.beat < 150:
                sd_notify("WATCHDOG=1")
            time.sleep(15)

    threading.Thread(target=heartbeat, daemon=True).start()
    sd_notify("READY=1")
    log("started", "(dry-run)" if DRY else "", "iface", ctrl.iface)
    ctrl.run_loop()
    ctrl.stop_server()                               # leave the hotspot as it is; the next start picks it up


if __name__ == "__main__":
    main()
