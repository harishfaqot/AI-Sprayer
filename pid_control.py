"""
PID control for the AI Sprayer (software only, servos are simulated)
=====================================================================

WHAT IS CONTROLLED
  1. Angle servo   : turns the nozzle left / right   (target comes from grass position)
  2. Nozzle servo  : opens / closes the nozzle        (target comes from grass area)

WHY PID IF WE USE SERVOS?
  A hobby servo goes to whatever angle you send it, but the camera gives a new
  target ~10 times per second and the target jumps around. If we send the raw
  target directly, the nozzle jerks, overshoots and shakes.
  The PID decides HOW FAST the servo should move toward the target every 5 ms:
      far from target  → move fast
      close to target  → slow down and stop smoothly

THE PID IN ONE LINE
      speed = Kp * error  +  Ki * (sum of error over time)  +  Kd * (how fast it's moving, braking)
      error = target - actual

      Kp (Proportional) : main push, bigger error → faster move
      Ki (Integral)     : removes the small error that is left at the end
                          (keep it small here: the servo command already adds up
                           the PID output, so a big Ki causes overshoot)
      Kd (Derivative)   : brake, stops it from overshooting

  The PID output (speed) is added to the servo command angle:
      command_angle += speed * dt        ← this is the value sent to the real servo

NO HARDWARE YET
  ServoModel below imitates a real hobby servo (it needs a little time to reach
  the command, has a max speed and ignores very small changes). When hardware is
  ready, send `command` to the servo instead of using the model.
"""

import csv
import os
import threading
import time
from collections import deque


# ── SETTINGS ────────────────────────────────────────────────────────────────
CONTROL_HZ = 200                 # PID runs 200 times per second (every 5 ms)

# Angle servo: values in degrees (0 = nozzle pointing straight down)
ANGLE_GAINS = dict(kp=6.0, ki=0.1, kd=0.2)
ANGLE_SERVO = dict(max_speed=300.0,   # deg/s   (fastest the servo can turn)
                   lag=0.10,          # s       (how long the servo needs to respond)
                   deadband=0.5,      # deg     (servo ignores smaller changes)
                   lo=-60.0, hi=60.0)

# Nozzle servo: values in % opening (0 = closed, 100 = fully open)
NOZZLE_GAINS = dict(kp=8.0, ki=0.1, kd=0.2)
NOZZLE_SERVO = dict(max_speed=250.0,  # %/s
                    lag=0.08,
                    deadband=0.5,     # %
                    lo=0.0, hi=100.0)

HISTORY_HZ  = 50                 # points per second kept for the dashboard chart
HISTORY_SEC = 20
LOG_DIR     = "logs"

# Step test: at these times (s) the target changes to (angle deg, nozzle %)
STEP_SEQUENCE = [
    (0.0,   0.0,  0.0),
    (0.5,  30.0, 80.0),
    (2.5, -30.0, 30.0),
    (4.5,   0.0,  0.0),
]
STEP_DURATION = 6.5
# ────────────────────────────────────────────────────────────────────────────


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class PID:
    def __init__(self, kp, ki, kd, out_limit):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_limit = out_limit          # max output (= servo max speed)
        self.integral  = 0.0
        self.prev_actual = None
        self.p = self.i = self.d = 0.0      # last P, I, D parts (for logging)

    def update(self, target, actual, dt):
        error = target - actual

        # I: add up the error over time. Limited so it can't grow forever (anti-windup).
        self.integral += error * dt
        if self.ki:
            i_max = 0.3 * self.out_limit / self.ki
            self.integral = clamp(self.integral, -i_max, i_max)

        # D: how fast the actual value is changing → used as a brake
        if self.prev_actual is None:
            self.prev_actual = actual
        speed_now = (actual - self.prev_actual) / dt
        self.prev_actual = actual

        self.p = self.kp * error
        self.i = self.ki * self.integral
        self.d = -self.kd * speed_now

        return clamp(self.p + self.i + self.d, -self.out_limit, self.out_limit)


class ServoModel:
    """Imitates a hobby servo. `command` = angle we send, `actual` = where it really is."""

    def __init__(self, max_speed, lag, deadband, lo, hi):
        self.max_speed, self.lag, self.deadband = max_speed, lag, deadband
        self.lo, self.hi = lo, hi
        self.command = 0.0
        self.actual  = 0.0

    def step(self, speed, dt):
        # PID output (speed) moves the command angle
        self.command = clamp(self.command + speed * dt, self.lo, self.hi)

        # Servo follows the command with a small delay and a max speed
        diff = self.command - self.actual
        if abs(diff) > self.deadband:
            move = clamp(diff / self.lag, -self.max_speed, self.max_speed) * dt
            self.actual = clamp(self.actual + move, self.lo, self.hi)


class Loop:
    """One PID + one servo."""

    def __init__(self, gains, servo):
        self.servo  = ServoModel(**servo)
        self.pid    = PID(**gains, out_limit=servo["max_speed"])
        self.target = 0.0
        self.output = 0.0

    def step(self, dt):
        self.output = self.pid.update(self.target, self.servo.actual, dt)
        self.servo.step(self.output, dt)


class ControlSystem:
    def __init__(self, realtime=True):
        self.loops = {"servo":  Loop(ANGLE_GAINS,  ANGLE_SERVO),
                      "nozzle": Loop(NOZZLE_GAINS, NOZZLE_SERVO)}
        self.dt    = 1.0 / CONTROL_HZ
        self.t     = 0.0
        self.lock  = threading.Lock()

        self.vision_target = {"servo": 0.0, "nozzle": 0.0}
        self.mode    = "vision"             # "vision" = follow camera, "step" = step test
        self.step_t0 = 0.0
        self.metrics = {}

        self.history     = deque(maxlen=HISTORY_HZ * HISTORY_SEC)
        self._hist_every = max(1, CONTROL_HZ // HISTORY_HZ)
        self._n          = 0
        self._step_log   = []
        self._step_owns_log = False
        self._log_file = self._log_writer = self.log_path = None

        if realtime:
            threading.Thread(target=self._run, daemon=True).start()

    # ── Used by main_web.py ───────────────────────────────────────────────
    def set_targets(self, angle_deg, nozzle_pct):
        with self.lock:
            self.vision_target["servo"]  = angle_deg
            self.vision_target["nozzle"] = nozzle_pct

    def set_gains(self, loop, kp=None, ki=None, kd=None):
        with self.lock:
            pid = self.loops[loop].pid
            if kp is not None: pid.kp = float(kp)
            if ki is not None: pid.ki = float(ki)
            if kd is not None: pid.kd = float(kd)

    def gains(self):
        with self.lock:
            return {n: dict(kp=l.pid.kp, ki=l.pid.ki, kd=l.pid.kd) for n, l in self.loops.items()}

    def snapshot(self):
        with self.lock:
            a, n = self.loops["servo"], self.loops["nozzle"]
            return {"t": round(self.t, 3), "mode": self.mode,
                    "servo_sp":  round(a.target, 2), "servo_pv":  round(a.servo.actual, 2),
                    "servo_cmd": round(a.servo.command, 2), "servo_u": round(a.output, 2),
                    "nozzle_sp": round(n.target, 2), "nozzle_pv": round(n.servo.actual, 2),
                    "nozzle_cmd": round(n.servo.command, 2), "nozzle_u": round(n.output, 2)}

    def get_history(self, since=-1.0):
        with self.lock:
            return [h for h in self.history if h[0] > since]

    def start_step_test(self):
        with self.lock:
            if self.mode == "step":
                return False
            self.mode, self.step_t0 = "step", self.t
            self._step_log, self.metrics = [], {}
        if self.log_path is None:
            self.start_log("step")
            self._step_owns_log = True
        return True

    # ── CSV log (for reports / thesis graphs) ─────────────────────────────
    def start_log(self, tag="run"):
        self.stop_log()
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, f"pid_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        f = open(path, "w", newline="")
        w = csv.writer(f)
        g = self.gains()
        w.writerow([f"# sample time {self.dt}s",
                    f"angle Kp={g['servo']['kp']} Ki={g['servo']['ki']} Kd={g['servo']['kd']}",
                    f"nozzle Kp={g['nozzle']['kp']} Ki={g['nozzle']['ki']} Kd={g['nozzle']['kd']}"])
        w.writerow(["time_s", "mode",
                    "angle_target", "angle_actual", "angle_error", "angle_command", "angle_pid_out",
                    "angle_P", "angle_I", "angle_D",
                    "nozzle_target", "nozzle_actual", "nozzle_error", "nozzle_command", "nozzle_pid_out",
                    "nozzle_P", "nozzle_I", "nozzle_D"])
        with self.lock:
            self._log_file, self._log_writer, self.log_path = f, w, path
        return path

    def stop_log(self):
        with self.lock:
            f, path = self._log_file, self.log_path
            self._log_file = self._log_writer = self.log_path = None
        if f:
            f.close()
        return path

    # ── PID thread ────────────────────────────────────────────────────────
    def _run(self):
        next_t = time.perf_counter()
        while True:
            next_t += self.dt
            self._tick()
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:                 # computer was busy: resync
                next_t = time.perf_counter()

    def _tick(self):
        finished = False
        with self.lock:
            dt = self.dt
            self.t += dt
            a, n = self.loops["servo"], self.loops["nozzle"]

            # 1. Choose targets
            if self.mode == "step":
                ts = self.t - self.step_t0
                for t_chg, a_sp, n_sp in STEP_SEQUENCE:
                    if ts >= t_chg:
                        a.target, n.target = a_sp, n_sp
                if ts >= STEP_DURATION:
                    self.mode, finished = "vision", True
            else:
                a.target = self.vision_target["servo"]
                n.target = self.vision_target["nozzle"]

            # 2. Run PID + servo
            a.step(dt)
            n.step(dt)

            # 3. Record
            if self._log_writer:
                row = [round(self.t, 4), self.mode]
                for l in (a, n):
                    row += [l.target, round(l.servo.actual, 3), round(l.target - l.servo.actual, 3),
                            round(l.servo.command, 3), round(l.output, 3),
                            round(l.pid.p, 3), round(l.pid.i, 3), round(l.pid.d, 3)]
                self._log_writer.writerow(row)
            if self.mode == "step" or finished:
                self._step_log.append((self.t - self.step_t0, a.target, a.servo.actual,
                                       n.target, n.servo.actual))
            self._n += 1
            if self._n % self._hist_every == 0:
                # (t, angle target, angle actual, angle out %, nozzle target, nozzle actual, nozzle out %)
                self.history.append((round(self.t, 3),
                                     a.target, round(a.servo.actual, 2), round(a.output / a.pid.out_limit, 3),
                                     n.target, round(n.servo.actual, 2), round(n.output / n.pid.out_limit, 3)))

        if finished:
            self.metrics = {"servo":  step_metrics(self._step_log, 1, 2),
                            "nozzle": step_metrics(self._step_log, 3, 4)}
            if self._step_owns_log:
                self.metrics["log"] = self.stop_log()
                self._step_owns_log = False


def simulate_step_test():
    """Run the step test instantly (no waiting). Returns the result table."""
    c = ControlSystem(realtime=False)
    c.mode, c.step_t0 = "step", 0.0
    while c.mode == "step":
        c._tick()
    return c.metrics


def step_metrics(samples, i_sp, i_pv, band=0.02):
    """
    For every target change, measure:
      rise_time     : time to go from 10% to 90% of the way   (speed)
      overshoot_pct : how far it went past the target          (smaller = better)
      settling_time : time until it stays within ±2% of target (when it is "done")
      sse           : error left at the end (steady-state error)
    """
    results = []
    changes = [k for k in range(1, len(samples)) if samples[k][i_sp] != samples[k - 1][i_sp]]
    for a, b in zip(changes, changes[1:] + [len(samples)]):
        seg  = samples[a:b]
        t0   = seg[0][0]
        y0   = samples[a - 1][i_pv]
        sp   = seg[0][i_sp]
        step = sp - y0
        if abs(step) < 1e-6:
            continue
        norm   = [(s[0] - t0, (s[i_pv] - y0) / step) for s in seg]   # 0 = start, 1 = target
        t10    = next((t for t, y in norm if y >= 0.1), None)
        t90    = next((t for t, y in norm if y >= 0.9), None)
        peak   = max(y for _, y in norm)
        settle = next((t for t, y in reversed(norm) if abs(y - 1) > band), 0.0)
        tail   = [s[i_pv] for s in seg if s[0] >= seg[-1][0] - 0.2]
        results.append({
            "from": round(y0, 2), "to": sp,
            "rise_time":     round(t90 - t10, 3) if t10 is not None and t90 is not None else None,
            "overshoot_pct": round(max(0.0, peak - 1) * 100, 2),
            "settling_time": round(settle, 3),
            "sse":           round(sp - sum(tail) / len(tail), 3),
        })
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(simulate_step_test(), indent=1))
