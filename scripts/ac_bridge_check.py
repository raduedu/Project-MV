"""Live diagnostic for the Assetto Corsa bridge (run by the user, game on track).

Usage (from the project root, AC running, car on track, session LIVE)::

    .venv\\Scripts\\python.exe scripts\\ac_bridge_check.py                 # 10 s telemetry
    .venv\\Scripts\\python.exe scripts\\ac_bridge_check.py --drive         # + 2 s gentle throttle (CSP app)
    .venv\\Scripts\\python.exe scripts\\ac_bridge_check.py --drive --backend vjoy
    .venv\\Scripts\\python.exe scripts\\ac_bridge_check.py --vjoy-sweep X  # help binding vJoy axes in CM

What it reports
* static info (car, track, track length, shared-memory and AC versions);
* whether the MV Bridge CSP Lua app is running (and physics.allowed / reset allowed);
* live telemetry lines at ~5 Hz for ``--seconds``;
* physics and graphics packet rates;
* convention checks (need the car to move: drive yourself during the 10 s,
  ideally accelerate, brake and take a left and a right corner):
  ax vs d(vx)/dt, ay vs yaw_rate*vx, yaw_rate vs d(yaw)/dt, heading calibration,
  plus the sign of steering you applied.
* with ``--drive``: 2 s of throttle (default 0.3) with the wheel straight,
  then 1.5 s of full brake, loop timing/latency statistics, then control is
  released back to you.  Nothing is sent without ``--drive``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mv.ac.bridge import (  # noqa: E402
    ACReader, ACSession, ACTimeoutError, CSPLuaBackend, PluginLink, SharedMemoryUnavailable,
    make_backend,
)
from mv.ac.bridge.shm_layout import AC_LIVE, AC_SESSION_NAMES, AC_STATUS_NAMES  # noqa: E402


def corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 10 or a.std() < 1e-6 or b.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def verdict(c: float, what: str) -> str:
    if math.isnan(c):
        return f"  [ ?  ] {what}: not enough motion to judge"
    tag = "PASS" if c > 0.6 else ("FAIL (sign flipped?)" if c < -0.6 else "WEAK")
    return f"  [{tag}] {what}: correlation {c:+.2f}"


def print_static(r: ACReader, link: PluginLink | None) -> None:
    s = r.static
    print(f"car={s.carModel!r} track={s.track!r} config={s.trackConfiguration!r} "
          f"length={s.trackSPlineLength:.1f} m maxRpm={s.maxRpm}")
    print(f"shared memory v{s.smVersion}  AC v{s.acVersion}  sessions={s.numberOfSessions} "
          f"cars={s.numCars}")
    g = r.graphics.snapshot()
    print(f"status={AC_STATUS_NAMES.get(g.status, g.status)} "
          f"session={AC_SESSION_NAMES.get(g.session, g.session)}")
    if link is None:
        return
    if link.alive(1.0):
        p = link.read_state()
        print(f"MV Bridge Lua app: RUNNING (CSP code {p.csp_version}, fps {p.fps:.0f}) "
              f"physics.allowed={p.physics_allowed} reset_allowed={p.reset_allowed}")
    else:
        print("MV Bridge Lua app: NOT DETECTED (install with mv/ac/plugin/install_plugin.ps1, "
              "then restart the session; the app loads automatically).")


def monitor(r: ACReader, seconds: float) -> None:
    print(f"\n--- live telemetry for {seconds:.0f} s (drive around: accelerate, brake, "
          "turn LEFT and RIGHT) ---")
    t0 = time.perf_counter()
    p0, g0 = r.physics_packet_id(), r.graphics.packet_id()
    last_print = 0.0
    S = {k: [] for k in ("t", "vx", "vy", "ax", "ay", "r", "yaw", "src", "steer", "sim")}
    while (el := time.perf_counter() - t0) < seconds:
        t = r.read()
        S["t"].append(t.wall_time)
        S["vx"].append(t.velocity_local[0])
        S["vy"].append(t.velocity_local[1])
        S["ax"].append(t.acceleration_local[0])
        S["ay"].append(t.acceleration_local[1])
        S["r"].append(t.yaw_rate)
        S["yaw"].append(t.yaw)
        S["src"].append(t.yaw_source)
        S["steer"].append(t.steer)
        S["sim"].append(t.sim_time)
        if el - last_print >= 0.2:
            last_print = el
            print(f"pkt {t.packet_id:8d} {AC_STATUS_NAMES.get(t.status, '?'):5s} "
                  f"v={t.speed * 3.6:6.1f}km/h vx={t.velocity_local[0]:6.2f} "
                  f"vy={t.velocity_local[1]:+5.2f} yaw={math.degrees(t.yaw):+7.1f}({t.yaw_source}) "
                  f"r={t.yaw_rate:+5.2f} ax={t.acceleration_local[0]:+6.2f} "
                  f"ay={t.acceleration_local[1]:+6.2f} steer={t.steer:+.2f}(raw {t.steer_raw:+.2f}) "
                  f"thr={t.throttle:.2f} brk={t.brake:.2f} g={t.gear} rpm={t.rpm:5.0f} "
                  f"spl={t.spline:.4f} lap={t.lap_time_current:6.2f}s out={t.tyres_out} "
                  f"pos=({t.position[0]:.1f},{t.position[1]:.1f})", flush=True)
        time.sleep(0.01)
    el = time.perf_counter() - t0
    dp, dg = r.physics_packet_id() - p0, r.graphics.packet_id() - g0
    print(f"\npacket rates: physics {dp / el:.1f}/s  graphics {dg / el:.1f}/s  "
          f"(torn-read retries: {r.physics.torn_reads})")
    if dp == 0:
        print("  physics packetId did not change: game paused, in menu, or replay?")

    t = np.array(S["t"])
    if t.size < 20:
        return
    dt = np.gradient(t)
    ok = dt > 1e-4
    vx, vy, r_ = map(np.array, (S["vx"], S["vy"], S["r"]))
    dvx = np.gradient(vx) / np.where(ok, dt, np.inf)
    yaw = np.unwrap(np.array(S["yaw"]))
    good = np.array([s in ("velocity", "plugin") for s in S["src"]]) & ok
    dyaw = np.gradient(yaw) / np.where(ok, dt, np.inf)
    print("\nconvention checks (project: x fwd, y left, yaw CCW+, steer + = left):")
    print(verdict(corr(np.array(S["ax"])[ok], dvx[ok]), "ax (accG) vs d(vx)/dt"))
    print(verdict(corr(np.array(S["ay"])[ok & (vx > 5)], (r_ * vx)[ok & (vx > 5)]),
                  "ay (accG) vs yaw_rate*vx (centripetal)"))
    print(verdict(corr(r_[good], dyaw[good]), "yaw_rate vs d(yaw)/dt"))
    print(verdict(corr(np.array(S["steer"])[vx > 5], r_[vx > 5]),
                  "steer vs yaw_rate (steer left => turn left)"))
    print(f"  heading calibration: sign={r.decoder.heading_sign} "
          f"offset={math.degrees(r.decoder.heading_offset):+.1f} deg "
          "(None = not enough driving above 3 m/s yet)")
    sim = np.array(S["sim"])
    if sim.size > 2 and t[-1] > t[0]:
        print(f"  sim_time advanced {sim[-1] - sim[0]:.2f} s over {t[-1] - t[0]:.2f} s wall "
              f"(source: {r.read().sim_time_source})")


def drive(r: ACReader, backend_name: str, link: PluginLink | None, throttle: float,
          seconds: float) -> None:
    print(f"\n--- --drive: backend {backend_name}, throttle {throttle:.2f} for {seconds:.1f} s, "
          "then brake 1.5 s ---")
    if r.graphics.snapshot().status != AC_LIVE:
        print("Game is not LIVE (paused/replay/menu); not driving.")
        return
    if backend_name == "csp":
        if link is None or not link.alive(1.0):
            print("MV Bridge Lua app not running; cannot drive with the csp backend.")
            return
        backend = CSPLuaBackend(link)
    else:
        backend = make_backend(backend_name)
    s = ACSession(r, backend, rate_hz=50.0)
    try:
        start = r.read()
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            tel = s.step(0.0, throttle, 0.0)
        print(f"after throttle: speed {start.speed * 3.6:.1f} -> {tel.speed * 3.6:.1f} km/h, "
              f"reported throttle {tel.throttle:.2f}, gear {tel.gear}")
        if tel.plugin is not None:
            print(f"  Lua override_active={tel.plugin.override_active} "
                  f"failsafe={tel.plugin.failsafe_active}")
        t_end = time.perf_counter() + 1.5
        while time.perf_counter() < t_end:
            tel = s.step(0.0, 0.0, 1.0)
        print(f"after braking: speed {tel.speed * 3.6:.1f} km/h, reported brake {tel.brake:.2f}")
        print("loop: " + s.stats.format())
        if tel.throttle < 0.05 and tel.speed < start.speed + 0.5:
            print("!! the car did not respond: check the backend setup (docs/07 section 5).")
    except ACTimeoutError as e:
        print(f"timeout: {e}")
    finally:
        s.close()   # brake, then release control back to you
        if backend_name != "csp":
            backend.close()
    print("control released.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--drive", action="store_true", help="apply gentle throttle for 2 s")
    ap.add_argument("--backend", default="csp", choices=["csp", "vjoy", "vgamepad"])
    ap.add_argument("--throttle", type=float, default=0.3)
    ap.add_argument("--drive-seconds", type=float, default=2.0)
    ap.add_argument("--vjoy-sweep", choices=["X", "Y", "Z"],
                    help="sweep one vJoy axis for 4 s (click 'bind' in CM first) and exit")
    ap.add_argument("--wait", type=float, default=5.0, help="seconds to wait for the game")
    a = ap.parse_args()

    if a.vjoy_sweep:
        from mv.ac.bridge import VJoyBackend
        be = VJoyBackend()
        print(f"sweeping vJoy axis {a.vjoy_sweep} ...")
        be.sweep_axis(a.vjoy_sweep)
        be.close()
        return 0

    link = PluginLink()
    try:
        r = ACReader.wait_for_game(a.wait, plugin=link)
    except SharedMemoryUnavailable as e:
        print(f"{e}\nStart Assetto Corsa, load a session and drive onto the track, then retry.")
        link.close()
        return 2
    try:
        print_static(r, link)
        monitor(r, a.seconds)
        if a.drive:
            drive(r, a.backend, link, a.throttle, a.drive_seconds)
    finally:
        r.close()
        link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
