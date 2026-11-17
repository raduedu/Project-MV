"""Drive a car through standard performance tests in the MV simulator.

Prints numbers you can hold against a magazine road test: 0-100 km/h,
0-200 km/h, top speed, braking distance and steady cornering grip.  If these
are wrong the parameter file (or the physics) is wrong.

Usage::

    python scripts/vehicle_check.py ks_porsche_911_gt3_rs
    python scripts/vehicle_check.py porsche_taycan_turbo_gt_weissach --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mv.sim.vehicle_sim import VehicleSim  # noqa: E402
from mv.vehicle.params import find_vehicle, load_vehicle  # noqa: E402
from mv.vehicle.summary import describe  # noqa: E402


def _fresh(vp, n, device, v0=0.0):
    sim = VehicleSim(vp, n, device)
    ids = torch.arange(n, device=sim.device)
    zero = torch.zeros(n, device=sim.device)
    sim.reset(ids, zero, zero, zero, torch.full((n,), float(v0), device=sim.device))
    return sim


def acceleration(vp, device) -> dict:
    sim = _fresh(vp, 1, device)
    one, zero = torch.ones(1, device=sim.device), torch.zeros(1, device=sim.device)
    out, t, vmax = {}, 0.0, 0.0
    marks = {"0-100 km/h": 100 / 3.6, "0-200 km/h": 200 / 3.6, "0-300 km/h": 300 / 3.6}
    while t < 90.0:
        sim.step(zero, one, zero)
        t += sim.dt
        v = float(sim.vx)
        for k, target in marks.items():
            if k not in out and v >= target:
                out[k] = t
        if "400 m" not in out and float(sim.x) >= 400.0:
            out["400 m"] = t
        vmax = max(vmax, v)
    out["top speed km/h"] = vmax * 3.6
    out["soc_end"] = float(sim.soc)
    return out


def braking(vp, device, v0_kmh=200.0) -> dict:
    sim = _fresh(vp, 1, device, v0_kmh / 3.6)
    one, zero = torch.ones(1, device=sim.device), torch.zeros(1, device=sim.device)
    t, peak = 0.0, 0.0
    while float(sim.vx) > 0.5 and t < 30:
        sim.step(zero, zero, one)
        t += sim.dt
        peak = max(peak, -float(sim.ax))
    return {"distance_m": float(sim.x), "time_s": t, "peak_g": peak / 9.81}


def cornering(vp, device, speeds_kmh=(60, 100, 150, 200)) -> dict:
    """Hold speed with a simple speed controller and slowly wind on steering.

    The largest lateral acceleration reached before the car runs out of grip is
    the steady-state cornering limit at that speed.
    """
    n = len(speeds_kmh)
    target = torch.tensor([s / 3.6 for s in speeds_kmh], device=device)
    sim = _fresh(vp, n, device)
    sim.vx[:] = target
    g0 = (target[:, None, None] > sim.gb_up_v[None]).sum(-1)
    sim.gear[:] = torch.minimum(g0, sim.gb_ngears[None] - 1)
    best = torch.zeros(n, device=sim.device)
    understeer = torch.zeros(n, device=sim.device)
    steps = int(25.0 / sim.dt)
    for i in range(steps):
        steer = torch.full((n,), 0.6 * i / steps, device=sim.device)
        err = target - sim.vx
        thr = (0.3 * err).clamp(0, 1)
        brk = (-0.1 * err).clamp(0, 1)
        sim.step(steer, thr, brk)
        ay = sim.vx * sim.r                       # centripetal acceleration actually achieved
        better = ay > best
        best = torch.where(better, ay, best)
        # balance at the limit: front minus rear grip usage (>0 = understeer)
        bal = sim.tire_util[:, :2].mean(1) - sim.tire_util[:, 2:].mean(1)
        understeer = torch.where(better, bal, understeer)
    return {f"{s} km/h": (float(best[i]) / 9.81, float(understeer[i])) for i, s in enumerate(speeds_kmh)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vehicle")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    vp = load_vehicle(find_vehicle(args.vehicle))
    print(describe(vp))
    print("\nSimulated performance")
    for k, v in acceleration(vp, args.device).items():
        print(f"  {k:<16}{v:8.2f}")
    b = braking(vp, args.device)
    print(f"  200-0 km/h      {b['distance_m']:8.1f} m in {b['time_s']:.2f} s (peak {b['peak_g']:.2f} g)")
    b = braking(vp, args.device, 100.0)
    print(f"  100-0 km/h      {b['distance_m']:8.1f} m")
    print("  cornering limit (g) and balance (+ understeer / - oversteer):")
    for k, (g, bal) in cornering(vp, args.device).items():
        print(f"    {k:<10} {g:5.2f} g   balance {bal:+.2f}")


if __name__ == "__main__":
    main()
