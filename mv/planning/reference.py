"""One call from (track, car) to a reference lap.

``reference_lap(track, vp)`` answers "what lap time is theoretically achievable
for this car on this track, and along which line?" by chaining the three
building blocks of this package:

1. :func:`mv.planning.ggv.build_ggv`            - what the car can do (g-g-v envelope);
2. :mod:`mv.planning.raceline`                  - where to drive (centre line,
   minimum-curvature line, minimum-time refinement);
3. :func:`mv.planning.speed_profile.solve_speed_profile` - how fast (forward-backward pass).

The result is a :class:`ReferenceLap`: the line, the speed along it and the
lap time, sampled at the stations of the track so that "index i of the track"
and "index i of the reference" mean the same place.  It can be saved to and
loaded from a single ``.npz`` file.
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mv.planning import raceline as rl
from mv.planning.ggv import GGV, PowertrainMap, build_ggv
from mv.planning.speed_profile import ShiftModel, solve_speed_profile
from mv.track.track import Track
from mv.vehicle.params import G, VehicleParams

__all__ = ["ReferenceLap", "reference_lap", "reference_lines", "default_margin"]

METHODS = ("centre", "min_curvature", "min_time")


@dataclass
class ReferenceLap:
    """A line around a track and the fastest way to drive it.

    All arrays have one entry per track station (same indexing as ``Track``).
    """
    s: np.ndarray           # (M,) centre-line distance of each station, m
    xy: np.ndarray          # (M, 2) coordinates of the line
    n: np.ndarray           # (M,) offset from the centre line, m (+ = left)
    kappa: np.ndarray       # (M,) curvature of the line, 1/m (+ = left turn)
    dist: np.ndarray        # (M,) distance travelled along the line, m
    speed: np.ndarray       # (M,) m/s
    ax: np.ndarray          # (M,) longitudinal acceleration towards the next station, m/s^2
    ay: np.ndarray          # (M,) lateral acceleration, m/s^2 (+ = left)
    time: np.ndarray        # (M,) time of arrival at each station, s
    lap_time: float
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, s=self.s, xy=self.xy, n=self.n, kappa=self.kappa, dist=self.dist,
                            speed=self.speed, ax=self.ax, ay=self.ay, time=self.time,
                            lap_time=self.lap_time, meta=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceLap":
        z = np.load(path, allow_pickle=False)
        return cls(s=z["s"], xy=z["xy"], n=z["n"], kappa=z["kappa"], dist=z["dist"], speed=z["speed"],
                   ax=z["ax"], ay=z["ay"], time=z["time"], lap_time=float(z["lap_time"]),
                   meta=json.loads(str(z["meta"])))


def default_margin(vp: VehicleParams, safety_m: float = 0.2) -> float:
    """Distance the car's centre keeps from the track edge: half the body width plus a buffer."""
    return 0.5 * vp.chassis.width_m + safety_m


def _battery_report(vp: VehicleParams, speed: np.ndarray, ax: np.ndarray, dt: np.ndarray) -> dict:
    """Rough electrical energy balance of the lap for cars with a battery.

    The envelope assumes the battery never runs low.  Here we estimate the
    throttle and brake pedal from the acceleration, integrate discharge and
    regeneration as the simulator would, and report whether one lap from the
    initial state of charge is actually possible.
    """
    pt = PowertrainMap(vp)
    if not pt.has_motor:
        return {}
    m = vp.chassis.mass_kg
    resist = 0.5 * vp.aero.air_density * vp.aero.cda_m2 * speed ** 2 + vp.tire_rear.rolling_resistance * m * G
    force = m * ax + resist                                  # what the powertrain / brakes must supply
    front, rear = pt.axle_forces(speed, 1.0)
    thr = np.clip(force / np.maximum(front + rear, 1.0), 0.0, 1.0)
    bk = vp.brakes
    brake_max = bk.max_torque_nm * (bk.front_share / vp.tire_front.radius_m + (1 - bk.front_share) / vp.tire_rear.radius_m)
    brk = np.clip(-force / brake_max, 0.0, 1.0)
    e_out = float((pt.electric_power(speed, thr) * dt).sum())
    e_in = float((np.minimum(brk * pt.regen_capacity(speed), pt.bat_regen_w) * dt).sum())
    usable = pt.bat_usable_j
    return {"battery_used_kwh": (e_out - e_in) / 3.6e6, "battery_discharged_kwh": e_out / 3.6e6,
            "battery_regenerated_kwh": e_in / 3.6e6, "battery_usable_kwh": usable / 3.6e6,
            "battery_lasts_the_lap": bool(e_out - e_in <= usable)}


def _evaluate(track: Track, vp: VehicleParams, line: rl.Line, ggv: GGV, shift: ShiftModel | None, meta: dict) -> ReferenceLap:
    prof = solve_speed_profile(line.kappa, line.ds, ggv, shift=shift)
    dt = 2.0 * line.ds / (prof.v + np.roll(prof.v, -1))
    info = dict(meta)
    info.update(line.meta)
    info.update({"top_speed": float(prof.v.max()), "min_speed": float(prof.v.min()), "line_length": line.length,
                 "upshifts": prof.upshifts, "max_ay": float(np.abs(prof.ay).max()),
                 "max_ax": float(prof.ax.max()), "min_ax": float(prof.ax.min())})
    info.update(_battery_report(vp, prof.v, prof.ax, dt))
    return ReferenceLap(s=np.arange(track.n) * track.ds, xy=line.xy, n=line.n, kappa=line.kappa,
                        dist=np.concatenate([[0.0], np.cumsum(line.ds)[:-1]]), speed=prof.v, ax=prof.ax,
                        ay=prof.ay, time=prof.t, lap_time=prof.lap_time, meta=info)


def reference_lines(track: Track, vp: VehicleParams, methods: tuple[str, ...] = METHODS, ggv: GGV | None = None,
                    margin: float | None = None, safety_m: float = 0.2, step_m: float = 3.0,
                    shift_losses: bool = True, min_time_iterations: int = 20) -> dict[str, ReferenceLap]:
    """Compute the requested lines (any of ``centre``, ``min_curvature``, ``min_time``).

    Returns ``{method: ReferenceLap}``.  ``meta['compute_s']`` of each lap is
    the time spent on that line (the envelope's build time is in
    ``meta['ggv_s']``; pass a ready ``ggv`` to reuse it across tracks).
    """
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown method(s) {sorted(unknown)}; choose from {METHODS}")
    t0 = _time.perf_counter()
    ggv_s = 0.0
    if ggv is None:
        ggv = build_ggv(vp)
        ggv_s = _time.perf_counter() - t0
    margin = default_margin(vp, safety_m) if margin is None else margin
    shift = ShiftModel.from_vehicle(vp) if shift_losses else None
    grid = rl.make_grid(track, margin, step_m)
    base = {"track": track.name, "vehicle": vp.name, "margin_m": margin, "grid_step_m": step_m,
            "shift_losses": shift_losses, "ggv_s": ggv_s, "track_length": track.length}
    out: dict[str, ReferenceLap] = {}

    def finish(name: str, line: rl.Line, started: float, extra: dict | None = None) -> ReferenceLap:
        lap = _evaluate(track, vp, line, ggv, shift, {**base, "method": name, **(extra or {})})
        lap.meta["compute_s"] = _time.perf_counter() - started
        return lap

    if "centre" in methods:
        t = _time.perf_counter()
        out["centre"] = finish("centre", rl.centre_line(track, grid), t)
    if "min_curvature" in methods or "min_time" in methods:
        t = _time.perf_counter()
        alpha_mc, info_mc = rl.min_curvature_offsets(grid)
        lap_mc = finish("min_curvature", rl.line_from_offsets(track, grid, alpha_mc, margin), t,
                        {"qp_rounds": len(info_mc["cost_history"]) - 1})
        if "min_curvature" in methods:
            out["min_curvature"] = lap_mc
        if "min_time" in methods:
            t = _time.perf_counter()
            candidates, info_mt = rl.min_time_offsets(grid, ggv, alpha_mc, iterations=min_time_iterations)
            # The optimiser ranks lines on the coarse grid; settle the final order on the full-resolution lap.
            best = None
            for alpha in candidates:
                lap = finish("min_time", rl.line_from_offsets(track, grid, alpha, margin), t,
                             {"rounds": len(info_mt["coarse_time_history"]) - 1})
                if best is None or lap.lap_time < best.lap_time:
                    best = lap
            if best.lap_time >= lap_mc.lap_time:        # refinement did not help: keep minimum curvature
                best = ReferenceLap(**{**lap_mc.__dict__, "meta": {**lap_mc.meta, "method": "min_time",
                                                                    "fell_back_to": "min_curvature"}})
            best.meta["compute_s"] = _time.perf_counter() - t
            best.meta["gain_vs_min_curvature_s"] = lap_mc.lap_time - best.lap_time
            out["min_time"] = best
    return out


def reference_lap(track: Track, vp: VehicleParams, method: str = "auto", **kw) -> ReferenceLap:
    """The reference lap of ``vp`` on ``track``.

    ``method``: ``"auto"`` (default; compute everything and return the fastest
    line, normally the minimum-time one), ``"min_time"``, ``"min_curvature"``
    or ``"centre"``.  Keyword arguments are passed to :func:`reference_lines`
    (``ggv``, ``margin``, ``safety_m``, ``step_m``, ``shift_losses``, ...).

    ``meta['lap_times']`` lists the lap time of every line that was computed
    on the way, so the gain of each stage is on record.
    """
    methods = METHODS if method == "auto" else {"centre": ("centre",), "min_curvature": ("min_curvature",),
                                                "min_time": ("min_curvature", "min_time")}[method]
    laps = reference_lines(track, vp, methods, **kw)
    best = min(laps.values(), key=lambda lap: lap.lap_time) if method == "auto" else laps[method]
    best.meta["lap_times"] = {name: lap.lap_time for name, lap in laps.items()}
    best.meta["compute_total_s"] = float(sum(lap.meta["compute_s"] for lap in laps.values()) + best.meta["ggv_s"])
    return best
