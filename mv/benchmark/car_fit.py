"""Equal machinery: tune a car file until it can do what a real car did on a real lap.

Why
---
"Beat Verstappen" only means something in the same car. Nobody publishes an F1
car's downforce, drag or tyre grip, but his lap tells us what the car *could
do*: how fast it reached the end of each straight (power against drag) and how
fast it went through each corner (grip plus downforce - fast corners show
downforce, slow ones mostly tyre grip). We take a detailed F1 car from the
owner's Assetto Corsa as a starting shape and scale four numbers - power, drag
area, downforce area, tyre grip - until our physics reproduces those speeds.

What exactly is matched (and why not the whole speed trace)
-----------------------------------------------------------
Only two kinds of points, both places where a top driver in qualifying is
assumed to be *at the car's limit*:

* the **minimum speed of every corner** (apex), and
* the **top speed at the end of every straight** (flat out: limited by the car,
  not the driver), and
* the **speed one third and two thirds of the way down each straight**: how
  quickly the car gathers speed. Without this, power and drag cannot be told
  apart - half the power with half the drag reaches the same top speed (first
  fit at Monza: power x0.52, drag x0.46 matched the top speeds equally well).

Everything in between - how early he brakes, how he releases the brake, how he
feeds in the throttle - is left free. Our physics-perfect lap of the fitted car
then drives those transitions perfectly, so it comes out somewhat *faster*
than his lap. That difference is measured, not assumed, and tells how much of
the car he left in the transitions (or how optimistic our quasi-steady
physics is there; we cannot separate the two).

Limits (recorded in the engineering log, WP43)
----------------------------------------------
* DRS is open on qualifying straights and our simulator has no DRS: the fitted
  drag is a mix of open and closed wing, i.e. too low for corners' exits and
  too high in DRS zones.
* Each weekend has its own aero package (Monza: low downforce, low drag), so
  each circuit gets its own fit - the "car of that weekend". Power and tyre
  grip should come out similar across circuits; how similar they are is a
  check on the method.
* Mass is not fitted (it is coupled with everything else); it is set from the
  regulations by the caller.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from mv.planning import raceline as rl
from mv.planning.ggv import build_ggv
from mv.planning.reference import default_margin
from mv.planning.speed_profile import ShiftModel, solve_speed_profile
from mv.sysid.replay import apply_factors
from mv.track.track import Track
from mv.vehicle.params import VehicleParams

__all__ = ["FIT_FACTORS", "LapTarget", "FitOutcome", "lap_target", "ideal_speed", "fit_car", "with_mass"]

#: the four numbers that are fitted (names as in mv.sysid.replay.apply_factors)
FIT_FACTORS = ("power", "drag", "downforce", "grip")


def with_mass(vp: VehicleParams, mass_kg: float) -> VehicleParams:
    """Copy of a car at another mass; yaw inertia scales with it (same shape, heavier)."""
    v = copy.deepcopy(vp)
    k = mass_kg / v.chassis.mass_kg
    v.chassis.mass_kg = float(mass_kg)
    v.chassis.yaw_inertia_kgm2 = float(v.chassis.yaw_inertia_kgm2 * k)
    return v


@dataclass
class LapTarget:
    """What is matched on one circuit: his speeds at apexes and at the ends of straights."""
    track: Track
    line: rl.Line                    # fixed line the physics drives (minimum curvature)
    s: np.ndarray                    # his speed trace: centre-line distance [m] ...
    v: np.ndarray                    # ... and speed [m/s]
    lap_time: float
    apex_s: np.ndarray               # [m] centre-line distance of each apex
    apex_v: np.ndarray               # [m/s] his minimum speed there
    top_s: np.ndarray                # [m] end of each straight
    top_v: np.ndarray                # [m/s] his top speed there
    acc_s: np.ndarray = None         # [m] points one and two thirds of the way down each straight
    acc_v: np.ndarray = None         # [m/s] his speed there
    excluded_m: float = 0.0          # length of track left out because the layouts differ


def lap_target(track: Track, s: np.ndarray, v: np.ndarray, lap_time: float, vp: VehicleParams,
               bad_s: np.ndarray | None = None, prominence_kmh: float = 12.0) -> LapTarget:
    """Find his apexes and straight-end speeds; ``bad_s`` = stretches where the circuits differ."""
    margin = default_margin(vp)
    grid = rl.make_grid(track, margin, 3.0)
    alpha, _ = rl.min_curvature_offsets(grid)
    line = rl.line_from_offsets(track, grid, alpha, margin)
    st = np.arange(track.n) * track.ds
    order = np.argsort(s)
    vs = np.interp(st, np.asarray(s)[order], np.asarray(v)[order])
    k = max(int(round(15.0 / track.ds)), 1)                     # smooth over ~30 m: the samples are 20 m apart
    vsm = np.convolve(np.concatenate([vs[-k:], vs, vs[:k]]), np.ones(2 * k + 1) / (2 * k + 1), "same")[k:-k]
    dist = int(round(80.0 / track.ds))
    lo, _ = find_peaks(-vsm, prominence=prominence_kmh / 3.6, distance=dist)
    hi, _ = find_peaks(vsm, prominence=prominence_kmh / 3.6, distance=dist)
    ok = np.ones(track.n, bool)
    excluded = 0.0
    if bad_s is not None and len(bad_s):
        w = int(round(60.0 / track.ds))
        for b in np.asarray(bad_s):
            i = int(b / track.ds) % track.n
            ok[np.arange(i - w, i + w + 1) % track.n] = False
        excluded = float((~ok).sum() * track.ds)
    lo, hi = lo[ok[lo]], hi[ok[hi]]
    # the raw trace (not the smoothed one) at the extreme: smoothing would lift apexes / lower top speeds
    win = max(int(round(25.0 / track.ds)), 1)
    take = lambda idx, f: np.array([f(vs[np.arange(i - win, i + win + 1) % track.n]) for i in idx])  # noqa: E731
    # points along each straight: from an apex to the next top speed (both kept)
    acc = []
    for j in hi:
        prev = lo[lo < j]
        i = prev[-1] if len(prev) else (lo[-1] - track.n if len(lo) else None)
        if i is None or j - i < int(round(150.0 / track.ds)):
            continue
        for f in (1 / 3, 2 / 3):
            k = int(round(i + f * (j - i))) % track.n
            if ok[k]:
                acc.append(k)
    acc = np.array(acc, int)
    return LapTarget(track, line, st, vs, lap_time, lo * track.ds, take(lo, np.min), hi * track.ds, take(hi, np.max),
                     acc_s=acc * track.ds, acc_v=vs[acc], excluded_m=excluded)


def ideal_speed(vp: VehicleParams, tg: LapTarget, ggv=None) -> tuple[np.ndarray, float]:
    """Physics-perfect speed at every centre-line station along the fixed line, and the lap time."""
    ggv = ggv or build_ggv(vp)
    prof = solve_speed_profile(tg.line.kappa, tg.line.ds, ggv, shift=ShiftModel.from_vehicle(vp))
    return prof.v, float(prof.lap_time)


def _features(v_ideal: np.ndarray, tg: LapTarget) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, ds = tg.track.n, tg.track.ds
    w_apex, w_top = int(round(40.0 / ds)), int(round(80.0 / ds))
    a = np.array([v_ideal[np.arange(int(s / ds) - w_apex, int(s / ds) + w_apex + 1) % n].min() for s in tg.apex_s])
    t = np.array([v_ideal[np.arange(int(s / ds) - w_top, int(s / ds) + w_top + 1) % n].max() for s in tg.top_s])
    m = np.array([v_ideal[int(s / ds) % n] for s in tg.acc_s]) if tg.acc_s is not None else np.zeros(0)
    return a, t, m


@dataclass
class FitOutcome:
    factors: dict[str, float]
    vehicle: VehicleParams
    ideal_lap_time: float
    real_lap_time: float
    apex_err_kmh: np.ndarray         # ideal - his, per apex (after the fit)
    top_err_kmh: np.ndarray          # ideal - his, per straight end
    acc_err_kmh: np.ndarray          # ideal - his, part-way down each straight
    rms_kmh: float
    evaluations: int
    history: list = field(default_factory=list)

    def summary(self) -> dict:
        return {"factors": {k: round(v, 3) for k, v in self.factors.items()},
                "ideal_lap_s": round(self.ideal_lap_time, 3), "real_lap_s": round(self.real_lap_time, 3),
                "ideal_faster_by_pct": round(100 * (1 - self.ideal_lap_time / self.real_lap_time), 2),
                "apexes": len(self.apex_err_kmh), "straights": len(self.top_err_kmh),
                "rms_kmh": round(self.rms_kmh, 2),
                "apex_err_kmh": np.round(self.apex_err_kmh, 1).tolist(),
                "top_err_kmh": np.round(self.top_err_kmh, 1).tolist(),
                "acc_err_kmh": np.round(self.acc_err_kmh, 1).tolist(), "evaluations": self.evaluations}


def fit_car(vp: VehicleParams, tg: LapTarget, names: tuple[str, ...] = FIT_FACTORS,
            bounds: tuple[float, float] = (0.4, 2.5), max_evals: int = 120, starts: int = 12, seed: int = 0,
            printer=None) -> FitOutcome:
    """Least-squares fit of the factors so apex and straight-end speeds match his (km/h residuals).

    Works on the logarithm of each factor (a factor of 2 and of 1/2 are equally
    far from 1). Every evaluation builds the car's g-g-v envelope (a few seconds).
    The error landscape has several valleys (e.g. more grip vs more downforce),
    so ``starts`` random cars (factors 0.6-1.7) are tried first and the fit begins
    from the best of them and from the unchanged car, keeping the better result.
    """
    hist = []

    def car(x):
        return apply_factors(vp, dict(zip(names, np.exp(x))))

    def resid(x):
        v_i, lt = ideal_speed(car(x), tg)
        a, t, m = _features(v_i, tg)
        r = np.concatenate([a - tg.apex_v, t - tg.top_v, m - tg.acc_v]) * 3.6
        hist.append((np.exp(x).round(4).tolist(), float(np.sqrt(np.mean(r ** 2))), lt))
        if printer:
            printer(f"    {dict(zip(names, np.exp(x).round(3)))} rms {hist[-1][1]:.2f} km/h, ideal lap {lt:.3f} s")
        return r

    lb, ub = np.log(bounds[0]) * np.ones(len(names)), np.log(bounds[1]) * np.ones(len(names))
    rng = np.random.default_rng(seed)
    cands = [np.zeros(len(names))] + [rng.uniform(np.log(0.6), np.log(1.7), len(names)) for _ in range(starts)]
    scores = [float(np.sqrt(np.mean(resid(x) ** 2))) for x in cands]
    order = np.argsort(scores)
    firsts = [cands[order[0]]] + ([np.zeros(len(names))] if order[0] != 0 else [cands[order[1]]])
    sol = None
    for x0 in firsts:
        cand = least_squares(resid, x0, bounds=(lb, ub), diff_step=0.03, x_scale=0.2,
                             max_nfev=max_evals // 2, xtol=2e-3, ftol=1e-3)
        if sol is None or cand.cost < sol.cost:
            sol = cand
    best = car(sol.x)
    v_i, lt = ideal_speed(best, tg)
    a, t, m = _features(v_i, tg)
    r = np.concatenate([a - tg.apex_v, t - tg.top_v, m - tg.acc_v]) * 3.6
    return FitOutcome(dict(zip(names, np.exp(sol.x).tolist())), best, lt, tg.lap_time, (a - tg.apex_v) * 3.6,
                      (t - tg.top_v) * 3.6, (m - tg.acc_v) * 3.6, float(np.sqrt(np.mean(r ** 2))), len(hist), hist)
