"""System identification: make the simulator behave like Assetto Corsa (or a real car).

The idea in plain words
-----------------------
Record a few laps in the target world: at every tick, what the car did (speed,
sideways speed, yaw rate, accelerations) and what it was told to do (steering,
throttle, brake).  Then cut the recording into short slices (~1.5 s).  For each
slice, start our simulator in exactly the recorded state and feed it exactly
the recorded inputs.  If our car parameters were perfect, the simulated motion
would follow the recorded one.  The differences tell us what is wrong; a
handful of correction factors (grip front/rear, downforce, drag, power,
braking) is then adjusted until the differences are as small as possible.

Why short slices: over a long replay, tiny errors grow until the simulated car
is somewhere else entirely and nothing can be compared.  Restarting from the
measured state every slice keeps each comparison meaningful ("multiple
shooting", in textbook terms).

All slices are simulated at once as a batch of cars, so one evaluation of a
parameter set takes a fraction of a second on the CPU.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from mv.sim.vehicle_sim import VehicleSim
from mv.vehicle.params import VehicleParams

__all__ = ["Recording", "FACTORS", "apply_factors", "replay_error", "fit", "FitResult"]

# Quantities recorded every tick (all SI, body frame x forward / y left):
CHANNELS = ("t", "vx", "vy", "r", "ax", "ay", "steer", "throttle", "brake", "gear")


@dataclass
class Recording:
    """Time series of one recording session.  ``steer`` is the fraction of full
    steering lock (+ = left); ``throttle``/``brake`` are the pedal positions the
    car actually received (after any driver aids); ``gear`` is 1 = first gear."""
    t: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    r: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    steer: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray
    gear: np.ndarray
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=np.array([repr(self.meta)]), **{c: getattr(self, c) for c in CHANNELS})

    @classmethod
    def load(cls, path: str | Path) -> "Recording":
        import ast
        z = np.load(path, allow_pickle=False)
        return cls(**{c: z[c].astype(float) for c in CHANNELS}, meta=ast.literal_eval(str(z["meta"][0])))

    @classmethod
    def from_rows(cls, rows: list[dict], meta: dict | None = None) -> "Recording":
        return cls(**{c: np.array([r[c] for r in rows], float) for c in CHANNELS}, meta=meta or {})

    def __len__(self) -> int:
        return len(self.t)


# --------------------------------------------------------------------------
# Correction factors
# --------------------------------------------------------------------------
#: name -> what it scales.  1.0 = the vehicle file unchanged.
#: Grip is expressed as an overall level and a front/rear balance rather than per
#: axle: raising front grip and lowering rear grip both shift the balance the same
#: way, so per-axle factors are strongly coupled and hard to fit independently
#: (engineering log WP19).  front = grip * sqrt(balance), rear = grip / sqrt(balance).
FACTORS = {
    "grip": "peak friction of all tyres (lateral and longitudinal)",
    "grip_balance": "front tyre grip relative to rear (>1 = more front grip, less oversteer)",
    "downforce": "downforce area (front and rear together)",
    "drag": "drag area",
    "power": "torque of every power unit",
    "brakes": "maximum brake torque",
}


def apply_factors(vp: VehicleParams, f: dict[str, float]) -> VehicleParams:
    """Return a copy of ``vp`` with the correction factors applied."""
    v = copy.deepcopy(vp)
    g, b = f.get("grip", 1.0), f.get("grip_balance", 1.0)
    front = g * b ** 0.5 * f.get("grip_front", 1.0)
    rear = g / b ** 0.5 * f.get("grip_rear", 1.0)
    v.tire_front.mu_lat *= front
    v.tire_front.mu_long *= front
    v.tire_rear.mu_lat *= rear
    v.tire_rear.mu_long *= rear
    v.aero.cla_front_m2 *= f.get("downforce", 1.0)
    v.aero.cla_rear_m2 *= f.get("downforce", 1.0)
    v.aero.cda_m2 *= f.get("drag", 1.0)
    for u in v.powertrain.units:
        u.torque_nm = [t * f.get("power", 1.0) for t in u.torque_nm]
    v.brakes.max_torque_nm *= f.get("brakes", 1.0)
    return v


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------
def _windows(rec: Recording, window_s: float, stride_s: float, min_speed: float) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(rec)
    t = rec.t
    while i < n - 2:
        j = int(np.searchsorted(t, t[i] + window_s))
        if j >= n:
            break
        seg = slice(i, j + 1)
        ok = (rec.vx[seg].min() > min_speed) and np.all(np.diff(t[seg]) > 0) and np.all(np.diff(t[seg]) < 0.2)
        if ok:
            out.append((i, j))
        i = int(np.searchsorted(t, t[i] + stride_s))
    return out


# error scales: how big a difference counts as "1" for each channel
_SCALE = {"vx": 1.0, "vy": 0.3, "r": 0.05, "ax": 1.5, "ay": 1.5}


def replay_error(vp: VehicleParams, rec: Recording, window_s: float = 1.5, stride_s: float = 0.75,
                 min_speed: float = 8.0, substeps_per_tick: int = 4, return_traces: bool = False,
                 return_residuals: bool = False):
    """Mean squared, scaled difference between recorded and replayed motion.

    Returns ``(error, per_channel)`` where ``per_channel`` maps each channel to the
    RMS error of a typical (median) window in physical units; with
    ``return_traces`` also the simulated and measured traces (useful for plots).
    """
    wins = _windows(rec, window_s, stride_s, min_speed)
    if not wins:
        raise ValueError("no usable replay windows (car too slow or recording too short)")
    lengths = [j - i for i, j in wins]
    steps = min(lengths)
    w = len(wins)
    start = np.array([i for i, _ in wins])
    dt_tick = float(np.median(np.diff(rec.t)))
    sim = VehicleSim(vp, w, "cpu", dt=dt_tick / substeps_per_tick)
    # the recorded pedals and steering are what the car actually received: no extra lag
    sim.k_thr = sim.k_brk = 1.0
    sim.steer_rate = 1e3
    ids = torch.arange(w)
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32))  # noqa: E731
    z = torch.zeros(w)
    sim.reset(ids, z, z, z, t(rec.vx[start]))
    sim.vy[:], sim.r[:] = t(rec.vy[start]), t(rec.r[start])
    sim.ax[:], sim.ay[:] = t(rec.ax[start]), t(rec.ay[start])
    sim.steer[:] = t(rec.steer[start]) * sim.steer_max
    gear = np.clip(rec.gear[start].astype(int) - 1, 0, int(sim.gb_ngears[0]) - 1)
    sim.gear[:, 0] = torch.as_tensor(gear)
    sims = {c: np.zeros((w, steps)) for c in _SCALE}
    meas = {c: np.stack([getattr(rec, c)[i + 1:i + 1 + steps] for i in start]) for c in _SCALE}
    for k in range(steps):
        idx = start + k
        # Inputs are sampled once per tick, but during the tick they move from one sample
        # to the next; the mid-tick value (average of both ends) is the best constant
        # stand-in.  Using the start value alone lags the car by half a tick.
        mid = lambda a: t(0.5 * (a[idx] + a[idx + 1]))  # noqa: E731
        st, th, br = mid(rec.steer), mid(rec.throttle), mid(rec.brake)
        for _ in range(substeps_per_tick):
            sim.step(st, th, br)
        sims["vx"][:, k], sims["vy"][:, k], sims["r"][:, k] = sim.vx.numpy(), sim.vy.numpy(), sim.r.numpy()
        sims["ax"][:, k], sims["ay"][:, k] = sim.ax.numpy(), sim.ay.numpy()
    # Robust error: per window, the mean scaled squared difference; windows are then
    # combined with a soft cap (Cauchy-type), so a few chaotic slices (a spin, a kerb
    # strike, a crash) cannot outweigh hundreds of ordinary ones.
    win_err = np.zeros(w)
    per = {}
    resid = []
    for c, s in _SCALE.items():
        d = sims[c] - meas[c]
        d = np.nan_to_num(d, nan=10.0 * s, posinf=10.0 * s, neginf=-10.0 * s)
        win_err += np.mean((d / s) ** 2, axis=1)
        per[c] = float(np.sqrt(np.median(np.mean(d ** 2, axis=1))))     # typical window, physical units
        resid.append((d / s).ravel() / np.sqrt(steps))                  # per-sample, for least squares
    err = float(np.mean(np.log1p(win_err)))
    if return_residuals:
        return np.concatenate(resid)
    if return_traces:
        return err, per, sims, meas
    return err, per


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------
@dataclass
class FitResult:
    factors: dict[str, float]
    error_before: float
    error_after: float
    per_channel_before: dict[str, float]
    per_channel_after: dict[str, float]
    evaluations: int
    sensitivity: dict[str, float] = field(default_factory=dict)   # error rise for a +-5 % change of each factor

    def well_determined(self, threshold: float = 0.01) -> dict[str, bool]:
        """A factor is trustworthy only if changing it by 5 % clearly worsens the match.

        Factors the recordings hardly exercise (e.g. downforce when the car never
        went fast) stay near 1.0 or drift; they must not be believed."""
        return {n: s >= threshold for n, s in self.sensitivity.items()}


def fit(vp: VehicleParams, recordings: list[Recording], names: tuple[str, ...] = tuple(FACTORS),
        bounds: tuple[float, float] = (0.7, 1.4), rounds: int = 3, first_step: float = 0.04,
        global_generations: int = 12, global_popsize: int = 8, seed: int = 0, printer=print) -> FitResult:
    """Find correction factors that make the simulator reproduce the recordings.

    Two stages, both on the logarithm of the factors (so 0.8 and 1.25 are equally
    far from 1):

    1. *Global search* (differential evolution): a population of candidate
       factor sets is improved generation by generation by mixing good
       candidates.  It copes with factors that can partly stand in for each
       other (less grip vs less downforce), which create long diagonal valleys.
    2. *Polish* by coordinate search: one factor at a time, try it a little lower
       and higher, keep the best (refined with a parabola), halve the step,
       repeat.

    Why not a standard gradient or simplex optimiser: the error surface is steep
    and slightly jagged (gear shifts make it jump); those were tried and stalled
    (engineering log WP19).
    """
    from scipy.optimize import differential_evolution
    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    count = [0]

    def summary(x: np.ndarray) -> tuple[float, dict[str, float]]:
        count[0] += 1
        v = apply_factors(vp, dict(zip(names, np.exp(x))))
        res = [replay_error(v, rec) for rec in recordings]
        per = {c: float(np.mean([p[c] for _, p in res])) for c in _SCALE}
        return float(np.mean([e for e, _ in res])), per

    x = np.zeros(len(names))
    e0, p0 = summary(x)
    if global_generations > 0:
        de = differential_evolution(lambda z: summary(z)[0], [(lo, hi)] * len(names), maxiter=global_generations,
                                    popsize=global_popsize, init="sobol", seed=seed, polish=False, tol=1e-4,
                                    mutation=(0.4, 0.9), recombination=0.7, x0=x)
        x = np.asarray(de.x)
        printer("  global search: error %.4f  " % de.fun + "  ".join(f"{n} {np.exp(v):.3f}" for n, v in zip(names, x)))
    best = summary(x)[0]
    step = float(np.log(1.0 + first_step))
    for rnd in range(rounds):
        for i, name in enumerate(names):
            offsets = np.array([-2.0, -1.0, 1.0, 2.0]) * step
            trial = {0.0: best}
            for o in offsets:
                xt = x.copy(); xt[i] = np.clip(x[i] + o, lo, hi)
                trial[float(xt[i] - x[i])] = summary(xt)[0]
            ks = sorted(trial)
            es = np.array([trial[k] for k in ks])
            j = int(np.argmin(es))
            shift = ks[j]
            if 0 < j < len(ks) - 1:                       # parabola through the best point and its neighbours
                a, b, c = ks[j - 1], ks[j], ks[j + 1]
                fa, fb, fc = es[j - 1], es[j], es[j + 1]
                den = (b - a) * (fb - fc) - (b - c) * (fb - fa)
                if abs(den) > 1e-12:
                    v = b - 0.5 * ((b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)) / den
                    if a < v < c:
                        xt = x.copy(); xt[i] = np.clip(x[i] + v, lo, hi)
                        ev = summary(xt)[0]
                        if ev < es[j]:
                            shift, es[j] = float(xt[i] - x[i]), ev
            x[i] = np.clip(x[i] + shift, lo, hi)
            best = float(es[j])
        printer(f"  round {rnd + 1}: error {best:.4f}  " + "  ".join(f"{n} {np.exp(v):.3f}" for n, v in zip(names, x)))
        step *= 0.5
    factors = {n: float(np.exp(v)) for n, v in zip(names, x)}
    e1, p1 = summary(x)
    # How well does the data pin down each factor?  Error rise for +-5 %.
    sens = {}
    for i, n in enumerate(names):
        rise = []
        for d in (np.log(1.05), -np.log(1.05)):
            xt = x.copy(); xt[i] = xt[i] + d
            rise.append(summary(xt)[0] - e1)
        sens[n] = float(min(rise))
    printer(f"replay error {e0:.4f} -> {e1:.4f} after {count[0]} evaluations")
    return FitResult(factors, e0, e1, p0, p1, count[0], sens)
