"""Fastest speed along a given path: the forward-backward pass.

The question
------------
The path is fixed (curvature ``kappa`` at stations along its length).  The car
is described by its g-g-v envelope (:mod:`mv.planning.ggv`).  How fast can it
go at every station, and how long does the lap take?

The answer in three sweeps
--------------------------
1. **Corner limit.**  Driving a curve of curvature ``kappa`` at speed ``v``
   needs the lateral acceleration ``v^2 * kappa``.  The envelope says how much
   is available, so every station has a top speed ``v_lim`` - the speed at
   which the car would use *all* its grip just for cornering.
2. **Backward (braking) pass.**  The car cannot shed speed instantly.  Walk
   the lap backwards: "if I must be at speed ``v[i+1]`` at the next station,
   and I brake as hard as the envelope allows while still cornering, how fast
   may I be here?"  Take the smaller of that and ``v_lim``.
3. **Forward (acceleration) pass.**  The car cannot gain speed instantly
   either.  Walk the lap forwards: "from ``v[i]``, with the most acceleration
   the envelope allows while cornering, how fast am I at the next station?"
   Take the smaller of that and the result of step 2.

Each step uses constant acceleration over one segment:
``v_next^2 = v^2 + 2 * a * ds``.  Because the lap is closed, the sweeps run
around the lap until nothing changes any more (one extra partial lap).

What is included / neglected
----------------------------
* Included: combined braking+cornering and acceleration+cornering (the full
  envelope, not just a friction circle), drag, downforce, power and traction
  limits, and - optionally - the speed lost during each up-shift.
* Neglected: transients (the car is assumed to move from one steady state to
  the next instantly - no yaw inertia, no actuator or load-transfer lag),
  the gearbox's down-shift hysteresis, battery depletion, elevation, banking.

All functions accept one path (1-D arrays) or a batch of paths (2-D arrays,
one path per row) - the race-line optimiser evaluates hundreds of slightly
different lines at once that way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mv.planning.ggv import GGV

__all__ = ["SpeedProfile", "ShiftModel", "solve_speed_profile", "lap_times"]

_V_FLOOR = 1.0     # m/s; keeps divisions finite on absurdly tight paths
_DOWNSHIFT = 0.93  # the simulator changes down at 93 % of the up-shift speed


@dataclass
class SpeedProfile:
    """Result of the forward-backward pass for one closed path with M stations."""
    v: np.ndarray           # (M,) speed at each station, m/s
    ax: np.ndarray          # (M,) longitudinal acceleration on the segment to the next station, m/s^2
    ay: np.ndarray          # (M,) lateral acceleration, m/s^2 (positive = turning left)
    t: np.ndarray           # (M,) time of arrival at each station, s (t[0] = 0)
    lap_time: float
    v_limit: np.ndarray     # (M,) corner-limited speed (sweep 1)
    upshifts: int = 0       # number of up-shifts counted on the lap


@dataclass
class ShiftModel:
    """Torque interruption at every up-shift.

    While the gearbox changes up, the simulator cuts the torque of everything
    connected to that gearbox for ``upshift_time_s``.  In the forward pass we
    do the same: when the speed crosses a shift speed, the share of the drive
    force that goes through the shifting gearbox is removed for the distance
    covered in that time.

    ``speeds`` are all up-shift speeds of the car (sorted), ``time[k]`` the
    duration of shift k, ``share[k]`` the fraction of the total drive force
    that is interrupted (1 for a single-gearbox car, less when another axle
    keeps pulling), ``resist[k]`` drag plus rolling-resistance deceleration at
    that speed (the drive force in use is ``m * (ax + resist)``).
    """
    speeds: np.ndarray
    time: np.ndarray
    share: np.ndarray
    resist: np.ndarray

    @classmethod
    def from_vehicle(cls, vp) -> "ShiftModel":
        from mv.planning.ggv import PowertrainMap
        from mv.vehicle.params import G
        pt = PowertrainMap(vp)
        rows = []
        for gb, ups in enumerate(pt.upshift_v):
            for v_up in ups[:-1]:
                v_before = np.array(v_up - 0.05)
                front, rear = pt.axle_forces(v_before, 1.0)
                total = float(front + rear)
                share = float(pt.gearbox_force(gb, v_before)) / total if total > 0 else 0.0
                resist = (0.5 * vp.aero.air_density * vp.aero.cda_m2 * v_up ** 2) / vp.chassis.mass_kg                     + vp.tire_rear.rolling_resistance * G
                rows.append((float(v_up), pt.shift_time_up[gb], min(share, 1.0), resist))
        rows.sort()
        arr = np.array(rows, float).reshape(-1, 4)
        return cls(speeds=arr[:, 0], time=arr[:, 1], share=arr[:, 2], resist=arr[:, 3])


def _make_lookup(ggv: GGV, table: np.ndarray):
    """Fast bilinear look-up ``a(v, |kappa|)`` into one of the envelope tables.

    Equivalent to ``GGV._lookup`` with ``ay = v^2 * kappa`` but with the grid
    arithmetic inlined - this sits in the innermost loop.
    """
    dv, nv = float(ggv.v[1] - ggv.v[0]), len(ggv.v)
    dth, nt = float(ggv.theta[1] - ggv.theta[0]), len(ggv.theta)
    ay_max, flat = ggv.ay_max, np.ascontiguousarray(table).ravel()
    x_top, y_top = nv - 1.000001, nt - 1.000001

    def lookup(v: np.ndarray, kap: np.ndarray) -> np.ndarray:
        x = np.minimum(v / dv, x_top)
        i = x.astype(np.intp)
        fx = x - i
        ay_lim = ay_max[i] + fx * (ay_max[i + 1] - ay_max[i])
        y = np.minimum(np.arcsin(np.minimum(v * v * kap / ay_lim, 1.0)) / dth, y_top)
        j = y.astype(np.intp)
        fy = y - j
        k = i * nt + j
        lo = flat[k] + fy * (flat[k + 1] - flat[k])
        hi = flat[k + nt] + fy * (flat[k + nt + 1] - flat[k + nt])
        return lo + fx * (hi - lo)

    return lookup


def _sweeps(kap: np.ndarray, ds: np.ndarray, ggv: GGV, order: int, shift: ShiftModel | None,
            max_laps: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core solver on a batch.  ``kap`` (B, M) is |curvature|, ``ds`` (B, M) the segment lengths.

    ``order=1``: acceleration evaluated at the start of each segment (classic).
    ``order=2``: averaged with the acceleration at the predicted end of the
    segment (Heun's method) - noticeably more accurate on coarse grids.
    """
    n_batch, m = kap.shape
    v_lim = np.maximum(ggv.corner_speed(kap), _V_FLOOR)
    brake, accel = _make_lookup(ggv, ggv.ax_min), _make_lookup(ggv, ggv.ax_max)
    floor2 = _V_FLOOR ** 2

    # ---- backward: braking-limited speed --------------------------------------
    vb = v_lim.copy()
    done = np.zeros(n_batch, bool)
    for step in range(max_laps * m):
        i = m - 1 - step % m
        j = (i + 1) % m
        v_next, d = vb[:, j], ds[:, i]
        a = brake(v_next, kap[:, j])                       # negative = braking
        v2 = v_next * v_next - 2.0 * a * d
        if order == 2:
            v_pred = np.minimum(np.sqrt(np.maximum(v2, floor2)), v_lim[:, i])
            v2 = v_next * v_next - (a + brake(v_pred, kap[:, i])) * d
        new = np.minimum(np.sqrt(np.maximum(v2, floor2)), v_lim[:, i])
        if step >= m:                                      # second lap: stop once nothing changes
            done |= np.abs(new - vb[:, i]) < 1e-9
        vb[:, i] = new
        if step >= m and done.all():
            break

    # ---- forward: acceleration-limited speed --------------------------------------
    vf = vb.copy()
    shifts = np.zeros(n_batch, int)
    use_shift = shift is not None and len(shift.speeds) > 0
    if use_shift:
        gear = np.searchsorted(shift.speeds, vf[:, 0])     # number of shift speeds below the current speed
        coast = np.zeros(n_batch)                          # metres of torque interruption still to go
        cut, resist = np.zeros(n_batch), np.zeros(n_batch)
    done[:] = False
    for step in range(max_laps * m):
        i = step % m
        j = (i + 1) % m
        v_here, d = vf[:, i], ds[:, i]
        a = accel(v_here, kap[:, i])
        v2 = v_here * v_here + 2.0 * a * d
        if order == 2:
            v_pred = np.minimum(np.sqrt(np.maximum(v2, floor2)), vb[:, j])
            a = 0.5 * (a + accel(v_pred, kap[:, j]))
            v2 = v_here * v_here + 2.0 * a * d
        if use_shift:
            # part of this segment is driven with the shifting gearbox's force missing
            part = np.minimum(coast, d) / d
            a = a - part * cut * np.maximum(a + resist, 0.0)
            v2 = v_here * v_here + 2.0 * a * d
            coast = np.maximum(coast - d, 0.0)
            v_new = np.sqrt(np.maximum(v2, floor2))
            new = np.minimum(v_new, vb[:, j])
            up = np.searchsorted(shift.speeds, new)        # gear the box wants while speeding up ...
            crossed = (up > gear) & (v_new <= vb[:, j])    # (a shift under braking/part throttle costs nothing)
            if crossed.any():
                k = np.minimum(up - 1, len(shift.speeds) - 1)
                coast = np.where(crossed, new * shift.time[k], coast)
                cut, resist = np.where(crossed, shift.share[k], cut), np.where(crossed, shift.resist[k], resist)
                if step < m:
                    shifts += crossed
            # ... and it only changes down again well below the shift speed (hysteresis, as in the
            # simulator), otherwise the speed lost during a shift would trigger the same shift again
            gear = np.minimum(np.maximum(gear, up), np.searchsorted(shift.speeds * _DOWNSHIFT, new))
        else:
            new = np.minimum(np.sqrt(np.maximum(v2, floor2)), vb[:, j])
        if step >= m:
            done |= np.abs(new - vf[:, j]) < 1e-9
        vf[:, j] = new
        if step >= m and done.all():
            break
    return vf, v_lim, shifts


def _as_batch(kappa, ds) -> tuple[np.ndarray, np.ndarray, bool]:
    kappa = np.asarray(kappa, float)
    single = kappa.ndim == 1
    kappa = np.atleast_2d(kappa)
    ds = np.broadcast_to(np.asarray(ds, float), kappa.shape)
    return kappa, ds, single


def lap_times(kappa, ds, ggv: GGV, order: int = 2, shift: ShiftModel | None = None) -> np.ndarray:
    """Lap time of each path in a batch.  ``kappa`` and ``ds`` have shape (B, M)."""
    kappa, ds, _ = _as_batch(kappa, ds)
    v, _, _ = _sweeps(np.abs(kappa), ds, ggv, order, shift)
    return (2.0 * ds / (v + np.roll(v, -1, axis=1))).sum(1)


def solve_speed_profile(kappa, ds, ggv: GGV, order: int = 2, shift: ShiftModel | None = None) -> SpeedProfile:
    """Speed profile of one closed path.

    ``kappa[i]`` is the signed curvature (1/m, positive = left) at station i,
    ``ds[i]`` (scalar or array) the distance from station i to station i+1;
    the last station connects back to the first.
    """
    kappa2, ds2, single = _as_batch(kappa, ds)
    if not single:
        raise ValueError("solve_speed_profile takes one path; use lap_times() for a batch")
    v2d, v_lim, shifts = _sweeps(np.abs(kappa2), ds2, ggv, order, shift)
    v, d = v2d[0], ds2[0]
    v_next = np.roll(v, -1)
    dt = 2.0 * d / (v + v_next)
    return SpeedProfile(v=v, ax=(v_next ** 2 - v ** 2) / (2.0 * d), ay=v * v * kappa2[0],
                        t=np.concatenate([[0.0], np.cumsum(dt)[:-1]]), lap_time=float(dt.sum()),
                        v_limit=v_lim[0], upshifts=int(shifts[0]))
