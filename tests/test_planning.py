"""Tests for mv.planning: g-g-v envelope, speed profile, racing line, reference lap.

Run with:  .venv/Scripts/python.exe -m pytest tests/test_planning.py -q

Several tests compare the planner against ``mv.sim.vehicle_sim.VehicleSim``.
They are the tripwire for physics drift: if the simulator's tyre or powertrain
model changes and ``mv/planning/ggv.py`` is not updated, they fail.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mv.planning import raceline as rl  # noqa: E402
from mv.planning.ggv import QuasiSteadyCar, build_ggv  # noqa: E402
from mv.planning.reference import ReferenceLap, default_margin, reference_lap, reference_lines  # noqa: E402
from mv.planning.speed_profile import ShiftModel, solve_speed_profile  # noqa: E402
from mv.sim.vehicle_sim import VehicleSim  # noqa: E402
from mv.track.generator import random_track  # noqa: E402
from mv.track.track import Track  # noqa: E402
from mv.vehicle.params import G, find_vehicle, load_vehicle  # noqa: E402

CAR = "porsche_911_gt3_rs_992"          # in vehicles/library, so the tests run on any machine


@pytest.fixture(scope="module")
def vp():
    return load_vehicle(find_vehicle(CAR))


@pytest.fixture(scope="module")
def ggv(vp):
    return build_ggv(vp)


def circle_track(radius: float, half_width: float = 5.0) -> Track:
    m = int(round(2 * math.pi * radius))
    th = np.linspace(0.0, 2 * math.pi, m, endpoint=False)
    return Track.from_centerline(np.stack([radius * np.cos(th), radius * np.sin(th)], 1), half_width, half_width,
                                 ds=1.0, name=f"circle_{radius:.0f}")


# --------------------------------------------------------------------------
# g-g-v envelope
# --------------------------------------------------------------------------
def test_ggv_shape_and_signs(ggv):
    assert np.all(ggv.ay_max > 0.8 * G) and np.all(ggv.ay_max < 3.0 * G)
    assert np.all(ggv.ax_min <= ggv.ax_max + 1e-9)
    assert np.all(ggv.ax_min[:, 0] < -0.9 * G)                  # straight-line braking beats 0.9 g everywhere
    assert ggv.ax_max[5, 0] > 0.5 * G                           # and it accelerates hard at low speed
    assert 250 / 3.6 < ggv.top_speed() < 360 / 3.6
    # the more the car corners, the less it can accelerate (allowing small numerical ripple)
    assert np.all(np.diff(ggv.ax_max[10:60], axis=1) < 0.3)
    # steady cornering is a little below the absolute cornering limit
    v = np.array([20.0, 40.0, 60.0])
    assert np.all(ggv.ay_steady(v) <= ggv.ay_limit(v) + 1e-9)
    assert np.all(ggv.ay_steady(v) > 0.9 * ggv.ay_limit(v))


def test_more_downforce_gives_more_lateral_grip(vp):
    pedal = np.linspace(-0.2, 0.6, 9)
    base = QuasiSteadyCar(vp)
    more = copy.deepcopy(vp)
    more.aero.cla_front_m2 *= 2.0
    more.aero.cla_rear_m2 *= 2.0
    winged = QuasiSteadyCar(more)
    fast, slow = np.full(9, 60.0), np.full(9, 8.0)
    assert winged.max_lateral(fast, pedal).max() > 1.03 * base.max_lateral(fast, pedal).max()
    # at walking pace downforce is irrelevant
    assert abs(winged.max_lateral(slow, pedal).max() / base.max_lateral(slow, pedal).max() - 1.0) < 0.01
    # and for one car, grip grows with speed
    assert base.max_lateral(fast, pedal).max() > base.max_lateral(slow, pedal).max()


def test_ggv_straight_line_acceleration_matches_simulator(vp, ggv):
    """0-200 km/h from the envelope's forward acceleration vs. the simulator at full throttle."""
    sim = VehicleSim(vp, 1, "cpu")
    zero, one = torch.zeros(1), torch.ones(1)
    sim.reset(torch.arange(1), zero, zero, zero, zero.clone())
    t_sim = 0.0
    while float(sim.vx) < 200 / 3.6 and t_sim < 60.0:
        sim.step(zero, one, zero)
        t_sim += sim.dt
    v, t, dt = 0.0, 0.0, 0.002
    while v < 200 / 3.6 and t < 60.0:
        v += float(ggv.ax_limits(v, 0.0)[1]) * dt
        t += dt
    assert abs(t / t_sim - 1.0) < 0.05, f"envelope {t:.2f} s vs simulator {t_sim:.2f} s"


def test_ggv_braking_matches_simulator(vp, ggv):
    """200-0 km/h stopping distance: envelope vs. simulator at full brake pedal."""
    sim = VehicleSim(vp, 1, "cpu")
    zero, one = torch.zeros(1), torch.ones(1)
    sim.reset(torch.arange(1), zero, zero, zero, torch.tensor([200 / 3.6]))
    while float(sim.vx) > 0.5:
        sim.step(zero, zero, one)
    v, x, dt = 200 / 3.6, 0.0, 0.002
    while v > 0.5:
        x += v * dt
        v += float(ggv.ax_limits(v, 0.0)[0]) * dt
    assert abs(x / float(sim.x) - 1.0) < 0.05, f"envelope {x:.1f} m vs simulator {float(sim.x):.1f} m"


@pytest.mark.parametrize("kmh,pedal,frac", [(80, 0.3, 0.9), (150, -0.4, 0.9), (200, 1.0, 0.95), (120, 0.0, 0.6)])
def test_force_balance_matches_one_simulator_step(vp, kmh, pedal, frac):
    """Put the simulator into the steady state the planner predicts and take one step.

    If the planner's equations are the simulator's equations, the car must
    (a) gain speed at the predicted rate and (b) neither start to spin nor to
    straighten (yaw acceleration ~ 0).
    """
    car = QuasiSteadyCar(vp)
    v = kmh / 3.6
    ay = frac * float(car.max_lateral(np.array(v), np.array(pedal)))
    res = car.solve(v, ay, pedal, n_iter=200)
    beta, steer = float(res["beta"]), float(res["steer"])
    sim = VehicleSim(vp, 1, "cpu")
    zero = torch.zeros(1)
    sim.reset(torch.arange(1), zero, zero, zero, torch.tensor([v * math.cos(beta)]))
    sim.vy[:], sim.r[:] = v * math.sin(beta), ay / v
    sim.steer[:], sim.thr[:], sim.brk[:] = steer, max(pedal, 0.0), max(-pedal, 0.0)
    sim.ax[:], sim.ay[:] = float(res["ax_body"]), float(res["ay_body"])
    v0, r0 = math.hypot(float(sim.vx), float(sim.vy)), float(sim.r)
    sim.step(torch.tensor([steer / sim.steer_max]), torch.tensor([max(pedal, 0.0)]), torch.tensor([max(-pedal, 0.0)]))
    ax_sim = (math.hypot(float(sim.vx), float(sim.vy)) - v0) / sim.dt
    yaw_acc = (float(sim.r) - r0) / sim.dt
    assert abs(ax_sim - float(res["ax"])) < 0.15, f"ax: planner {float(res['ax']):.3f}, simulator {ax_sim:.3f}"
    assert abs(yaw_acc) < 0.15, f"yaw acceleration {yaw_acc:.3f} rad/s^2 (should be ~0)"


# --------------------------------------------------------------------------
# speed profile
# --------------------------------------------------------------------------
@pytest.mark.parametrize("radius", [30.0, 80.0, 200.0])
def test_circle_speed_matches_analytic(ggv, radius):
    """On a constant-radius circle the car settles at v = sqrt(ay * R).

    ``ay`` is the lateral acceleration the envelope can hold at CONSTANT speed
    (``ay_steady``): the absolute maximum ``ay_max`` is only available while
    the car is slowing down (tyre drag), so it bounds the speed from above.
    """
    track = circle_track(radius)
    prof = solve_speed_profile(track.kappa, track.ds, ggv)
    v = 20.0
    for _ in range(100):                                       # fixed point: ay depends on speed (downforce)
        v = math.sqrt(float(ggv.ay_steady(v)) * radius)
    assert np.ptp(prof.v) < 1e-3 * v                           # constant speed all the way round
    assert abs(prof.v.mean() / v - 1.0) < 0.01
    v_cap = 20.0
    for _ in range(100):
        v_cap = math.sqrt(float(ggv.ay_limit(v_cap)) * radius)
    assert v_cap * 0.95 < prof.v.mean() <= v_cap * (1 + 1e-6)
    assert abs(prof.lap_time - track.length / prof.v.mean()) < 1e-3 * prof.lap_time


def test_speed_profile_respects_envelope_and_is_periodic(ggv):
    track = random_track(1, "mixed")
    prof = solve_speed_profile(track.kappa, track.ds, ggv)
    assert np.all(np.abs(prof.ay) <= ggv.ay_limit(prof.v) * (1 + 1e-3))    # 1e-3: table interpolation
    # The acceleration over a segment is the mean of what the envelope allows at its two ends
    # (second-order integration), so it must lie within the larger of the two end values.
    lo, hi = ggv.ax_limits(prof.v, prof.ay)
    hi2, lo2 = np.maximum(hi, np.roll(hi, -1)), np.minimum(lo, np.roll(lo, -1))
    assert np.mean(prof.ax > hi2 + 0.3) < 0.01
    assert np.mean(prof.ax < lo2 - 0.3) < 0.01
    # closed lap: the last segment leads back to the first station at the same speed (by construction),
    # and the lap time is the sum of the segment times
    dt = 2.0 * track.ds / (prof.v + np.roll(prof.v, -1))
    assert abs(dt.sum() - prof.lap_time) < 1e-9
    # finer sampling changes the lap time only marginally (second-order integration)
    fine = Track.from_centerline(track.xy, track.w_left, track.w_right, ds=0.5)
    grid = rl.make_grid(track, 1.2)
    spline, t = rl._closed_spline(grid.p)
    laps = []
    for n in (len(track.xy), 2 * len(track.xy)):
        tt = np.linspace(0.0, t[-1], n, endpoint=False)
        laps.append(solve_speed_profile(rl._spline_curvature(spline, tt), t[-1] / n, ggv).lap_time)
    assert abs(laps[0] / laps[1] - 1.0) < 2e-3
    assert fine.n == pytest.approx(2 * track.n, abs=2)


def test_upshift_losses_cost_time(vp, ggv):
    track = random_track(2, "mixed")
    plain = solve_speed_profile(track.kappa, track.ds, ggv)
    shifted = solve_speed_profile(track.kappa, track.ds, ggv, shift=ShiftModel.from_vehicle(vp))
    assert shifted.upshifts > 0
    assert plain.lap_time < shifted.lap_time < plain.lap_time * 1.02


# --------------------------------------------------------------------------
# racing line
# --------------------------------------------------------------------------
def test_box_qp_matches_scipy():
    from scipy.optimize import lsq_linear
    rng = np.random.default_rng(0)
    a = rng.normal(size=(60, 40))
    b = rng.normal(size=60)
    lo, hi = -0.1 * np.ones(40), 0.15 * np.ones(40)
    import scipy.sparse as sp
    x, _ = rl.solve_box_qp(sp.csc_matrix(a.T @ a), -a.T @ b, lo, hi, np.zeros(40))
    ref = lsq_linear(a, b, bounds=(lo, hi), method="bvls").x
    assert np.all(x >= lo - 1e-12) and np.all(x <= hi + 1e-12)
    assert np.linalg.norm(a @ x - b) <= np.linalg.norm(a @ ref - b) * (1 + 1e-6)
    assert np.abs(x - ref).max() < 1e-5


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_min_curvature_line_inside_bounds_and_faster_than_centre(vp, ggv, seed):
    track = random_track(seed, "mixed")
    margin = default_margin(vp)
    laps = reference_lines(track, vp, ("centre", "min_curvature"), ggv=ggv)
    line = laps["min_curvature"]
    lo, hi = rl.offset_limits(track, margin)
    assert np.all(line.n >= lo - 1e-9) and np.all(line.n <= hi + 1e-9)
    assert np.all(line.n <= track.w_left - margin + 1e-9) and np.all(line.n >= -(track.w_right - margin) - 1e-9)
    assert np.allclose(line.xy, track.xy + track.normal * line.n[:, None])
    assert line.meta["clip_max_m"] < 0.05                      # the hard clip is only a safety net
    assert line.lap_time <= laps["centre"].lap_time
    assert np.mean(line.kappa ** 2) < np.mean(laps["centre"].kappa ** 2)


def test_reference_lap_min_time_roundtrip(vp, ggv, tmp_path):
    track = random_track(1, "mixed")                           # ~750 m: quick
    ref = reference_lap(track, vp, ggv=ggv)
    times = ref.meta["lap_times"]
    assert times["min_time"] <= times["min_curvature"] <= times["centre"]
    assert ref.lap_time == pytest.approx(min(times.values()))
    lo, hi = rl.offset_limits(track, default_margin(vp))
    assert np.all(ref.n >= lo - 1e-9) and np.all(ref.n <= hi + 1e-9)
    assert len(ref.s) == track.n and ref.xy.shape == (track.n, 2)
    assert ref.time[0] == 0.0 and np.all(np.diff(ref.time) > 0) and ref.time[-1] < ref.lap_time
    assert ref.speed.min() > 3.0 and ref.speed.max() < 110.0
    ref.save(tmp_path / "ref.npz")
    back = ReferenceLap.load(tmp_path / "ref.npz")
    assert back.lap_time == pytest.approx(ref.lap_time)
    assert np.allclose(back.speed, ref.speed) and np.allclose(back.n, ref.n)
    assert back.meta["method"] == ref.meta["method"]
    assert float(np.load(tmp_path / "ref.npz")["lap_time"]) == pytest.approx(ref.lap_time)   # what evaluate.py reads
