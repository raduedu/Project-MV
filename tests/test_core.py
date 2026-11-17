"""Tests for the core of Project MV: parameters, tracks, physics, sensors, environment, learning.

Run from the project root:  .venv\\Scripts\\python.exe -m pytest tests/test_core.py -q

Each test checks one thing against a number that can be worked out by hand,
so a failure points at a specific piece of the simulator.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mv.ac.acd import derive_key
from mv.ac.car_import import parse_ini, parse_lut
from mv.control.scripted import ScriptedDriver
from mv.envs.racing_env import EnvConfig, RacingEnv
from mv.rl.evaluate import eval_config, evaluate
from mv.rl.ppo import PPO, PPOConfig
from mv.sim.vehicle_sim import VehicleSim, _pacejka_bc
from mv.track.generator import PRESETS, random_tracks
from mv.track.track import Track, TrackBatch
from mv.vehicle.params import G, load_vehicle

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = sorted((ROOT / "vehicles" / "library").glob("*.yaml"))


def circle_track(radius: float = 100.0, half_width: float = 6.0, ds: float = 1.0, clockwise: bool = False) -> Track:
    a = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    if clockwise:
        a = -a
    xy = np.stack([radius * np.cos(a), radius * np.sin(a)], axis=1)
    return Track.from_centerline(xy, half_width, half_width, ds=ds, name="circle")


@pytest.fixture(scope="module")
def car():
    assert LIBRARY, "no vehicle files in vehicles/library"
    path = next((p for p in LIBRARY if "gt3" in p.name), LIBRARY[0])
    return load_vehicle(path)


def quiet_config(**kw) -> EnvConfig:
    """Environment config without randomness, for repeatable tests."""
    return eval_config(EnvConfig(**kw), max_time_s=kw.get("max_episode_s", 600.0))


# ---------------------------------------------------------------- data import
def test_acd_key_is_stable():
    # This key decodes the real archive (checked by reading car.ini from the game).
    assert derive_key("ks_porsche_911_gt3_rs") == "220-73-230-18-22-143-15-116"


def test_ini_and_lut_parsers():
    ini = parse_ini("[A]\nX=1 ; comment\nY = 2,3\n\n[B]\nZ=hello // other comment\n")
    assert ini == {"A": {"X": "1", "Y": "2,3"}, "B": {"Z": "hello"}}
    x, y = parse_lut("1000|50\n0|10 ; c\nbad line\n2000|80\n")
    assert x.tolist() == [0, 1000, 2000] and y.tolist() == [10, 50, 80]


# ---------------------------------------------------------------- vehicles
@pytest.mark.parametrize("path", LIBRARY, ids=lambda p: p.stem)
def test_library_vehicle_loads_and_simulates(path):
    vp = load_vehicle(path)
    sim = VehicleSim(vp, 4, "cpu")
    ids = torch.arange(4)
    sim.reset(ids, torch.zeros(4), torch.zeros(4), torch.zeros(4), torch.tensor([0.0, 10.0, 30.0, 60.0]))
    steer = torch.tensor([0.0, 0.2, -0.2, 0.05])
    for _ in range(500):                                     # 2 s of driving
        sim.step(steer, torch.ones(4) * 0.7, torch.zeros(4))
    for a in (sim.x, sim.y, sim.vx, sim.vy, sim.r, sim.soc):
        assert torch.isfinite(a).all()
    assert (sim.vx > 1.0).all(), "every car should be moving forward"


def test_pacejka_shape_factors():
    b, c = _pacejka_bc(peak_slip_deg=8.0, falloff=0.9)
    f = lambda a: math.sin(c * math.atan(b * a))             # noqa: E731
    assert f(math.radians(8.0)) == pytest.approx(1.0, abs=1e-6)       # peak where we asked for it
    assert f(math.radians(89.0)) == pytest.approx(0.9, abs=0.02)      # sliding grip = falloff level


def test_sign_conventions(car):
    """Positive steering = left turn = positive yaw rate, car moves towards +y."""
    sim = VehicleSim(car, 1, "cpu")
    sim.reset(torch.arange(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.tensor([20.0]))
    for _ in range(250):
        sim.step(torch.tensor([0.15]), torch.tensor([0.2]), torch.zeros(1))
    assert float(sim.r) > 0 and float(sim.y) > 0 and float(sim.yaw) > 0
    assert float(sim.ay) > 0


def test_braking_matches_tyre_grip(car):
    """A full stop from 100 km/h should average close to the tyre friction limit."""
    sim = VehicleSim(car, 1, "cpu")
    v0 = 100 / 3.6
    sim.reset(torch.arange(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.tensor([v0]))
    while float(sim.vx) > 0.5:
        sim.step(torch.zeros(1), torch.zeros(1), torch.ones(1))
    decel_g = v0 ** 2 / (2 * float(sim.x)) / G
    mu = 0.5 * (car.tire_front.mu_long + car.tire_rear.mu_long)
    assert 0.75 * mu < decel_g < 1.25 * mu, f"mean deceleration {decel_g:.2f} g vs tyre mu {mu:.2f}"
    assert abs(float(sim.y)) < 0.05, "straight-line braking must stay straight"


def test_standstill_is_stable(car):
    sim = VehicleSim(car, 1, "cpu")
    sim.reset(torch.arange(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    for _ in range(1000):
        sim.step(torch.tensor([0.5]), torch.zeros(1), torch.ones(1))
    assert abs(float(sim.vx)) < 0.05 and abs(float(sim.x)) < 0.05 and abs(float(sim.r)) < 0.01


def test_energy_bookkeeping(car):
    """Kinetic energy gained can never exceed the energy the powertrain could deliver."""
    from mv.vehicle.summary import unit_peak_power_kw
    sim = VehicleSim(car, 1, "cpu")
    sim.reset(torch.arange(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    t = 0.0
    while t < 8.0:
        sim.step(torch.zeros(1), torch.ones(1), torch.zeros(1))
        t += sim.dt
    kinetic = 0.5 * car.chassis.mass_kg * float(sim.vx) ** 2
    available = sum(unit_peak_power_kw(u) for u in car.powertrain.units) * 1000.0 * t
    assert kinetic < available


# ---------------------------------------------------------------- tracks
def test_circle_track_geometry():
    t = circle_track(100.0, 6.0)
    assert t.length == pytest.approx(2 * math.pi * 100, rel=1e-3)
    assert np.allclose(t.kappa, 1 / 100, rtol=0.01)
    assert t.turning_direction() == "counter-clockwise"
    # counter-clockwise: the left edge is the inside of the circle
    assert np.allclose(np.linalg.norm(t.left_xy, axis=1), 94.0, atol=0.05)
    assert np.allclose(np.linalg.norm(t.right_xy, axis=1), 106.0, atol=0.05)
    cw = circle_track(100.0, 6.0, clockwise=True)
    assert cw.turning_direction() == "clockwise" and np.allclose(cw.kappa, -1 / 100, rtol=0.01)


@pytest.mark.parametrize("preset", ["circuit", "fs", "mixed"])
def test_generated_tracks_are_valid(preset):
    cfg = PRESETS[preset]
    for t in random_tracks(6, seed=3, cfg=preset):
        assert cfg.length_m[0] * 0.7 < t.length < cfg.length_m[1] * 1.15
        assert t.min_radius() >= cfg.min_radius_m[0] * 0.99
        assert t.self_clearance() >= cfg.min_clearance_m
        assert np.isfinite(t.xy).all() and (t.w_left > 0).all()
        # closed and evenly sampled
        seg = np.linalg.norm(np.diff(np.vstack([t.xy, t.xy[:1]]), axis=0), axis=1)
        assert np.allclose(seg, t.ds, rtol=0.05)


def test_track_save_load(tmp_path):
    t = random_tracks(1, seed=1, cfg="fs")[0]
    t.save(tmp_path / "t.npz")
    u = Track.load(tmp_path / "t.npz")
    assert np.allclose(t.xy, u.xy) and np.allclose(t.kappa, u.kappa) and u.ds == pytest.approx(t.ds)


def test_track_batch_replace_and_wrap():
    a, b = circle_track(50.0), circle_track(120.0)
    batch = TrackBatch([a, a], ds=1.0, min_points=b.n)
    batch.replace(1, b)
    assert int(batch.n[1]) == b.n and float(batch.length[1]) == pytest.approx(b.length)
    tid = torch.tensor([0, 1])
    idx = torch.tensor([a.n + 3, -1])
    assert batch.wrap(tid, idx).tolist() == [3, b.n - 1]


# ---------------------------------------------------------------- sensors + environment
def make_env(car, track, n=1, **kw) -> RacingEnv:
    env = RacingEnv(car, TrackBatch([track], ds=track.ds), n, quiet_config(**kw), device="cpu")
    env.fixed_start = (0, 0.0)
    env.reset(torch.arange(n))
    return env


def test_observation_layout(car):
    env = make_env(car, circle_track(), n=3)
    obs = env.observe()
    assert obs.shape == (3, env.num_obs) and torch.isfinite(obs).all()
    assert sum(s.stop - s.start for s in env.obs_layout.values()) == env.num_obs
    assert torch.allclose(obs[:, env.obs_layout["kappa"]], torch.full((3, len(env.cfg.lookahead_m)), 0.25), atol=0.01)


def test_lidar_on_circle_matches_geometry(car):
    """Car on the centre line of a circular track, pointing along it."""
    radius, half = 100.0, 6.0
    env = make_env(car, circle_track(radius, half))
    d = env._lidar(torch.arange(1))[0]
    ang = list(env.cfg.lidar_angles_deg)
    assert float(d[ang.index(90)]) == pytest.approx(half, rel=0.03)        # straight left: inner edge
    assert float(d[ang.index(-90)]) == pytest.approx(half, rel=0.03)       # straight right: outer edge
    ahead = math.sqrt((radius + half) ** 2 - radius ** 2)                  # tangent ray meets the outer edge
    assert float(d[ang.index(0)]) == pytest.approx(ahead, rel=0.03)
    assert (d <= env.cfg.lidar_range_m + 1e-3).all() and (d > 0).all()


def test_blind_mode_hides_far_preview(car):
    env = make_env(car, circle_track())
    env.horizon[:] = 40.0
    k = env.observe()[0, env.obs_layout["kappa"]]
    far = torch.tensor(env.cfg.lookahead_m) > 40.0
    assert (k[far] == 0).all() and (k[~far] != 0).all()


def test_scripted_driver_laps_a_circle_and_timing_is_consistent(car):
    track = circle_track(80.0, 6.0)
    env = make_env(car, track)
    res = evaluate(ScriptedDriver(env), car, [track], "cpu", laps=2, record=True)[0]
    assert res.finished, f"scripted driver failed: {res.end_reason} after {res.distance_m:.0f} m"
    assert res.off_track_steps == 0
    # flying lap time must equal lap length / mean speed over that lap
    tr = res.trace
    lap2 = tr["t"] > res.lap_times[0]
    mean_speed = tr["speed"][lap2].mean()
    assert res.flying_lap == pytest.approx(track.length / mean_speed, rel=0.03)
    # and the speed must respect the grip limit: v^2 / R below the tyre limit (plus downforce margin)
    assert tr["speed"].max() ** 2 / 80.0 < 1.6 * car.tire_rear.mu_lat * G


def test_scripted_driver_completes_random_tracks(car):
    tracks = random_tracks(3, seed=11, cfg="mixed")
    env = make_env(car, tracks[0])
    res = evaluate(ScriptedDriver(env), car, tracks, "cpu", laps=1)
    assert all(r.finished for r in res), [(r.track, r.end_reason, round(r.distance_m)) for r in res]


def test_leaving_the_track_ends_the_episode(car):
    env = make_env(car, circle_track(80.0, 6.0))
    env.sim.vx[:] = 30.0
    terminated = torch.zeros(1, dtype=torch.bool)
    total = 0.0
    for _ in range(200):                                     # full throttle, no steering, on a circle
        _, rew, terminated, _, info = env.step(torch.tensor([[0.0, 1.0]]))
        total += float(rew)
        if terminated.any():
            break
    assert terminated.any() and bool(info["off"][0])
    assert float(rew) < -env.cfg.crash_penalty * 0.9


def test_progress_reward_equals_distance(car):
    track = circle_track(200.0, 8.0)
    env = make_env(car, track, smooth_penalty=0.0)
    env.sim.vx[:] = 20.0
    driver = ScriptedDriver(env)
    obs, total = env.observe(), 0.0
    for _ in range(100):
        a, _, _ = driver.act(obs)
        obs, rew, *_ = env.step(a)
        total += float(rew)
    assert total == pytest.approx(env.cfg.progress_scale * float(env.s_total[0]), rel=1e-3)
    # 100 decisions = 4 s starting at 20 m/s: somewhere between coasting and hard acceleration
    assert 60.0 < float(env.s_total[0]) < 140.0


# ---------------------------------------------------------------- learning
def test_ppo_runs_and_checkpoints(car, tmp_path):
    tracks = TrackBatch(random_tracks(4, seed=2, cfg="mixed"), ds=1.0)
    env = RacingEnv(car, tracks, 32, EnvConfig(), device="cpu", seed=1)
    ppo = PPO(env, PPOConfig(rollout_steps=8, epochs=2, minibatches=2, hidden=(32, 32)))
    for _ in range(2):
        stats = ppo.collect()
        out = ppo.update()
    assert math.isfinite(out["kl"]) and math.isfinite(out["value_loss"]) and stats["speed"] >= 0
    ppo.save(tmp_path / "m.pt")
    before = ppo.net.act(ppo.obs, deterministic=True)[0]
    other = PPO(env, ppo.cfg)
    other.load(tmp_path / "m.pt")
    assert torch.allclose(before, other.net.act(ppo.obs, deterministic=True)[0], atol=1e-5)


def test_evaluate_gives_up_on_a_driver_that_does_not_move(car):
    env = make_env(car, circle_track())

    class Parked:
        def act(self, obs, deterministic=True):
            z = torch.zeros(obs.shape[0])
            return torch.zeros(obs.shape[0], 2), z, z

    res = evaluate(Parked(), car, [circle_track()], "cpu")[0]
    assert not res.finished and res.end_reason in ("stalled", "slow")
