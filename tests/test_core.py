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
    from dataclasses import replace
    # eval_config is strict (no rejoining); settings given explicitly by a test win
    return replace(eval_config(EnvConfig(**kw), max_time_s=kw.get("max_episode_s", 600.0)), **kw)


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
    from mv.vehicle.summary import top_speed
    v_top = top_speed(vp)                                    # speeds scaled to what the car can do
    sim.reset(ids, torch.zeros(4), torch.zeros(4), torch.zeros(4), torch.tensor([0.0, 0.12, 0.35, 0.7]) * v_top)
    steer = torch.tensor([0.0, 0.2, -0.1, 0.02])
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


def test_car_without_abs_loses_steering_when_brakes_lock():
    """No ABS (the generic FS car): stamping on the brakes mid-corner locks the wheels,
    and locked wheels cannot steer - the car must run wider than with gentle braking."""
    vp = load_vehicle(ROOT / "vehicles" / "library" / "fs_generic_ev.yaml")
    assert not vp.electronics.abs
    sim = VehicleSim(vp, 2, "cpu")
    ids = torch.arange(2)
    sim.reset(ids, torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.full((2,), 12.0))    # slow enough that downforce is small
    steer = torch.full((2,), 0.35)
    brake = torch.tensor([1.0, 0.3])                       # car 0 stamps on the pedal, car 1 brakes gently
    for _ in range(int(0.6 / sim.dt)):
        sim.step(steer, torch.zeros(2), brake)
    assert float(sim.yaw[0]) < 0.6 * float(sim.yaw[1]), "locked wheels should turn the car far less"
    assert torch.isfinite(sim.x).all() and torch.isfinite(sim.yaw).all()


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


def test_one_driver_can_train_on_several_cars():
    """MultiCarEnv presents several car models as one environment to PPO."""
    from mv.envs.multi_car import MultiCarEnv
    from mv.rl.pretrain import warm_start
    vps = [load_vehicle(p) for p in LIBRARY]
    tracks = TrackBatch(random_tracks(4, seed=8, cfg="mixed"), ds=1.0)
    env = MultiCarEnv(vps, tracks, 8 * len(vps), EnvConfig(), device="cpu", seed=3)
    obs = env.observe()
    assert obs.shape == (env.n, env.num_obs)
    desc = obs[:, env.obs_layout["car_desc"]]
    assert not torch.allclose(desc[0], desc[-1]), "different cars must have different descriptors"
    ppo = PPO(env, PPOConfig(rollout_steps=8, epochs=1, minibatches=2, hidden=(32, 32)))
    info = warm_start(ppo, env.scripted_teacher(), steps=20, critic_updates=1, printer=lambda *_: None)
    assert math.isfinite(info["imitation_error"])
    ppo.collect()
    out = ppo.update()
    assert math.isfinite(out["value_loss"])
    env.tracks_replaced(torch.tensor([0, 1]))
    assert torch.isfinite(env.observe()).all()


@pytest.mark.parametrize("preset", ["gp", "fs_segments"])
def test_segment_tracks_are_closed_valid_and_have_braking_zones(preset):
    from mv.track.segments import SEGMENT_PRESETS, segment_track
    cfg = SEGMENT_PRESETS[preset]
    rng = np.random.default_rng(5)
    long_straights = 0
    for _ in range(6):
        t = segment_track(rng, preset)
        assert t.min_radius() >= cfg.min_radius_m * 0.99
        assert t.self_clearance() >= cfg.min_clearance_m
        assert t.length <= cfg.max_length_m * 1.02
        seg = np.linalg.norm(np.diff(np.vstack([t.xy, t.xy[:1]]), axis=0), axis=1)
        assert np.allclose(seg, t.ds, rtol=0.05)            # closed: last point joins the first
        straight = np.abs(t.kappa) < 1 / 800.0
        run = best = 0
        for s in np.concatenate([straight, straight]):
            run = run + 1 if s else 0
            best = max(best, run)
        long_straights += min(best, t.n) * t.ds >= cfg.long_straight_m[0] * 0.8
    assert long_straights >= 3, "most circuits should contain a long straight"


def test_preset_mix_draws_from_both_generators():
    ts = random_tracks(12, seed=4, cfg="mixed+gp")
    kinds = {t.meta.get("generator") for t in ts}
    assert "segments" in kinds and kinds & {"polygon", "spline"}


def test_blind_view_limits_preview_and_lidar(car):
    env = make_env(car, circle_track(300.0, 6.0))
    env.set_view(30.0)
    obs = env.observe()[0]
    lidar_m = (obs[env.obs_layout["lidar"]] ** 2) * env.cfg.lidar_range_m       # undo the sqrt scaling
    assert float(lidar_m.max()) <= 30.0 + 1e-3
    far = torch.tensor(env.cfg.lookahead_m) > 30.0
    assert (obs[env.obs_layout["kappa"]][far] == 0).all()
    env.set_view(None)
    assert float(((env.observe()[0][env.obs_layout["lidar"]] ** 2) * env.cfg.lidar_range_m).max()) > 30.0


def test_memory_holds_past_car_states(car):
    cfg = quiet_config(history_steps=4, history_every=2)
    env = RacingEnv(car, TrackBatch([circle_track()]), 1, cfg, device="cpu")
    env.fixed_start = (0, 0.0)
    env.reset(torch.arange(1))
    lay = env.obs_layout
    assert lay["history"].stop == env.num_obs and lay["history"].stop - lay["history"].start == 4 * 15
    cores = []
    for i in range(12):
        cores.append(env._core(torch.arange(1))[0].clone())     # state before this decision's physics...
        obs, *_ = env.step(torch.tensor([[0.0, 1.0]]))
    cores.append(env._core(torch.arange(1))[0].clone())         # ...and the current one
    hist = obs[0, lay["history"]].reshape(4, 15)
    # sample j (1-based) is the state j*every decisions before the current one
    for j in range(1, 5):
        assert torch.allclose(hist[j - 1], cores[-1 - 2 * j].clamp(-5, 5), atol=1e-5)


def test_recover_on_crash_puts_the_car_back_and_continues(car):
    cfg = quiet_config(recover_on_crash=True, max_recoveries=2)
    env = RacingEnv(car, TrackBatch([circle_track(80.0, 6.0)]), 1, cfg, device="cpu")
    env.fixed_start = (0, 0.0)
    env.reset(torch.arange(1))
    env.sim.vx[:] = 35.0                                     # far too fast for an 80 m circle, no steering
    recovered, terminated = 0, False
    for _ in range(600):
        _, rew, term, _, info = env.step(torch.tensor([[0.0, 1.0]]))
        if bool(info["recovered"][0]):
            recovered += 1
            assert float(rew[0]) < -env.cfg.crash_penalty * 0.9          # the mistake still costs
            assert abs(float(env.lat[0])) < 0.5 and abs(float(env.head_err[0])) < 0.05   # back on, pointing the right way
        if bool(term[0]):
            terminated = True
            break
    assert recovered == 2 and terminated, "two rejoins, then the third off ends the episode"


def test_lap_memory_remembers_last_lap(car):
    track = circle_track(120.0, 8.0)
    cfg = quiet_config(lap_memory=True)
    env = RacingEnv(car, TrackBatch([track]), 1, cfg, device="cpu")
    env.fixed_start = (0, 0.0)
    env.reset(torch.arange(1))
    lay = env.obs_layout["lap_memory"]
    n = len(env.cfg.lookahead_m)
    driver = ScriptedDriver(env)
    obs = env.observe()
    assert (obs[0, lay][2 * n:] == 0).all(), "nothing is known before the first lap"
    speeds = []
    while int(env.lap_count[0]) < 2:
        a, _, _ = driver.act(obs)
        obs, *_ = env.step(a)
        speeds.append(float(env.sim.vx[0]))
    block = obs[0, lay]
    known = block[2 * n:]
    assert (known == 1).all(), "after a full lap every section ahead has a memory"
    v_last = block[:n] * 50.0
    # on a constant circle the remembered speed ahead is close to the speed actually driven
    assert float(v_last[:10].mean()) == pytest.approx(float(np.mean(speeds[-50:])), rel=0.1)
    u_last = block[n:2 * n]
    assert 0.2 < float(u_last.mean()) < 1.5                  # a cautious driver: below the limit, but not idle
