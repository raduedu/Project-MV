"""Race bot stage 2 (engineering log WP61): grass/gravel past the edge and rejoining by itself.

Checks, each against a number worked out by hand:
* which wheels are off the road, from the car's position and angle;
* ploughing drag and lower grip slow a car down by the expected amount;
* with self-recovery, the grass is not the end - the barrier is; a spun car gets time to turn around;
* practice episodes start off the road;
* the default (no self-recovery) is unchanged: every wheel keeps road grip.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from mv.envs.racing_env import EnvConfig, RacingEnv
from mv.rl.evaluate import eval_config
from mv.sim.vehicle_sim import VehicleSim
from mv.track.track import Track, TrackBatch
from mv.vehicle.params import G
from tests.test_core import car, circle_track, quiet_config  # noqa: F401  (fixture)

R, HALF = 300.0, 6.0          # a big circle: nearly straight road, 12 m wide


def _env(car, n=1, **kw) -> RacingEnv:  # noqa: F811
    cfg = quiet_config(self_recovery=True, offroad_grip=(0.5, 0.5), offroad_drag=(0.2, 0.2), runoff_m=(10.0, 10.0),
                       recovery_start_prob=0.0, **kw)
    env = RacingEnv(car, TrackBatch([circle_track(R, HALF)]), n, cfg, device="cpu")
    env.fixed_start = None                 # random draws switched off by the ranges above
    return env


def _place(env: RacingEnv, lat: float, head_err: float = 0.0, speed: float = 0.0) -> None:
    """Put car 0 at the start of the circle, ``lat`` metres left of the centre line (left = towards the
    centre of this anticlockwise circle), at ``head_err`` to the road, rolling forward at ``speed``."""
    sim = env.sim
    ids = torch.tensor([0])
    sim.reset(ids, torch.tensor([R - lat]), torch.tensor([0.0]), torch.tensor([math.pi / 2 + head_err]),
              torch.tensor([speed]))
    env.idx[0], env.along[0] = 0, 0.0
    env._locate()
    env.stall_t[0], env.wrong_t[0] = 0.0, 0.0


def test_wheels_past_the_edge_are_on_the_grass(car):  # noqa: F811
    env = _env(car)
    half_track = 0.5 * car.chassis.track_front_m
    _place(env, lat=HALF - 0.3)                    # left wheels (y = +track/2) are past the left edge
    env._update_surface()
    assert half_track > 0.3
    assert env.sim.surface_grip[0].tolist() == pytest.approx([0.5, 1.0, 0.5, 1.0])     # FL, FR, RL, RR
    assert env.sim.surface_drag[0].tolist() == pytest.approx([0.2, 0.0, 0.2, 0.0])
    assert int(env.wheels_off[0]) == 2
    # turned 90 degrees to the left, at the same place: now the FRONT wheels point over the edge
    # (front axle a metres ahead of the centre of mass -> a metres further left)
    _place(env, lat=HALF - 0.5 * car.chassis.a, head_err=math.pi / 2)
    env._update_surface()
    assert env.sim.surface_grip[0].tolist() == pytest.approx([0.5, 0.5, 1.0, 1.0])
    _place(env, lat=0.0)
    env._update_surface()
    assert int(env.wheels_off[0]) == 0 and torch.all(env.sim.surface_grip == 1.0)


def test_default_mode_keeps_road_grip_everywhere(car):  # noqa: F811
    env = RacingEnv(car, TrackBatch([circle_track(R, HALF)]), 1, quiet_config(), device="cpu")
    _place(env, lat=HALF + 0.5)                    # most of the car off the road
    env.step(torch.zeros(1, 2))
    assert torch.all(env.sim.surface_grip == 1.0) and torch.all(env.sim.surface_drag == 0.0)


def test_ploughing_and_low_grip_slow_the_car(car):  # noqa: F811
    """Coasting at 30 m/s: ploughing drag of 0.2 x load costs about 0.2 g = 1.96 m/s^2 more (a little more
    with downforce, which adds load).  Braking on half the grip stops the car about half as hard."""
    dt, steps = 0.008, 125                         # 1 s
    sim = VehicleSim(car, 4, "cpu", dt=dt)
    ids = torch.arange(4)
    sim.reset(ids, torch.zeros(4), torch.zeros(4), torch.zeros(4), torch.full((4,), 30.0))
    sim.surface_drag[1] = 0.2
    sim.surface_grip[3] = 0.5
    z = torch.zeros(4)
    brake = torch.tensor([0.0, 0.0, 1.0, 1.0])
    for _ in range(steps):
        sim.step(z, z, brake)
    v = sim.vx
    extra = float(v[0] - v[1])                     # m/s lost to ploughing in 1 s
    assert 1.8 < extra < 2.8
    decel_road, decel_grass = 30.0 - float(v[2]), 30.0 - float(v[3])
    assert 0.4 < decel_grass / decel_road < 0.65


def test_grass_is_not_the_end_but_the_barrier_is(car):  # noqa: F811
    env = _env(car)
    _place(env, lat=HALF + 4.0, speed=15.0)       # centre 4 m onto the 10 m run-off
    _, _, term, trunc, info = env.step(torch.zeros(1, 2))
    assert not bool(term[0]) and int(info["wheels_off"][0]) == 4
    assert int(env.excursions[0]) == 1
    _place(env, lat=HALF + 9.5, speed=15.0)       # body reaches past 10 m: the barrier
    _, _, term, _, info = env.step(torch.zeros(1, 2))
    assert bool(term[0]) and bool(info["off"][0])


def test_a_spun_car_gets_time_to_turn_around(car):  # noqa: F811
    env = _env(car, wrong_way_time_s=2.0)
    _place(env, lat=0.0, head_err=math.pi)         # facing backwards, standing
    ended_at = None
    for k in range(100):                           # 4 s
        _, _, term, _, info = env.step(torch.zeros(1, 2))
        if bool(term[0]):
            ended_at = (k + 1) * env.cfg.control_dt
            assert bool(info["spun"][0])
            break
    assert ended_at is not None and 2.0 < ended_at < 2.2


def test_practice_episodes_start_off_the_road(car):  # noqa: F811
    cfg = replace(EnvConfig(), self_recovery=True, recovery_start_prob=1.0, runoff_m=(6.0, 12.0))
    env = RacingEnv(car, TrackBatch([circle_track(R, HALF)]), 64, cfg, device="cpu", seed=3)
    lat = env.lat
    past = torch.where(lat >= 0, lat - env.w_left, -lat - env.w_right)          # centre past the edge, m
    assert torch.all(past > 0) and torch.all(past + env.half_width < env.runoff)
    assert env.head_err.abs().max() > 2.0         # some start (nearly) backwards


def test_own_map_shows_track_already_seen_beyond_the_sensors(car):  # noqa: F811
    """WP62: with a 50 m sensor range the preview stops at 50 m; once the sensors have seen further
    (map_reach) the remembered stretch appears, and after a full lap the whole preview is back."""
    cfg = quiet_config(own_map=True)
    env = RacingEnv(car, TrackBatch([circle_track(R, HALF)]), 1, cfg, device="cpu")
    env.set_view(50.0)
    lay, look = env.obs_layout, torch.tensor(cfg.lookahead_m)
    L = float(env.tracks.length[0])

    def preview():
        o = env.observe()[0]
        return (o[lay["kappa"]] != 0), float(o[lay["horizon"]][0])

    env.s_total[0], env.map_reach[0] = 100.0, 0.0       # nothing mapped: only the sensors
    known, hor = preview()
    assert torch.equal(known, look <= 50.0) and hor == pytest.approx(50.0 / 340.0)
    env.map_reach[0] = 100.0 + 200.0                    # sensors have seen up to 200 m ahead of here
    known, hor = preview()
    assert torch.equal(known, look <= 200.0) and hor == pytest.approx(190.0 / 340.0)
    env.s_total[0], env.map_reach[0] = L - 60.0, L - 10.0     # 10 m before a whole lap is seen: not yet
    known, _ = preview()
    assert torch.equal(known, look <= 50.0)
    env.s_total[0], env.map_reach[0] = L - 30.0, L + 20.0     # the sensors see past the start line: lap complete
    known, hor = preview()
    assert bool(known.all()) and hor == pytest.approx(1.0)
    # driving updates the map: one step from a fresh start maps the sensor range
    env.reset(torch.arange(1)); env.set_view(50.0)
    env.step(torch.zeros(1, 2))
    assert float(env.map_reach[0]) == pytest.approx(float(env.s_total[0]) + 50.0)


def test_strict_evaluation_switches_self_recovery_off():
    base = replace(EnvConfig(), self_recovery=True)
    assert not eval_config(base).self_recovery
    assert eval_config(base, strict=False).self_recovery
    assert not eval_config(EnvConfig(), strict=False).self_recovery
