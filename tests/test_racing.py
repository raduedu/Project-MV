"""Race bot stage 4 (engineering log WP64): cars of the same race see and touch each other.

Hand-checkable numbers:
* a rear-end hit between two equal cars keeps the total momentum and leaves them separating at
  restitution x the closing speed; only the car behind is "at fault";
* the opponents input shows a car 50 m ahead at +1.0 and hides one 50 m behind (mirrors see 30 m);
* restarts are placed near a group mate on the same circuit.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from mv.envs.racing_env import EnvConfig, RacingEnv
from mv.track.track import TrackBatch
from tests.test_core import car, circle_track, quiet_config  # noqa: F401  (fixture)

R, HALF = 400.0, 7.0


def _pair(car, **kw) -> RacingEnv:  # noqa: F811
    cfg = quiet_config(race_group=2, opponents=2, **kw)
    return RacingEnv(car, TrackBatch([circle_track(R, HALF)]), 2, cfg, device="cpu")


def _put(env: RacingEnv, i: int, s: float, lat: float, speed: float) -> None:
    """Car i at s metres along the anticlockwise circle (start at (R, 0)), lat metres left, at speed."""
    a = s / R
    rr = R - lat
    env.sim.reset(torch.tensor([i]), torch.tensor([rr * math.cos(a)]), torch.tensor([rr * math.sin(a)]),
                  torch.tensor([a + math.pi / 2]), torch.tensor([speed]))
    env.idx[i], env.along[i] = int(round(s / env.tracks.ds)) % int(env.tracks.n[0]), 0.0


def test_rear_end_keeps_momentum_and_blames_the_car_behind(car):  # noqa: F811
    env = _pair(car, contact_restitution=0.3)
    gap = car.chassis.length_m - 0.3                    # bodies overlap by 0.3 m, nose to tail
    _put(env, 0, 100.0, 0.0, 30.0)                       # behind, fast
    _put(env, 1, 100.0 + gap, 0.0, 10.0)                 # ahead, slow
    env._locate()
    sim = env.sim
    p_before = float(sim.vx.sum())                       # equal masses: momentum ~ sum of speeds
    closing, at_fault = env._collide()
    assert float(closing[0]) == pytest.approx(20.0, rel=0.02) and float(closing[1]) == pytest.approx(20.0, rel=0.02)
    assert float(at_fault[0]) == pytest.approx(20.0, rel=0.02) and float(at_fault[1]) == 0.0
    assert float(sim.vx.sum()) == pytest.approx(p_before, rel=1e-3)
    # after the hit they move apart at 0.3 x 20 m/s
    assert float(sim.vx[1] - sim.vx[0]) == pytest.approx(0.3 * 20.0, rel=0.03)
    # pushed apart: no overlap left along the line of the hit (the arc bends it very slightly)
    d = math.hypot(float(sim.x[1] - sim.x[0]), float(sim.y[1] - sim.y[0]))
    assert d >= car.chassis.length_m - 0.02


def test_cars_apart_do_not_touch(car):  # noqa: F811
    env = _pair(car)
    _put(env, 0, 100.0, -2.5, 30.0)
    _put(env, 1, 101.0, 2.5, 30.0)                       # side by side, 5 m apart
    env._locate()
    closing, _ = env._collide()
    assert float(closing.abs().sum()) == 0.0


def test_opponents_seen_within_sensor_range(car):  # noqa: F811
    env = _pair(car)
    _put(env, 0, 100.0, 0.0, 30.0)
    _put(env, 1, 150.0, 2.0, 35.0)
    env._locate()
    env.set_view(120.0)
    opp = env.observe()[:, env.obs_layout["opponents"]].view(2, 2, 5)
    seen, ahead, left, faster, _ = opp[0, 0].tolist()
    assert seen == 1.0 and ahead == pytest.approx(1.0, abs=0.03) and left == pytest.approx(0.4, abs=0.02)
    assert faster == pytest.approx(0.5, abs=0.03)
    assert torch.all(opp[0, 1] == 0)                     # only one other car: second slot empty
    assert torch.all(opp[1] == 0)                        # car 1 sees car 0 50 m behind: beyond the 30 m mirrors
    env.set_view(40.0)                                   # sensors shorter than the gap: not seen either
    opp = env.observe()[:, env.obs_layout["opponents"]].view(2, 2, 5)
    assert torch.all(opp[0] == 0)


def test_restarts_join_a_group_mate(car):  # noqa: F811
    cfg = replace(EnvConfig(), race_group=4, opponents=3, spawn_near_prob=1.0)
    tracks = TrackBatch([circle_track(R, HALF), circle_track(300.0, HALF)])
    env = RacingEnv(car, tracks, 64, cfg, device="cpu", seed=2)
    for _ in range(10):
        env.step(torch.zeros(64, 2))
    ids = torch.tensor([1, 6, 11, 13, 30, 47])           # not the first of their race: they follow its circuit
    mates_tid = env.tid.view(16, 4)[ids // 4]
    env.reset(ids)
    assert torch.all(env.tid[ids] == env.group_tid[ids // 4])
    gap, same = env._gap_ahead()
    g, i = ids // 4, ids % 4
    nearest = torch.where(same[g, i], gap[g, i].abs(), torch.full_like(gap[g, i], 1e9)).amin(1)
    on_same = (mates_tid == env.tid[ids][:, None]).sum(1) > 1
    assert torch.all(nearest[on_same] <= 62.0)


def test_single_cars_unchanged_by_racing_code(car):  # noqa: F811
    """race_group=1 (the default) must not change a single number of the old environment."""
    tracks = TrackBatch([circle_track(R, HALF)])
    a = RacingEnv(car, tracks, 8, EnvConfig(), device="cpu", seed=4)
    b = RacingEnv(car, tracks, 8, replace(EnvConfig(), contact_penalty=7.0), device="cpu", seed=4)
    for _ in range(30):
        act = torch.full((8, 2), 0.3)
        oa, ra, *_ = a.step(act)
        ob, rb, *_ = b.step(act)
        assert torch.equal(oa, ob) and torch.equal(ra, rb)
