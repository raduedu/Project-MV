"""WP63: a step replayed as a CUDA graph gives bit-for-bit the same results as a normal step.

Runs only where a CUDA GPU is available.  Two identical environments (same seed) are driven with the
same random actions through many episode ends (short episodes, cars put back, practice starts off the
road); every observation, reward and end flag must match exactly.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mv.envs.multi_car import MultiCarEnv
from mv.envs.racing_env import EnvConfig, RacingEnv
from mv.track.generator import random_tracks
from mv.track.track import TrackBatch
from tests.test_core import LIBRARY, car  # noqa: F401  (fixture)
from mv.vehicle.params import load_vehicle

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _same_run(make, steps: int = 150):
    a, b = make(), make()
    b.use_cuda_graph()
    gen = torch.Generator(device="cuda").manual_seed(5)
    ends = 0
    for _ in range(steps):
        act = torch.rand(a.n, 2, device="cuda", generator=gen) * 2 - 1
        oa, ra, ta, ua, ia = a.step(act)
        ob, rb, tb, ub, ib = b.step(act)
        assert torch.equal(torch.nan_to_num(oa), torch.nan_to_num(ob))
        assert torch.equal(ra, rb) and torch.equal(ta, tb) and torch.equal(ua, ub)
        assert torch.equal(ia["speed"], ib["speed"])
        ends += int((ta | ua).sum())
    return ends


@cuda
@pytest.mark.parametrize("mode", ["self_recovery", "limit_seeking", "racing"])
def test_graph_step_matches_normal_step(car, mode):  # noqa: F811
    tracks = TrackBatch(random_tracks(3, seed=4, cfg="mixed"), device="cuda", ds=1.0)
    cfg = replace(EnvConfig(), history_steps=4, lap_memory=True, max_episode_s=1.5, blind_prob=0.5,
                  blind_horizon_m=(40.0, 150.0), steer_filter_s=0.03)
    if mode == "self_recovery":
        cfg = replace(cfg, self_recovery=True, own_map=True, recovery_start_prob=0.3)
    elif mode == "racing":                 # groups of 4 sharing a circuit, contacts, opponents input
        cfg = replace(cfg, self_recovery=True, race_group=4, opponents=3, spawn_gap_m=(3.0, 10.0))
    else:
        cfg = replace(cfg, recover_on_crash=True, max_recoveries=2)
    ends = _same_run(lambda: RacingEnv(car, tracks, 96, cfg, device="cuda", seed=7))
    assert ends > 50                    # the comparison really went through many restarts


@cuda
def test_graph_step_matches_for_several_cars():
    vps = [load_vehicle(p) for p in LIBRARY[:2]]
    tracks = TrackBatch(random_tracks(2, seed=6, cfg="mixed"), device="cuda", ds=1.0)
    cfg = replace(EnvConfig(), max_episode_s=1.0, vary_car=True)
    _same_run(lambda: MultiCarEnv(vps, tracks, 64, cfg, device="cuda", seed=2), steps=60)
