"""Train one driver on several different cars at once.

:class:`mv.envs.racing_env.RacingEnv` simulates many copies of *one* car.
This wrapper runs several of them side by side - one per car model - and makes
them look like a single big environment to the learning algorithm.  The
network therefore sees, in every batch, a combustion car, a hybrid and an EV
(or whatever list it is given) and must drive them all.  The "car descriptor"
block of the observation tells it which car it is in.

Why this matters: it moves "works with any car" from the *code* (already true:
a car is a parameter file) to the *trained driver* itself (roadmap phase 4).

All cars share the same pool of tracks.  The fleet is split evenly between the
car models.
"""

from __future__ import annotations

import torch

from mv.envs.racing_env import EnvConfig, RacingEnv
from mv.track.track import TrackBatch
from mv.vehicle.params import VehicleParams

__all__ = ["MultiCarEnv"]


class MultiCarEnv:
    def __init__(self, vps: list[VehicleParams], tracks: TrackBatch, num_envs: int, cfg: EnvConfig | None = None,
                 device: torch.device | str = "cpu", seed: int = 0):
        assert vps, "need at least one car"
        per_car = num_envs // len(vps)
        self.envs = [RacingEnv(vp, tracks, per_car, cfg, device=device, seed=seed + 1000 * i)
                     for i, vp in enumerate(vps)]
        first = self.envs[0]
        self.vps, self.tracks, self.cfg, self.device = vps, tracks, first.cfg, first.device
        self.n = per_car * len(vps)
        self.num_obs, self.num_actions, self.obs_layout = first.num_obs, first.num_actions, first.obs_layout
        assert all(e.num_obs == self.num_obs for e in self.envs)
        self.per_car = per_car

    # the three methods the learner uses -------------------------------------
    def observe(self) -> torch.Tensor:
        return torch.cat([e.observe() for e in self.envs], 0)

    def step(self, action: torch.Tensor):
        obs, rew, term, trunc = [], [], [], []
        info: dict = {"speed": [], "progress": []}
        final_obs, done_ids, ep_return, ep_len, ep_dist = [], [], [], [], []
        for i, env in enumerate(self.envs):
            lo = i * self.per_car
            o, r, te, tr, inf = env.step(action[lo:lo + self.per_car])
            obs.append(o); rew.append(r); term.append(te); trunc.append(tr)
            info["speed"].append(inf["speed"]); info["progress"].append(inf["progress"])
            if "final_obs" in inf:
                final_obs.append(inf["final_obs"]); done_ids.append(inf["done_ids"] + lo)
                ep_return.append(inf["ep_return"]); ep_len.append(inf["ep_len"]); ep_dist.append(inf["ep_dist"])
        info["speed"], info["progress"] = torch.cat(info["speed"]), torch.cat(info["progress"])
        if final_obs:
            info.update(final_obs=torch.cat(final_obs), done_ids=torch.cat(done_ids), ep_return=torch.cat(ep_return),
                        ep_len=torch.cat(ep_len), ep_dist=torch.cat(ep_dist))
        return torch.cat(obs), torch.cat(rew), torch.cat(term), torch.cat(trunc), info

    def use_cuda_graph(self, on: bool = True) -> None:
        for env in self.envs:
            env.use_cuda_graph(on)

    def tracks_replaced(self, track_ids: torch.Tensor) -> None:
        for env in self.envs:
            env.tracks_replaced(track_ids)

    # a teacher for the warm start ----------------------------------------------
    def scripted_teacher(self):
        """One rule-based driver per car model, presented as a single driver."""
        from mv.control.scripted import ScriptedDriver
        teachers = [ScriptedDriver(e) for e in self.envs]
        per_car = self.per_car

        class _Teacher:
            def act(self, obs: torch.Tensor, deterministic: bool = True):
                acts = [t.act(obs[i * per_car:(i + 1) * per_car])[0] for i, t in enumerate(teachers)]
                zeros = torch.zeros(obs.shape[0], device=obs.device)
                return torch.cat(acts, 0), zeros, zeros

        return _Teacher()
