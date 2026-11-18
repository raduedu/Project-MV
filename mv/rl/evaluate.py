"""Evaluate a trained driver: timed laps on a fixed set of tracks.

One car per track, standing start on the start line, no randomisation of the
car, full track preview, no exploration noise.  Lap 1 is therefore a
standing-start lap and lap 2 onward are flying laps.  The flying lap time is
the number to compare between drivers (and with the physics reference in
``mv/planning``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from mv.envs.racing_env import EnvConfig, RacingEnv
from mv.track.track import Track, TrackBatch
from mv.vehicle.params import VehicleParams

__all__ = ["LapResult", "evaluate", "eval_config"]


@dataclass
class LapResult:
    track: str
    length_m: float
    lap_times: list[float]            # completed laps, in order (first = standing start)
    finished: bool                    # completed the requested number of laps
    end_reason: str                   # "laps", "off", "spun", "stalled", "slow", "timeout"
    distance_m: float                 # progress made before the run ended
    max_speed: float
    off_track_steps: int              # decisions with any part of the car over the edge
    recoveries: int = 0               # limit-seeking mode: times the car went off and was put back on
    trace: dict = field(default_factory=dict, repr=False)   # t, x, y, speed, steer, pedal, lat (numpy)

    @property
    def flying_lap(self) -> float:
        return min(self.lap_times[1:]) if len(self.lap_times) > 1 else float("nan")


def eval_config(base: EnvConfig | None = None, max_time_s: float = 900.0) -> EnvConfig:
    """The training config with every source of randomness switched off - and strict.

    Strict means: leaving the track ends the run, even for drivers trained in
    limit-seeking mode (where training lets the car rejoin).  So "completed" and
    every recorded lap time always mean a CLEAN lap (engineering log WP25).
    """
    one = (1.0, 1.0)
    return replace(base or EnvConfig(), blind_prob=0.0, rand_grip=one, rand_mass=one, rand_power=one,
                   rand_aero=one, max_episode_s=max_time_s, standing_start_prob=0.0, recover_on_crash=False)


@torch.no_grad()
def evaluate(net, vp: VehicleParams, tracks: list[Track], device, laps: int = 2, cfg: EnvConfig | None = None,
             record: bool = False, blind: bool = False, grip: float = 1.0,
             min_avg_speed: float = 8.0) -> list[LapResult]:
    """Drive ``laps`` laps on each track with the deterministic policy.

    A run is abandoned as "slow" once its average speed since the start is below
    ``min_avg_speed`` (m/s) after the first 20 s; an untrained driver would
    otherwise crawl around for the full time limit.

    ``blind=True`` shortens the track preview to 40 m to mimic a first lap on an
    unknown track.  ``grip`` scales tyre grip (e.g. 0.9 for a dusty track).
    """
    cfg = eval_config(cfg)
    # A handful of cars is much faster on the CPU than on the GPU (the GPU has a
    # fixed cost per operation that only pays off with thousands of cars).
    device = torch.device(device)
    if isinstance(net, torch.nn.Module) and next(net.parameters()).device != device:
        net = copy.deepcopy(net).to(device)
    threads = torch.get_num_threads()
    if device.type == "cpu":
        torch.set_num_threads(1)          # tiny tensors: threading overhead costs more than it gains
    try:
        return _run(net, vp, tracks, device, laps, cfg, record, blind, grip, min_avg_speed)
    finally:
        torch.set_num_threads(threads)


def _run(net, vp, tracks, device, laps, cfg, record, blind, grip, min_avg_speed) -> list[LapResult]:
    batch = TrackBatch(tracks, device=device, ds=cfg_ds(tracks))
    env = RacingEnv(vp, batch, len(tracks), cfg, device=device)
    env.fixed_start = (0, 0.0)
    env.reset(torch.arange(env.n, device=env.device))
    env.sim.grip[:] = grip
    if blind:
        env.set_view(40.0)
    n = env.n
    active = torch.ones(n, dtype=torch.bool, device=env.device)
    lap_times: list[list[float]] = [[] for _ in range(n)]
    reason = ["timeout"] * n
    dist = torch.zeros(n, device=env.device)
    vmax = torch.zeros(n, device=env.device)
    off_steps = torch.zeros(n, dtype=torch.long, device=env.device)
    # Full telemetry for lap replays (mv/laps.py): what the car did and what it actually
    # received (steering/throttle/brake after the actuators), 25 times per second.
    fields = ("t", "s", "x", "y", "yaw", "speed", "steer", "thr", "brk", "gear", "pedal", "lat")
    traces: list[dict[str, list]] = [{f: [] for f in fields} for _ in range(n)]
    obs = env.observe()
    max_steps = int(cfg.max_episode_s / cfg.control_dt)
    for _step in range(max_steps):
        action = _act_per_car(net, obs)
        # read state that step() would overwrite when an episode ends
        obs, _rew, term, trunc, info = env.step(action)
        done = term | trunc
        live = active & ~done
        dist = torch.where(live, env.s_total, dist)
        vmax = torch.where(live, torch.maximum(vmax, info["speed"]), vmax)
        over = torch.maximum(env.lat - env.w_left, -env.lat - env.w_right) + env.half_width
        off_steps += (live & (over > 0)).long()
        if record:
            sim = env.sim
            snap = {"t": env.t_ep, "s": env.s_total, "x": sim.x, "y": sim.y, "yaw": sim.yaw, "speed": info["speed"],
                    "steer": sim.steer / sim.steer_max, "thr": sim.thr, "brk": sim.brk, "gear": sim.gear[:, 0] + 1,
                    "pedal": action.clamp(-1, 1)[:, 1], "lat": env.lat}
            snap = {k: v.detach().cpu().numpy() for k, v in snap.items()}
            for i in live.nonzero().flatten().tolist():
                tr = traces[i]
                for f in fields:
                    tr[f].append(float(snap[f][i]))
        for i in (active & info["lap_done"] & ~done).nonzero().flatten().tolist():
            lap_times[i].append(float(info["lap_time"][i]))
            if len(lap_times[i]) >= laps:
                active[i] = False
                reason[i] = "laps"
        for i in (active & done).nonzero().flatten().tolist():
            active[i] = False
            reason[i] = "off" if info["off"][i] else "spun" if info["spun"][i] else "stalled" if info["stalled"][i] else "timeout"
        slow = active & (env.t_ep > 20.0) & (env.s_total < min_avg_speed * env.t_ep)
        for i in slow.nonzero().flatten().tolist():
            active[i] = False
            reason[i] = "slow"
        if blind:
            env.set_view(40.0)
        env.sim.grip[:] = grip
        if not active.any():
            break
    out = []
    for i, t in enumerate(batch.tracks):
        out.append(LapResult(track=t.name, length_m=t.length, lap_times=lap_times[i], finished=reason[i] == "laps",
                             end_reason=reason[i], distance_m=float(dist[i]), max_speed=float(vmax[i]),
                             off_track_steps=int(off_steps[i]), recoveries=int(env.recoveries[i]),
                             trace={k: np.asarray(v) for k, v in traces[i].items()} if record else {}))
    return out


def _act_per_car(net, obs: torch.Tensor) -> torch.Tensor:
    """The driver's actions, computed for each car on its own.

    Why: the maths library may round a matrix product very slightly differently
    depending on how many rows (cars) are computed together.  Over a two-minute
    lap such differences grow (the driving is chaotic), so the same driver could
    post a lap 0.1 s different just because other tracks were evaluated in the
    same batch.  Computing every car separately makes a lap time independent of
    what else was evaluated - so any lap can be reproduced exactly
    (scripts/reproduce_lap.py; engineering log WP26).
    """
    if not isinstance(net, torch.nn.Module):          # rule-based driver: element-wise maths, already independent
        return net.act(obs, deterministic=True)[0]
    return torch.cat([net.act(obs[i:i + 1], deterministic=True)[0] for i in range(obs.shape[0])], 0)


def cfg_ds(tracks: list[Track]) -> float:
    """Common sample spacing for a set of tracks (the finest one)."""
    return float(min(t.ds for t in tracks))
