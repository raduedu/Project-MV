"""Make recordings from the simulator itself.

Used to *test* system identification: record a car whose parameters we changed
on purpose, then check that the fit finds exactly those changes.  Also a quick
way to see what a recording looks like before driving in the game.
"""

from __future__ import annotations

import torch

from mv.control.scripted import ScriptedDriver
from mv.envs.racing_env import RacingEnv
from mv.rl.evaluate import eval_config
from mv.sysid.replay import Recording
from mv.track.track import Track, TrackBatch
from mv.vehicle.params import VehicleParams

__all__ = ["record_in_simulator"]


@torch.no_grad()
def record_in_simulator(vp: VehicleParams, track: Track, seconds: float = 60.0, grip_use: float = 0.75,
                        start_index: int = 0) -> Recording:
    """Let the rule-based driver lap ``track`` in car ``vp`` and record it like the game bridge would."""
    cfg = eval_config(max_time_s=seconds + 5)
    env = RacingEnv(vp, TrackBatch([track], ds=track.ds), 1, cfg, device="cpu")
    env.fixed_start = (start_index, 0.0)
    env.reset(torch.arange(1))
    driver = ScriptedDriver(env, grip_use=grip_use, brake_use=0.6)
    sim, rows = env.sim, []
    obs = env.observe()
    for k in range(int(seconds / cfg.control_dt)):
        action, _, _ = driver.act(obs)
        obs, _, term, trunc, _ = env.step(action)
        if bool(term[0] | trunc[0]):
            break
        rows.append(dict(t=(k + 1) * cfg.control_dt, vx=float(sim.vx[0]), vy=float(sim.vy[0]), r=float(sim.r[0]),
                         ax=float(sim.ax[0]), ay=float(sim.ay[0]), steer=float(sim.steer[0]) / sim.steer_max,
                         throttle=float(sim.thr[0]), brake=float(sim.brk[0]), gear=int(sim.gear[0, 0]) + 1))
    return Recording.from_rows(rows, meta={"source": "simulator", "car": vp.name, "track": track.name})
