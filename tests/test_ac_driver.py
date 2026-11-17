"""The Assetto Corsa driver adapter must reproduce training observations exactly.

No game needed: a simulated car plays the role of the game.  Its state is
packed into a telemetry-like object (the same fields the bridge delivers) and
fed to the adapter; the adapter's observation must equal the observation the
training environment itself produced for that state.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mv.ac.driver import ActuatorModel, ObservationAdapter
from mv.control.scripted import ScriptedDriver
from mv.envs.racing_env import RacingEnv
from mv.rl.evaluate import eval_config
from mv.sim.vehicle_sim import VehicleSim
from mv.track.generator import random_tracks
from mv.track.track import TrackBatch
from mv.vehicle.params import load_vehicle

ROOT = Path(__file__).resolve().parents[1]
CARS = sorted((ROOT / "vehicles" / "library").glob("*.yaml"))


def fake_telemetry(env: RacingEnv):
    """What the bridge would report if the simulated car were the game car."""
    s = env.sim
    return SimpleNamespace(
        position=np.array([float(s.x[0]), float(s.y[0]), 0.0]), yaw=float(s.yaw[0]),
        velocity_local=np.array([float(s.vx[0]), float(s.vy[0]), 0.0]), yaw_rate=float(s.r[0]),
        acceleration_local=np.array([float(s.ax[0]), float(s.ay[0]), 0.0]),
        steer=float(s.steer[0]) / s.steer_max, throttle=float(s.thr[0]), brake=float(s.brk[0]),
        gear=int(s.gear[0, 0]) + 1, rpm=float(s.rpm_frac[0]) * float(s.u_top[0]), max_rpm=float(s.u_top[0]),
        kers_charge=float(s.soc[0]), speed=float(torch.sqrt(s.vx[0] ** 2 + s.vy[0] ** 2)), sim_time=float(env.t_ep[0]),
    )


@pytest.mark.parametrize("path", CARS, ids=lambda p: p.stem)
def test_adapter_reproduces_training_observation(path):
    vp = load_vehicle(path)
    track = random_tracks(1, seed=21, cfg="mixed")[0]
    cfg = eval_config()
    truth = RacingEnv(vp, TrackBatch([track]), 1, cfg, device="cpu")
    truth.fixed_start = (int(track.n * 0.37), 25.0)          # start somewhere in the middle of the lap
    truth.reset(torch.arange(1))
    adapter = ObservationAdapter(vp, track, cfg, accel_filter_s=0.0)
    driver = ScriptedDriver(truth)
    obs = truth.observe()
    worst = 0.0
    for _ in range(250):                                     # 10 s of driving
        action, _, _ = driver.act(obs)
        obs, _, term, trunc, _ = truth.step(action)
        assert not (term | trunc).any(), "reference run should not end"
        got = adapter.update(fake_telemetry(truth), prev_action=tuple(action[0].clamp(-1, 1).tolist()))
        worst = max(worst, float((got - obs).abs().max()))
    assert worst < 2e-3, f"largest observation mismatch {worst:.2e}"
    assert adapter.lateral_offset == pytest.approx(float(truth.lat[0]), abs=1e-3)


def test_adapter_relocates_after_teleport():
    vp = load_vehicle(CARS[0])
    track = random_tracks(1, seed=4, cfg="mixed")[0]
    adapter = ObservationAdapter(vp, track)
    env = adapter.env

    def at(index: int):
        p, psi = track.xy[index], track.psi[index]
        return SimpleNamespace(position=np.array([p[0], p[1], 0.0]), yaw=float(psi),
                               velocity_local=np.array([20.0, 0.0, 0.0]), yaw_rate=0.0,
                               acceleration_local=np.zeros(3), steer=0.0, throttle=0.3, brake=0.0, gear=3,
                               rpm=5000.0, max_rpm=9000.0, kers_charge=0.0, speed=20.0, sim_time=0.0)

    adapter.update(at(10))
    assert abs(int(env.idx[0]) - 10) <= 1
    far = track.n // 2
    adapter.update(at(far))                                   # jump half a lap away
    assert abs(int(env.idx[0]) - far) <= 1
    assert abs(adapter.lateral_offset) < 0.2 and abs(adapter.heading_error) < 0.05


def test_actuator_model_matches_simulator():
    """Commands sent to the game must lag exactly like the simulated actuators."""
    vp = load_vehicle(CARS[0])
    cfg = eval_config()
    sim = VehicleSim(vp, 1, "cpu", dt=cfg.control_dt / cfg.substeps)
    sim.reset(torch.arange(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.tensor([30.0]))
    act = ActuatorModel(vp, cfg.control_dt)
    rng = np.random.default_rng(0)
    for _ in range(60):
        a = rng.uniform(-1, 1, 2)
        steer, thr, brk = act(a)
        for _ in range(cfg.substeps):
            sim.step(torch.tensor([a[0]], dtype=torch.float32), torch.tensor([max(a[1], 0)], dtype=torch.float32),
                     torch.tensor([max(-a[1], 0)], dtype=torch.float32))
        assert steer == pytest.approx(float(sim.steer[0]) / sim.steer_max, abs=1e-4)
        assert thr == pytest.approx(float(sim.thr[0]), abs=1e-4)
        assert brk == pytest.approx(float(sim.brk[0]), abs=1e-4)
