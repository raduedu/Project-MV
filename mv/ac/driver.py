"""Let a Project MV driver drive the real Assetto Corsa car.

The learned driver only understands the observation vector of
:class:`mv.envs.racing_env.RacingEnv`.  To drive in the game we must build
exactly that vector from live game telemetry.  Re-implementing the observation
maths here would invite subtle mismatches, so instead we keep a *shadow* copy
of the training environment with a single car, overwrite that car's state with
what the game reports every tick, and ask the shadow environment for its
observation.  The observation code that runs in the game is therefore literally
the code that ran in training.

::

    Assetto Corsa --telemetry--> ObservationAdapter (shadow RacingEnv) --obs--> driver
          ^                                                                        |
          +---- steering / throttle / brake <---- actuator model <---- action ------+

The same class works for the neural network and for the rule-based
:class:`mv.control.scripted.ScriptedDriver`, which makes the scripted driver a
network-free end-to-end test of the whole chain in the game.

Telemetry is duck-typed: anything with the attributes used in
:meth:`ObservationAdapter.update` works (the bridge's ``Telemetry``, or a test
stand-in).  All quantities must be SI in the project's frames (see
docs/07-assetto-corsa-bridge.md).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from mv.envs.racing_env import EnvConfig, RacingEnv
from mv.rl.evaluate import eval_config
from mv.track.track import Track, TrackBatch
from mv.vehicle.params import VehicleParams

__all__ = ["ObservationAdapter", "ActuatorModel", "DriveResult", "drive"]


class ObservationAdapter:
    """Builds training-identical observations from live telemetry."""

    def __init__(self, vp: VehicleParams, track: Track, cfg: EnvConfig | None = None,
                 accel_filter_s: float = 0.06, blind: bool = False):
        self.cfg = eval_config(cfg)
        self.env = RacingEnv(vp, TrackBatch([track], ds=track.ds), 1, self.cfg, device="cpu")
        self.env.fixed_start = (0, 0.0)
        self.track = track
        self.accel_filter_s = accel_filter_s
        self.blind = blind
        self._located = False
        self._xy = torch.as_tensor(track.xy, dtype=torch.float32)
        self._last_time: float | None = None

    @property
    def num_obs(self) -> int:
        return self.env.num_obs

    def reset(self) -> None:
        """Forget where the car was (call after a teleport / session restart)."""
        self._located = False
        self._last_time = None
        self.env.prev_action[:] = 0.0
        self.env.sim.ax[:] = 0.0
        self.env.sim.ay[:] = 0.0

    def update(self, tel, prev_action: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
        """Copy telemetry into the shadow car and return the observation, shape (1, num_obs)."""
        env, sim = self.env, self.env.sim
        x, y = float(tel.position[0]), float(tel.position[1])
        sim.x[0], sim.y[0], sim.yaw[0] = x, y, float(tel.yaw)
        sim.vx[0], sim.vy[0] = float(tel.velocity_local[0]), float(tel.velocity_local[1])
        sim.r[0] = float(tel.yaw_rate)
        # body accelerations, smoothed like the simulator's load-transfer filter
        now = float(getattr(tel, "sim_time", 0.0))
        dt = self.cfg.control_dt if self._last_time is None else max(now - self._last_time, 1e-3)
        self._last_time = now
        k = 1.0 if self.accel_filter_s <= 0 else 1.0 - math.exp(-dt / self.accel_filter_s)
        sim.ax[0] += (float(tel.acceleration_local[0]) - float(sim.ax[0])) * k
        sim.ay[0] += (float(tel.acceleration_local[1]) - float(sim.ay[0])) * k
        sim.steer[0] = float(np.clip(tel.steer, -1.0, 1.0)) * sim.steer_max
        sim.thr[0], sim.brk[0] = float(tel.throttle), float(tel.brake)
        n_gears = int(sim.gb_ngears[0])
        sim.gear[0, 0] = int(np.clip(int(tel.gear) - 1, 0, n_gears - 1))      # game: 1 = first gear
        rpm = max(float(tel.rpm), float(sim.u_launch[0]))
        sim.rpm_frac[0] = min(rpm / float(sim.u_top[0]), 1.0)
        sim.soc[0] = float(tel.kers_charge) if sim.has_motor else 1.0
        sim.slip = sim.wheel_slip_angles()

        # ---- where on the track is the car? -----------------------------------------
        if self._located:
            near = self._xy[int(env.idx[0])]
            if (near[0] - x) ** 2 + (near[1] - y) ** 2 > 30.0 ** 2:           # teleported
                self._located = False
        fresh = not self._located
        if fresh:
            d2 = (self._xy[:, 0] - x) ** 2 + (self._xy[:, 1] - y) ** 2
            env.idx[0] = int(d2.argmin())
            env.along[0] = 0.0
            self._located = True
        progress = env._locate()
        if fresh:
            progress = torch.zeros_like(progress)
        env.t_ep[0] = now
        env.advance_laps(progress)
        # lap memory (if the driver was trained with one): this section's speed and grip use;
        # "off" = whole car beyond the edge, as in training
        over = torch.maximum(env.lat - env.w_left, -env.lat - env.w_right) + env.half_width
        env.update_lap_memory(over > env.half_width + env.cfg.off_track_margin_m)
        env.set_view(40.0 if self.blind else None)
        env.prev_action[0, 0], env.prev_action[0, 1] = float(prev_action[0]), float(prev_action[1])
        # memory (if the driver was trained with one): one sample per decision, as in training
        if fresh:
            env.reset_history(torch.arange(1))
        else:
            env.push_history()
        return env.observe()

    # handy read-outs for diagnostics
    @property
    def lateral_offset(self) -> float:
        return float(self.env.lat[0])

    @property
    def heading_error(self) -> float:
        return float(self.env.head_err[0])

    @property
    def track_position_m(self) -> float:
        return float(self.env.idx[0]) * self.track.ds

    def body_over_edge_m(self) -> float:
        env = self.env
        return float(torch.maximum(env.lat - env.w_left, -env.lat - env.w_right)[0]) + env.half_width


class ActuatorModel:
    """Turns driver actions into the commands sent to the game.

    The simulator the driver trained in has a steering-rate limit and a small
    lag on the pedals.  The game applies whatever it is sent instantly, so the
    same limits are applied here; otherwise the car would react faster than the
    driver has ever experienced.
    """

    def __init__(self, vp: VehicleParams, dt: float, steer_filter_s: float = 0.0):
        self.rate = vp.steering.max_rate_deg_s / vp.steering.max_wheel_angle_deg * dt   # lock fractions per tick
        self.k_thr = 1.0 - math.exp(-dt / max(vp.powertrain.throttle_response_time_s, 1e-3))
        self.k_brk = 1.0 - math.exp(-dt / max(vp.brakes.response_time_s, 1e-3))
        self.steer = self.thr = self.brk = 0.0
        self.k_filter = 1.0 - math.exp(-dt / steer_filter_s) if steer_filter_s > 0 else 1.0
        self.cmd = 0.0

    def sync(self, steer: float, throttle: float, brake: float) -> None:
        self.steer, self.thr, self.brk = float(steer), float(throttle), float(brake)
        self.cmd = float(steer)

    def __call__(self, action) -> tuple[float, float, float]:
        target = float(np.clip(action[0], -1, 1))
        if self.k_filter < 1.0:                      # same steering damper as in training (EnvConfig.steer_filter_s)
            self.cmd += (target - self.cmd) * self.k_filter
            target = self.cmd
        pedal = float(np.clip(action[1], -1, 1))
        self.steer += float(np.clip(target - self.steer, -self.rate, self.rate))
        self.thr += (max(pedal, 0.0) - self.thr) * self.k_thr
        self.brk += (max(-pedal, 0.0) - self.brk) * self.k_brk
        return self.steer, self.thr, self.brk


@dataclass
class DriveResult:
    laps: list[float]
    reason: str
    distance_m: float
    max_speed: float
    off_track_ticks: int
    ticks: int


def drive(raw_env, driver, adapter: ObservationAdapter, vp: VehicleParams, laps: int = 1,
          max_time_s: float = 600.0, max_speed: float | None = None, dry_run: bool = False,
          log_every_s: float = 1.0, printer=print, record: list | None = None) -> DriveResult:
    """Drive the game car with ``driver`` (anything with ``act(obs, deterministic=True)``).

    ``raw_env`` is a :class:`mv.ac.bridge.ACRawEnv`.  With ``dry_run=True`` no
    controls are sent: drive yourself and watch the printed track position,
    lateral offset and the action the driver *would* take - the safest first
    check that telemetry, track map and observation line up.

    ``max_speed`` (m/s) is a safety governor for first tests: above it the
    throttle is cut and the brakes are applied lightly.
    """
    dt = adapter.cfg.control_dt
    act = ActuatorModel(vp, dt, adapter.cfg.steer_filter_s)
    tel = raw_env.session.observe(timeout_s=5.0)
    act.sync(tel.steer, tel.throttle, tel.brake)
    adapter.reset()
    prev = (0.0, 0.0)
    lap0 = tel.lap_count
    lap_times: list[float] = []
    start_idx = None
    dist, vmax, off, ticks, reason = 0.0, 0.0, 0, 0, "time"
    next_log = 0.0
    next_tick = None
    ax_f = ay_f = None
    try:
        while ticks * dt < max_time_s:
            obs = adapter.update(tel, prev)
            action = driver.act(obs, deterministic=True)[0][0].clamp(-1, 1).tolist()
            if start_idx is None:
                start_idx = adapter.track_position_m
            steer, thr, brk = act(action)
            if max_speed is not None and tel.speed > max_speed:
                thr, brk = 0.0, max(brk, 0.15)
            if dry_run:
                # observe() returns at the next physics packet (~3 ms in AC); pace the loop
                # to the driver's decision rate ourselves, as step() does when driving
                act.sync(tel.steer, tel.throttle, tel.brake)
                next_tick = (next_tick or time.perf_counter()) + dt
                time.sleep(max(0.0, next_tick - time.perf_counter()))
                tel = raw_env.session.observe(timeout_s=2.0)
            else:
                tel = raw_env.step(steer, thr, brk)
            if record is not None:
                # what the car did and what it actually received, for system identification
                # (mv/sysid).  Accelerations get the same 60 ms smoothing as the simulator's.
                k = 1.0 - math.exp(-dt / 0.06)
                ax_f = (ax_f if ax_f is not None else float(tel.acceleration_local[0]))
                ay_f = (ay_f if ay_f is not None else float(tel.acceleration_local[1]))
                ax_f += (float(tel.acceleration_local[0]) - ax_f) * k
                ay_f += (float(tel.acceleration_local[1]) - ay_f) * k
                record.append(dict(t=ticks * dt + dt, vx=float(tel.velocity_local[0]), vy=float(tel.velocity_local[1]),
                                   r=float(tel.yaw_rate), ax=ax_f, ay=ay_f, steer=float(tel.steer),
                                   throttle=float(tel.throttle), brake=float(tel.brake), gear=int(tel.gear)))
            prev = (action[0], action[1])
            ticks += 1
            dist += tel.speed * dt
            vmax = max(vmax, tel.speed)
            off += adapter.body_over_edge_m() > 0
            if ticks * dt >= next_log:
                next_log += log_every_s
                printer(f"t={ticks * dt:6.1f}s  pos={adapter.track_position_m:7.1f} m  lat={adapter.lateral_offset:+5.2f} m  "
                        f"head={math.degrees(adapter.heading_error):+5.1f} deg  v={tel.speed * 3.6:5.1f} km/h  "
                        f"-> steer {action[0]:+.2f} pedal {action[1]:+.2f}")
            if tel.lap_count > lap0 + len(lap_times):
                lap_times.append(float(tel.lap_time_last))
                printer(f"lap {len(lap_times)}: {tel.lap_time_last:.3f} s")
                if len(lap_times) >= laps:
                    reason = "laps"
                    break
            if not dry_run and adapter.body_over_edge_m() > adapter.env.half_width + 3.0:
                reason = "off_track"
                break
    finally:
        if not dry_run:
            raw_env.session.failsafe()
    return DriveResult(lap_times, reason, dist, vmax, off, ticks)
