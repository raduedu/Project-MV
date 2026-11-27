"""The race bot in the game (stage 5, engineering log WP65): observation from live data, the drive loop.

The driver's inputs are built by the training environment itself (:class:`RacingEnv`), so they are
computed exactly as in training: car 0 of a small shadow environment is the bot car, the other slots are
the other cars of the session (the owner first), and every value - track position, preview, lidar, own
map, lap memory, the opponents block - comes from the same code.  Only the car's own state is copied in
from the game each tick.  Cars beyond the slots, or slots without a car, are parked on a dummy circuit
where nobody sees them.

What the bot does NOT get that the owner has: nothing.  It sees ``view_m`` metres ahead (sensors), has no
map of the circuit beyond what it has already driven (own map), and drives the same car model through
the game's own physics.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np
import torch

from mv.ac.driver import ActuatorModel
from mv.ac.racebot_link import RaceBotLink, RaceBotState, TAKEOVER_NO_INPUT, TAKEOVER_THROTTLE_LIMIT
from mv.envs.racing_env import EnvConfig, RacingEnv
from mv.rl.evaluate import eval_config
from mv.track.track import Track, TrackBatch
from mv.vehicle.params import VehicleParams

__all__ = ["RaceBotAdapter", "race"]


def _dummy_track(ds: float) -> Track:
    a = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    xy = np.stack([60 * np.cos(a), 60 * np.sin(a)], 1) + 1e6          # far from everything
    return Track.from_centerline(xy, 6.0, 6.0, ds=ds, name="parking")


class RaceBotAdapter:
    """Builds the bot's observation from a :class:`RaceBotState`, exactly as the training environment does."""

    def __init__(self, vp: VehicleParams, track: Track, cfg: EnvConfig, view_m: float = 120.0,
                 accel_filter_s: float = 0.06):
        slots = max(int(cfg.race_group), 1) if cfg.opponents > 0 else 1
        cfg = replace(eval_config(cfg, strict=False), race_group=slots)
        self.cfg, self.view_m, self.slots = cfg, float(view_m), slots
        self.env = env = RacingEnv(vp, TrackBatch([track, _dummy_track(track.ds)], ds=track.ds), slots, cfg,
                                   device="cpu")
        env.fixed_start = (0, 0.0)
        env.reset(torch.arange(slots))
        self.track = track
        self._xy = torch.as_tensor(track.xy, dtype=torch.float32)
        self.accel_filter_s = accel_filter_s
        self._fresh = [True] * slots
        self._last_t: float | None = None
        self._last_v: tuple[float, float] | None = None
        self._park(range(slots))

    def _park(self, ids) -> None:
        env, sim = self.env, self.env.sim
        for i in ids:
            env.tid[i], env.idx[i], env.along[i] = 1, 0, 0.0
            p = env.tracks.xy[1, 0]
            sim.x[i], sim.y[i], sim.yaw[i], sim.vx[i], sim.vy[i] = p[0], p[1], float(env.tracks.psi[1, 0]), 0.0, 0.0
            self._fresh[i] = True

    def _place(self, i: int, x: float, y: float) -> None:
        """Find car i on the circuit from scratch (first sight, or it jumped > 30 m: teleport, pit)."""
        env = self.env
        if not self._fresh[i] and int(env.tid[i]) == 0:
            near = self._xy[int(env.idx[i])]
            if (near[0] - x) ** 2 + (near[1] - y) ** 2 <= 30.0 ** 2:
                return
        d2 = (self._xy[:, 0] - x) ** 2 + (self._xy[:, 1] - y) ** 2
        env.tid[i], env.idx[i], env.along[i] = 0, int(d2.argmin()), 0.0
        self._fresh[i] = True

    def update(self, st: RaceBotState, act: ActuatorModel, prev_action=(0.0, 0.0)) -> torch.Tensor:
        """Copy the game state into the shadow cars and return the bot's observation, shape (1, num_obs)."""
        env, sim, b = self.env, self.env.sim, st.bot
        # ---- the bot car ------------------------------------------------------------------------
        now = float(st.game_time_s)
        dt = self.cfg.control_dt if self._last_t is None else max(now - self._last_t, 1e-3)
        sim.x[0], sim.y[0], sim.yaw[0], sim.vx[0], sim.vy[0] = b.x, b.y, b.yaw, b.vx, b.vy
        sim.r[0] = st.yaw_rate
        # body accelerations from the change of the body velocity (a = dv/dt + omega x v): no dependence
        # on the sign conventions of the game's g-meter; smoothed like the simulator's load transfer
        if self._last_v is not None:
            ax = (b.vx - self._last_v[0]) / dt - b.vy * st.yaw_rate
            ay = (b.vy - self._last_v[1]) / dt + b.vx * st.yaw_rate
            k = 1.0 - math.exp(-dt / self.accel_filter_s)
            sim.ax[0] += (ax - float(sim.ax[0])) * k
            sim.ay[0] += (ay - float(sim.ay[0])) * k
        self._last_t, self._last_v = now, (b.vx, b.vy)
        # actuators: what the bot commanded (as in training, the simulated car's actuator state)
        sim.steer[0], sim.thr[0], sim.brk[0] = act.steer * sim.steer_max, act.thr, act.brk
        n_gears = int(sim.gb_ngears[0])
        sim.gear[0, 0] = int(np.clip(st.gear - 1, 0, n_gears - 1))
        sim.rpm_frac[0] = min(max(st.rpm, float(sim.u_launch[0])) / float(sim.u_top[0]), 1.0)
        sim.soc[0] = float(st.kers_charge) if sim.has_motor else 1.0
        self._place(0, b.x, b.y)
        # ---- the other cars (nearest first would not matter: the opponents block sorts) ----------
        others = st.others[: self.slots - 1]
        for i, o in enumerate(others, start=1):
            sim.x[i], sim.y[i], sim.yaw[i], sim.vx[i], sim.vy[i] = o.x, o.y, o.yaw, o.vx, o.vy
            self._place(i, o.x, o.y)
        self._park(range(1 + len(others), self.slots))
        # ---- locate, laps, map, memory - the training code --------------------------------------
        progress = env._locate()
        fresh = torch.tensor(self._fresh)
        progress = torch.where(fresh, torch.zeros_like(progress), progress)
        env.t_ep[:] = now
        env.advance_laps(progress)
        env.set_view(self.view_m)
        if self.cfg.own_map:
            env.map_reach = torch.maximum(env.map_reach, env.s_total + env.horizon.clamp(max=1e5))
        if self.cfg.self_recovery:
            env._update_surface()
            off = env.wheels_off >= 3
        else:
            over = torch.maximum(env.lat - env.w_left, -env.lat - env.w_right) + env.half_width
            off = over > env.half_width + self.cfg.off_track_margin_m
        env.update_lap_memory(off)
        env.prev_action[0, 0], env.prev_action[0, 1] = float(prev_action[0]), float(prev_action[1])
        if self._fresh[0]:
            env.reset_history(torch.arange(1))
        else:
            env.push_history()
        self._fresh = [False] * self.slots
        return env._observe(torch.arange(1))

    @property
    def lateral_offset(self) -> float:
        return float(self.env.lat[0])

    @property
    def track_position_m(self) -> float:
        return float(self.env.idx[0]) * self.track.ds


def race(link: RaceBotLink, driver, adapter: RaceBotAdapter, vp: VehicleParams, dry_run: bool = False,
         max_time_s: float = 7200.0, printer=print, control_check_s: float = 3.0) -> str:
    """Drive the bot car until the session ends, the owner stops it (Ctrl+C) or something is wrong.

    Waits on the grid (brake held) until the session starts.  After ``control_check_s`` seconds of
    driving, the game must confirm that the car's pedals equal ours (``control_ok``); if the AI could not
    be silenced the fallback takeover is tried once, then the bot stops and gives the car back.
    Returns why it stopped.
    """
    dt = adapter.cfg.control_dt
    act = ActuatorModel(vp, dt, adapter.cfg.steer_filter_s)
    prev = (0.0, 0.0)
    t0 = time.perf_counter()
    next_tick = t0
    driving_since = None
    tried_fallback = False
    next_log = 0.0
    reason = "time"
    try:
        while time.perf_counter() - t0 < max_time_s:
            next_tick += dt
            time.sleep(max(0.0, next_tick - time.perf_counter()))
            st = link.read()
            if st is None:
                reason = "the MV Race Bot app stopped publishing (game closed or app not installed)"
                break
            if st.in_menu or st.is_paused or not st.session_started:
                if not dry_run:
                    link.drive(0.0, 0.0, 1.0)          # hold the car (grid / pause); keeps the app from timing out
                act.sync(0.0, 0.0, 1.0)
                driving_since = None
                continue
            obs = adapter.update(st, act, prev)
            action = driver.act(obs, deterministic=True)[0][0].clamp(-1, 1).tolist()
            steer, thr, brk = act(action)
            prev = (action[0], action[1])
            if not dry_run:
                link.drive(steer, thr, brk)
                now = time.perf_counter()
                driving_since = driving_since or now
                if now - driving_since > control_check_s and not st.control_ok:
                    if not tried_fallback and link.takeover == TAKEOVER_NO_INPUT:
                        printer("the game's AI still presses the pedals - trying the fallback takeover")
                        link.takeover, tried_fallback, driving_since = TAKEOVER_THROTTLE_LIMIT, True, None
                    else:
                        reason = ("the car does not follow the bot's pedals (the game's AI could not be silenced: "
                                  f"physics.allowed={st.physics_allowed}, takeover={st.takeover_active})")
                        break
            t = time.perf_counter() - t0
            if t >= next_log:
                next_log += 1.0
                gaps = []
                for o in st.others:
                    gaps.append(f"car {o.index} {math.hypot(o.x - st.bot.x, o.y - st.bot.y):5.0f} m")
                printer(f"t={t:6.1f}s lap {st.lap_count + 1} P{st.race_position}  pos {adapter.track_position_m:7.0f} m  "
                        f"lat {adapter.lateral_offset:+5.2f}  {math.hypot(st.bot.vx, st.bot.vy) * 3.6:5.1f} km/h  "
                        f"steer {action[0]:+.2f} pedal {action[1]:+.2f}  | " + ", ".join(gaps))
    except KeyboardInterrupt:
        reason = "stopped by the owner"
    finally:
        link.release()
    return reason
