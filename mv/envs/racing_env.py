"""The racing environment: thousands of cars, each on its own track.

This is the "game" the learning algorithm plays.  Each control step
(25 times per simulated second by default) it

1. takes an action per car: steering in [-1, 1] and a combined pedal in
   [-1, 1] (positive = throttle, negative = brake),
2. advances the physics (:class:`mv.sim.vehicle_sim.VehicleSim`),
3. works out where each car is relative to its track,
4. computes a reward (metres of progress, minus penalties) and decides whether
   the episode is over (left the track, spun, stalled, time limit),
5. builds the observation the driver sees next.

What the driver sees (the observation)
--------------------------------------
======================  ====================================================
Block                   Content
======================  ====================================================
car state               speed forward/sideways, yaw rate, accelerations,
                        steering angle, pedals, gear, rpm, battery charge,
                        front/rear tyre slip angle relative to the angle of
                        peak grip, its own previous action
position on track       lateral position between the edges, track width,
                        heading relative to the track direction
track preview ("map")   tightest curvature and narrowest width of the track
                        in 23 stretches ahead (up to 2 m, 2-5 m, ... 295-340 m).  Can be cut short to mimic a
                        first lap on an unknown track.
lidar                   distance to the track edge along 19 rays fanned
                        over the front 180 degrees, up to 150 m
car descriptor          a few numbers describing the car (mass, power to
                        weight, grip, downforce, ...) so one driver can
                        adapt its style to different cars
======================  ====================================================

Everything in the observation can be produced from real sensors or from
Assetto Corsa telemetry + the track map, which is what makes the trained
driver transferable (see docs/01-architecture.md, "observation contract").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from mv.sim.vehicle_sim import VehicleSim
from mv.track.track import TrackBatch
from mv.vehicle.params import G, VehicleParams
from mv.vehicle.summary import top_speed, unit_peak_power_kw

__all__ = ["EnvConfig", "RacingEnv"]


@dataclass
class EnvConfig:
    control_dt: float = 0.04                     # seconds between driver decisions (25 Hz)
    # physics steps per decision (-> 125 Hz physics).  Measured: lap times at 500, 250, 125 and
    # 100 Hz agree within 0.05 s, so 125 Hz buys double the training speed for free.
    substeps: int = 5
    max_episode_s: float = 60.0
    # --- sensors ---
    # The furthest point must cover the longest braking zone: a 300 -> 80 km/h stop takes ~230 m.
    lookahead_m: tuple[float, ...] = (2, 5, 8, 12, 16, 20, 25, 30, 36, 42, 50, 60, 70, 85, 100, 120, 140, 165,
                                      190, 220, 255, 295, 340)
    lidar_angles_deg: tuple[float, ...] = (-90, -70, -52, -38, -27, -18, -11, -6, -2.5, 0, 2.5, 6, 11, 18, 27, 38, 52, 70, 90)
    lidar_range_m: float = 150.0
    lidar_stride_m: float = 3.0                  # spacing of the edge points the rays are tested against
    lidar_back_m: float = 15.0
    blind_prob: float = 0.3                      # share of episodes driven with a short track preview ("first lap")
    blind_horizon_m: tuple[float, float] = (25.0, 60.0)
    blind_lidar: bool = True                     # blind episodes also cut the lidar range to the preview distance
    # --- memory: recent history of the car's own state and actions (0 = off) ---
    # Lets the driver infer what it cannot measure directly (grip, its car's response)
    # from how the car reacted over the last moments.  Changes the observation size:
    # a driver trained with one setting cannot be used with another.
    history_steps: int = 0                       # number of past samples
    history_every: int = 3                       # decisions between samples (3 x 0.04 s = 0.12 s)
    # --- reward ---
    progress_scale: float = 0.25                 # reward per metre of progress along the centre line
    edge_penalty: float = 0.5                    # per step, per metre the body hangs over the track edge
    edge_penalty_const: float = 0.1              # per step while any part of the car is over the edge
    crash_penalty: float = 5.0                   # when the episode ends by leaving the track / spinning
    smooth_penalty: float = 0.02                 # discourages sawing at the wheel and pedal
    # --- termination ---
    off_track_margin_m: float = 1.0              # car centre this far beyond the edge = crashed
    max_heading_error_rad: float = 1.8
    stall_speed: float = 1.0
    stall_time_s: float = 4.0
    # --- episode start ---
    start_speed_frac: tuple[float, float] = (0.3, 1.0)   # of the highest feasible speed at the start point
    standing_start_prob: float = 0.1
    # --- per-car variation (domain randomisation): multiplicative ranges ---
    rand_grip: tuple[float, float] = (0.88, 1.05)
    rand_mass: tuple[float, float] = (0.97, 1.06)
    rand_power: tuple[float, float] = (0.93, 1.03)
    rand_aero: tuple[float, float] = (0.9, 1.05)
    obs_clip: float = 5.0
    # --- limit-seeking mode (engineering log WP21) ---
    # The owner's principle: find the limit, sometimes overstep it, adjust lap after lap.
    recover_on_crash: bool = False               # leaving the track costs time (car put back on), episode continues
    recover_speed_frac: float = 0.4              # rejoin speed, as a fraction of the feasible speed there
    max_recoveries: int = 6                      # after this many offs the episode does end
    lap_memory: bool = False                     # per track section: last lap's min speed, closeness to the limit, offs
    lap_memory_cell_m: float = 5.0               # length of one remembered section


def _wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


class RacingEnv:
    def __init__(self, vp: VehicleParams, tracks: TrackBatch, num_envs: int, cfg: EnvConfig | None = None,
                 device: torch.device | str = "cpu", seed: int = 0):
        self.vp, self.tracks, self.n = vp, tracks, num_envs
        self.cfg = cfg = cfg or EnvConfig()
        self.device = dev = torch.device(device)
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        self.sim = VehicleSim(vp, num_envs, dev, dt=cfg.control_dt / cfg.substeps)
        ds = tracks.ds
        self.half_width = 0.5 * vp.chassis.width_m
        self.v_cap = 0.92 * top_speed(vp)
        self.mu_ref = vp.chassis.front_weight_frac * vp.tire_front.mu_lat + (1 - vp.chassis.front_weight_frac) * vp.tire_rear.mu_lat

        # index offsets used to look things up along the track
        fwd = int(math.ceil(110.0 * cfg.control_dt / ds)) + 3
        self.search_off = torch.arange(-3, fwd + 1, device=dev)
        self.look_m = torch.tensor(cfg.lookahead_m, device=dev)
        self.look_off = torch.round(self.look_m / ds).long()
        # The preview reports, for each stretch of road between two preview distances,
        # the TIGHTEST curvature in that stretch (and the narrowest width) - not the value
        # at one point.  Point samples are 20-45 m apart far ahead, so a short, sharp
        # corner (Monza's first chicane: ~15 m at 12 m radius) could fall between them and
        # flicker in and out of view exactly where the braking decision is made
        # (engineering log WP20).  Each stretch is scanned every ~1 m.
        stride = max(int(round(1.0 / ds)), 1)
        edges = [0.0] + list(cfg.lookahead_m)
        bins = []
        for a, b in zip(edges[:-1], edges[1:]):
            lo, hi = int(round(a / ds)) + (1 if a > 0 else 0), int(round(b / ds))
            bins.append(list(range(lo, hi + 1, stride)) or [hi])
        width_bins = max(len(b) for b in bins)
        self.bin_off = torch.tensor([b + [b[-1]] * (width_bins - len(b)) for b in bins], device=dev, dtype=torch.long)
        stride = max(int(round(cfg.lidar_stride_m / ds)), 1)
        n_back, n_fwd = int(cfg.lidar_back_m / (stride * ds)), int(cfg.lidar_range_m / (stride * ds)) + 1
        self.lidar_off = torch.arange(-n_back, n_fwd + 1, device=dev) * stride
        ang = torch.deg2rad(torch.tensor(cfg.lidar_angles_deg, device=dev))
        self.ray_x, self.ray_y = torch.cos(ang), torch.sin(ang)
        # preview used to pick a feasible starting speed (see reset)
        self.safe_m = torch.arange(0.0, 450.0, 5.0, device=dev)
        self.safe_off = torch.round(self.safe_m / ds).long()
        cla = vp.aero.cla_front_m2 + vp.aero.cla_rear_m2
        self.k_aero = 0.5 * vp.aero.air_density * cla / vp.chassis.mass_kg      # downforce "gravity" per (m/s)^2

        # per-car bookkeeping
        z = lambda dtype=torch.float32: torch.zeros(num_envs, device=dev, dtype=dtype)  # noqa: E731
        self.tid, self.idx = z(torch.long), z(torch.long)
        self.along, self.lat, self.head_err = z(), z(), z()
        self.w_left, self.w_right = z(), z()
        self.s_total, self.t_ep, self.stall_t = z(), z(), z()
        self.horizon = torch.full((num_envs,), 1e6, device=dev)
        self.lidar_range = torch.full((num_envs,), float(cfg.lidar_range_m), device=dev)
        self.hist_len = cfg.history_steps * cfg.history_every + 1 if cfg.history_steps > 0 else 0
        self.hist = torch.zeros(num_envs, max(self.hist_len, 1), 15, device=dev)
        self.hist_idx = torch.tensor([self.hist_len - 1 - cfg.history_every * j for j in range(1, cfg.history_steps + 1)],
                                     device=dev, dtype=torch.long)
        self.prev_action = torch.zeros(num_envs, 2, device=dev)
        self.lap_count, self.lap_start_t = z(torch.long), z()
        self.last_lap_time = torch.full((num_envs,), float("nan"), device=dev)
        self.ep_return, self.ep_len = z(), z()
        self.fixed_start: tuple[int, float] | None = None      # (point index, speed) for evaluation runs
        self.recoveries = z(torch.long)
        self.cell_pts = max(int(round(cfg.lap_memory_cell_m / ds)), 1)
        n_cells = tracks.m_max // self.cell_pts + 2 if cfg.lap_memory else 1
        self.mem_v = torch.full((num_envs, n_cells), -1.0, device=dev)   # slowest speed in the section last lap (-1 = none)
        self.mem_u = torch.zeros(num_envs, n_cells, device=dev)          # highest grip use there (3 = went off)
        self.mem_lap = torch.full((num_envs, n_cells), -1, device=dev, dtype=torch.long)

        self.car_desc = self._car_descriptor()
        self.num_actions = 2
        self.reset(torch.arange(num_envs, device=dev))
        self.num_obs = self._observe(torch.arange(1, device=dev)).shape[1]
        # where each block sits in the observation vector (name -> slice); used by tests,
        # the scripted baseline driver and the Assetto Corsa adapter
        sizes = [("car", 13), ("prev_action", 2), ("position", 5), ("kappa", len(cfg.lookahead_m)),
                 ("width", len(cfg.lookahead_m)), ("horizon", 1), ("lidar", len(cfg.lidar_angles_deg)),
                 ("car_desc", len(self.car_desc))]
        if cfg.history_steps > 0:
            sizes.append(("history", cfg.history_steps * 15))
        if cfg.lap_memory:
            sizes.append(("lap_memory", 3 * len(cfg.lookahead_m)))
        self.obs_layout, start = {}, 0
        for key, size in sizes:
            self.obs_layout[key] = slice(start, start + size)
            start += size
        assert start == self.num_obs, "observation layout out of sync with _observe()"

    # ------------------------------------------------------------------------
    def _car_descriptor(self) -> torch.Tensor:
        vp, c = self.vp, self.vp.chassis
        power_kw = sum(unit_peak_power_kw(u) for u in vp.powertrain.units)
        if vp.powertrain.battery and vp.powertrain.type == "ev":
            power_kw = min(power_kw, vp.powertrain.battery.max_discharge_kw)
        drive_front = sum(u.front_frac * unit_peak_power_kw(u) for u in vp.powertrain.units) / max(power_kw, 1e-6)
        cla = vp.aero.cla_front_m2 + vp.aero.cla_rear_m2
        d = [c.mass_kg / 2000.0, power_kw / c.mass_kg * 3.0, self.mu_ref / 1.5, cla / c.mass_kg * 500.0,
             vp.aero.cda_m2 / c.mass_kg * 1000.0, c.wheelbase_m / 3.0, c.front_weight_frac * 2 - 1,
             math.radians(vp.steering.max_wheel_angle_deg), min(drive_front, 1.0),
             1.0 if vp.powertrain.battery else 0.0, c.width_m / 2.0, self.v_cap / 100.0]
        return torch.tensor(d, device=self.device)

    def _rand(self, k: int, lo: float, hi: float) -> torch.Tensor:
        return torch.rand(k, device=self.device, generator=self.gen) * (hi - lo) + lo

    # ------------------------------------------------------------------------
    def reset(self, ids: torch.Tensor) -> None:
        """Start a new episode for cars ``ids``: random track, random place, plausible speed."""
        k = len(ids)
        if k == 0:
            return
        cfg, tr, dev = self.cfg, self.tracks, self.device
        tid = torch.randint(0, tr.num_tracks, (k,), device=dev, generator=self.gen)
        if self.fixed_start is not None:
            tid = ids % tr.num_tracks
            idx = torch.full((k,), self.fixed_start[0], device=dev, dtype=torch.long) % tr.n[tid]
            v0 = torch.full((k,), float(self.fixed_start[1]), device=dev)
            lat0 = torch.zeros(k, device=dev)
            yaw_noise = torch.zeros(k, device=dev)
        else:
            idx = (torch.rand(k, device=dev, generator=self.gen) * tr.n[tid]).long()
            # Highest speed from which the car can still make every corner in the next
            # 450 m: each corner allows v_c = sqrt(mu*g / curvature) (more with downforce),
            # and from a distance d before it we may be doing sqrt(v_c^2 + 2*braking*d).
            # Starting faster than this guarantees a crash and teaches nothing; starting
            # at a random fraction of it puts many episodes right into braking zones at
            # racing speed, which is exactly the hard part to learn.
            v0 = self._feasible_speed(tid, idx) * self._rand(k, *cfg.start_speed_frac)
            v0 = torch.where(self._rand(k, 0, 1) < cfg.standing_start_prob, torch.zeros_like(v0), v0)
            room = (torch.minimum(tr.w_left[tid, idx], tr.w_right[tid, idx]) - self.half_width).clamp(min=0.0)
            lat0 = room * self._rand(k, -0.6, 0.6)
            yaw_noise = self._rand(k, -0.04, 0.04)
        psi = tr.psi[tid, idx]
        p = tr.xy[tid, idx]
        x = p[:, 0] - torch.sin(psi) * lat0
        y = p[:, 1] + torch.cos(psi) * lat0
        self.sim.reset(ids, x, y, psi + yaw_noise, v0)
        self.sim.grip[ids] = self._rand(k, *cfg.rand_grip)
        self.sim.mass_scale[ids] = self._rand(k, *cfg.rand_mass)
        self.sim.power_scale[ids] = self._rand(k, *cfg.rand_power)
        self.sim.aero_scale[ids] = self._rand(k, *cfg.rand_aero)
        self.tid[ids], self.idx[ids] = tid, idx
        self.along[ids], self.lat[ids], self.head_err[ids] = 0.0, lat0, yaw_noise
        self.w_left[ids], self.w_right[ids] = tr.w_left[tid, idx], tr.w_right[tid, idx]
        self.s_total[ids], self.t_ep[ids], self.stall_t[ids] = 0.0, 0.0, 0.0
        blind = self._rand(k, 0, 1) < cfg.blind_prob
        self.horizon[ids] = torch.where(blind, self._rand(k, *cfg.blind_horizon_m), torch.full((k,), 1e6, device=dev))
        full_range = torch.full((k,), float(cfg.lidar_range_m), device=dev)
        self.lidar_range[ids] = torch.where(blind, self.horizon[ids], full_range) if cfg.blind_lidar else full_range
        self.prev_action[ids] = 0.0
        self.lap_count[ids], self.lap_start_t[ids] = 0, 0.0
        self.ep_return[ids], self.ep_len[ids] = 0.0, 0.0
        self.recoveries[ids] = 0
        if cfg.lap_memory:
            self.mem_v[ids], self.mem_u[ids], self.mem_lap[ids] = -1.0, 0.0, -1
        self.reset_history(ids)

    def _feasible_speed(self, tid: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Highest speed from which every corner in the next 450 m can still be made.

        Each corner allows v_c = sqrt(mu*g / curvature) (more with downforce), and from a
        distance d before it we may be doing sqrt(v_c^2 + 2*braking*d).  Starting faster
        than this guarantees a crash and teaches nothing.
        """
        tr = self.tracks
        ahead = tr.wrap(tid, idx[:, None] + self.safe_off[None])
        kap = tr.kappa[tid[:, None], ahead].abs()
        mu = 0.85 * self.mu_ref
        v_c = torch.sqrt(mu * G / (kap - mu * self.k_aero).clamp(min=1e-5)).clamp(max=self.v_cap)
        return torch.sqrt(v_c ** 2 + 2.0 * 0.75 * self.mu_ref * G * self.safe_m[None]).amin(1).clamp(max=self.v_cap)

    def _recover(self, ids: torch.Tensor) -> None:
        """Put cars that left the track back on it, at the same place, slowly, pointing the right way."""
        if len(ids) == 0:
            return
        tr, sim = self.tracks, self.sim
        tid, idx = self.tid[ids], self.idx[ids]
        psi, p = tr.psi[tid, idx], tr.xy[tid, idx]
        v0 = self.cfg.recover_speed_frac * self._feasible_speed(tid, idx)
        soc = sim.soc[ids].clone()                      # the battery does not refill by going off
        sim.reset(ids, p[:, 0], p[:, 1], psi, v0)
        sim.soc[ids] = soc
        self.along[ids], self.lat[ids], self.head_err[ids] = 0.0, 0.0, 0.0
        self.stall_t[ids] = 0.0
        self.prev_action[ids] = 0.0
        self.reset_history(ids)

    # ------------------------------------------------------------------------
    def grip_use(self) -> torch.Tensor:
        """How close each car is to its grip limit right now (1.0 = at the limit).

        The larger of: front/rear slip angle relative to the slip angle of peak grip,
        and total acceleration relative to nominal grip (incl. downforce).  Both can be
        measured on a real car (speeds, yaw rate, steering, accelerometer).
        """
        sim = self.sim
        slip = sim.wheel_slip_angles()
        sf = (slip[:, :2].mean(1) / sim.slip_peak[0]).abs()
        sr = (slip[:, 2:].mean(1) / sim.slip_peak[2]).abs()
        v2 = sim.vx ** 2 + sim.vy ** 2
        g_use = torch.sqrt(sim.ax ** 2 + sim.ay ** 2) / (self.mu_ref * (G + self.k_aero * v2))
        return torch.maximum(torch.maximum(sf, sr), g_use)

    def advance_laps(self, progress: torch.Tensor) -> torch.Tensor:
        """Add progress, count laps, time them.  Returns which cars just completed a lap."""
        self.s_total += progress
        length = self.tracks.length[self.tid]
        lap_done = self.s_total >= (self.lap_count + 1).float() * length
        self.last_lap_time = torch.where(lap_done, self.t_ep - self.lap_start_t, self.last_lap_time)
        self.lap_start_t = torch.where(lap_done, self.t_ep, self.lap_start_t)
        self.lap_count += lap_done.long()
        return lap_done

    def update_lap_memory(self, off: torch.Tensor) -> None:
        """Record, for the section each car is in, its slowest speed and highest grip use this lap."""
        if not self.cfg.lap_memory:
            return
        row = torch.arange(self.n, device=self.device)
        cell = self.idx // self.cell_pts
        v = torch.sqrt(self.sim.vx ** 2 + self.sim.vy ** 2)
        u = torch.where(off, torch.full_like(v, 3.0), self.grip_use().clamp(max=3.0))
        u = torch.nan_to_num(u, nan=3.0)
        fresh = self.mem_lap[row, cell] != self.lap_count      # first visit of this section in this lap
        self.mem_v[row, cell] = torch.where(fresh, v, torch.minimum(self.mem_v[row, cell], v))
        self.mem_u[row, cell] = torch.where(fresh, u, torch.maximum(self.mem_u[row, cell], u))
        self.mem_lap[row, cell] = self.lap_count

    def set_view(self, distance_m: float | None, ids: torch.Tensor | None = None) -> None:
        """Limit how far ahead the car can see (track preview and, if enabled, lidar).

        ``None`` restores full view.  Used for blind-lap evaluation and in the game.
        """
        ids = torch.arange(self.n, device=self.device) if ids is None else ids
        if distance_m is None:
            self.horizon[ids] = 1e6
            self.lidar_range[ids] = float(self.cfg.lidar_range_m)
        else:
            self.horizon[ids] = float(distance_m)
            self.lidar_range[ids] = min(float(distance_m), self.cfg.lidar_range_m) if self.cfg.blind_lidar                 else float(self.cfg.lidar_range_m)

    # ------------------------------------------------------------------------
    def _core(self, ids: torch.Tensor) -> torch.Tensor:
        """Car-state block + previous action (15 values): what the memory remembers."""
        return torch.cat([self._car_state(ids), self.prev_action[ids]], 1)

    def reset_history(self, ids: torch.Tensor) -> None:
        """Fill the memory of cars ``ids`` with their current state (as if it had been steady)."""
        if self.hist_len and len(ids):
            self.hist[ids] = self._core(ids)[:, None, :].expand(-1, self.hist_len, -1)

    def push_history(self) -> None:
        """Append the current state of every car to its memory (once per decision)."""
        if self.hist_len:
            core = self._core(torch.arange(self.n, device=self.device))
            self.hist = torch.cat([self.hist[:, 1:], core[:, None, :]], 1)

    # ------------------------------------------------------------------------
    def _locate(self) -> torch.Tensor:
        """Find each car on its track; returns metres of progress since the last call."""
        tr, sim = self.tracks, self.sim
        cand = tr.wrap(self.tid, self.idx[:, None] + self.search_off[None])            # (N, C)
        pts = tr.xy[self.tid[:, None], cand]                                             # (N, C, 2)
        d2 = (pts[..., 0] - sim.x[:, None]) ** 2 + (pts[..., 1] - sim.y[:, None]) ** 2
        best = d2.argmin(1)
        row = torch.arange(self.n, device=self.device)
        new_idx = cand[row, best]
        moved = self.search_off[best].float() * tr.ds_each[self.tid]
        psi = tr.psi[self.tid, new_idx]
        dx, dy = sim.x - pts[row, best, 0], sim.y - pts[row, best, 1]
        cp, sp = torch.cos(psi), torch.sin(psi)
        along = dx * cp + dy * sp                 # distance past the nearest centre-line point
        progress = moved + along - self.along
        self.idx, self.along = new_idx, along
        self.lat = -dx * sp + dy * cp             # + = left of the centre line
        self.head_err = _wrap_angle(sim.yaw - psi)
        self.w_left, self.w_right = tr.w_left[self.tid, new_idx], tr.w_right[self.tid, new_idx]
        return progress

    def _lidar(self, ids: torch.Tensor) -> torch.Tensor:
        """Distance to the track edges along each ray, shape (len(ids), rays)."""
        tr, sim, rng = self.tracks, self.sim, self.cfg.lidar_range_m
        car_range = self.lidar_range[ids][:, None]                                       # (n, 1) sensor range per car
        tid = self.tid[ids]
        widx = tr.wrap(tid, self.idx[ids][:, None] + self.lidar_off[None])               # (n, P)
        left, right = tr.left[tid[:, None], widx], tr.right[tid[:, None], widx]          # (n, P, 2)

        def to_car_frame(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            dx, dy = p[..., 0] - sim.x[ids][:, None], p[..., 1] - sim.y[ids][:, None]
            c, s = torch.cos(sim.yaw[ids])[:, None], torch.sin(sim.yaw[ids])[:, None]
            return dx * c + dy * s, -dx * s + dy * c

        lx, ly = to_car_frame(left)
        rx, ry = to_car_frame(right)
        ax = torch.cat([lx[:, :-1], rx[:, :-1]], 1)                                      # segment start (n, S)
        ay = torch.cat([ly[:, :-1], ry[:, :-1]], 1)
        ex = torch.cat([lx[:, 1:], rx[:, 1:]], 1) - ax                                   # segment direction
        ey = torch.cat([ly[:, 1:], ry[:, 1:]], 1) - ay
        # Ray from the origin along d hits segment A + u*e at distance t:
        #   t = (A x e) / (d x e),   u = (A x d) / (d x e),   valid if t > 0 and 0 <= u <= 1
        dxe = self.ray_x[None, :, None] * ey[:, None, :] - self.ray_y[None, :, None] * ex[:, None, :]
        dxe = torch.where(dxe.abs() < 1e-9, torch.full_like(dxe, 1e-9), dxe)
        t = (ax * ey - ay * ex)[:, None, :] / dxe
        u = (ax[:, None, :] * self.ray_y[None, :, None] - ay[:, None, :] * self.ray_x[None, :, None]) / dxe
        hit = (t > 0) & (u >= 0) & (u <= 1)
        return torch.minimum(torch.where(hit, t, torch.full_like(t, rng)).amin(2), car_range)

    def _car_state(self, ids: torch.Tensor) -> torch.Tensor:
        sim = self.sim
        vx = sim.vx[ids]
        n_gears = float(max(int(sim.gb_ngears[0]) - 1, 1))
        slip = sim.wheel_slip_angles()[ids]          # from the current state, as a real car would compute it
        return torch.stack([
            vx / 50.0, sim.vy[ids] / 5.0, sim.r[ids], sim.ax[ids] / 15.0, sim.ay[ids] / 15.0,
            sim.steer[ids] / sim.steer_max, sim.thr[ids], sim.brk[ids],
            sim.gear[ids, 0].float() / n_gears, sim.rpm_frac[ids], sim.soc[ids],
            # tyre slip angle / slip angle of peak grip: +-1 = at the limit.  Purely kinematic
            # (from speeds, yaw rate and steering), so a real car or AC can supply it too.
            (slip[:, :2].mean(1) / sim.slip_peak[0]).clamp(-3, 3),
            (slip[:, 2:].mean(1) / sim.slip_peak[2]).clamp(-3, 3),
        ], 1)

    def _observe(self, ids: torch.Tensor) -> torch.Tensor:
        tr, cfg = self.tracks, self.cfg
        tid = self.tid[ids]
        car = self._car_state(ids)
        wl, wr = self.w_left[ids], self.w_right[ids]
        lat = self.lat[ids]
        pos = torch.stack([
            torch.where(lat >= 0, lat / wl, lat / wr),          # -1 = right edge, +1 = left edge
            (wl - lat - self.half_width) / 5.0,                 # room to the left edge
            (wr + lat - self.half_width) / 5.0,                 # room to the right edge
            torch.sin(self.head_err[ids]), torch.cos(self.head_err[ids]),
        ], 1)
        n = len(ids)
        widx = tr.wrap(tid, (self.idx[ids][:, None, None] + self.bin_off[None]).reshape(n, -1)).reshape(n, *self.bin_off.shape)
        k_all = tr.kappa[tid[:, None, None], widx]                                   # (n, bins, samples)
        pick = k_all.abs().argmax(-1, keepdim=True)
        k_peak = k_all.gather(-1, pick).squeeze(-1)                                  # signed tightest curvature
        w_min = (tr.w_left[tid[:, None, None], widx] + tr.w_right[tid[:, None, None], widx]).amin(-1)
        seen = (self.look_m[None] <= self.horizon[ids][:, None]).float()
        kappa = (k_peak * 25.0).clamp(-3, 3) * seen
        width = w_min / 15.0 * seen
        horizon = (self.horizon[ids] / self.look_m[-1]).clamp(max=1.0)[:, None]
        lidar = torch.sqrt(self._lidar(ids) / cfg.lidar_range_m)   # sqrt: finer resolution up close
        desc = self.car_desc[None].expand(len(ids), -1)
        parts = [car, self.prev_action[ids], pos, kappa, width, horizon, lidar, desc]
        if self.hist_len:
            parts.append(self.hist[ids][:, self.hist_idx].reshape(len(ids), -1))
        if cfg.lap_memory:
            cells = widx // self.cell_pts
            mv = self.mem_v[ids[:, None, None], cells]
            mu_ = self.mem_u[ids[:, None, None], cells]
            known = mv >= 0
            any_known = known.any(-1)
            v_min = torch.where(known, mv, torch.full_like(mv, 1e3)).amin(-1)
            u_max = torch.where(known, mu_, torch.zeros_like(mu_)).amax(-1)
            parts += [torch.where(any_known, v_min / 50.0, torch.zeros_like(v_min)), u_max, any_known.float()]
        obs = torch.cat(parts, 1)
        return obs.clamp(-cfg.obs_clip, cfg.obs_clip)

    # ------------------------------------------------------------------------
    def observe(self) -> torch.Tensor:
        return self._observe(torch.arange(self.n, device=self.device))

    def tracks_replaced(self, track_ids: torch.Tensor) -> None:
        """Restart every car that was driving on a track that has just been swapped out."""
        self.reset(torch.isin(self.tid, track_ids).nonzero(as_tuple=False).squeeze(1))

    def scripted_teacher(self):
        """The rule-based driver for this car (used by the warm start)."""
        from mv.control.scripted import ScriptedDriver
        return ScriptedDriver(self)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Returns (obs, reward, terminated, truncated, info).

        ``terminated`` = the episode ended because of what the car did (crash).
        ``truncated``  = it merely ran out of time.  ``info["final_obs"]`` holds
        the last observation of episodes that just ended, which the learner
        needs to treat time-outs correctly.
        """
        cfg, sim = self.cfg, self.sim
        action = action.clamp(-1.0, 1.0)
        steer, pedal = action[:, 0], action[:, 1]
        thr, brk = pedal.clamp(min=0.0), (-pedal).clamp(min=0.0)
        for _ in range(cfg.substeps):
            sim.step(steer, thr, brk)
        progress = self._locate()
        self.t_ep += cfg.control_dt
        self.ep_len += 1
        lap_done = self.advance_laps(progress)

        # ---- reward -------------------------------------------------------------
        over = (torch.maximum(self.lat - self.w_left, -self.lat - self.w_right) + self.half_width)  # body over the edge (m)
        over_pos = over.clamp(min=0.0)
        jerk = ((action - self.prev_action) ** 2).sum(1)
        reward = (cfg.progress_scale * progress
                  - cfg.edge_penalty * over_pos.clamp(max=3.0) - cfg.edge_penalty_const * (over > 0).float()
                  - cfg.smooth_penalty * jerk)

        # ---- end of episode? -------------------------------------------------------
        speed = torch.sqrt(sim.vx ** 2 + sim.vy ** 2)
        self.stall_t = torch.where(speed < cfg.stall_speed, self.stall_t + cfg.control_dt, torch.zeros_like(self.stall_t))
        off = over > self.half_width + cfg.off_track_margin_m
        spun = self.head_err.abs() > cfg.max_heading_error_rad
        stalled = self.stall_t > cfg.stall_time_s
        bad = ~torch.isfinite(sim.vx + sim.vy + sim.r + sim.x + sim.y)
        crash = off | spun | stalled
        self.update_lap_memory(crash)
        if cfg.recover_on_crash:
            # limit-seeking: a mistake costs the penalty and the time lost; then try again
            self.recoveries += crash.long()
            give_up = crash & (self.recoveries > cfg.max_recoveries)
            recover = crash & ~give_up & ~bad
            terminated = give_up | bad
        else:
            recover = torch.zeros_like(crash)
            terminated = crash | bad
        truncated = (self.t_ep >= cfg.max_episode_s) & ~terminated
        reward = torch.where(crash | bad, reward - cfg.crash_penalty, reward)
        reward = torch.nan_to_num(reward, nan=-cfg.crash_penalty, posinf=0.0, neginf=-cfg.crash_penalty)
        self.prev_action = action
        self.ep_return += reward
        self.push_history()

        done = terminated | truncated
        self._recover(recover.nonzero(as_tuple=False).squeeze(1))
        info: dict = {"off": off, "spun": spun, "stalled": stalled, "lap_done": lap_done,
                      "lap_time": self.last_lap_time, "speed": speed, "progress": progress, "recovered": recover}
        obs = self.observe()
        ids = done.nonzero(as_tuple=False).squeeze(1)
        if len(ids) > 0:
            info["final_obs"] = torch.nan_to_num(obs[ids])
            info["done_ids"] = ids
            info["ep_return"] = self.ep_return[ids].clone()
            info["ep_len"] = self.ep_len[ids].clone()
            info["ep_dist"] = self.s_total[ids].clone()
            if bad.any():       # scrub non-finite state before re-using those slots
                b = bad.nonzero(as_tuple=False).squeeze(1)
                for a in (sim.x, sim.y, sim.yaw, sim.vx, sim.vy, sim.r, sim.ax, sim.ay, sim.steer, sim.thr, sim.brk):
                    a[b] = 0.0
            self.reset(ids)
            obs[ids] = self._observe(ids)
        return obs, reward, terminated, truncated, info
