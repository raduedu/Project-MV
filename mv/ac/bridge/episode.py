"""Off-track / wrong-way / collision / stuck detection from :class:`Telemetry`.

These are *detectors*, not rewards: they tell the caller (the lead's env
wrapper) that something happened; what to do about it is not decided here.

Signals used, best first:

off-track
    * Kunos ``numberOfTyresOut`` (AC's own track-limits logic, the same that
      invalidates laps): off when ``>= off_track_tyres`` (default 3, i.e. AC's
      default of 2 allowed tyres out is exceeded).
    * Plugin ``track_lateral`` = ``ac.worldCoordinateToTrack(pos).x`` (-1 left
      AI-spline side, +1 right side): off when ``|x| > lateral_limit``.  The
      AI-spline "sides" are the AI's notion of the track edges, not always the
      painted limits, hence a margin (default 1.15).
wrong-way
    Progress rate along the spline (wrap-aware) converted to m/s with the
    track length: wrong-way when it is below ``-wrong_way_speed`` for
    ``wrong_way_hold_s``.  Uses only the Kunos pages.
collision
    * Plugin: ``collision_count`` increased (rising edges of CSP's
      ``car.collisionDepth > 0``); ``collided_with`` = 0 for track objects,
      non-zero for cars.
    * Fallback (Kunos only): any ``carDamage`` zone increased, or a velocity
      change larger than ``collision_accel`` m/s^2 between two packets (5 g by
      default — far above tyre-limited braking, so false positives are rare,
      but light scrapes are missed).
stuck
    Speed below ``stuck_speed`` for ``stuck_s`` seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .telemetry import Telemetry

__all__ = ["EpisodeFlags", "EpisodeMonitor"]


@dataclass
class EpisodeFlags:
    off_track: bool = False
    wrong_way: bool = False
    collision: bool = False
    collision_with: int | None = None   # plugin only: 0 track, >0 car
    stuck: bool = False
    progress_speed: float = 0.0         # m/s along the track (+ forward)
    reasons: list[str] = field(default_factory=list)

    @property
    def any(self) -> bool:
        return self.off_track or self.wrong_way or self.collision or self.stuck


class EpisodeMonitor:
    def __init__(self, off_track_tyres: int = 3, lateral_limit: float = 1.15,
                 off_track_hold_s: float = 0.0, wrong_way_speed: float = 2.0,
                 wrong_way_hold_s: float = 1.0, collision_accel: float = 5.0 * 9.80665,
                 stuck_speed: float = 0.5, stuck_s: float = 5.0,
                 default_track_length: float = 1000.0):
        self.off_track_tyres = off_track_tyres
        self.lateral_limit = lateral_limit
        self.off_track_hold_s = off_track_hold_s
        self.wrong_way_speed = wrong_way_speed
        self.wrong_way_hold_s = wrong_way_hold_s
        self.collision_accel = collision_accel
        self.stuck_speed = stuck_speed
        self.stuck_s = stuck_s
        self.default_track_length = default_track_length
        self.reset()

    def reset(self, tel: Telemetry | None = None) -> None:
        self._prev: Telemetry | None = tel
        self._off_t = 0.0
        self._wrong_t = 0.0
        self._stuck_t = 0.0

    @staticmethod
    def spline_delta(prev: float, cur: float) -> float:
        """Signed progress in laps, handling the 1 -> 0 wrap at the line."""
        d = cur - prev
        if d > 0.5:
            d -= 1.0
        elif d < -0.5:
            d += 1.0
        return d

    def update(self, tel: Telemetry) -> EpisodeFlags:
        f = EpisodeFlags()
        prev, self._prev = self._prev, tel
        dt = 0.0 if prev is None else max(0.0, tel.wall_time - prev.wall_time)
        # Prefer simulation time when both packets have it (immune to Python stalls).
        if prev is not None and tel.sim_time_source == prev.sim_time_source == "plugin":
            dt_sim = tel.sim_time - prev.sim_time
            if dt_sim > 0:
                dt = dt_sim

        # ---- off-track
        off_reasons = []
        if tel.tyres_out >= self.off_track_tyres:
            off_reasons.append(f"tyres_out={tel.tyres_out}")
        if tel.plugin is not None and abs(tel.plugin.track_lateral) > self.lateral_limit:
            off_reasons.append(f"lateral={tel.plugin.track_lateral:+.2f}")
        self._off_t = self._off_t + dt if off_reasons else 0.0
        if off_reasons and self._off_t >= self.off_track_hold_s:
            f.off_track = True
            f.reasons += off_reasons

        if prev is None or dt <= 0.0:
            return f

        # ---- wrong-way (progress rate along the spline)
        length = tel.track_length if tel.track_length > 0 else self.default_track_length
        f.progress_speed = self.spline_delta(prev.spline, tel.spline) * length / dt
        if f.progress_speed < -self.wrong_way_speed:
            self._wrong_t += dt
        else:
            self._wrong_t = 0.0
        if self._wrong_t >= self.wrong_way_hold_s:
            f.wrong_way = True
            f.reasons.append(f"progress_speed={f.progress_speed:.1f}m/s")

        # ---- collision
        if tel.plugin is not None and prev.plugin is not None \
                and tel.plugin.collision_count > prev.plugin.collision_count:
            f.collision = True
            f.collision_with = tel.plugin.collided_with
            f.reasons.append(f"plugin_collision(with={tel.plugin.collided_with})")
        if np.any(tel.car_damage > prev.car_damage + 1e-6):
            f.collision = True
            f.reasons.append("damage_increase")
        dv = float(np.linalg.norm(tel.velocity_world - prev.velocity_world))
        if dt > 0 and dv / dt > self.collision_accel and not math.isinf(dv / dt):
            f.collision = True
            f.reasons.append(f"accel_spike={dv / dt / 9.80665:.1f}g")

        # ---- stuck
        self._stuck_t = self._stuck_t + dt if tel.speed < self.stuck_speed else 0.0
        if self._stuck_t >= self.stuck_s:
            f.stuck = True
            f.reasons.append(f"stuck {self._stuck_t:.1f}s")
        return f
