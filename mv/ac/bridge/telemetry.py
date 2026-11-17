"""Typed, SI-unit telemetry in the project's axis convention.

Axis conventions
================
Project (``mv.vehicle.params``): x forward, y left, z up; yaw counter-clockwise
(seen from above) positive; positive steering turns left.  Angular rates use
the right-hand rule about these axes, so ``yaw_rate > 0`` = turning left,
``pitch`` > 0 = nose down, ``roll`` > 0 = right side down (ISO 8855).

Assetto Corsa: car-local **X = left, Y = up, Z = forward** (right-handed).
Verified, not assumed: in this install every left-hand-drive car's
``car.ini [GRAPHICS] DRIVEREYES`` has x > 0 (ferrari_458 0.34, bmw_m3_e30
0.31, ks_mazda_mx5_nd 0.34) and every right-hand-drive car x < 0
(ks_nissan_skyline_r34 -0.38, ks_toyota_supra_mkiv -0.34, ks_toyota_ae86
-0.32), and z < 0 (eyes behind the reference point).  The world frame is a
rigid rotation of the same handedness with Y up.  Two conversions follow
(:mod:`mv.ac.bridge.frames`), both proper rotations (determinant +1):

* body quantities (localVelocity, localAngularVel, accG)::

      v_body = (Z, X, Y)          # x fwd, y left, z up

* world quantities (position, velocity, orientation) go to the *map frame* of
  ``mv.ac.track_import.ac_world_to_map`` so live positions overlay imported
  tracks exactly::

      p_map = (X, -Z, Y)          # map_x right, map_y up on the map, z up

  ``yaw`` is measured in that map frame: CCW positive, 0 along +map_x.  A car
  facing AC +Z therefore has yaw = -pi/2; facing AC +X has yaw = 0.

What is *not* documented by Kunos and therefore handled defensively:

* ``heading`` zero/sign.  We do not trust it blindly: :class:`TelemetryDecoder`
  derives yaw from ``velocity`` (world) vs ``localVelocity`` (body) while
  the car moves (that relation is exact and convention-free), learns the
  sign/offset that maps ``heading`` onto it, and uses calibrated ``heading``
  at low speed.  If the MV Bridge Lua app is running, its orientation
  vectors give exact yaw/pitch/roll and take priority.
* ``pitch``/``roll`` signs (Kunos pages): exposed via ``PITCH_SIGN``/
  ``ROLL_SIGN`` assumptions and flagged; exact values come from the plugin.
* ``steerAngle`` normalisation/sign and ``accG`` signs: verified at run time
  by ``scripts/ac_bridge_check.py`` (it correlates ``accG`` with the
  numerical derivative of velocity and prints PASS/FAIL).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .frames import ac_local_to_body, ac_world_to_map3
from .plugin_link import PluginState
from .shm_layout import SPageFileGraphic, SPageFilePhysics, SPageFileStatic

__all__ = ["Telemetry", "TelemetryDecoder", "ac_local_to_body", "ac_world_to_map3", "G",
           "NOMINAL_PHYSICS_HZ"]

G = 9.80665
NOMINAL_PHYSICS_HZ = 333.0  # AC physics step; packetId rate is measured by the check script

# Sign assumptions for Kunos pitch/roll (unverified; exact values come from the plugin).
PITCH_SIGN = 1.0
ROLL_SIGN = 1.0
# AC steerAngle: normalised -1..1 with +1 = right is the working assumption
# (it matches CSP's controls convention).  Project convention is +1 = left.
STEER_SIGN_AC_TO_PROJECT = -1.0

_INT_MAX = 2**31 - 1


def _arr4(a) -> np.ndarray:
    return np.array([a[0], a[1], a[2], a[3]], dtype=np.float64)


def _ms_to_s(ms: int) -> float:
    """AC lap-time ints are ms; 0 / INT_MAX / negative mean 'no time'."""
    if ms <= 0 or ms >= _INT_MAX:
        return math.nan
    return ms / 1000.0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Telemetry:
    """One decoded physics packet (plus latest graphics/static/plugin data).

    All quantities SI (m, s, m/s, m/s^2, rad, rad/s, N, deg C, L) and in the
    project axis convention.  Wheel arrays: FL, FR, RL, RR.  Unavailable
    times are ``nan``.
    """

    # ---- timing / identity
    packet_id: int
    graphics_packet_id: int
    wall_time: float              # time.perf_counter() when read
    sim_time: float               # s; see sim_time_source
    sim_time_source: str          # "plugin" (CSP physics timestamp) or "packet_estimate"
    status: int                   # AC_STATUS: 0 OFF, 1 REPLAY, 2 LIVE, 3 PAUSE
    session_type: int
    # ---- pose & motion
    position: np.ndarray          # map frame (AC X, -AC Z, AC Y) m
    velocity_world: np.ndarray    # map frame, m/s
    velocity_local: np.ndarray    # (vx fwd, vy left, vz up) m/s
    speed: float                  # m/s (from speedKmh)
    yaw: float                    # rad, map frame, CCW positive, 0 = +map_x, (-pi, pi]
    yaw_source: str               # "plugin" | "velocity" | "heading_calibrated" | "heading_raw"
    pitch: float                  # rad, + nose down (exact only if plugin)
    roll: float                   # rad, + right side down (exact only if plugin)
    yaw_rate: float               # rad/s, + turning left
    angular_velocity_local: np.ndarray  # (p roll, q pitch, r yaw) rad/s
    acceleration_local: np.ndarray      # (ax fwd, ay left, az up) m/s^2 (from accG)
    heading_raw: float            # Kunos 'heading' rad, unconverted
    pitch_raw: float
    roll_raw: float
    # ---- driver / powertrain
    rpm: float
    max_rpm: float
    gear: int                     # -1 R, 0 N, 1 = first ...
    steer: float                  # normalised, + left (project convention)
    steer_raw: float              # AC steerAngle as published
    throttle: float               # 0..1 (actual, after assists)
    brake: float                  # 0..1
    clutch: float                 # AC: 1 = engaged (pedal up)
    fuel: float                   # L
    # ---- tyres
    wheel_slip: np.ndarray
    wheel_load: np.ndarray        # N
    slip_ratio: np.ndarray
    slip_angle: np.ndarray        # rad (per reference doc)
    tyre_temp_core: np.ndarray    # deg C
    tyre_temp_surface: np.ndarray  # deg C (AC 'tyreTemp')
    tyre_temp_imo: np.ndarray     # (4, 3) inner/middle/outer deg C
    wheel_angular_speed: np.ndarray  # rad/s
    # ---- track / lap
    spline: float                 # normalised track position 0..1
    lap_count: int
    lap_time_current: float       # s
    lap_time_last: float          # s
    lap_time_best: float          # s
    tyres_out: int                # Kunos numberOfTyresOut
    in_pit: bool
    in_pitlane: bool
    # ---- electronics / hybrid
    tc_level: float               # Kunos 'tc' (intervention)
    abs_level: float
    tc_active: bool
    abs_active: bool
    kers_charge: float            # 0..1
    kers_input: float
    kers_current_kj: float
    ers_recovery_level: int
    ers_power_level: int
    ers_heat_charging: bool
    ers_is_charging: bool
    car_damage: np.ndarray        # 5 zones
    is_ai_controlled: bool
    # ---- static
    car_model: str
    track: str
    track_config: str
    track_length: float           # m (static, or plugin if static is 0)
    # ---- optional CSP Lua app data (None if the app is not running)
    plugin: PluginState | None = field(default=None, repr=False)


class TelemetryDecoder:
    """Stateful converter ``(physics, graphics, static, plugin) -> Telemetry``.

    State is only used for yaw: the heading calibration described in the
    module docstring.  ``min_speed_for_yaw`` (m/s) is the speed above which
    the velocity-derived yaw is trusted (lateral slip makes the *course*
    angle differ from yaw, but body-vs-world velocity gives yaw itself, so
    the only requirement is a well-defined velocity direction).
    """

    def __init__(self, min_speed_for_yaw: float = 3.0, calib_samples: int = 50):
        self.min_speed_for_yaw = min_speed_for_yaw
        self.calib_samples = calib_samples
        self._resid = {+1: [], -1: []}  # yaw_vel - s*heading samples (complex unit)
        self.heading_sign: int | None = None
        self.heading_offset: float = 0.0

    # --------------------------------------------------------------- yaw calib
    def _update_calibration(self, heading: float, yaw_vel: float) -> None:
        if self.heading_sign is not None:
            return
        for s in (+1, -1):
            self._resid[s].append(complex(math.cos(yaw_vel - s * heading),
                                          math.sin(yaw_vel - s * heading)))
        if len(self._resid[+1]) < self.calib_samples:
            return
        best, best_r = None, 0.0
        for s in (+1, -1):
            m = sum(self._resid[s]) / len(self._resid[s])
            if abs(m) > best_r:
                best, best_r = s, abs(m)
        # Mean resultant length ~1 means a constant offset fits: accept.
        if best is not None and best_r > 0.98:
            m = sum(self._resid[best]) / len(self._resid[best])
            self.heading_sign = best
            self.heading_offset = math.atan2(m.imag, m.real)
        else:  # inconsistent (e.g. collected during a spin) -> start over
            self._resid = {+1: [], -1: []}

    @staticmethod
    def yaw_from_velocities(v_world: np.ndarray, v_local: np.ndarray) -> float:
        """Yaw such that Rz(yaw) @ v_local_xy == v_world_xy (exact for flat ground)."""
        return _wrap(math.atan2(v_world[1], v_world[0]) - math.atan2(v_local[1], v_local[0]))

    # ------------------------------------------------------------------ decode
    def decode(self, ph: SPageFilePhysics, gr: SPageFileGraphic, st: SPageFileStatic,
               plugin: PluginState | None = None, wall_time: float | None = None) -> Telemetry:
        v_world = ac_world_to_map3(ph.velocity)
        v_local = ac_local_to_body(ph.localVelocity)
        w_local = ac_local_to_body(ph.localAngularVel)
        acc = ac_local_to_body(ph.accG) * G
        speed = ph.speedKmh / 3.6

        # yaw: plugin (exact) > velocity-derived > calibrated heading > raw heading
        if plugin is not None:
            yaw, yaw_src = plugin.yaw, "plugin"
            pitch, roll = plugin.pitch, plugin.roll
        else:
            pitch, roll = PITCH_SIGN * ph.pitch, ROLL_SIGN * ph.roll
            hspeed = math.hypot(v_world[0], v_world[1])
            if hspeed >= self.min_speed_for_yaw and math.hypot(v_local[0], v_local[1]) > 0.5:
                yaw = self.yaw_from_velocities(v_world, v_local)
                yaw_src = "velocity"
                self._update_calibration(ph.heading, yaw)
            elif self.heading_sign is not None:
                yaw = _wrap(self.heading_sign * ph.heading + self.heading_offset)
                yaw_src = "heading_calibrated"
            else:
                yaw, yaw_src = _wrap(ph.heading), "heading_raw"

        if plugin is not None and plugin.car_timestamp_s > 0:
            sim_time, sim_src = plugin.car_timestamp_s, "plugin"
        else:
            sim_time, sim_src = ph.packetId / NOMINAL_PHYSICS_HZ, "packet_estimate"

        track_len = float(st.trackSPlineLength)
        if track_len <= 0 and plugin is not None:
            track_len = float(plugin.track_length_m)

        max_rpm = float(ph.currentMaxRpm) if ph.currentMaxRpm > 0 else float(st.maxRpm)

        return Telemetry(
            packet_id=int(ph.packetId), graphics_packet_id=int(gr.packetId),
            wall_time=time.perf_counter() if wall_time is None else wall_time,
            sim_time=sim_time, sim_time_source=sim_src,
            status=int(gr.status), session_type=int(gr.session),
            position=ac_world_to_map3(gr.carCoordinates),
            velocity_world=v_world, velocity_local=v_local, speed=speed,
            yaw=yaw, yaw_source=yaw_src, pitch=pitch, roll=roll,
            yaw_rate=float(w_local[2]), angular_velocity_local=w_local,
            acceleration_local=acc,
            heading_raw=float(ph.heading), pitch_raw=float(ph.pitch), roll_raw=float(ph.roll),
            rpm=float(ph.rpms), max_rpm=max_rpm, gear=int(ph.gear) - 1,
            steer=STEER_SIGN_AC_TO_PROJECT * float(ph.steerAngle),
            steer_raw=float(ph.steerAngle),
            throttle=float(ph.gas), brake=float(ph.brake), clutch=float(ph.clutch),
            fuel=float(ph.fuel),
            wheel_slip=_arr4(ph.wheelSlip), wheel_load=_arr4(ph.wheelLoad),
            slip_ratio=_arr4(ph.slipRatio), slip_angle=_arr4(ph.slipAngle),
            tyre_temp_core=_arr4(ph.tyreCoreTemperature), tyre_temp_surface=_arr4(ph.tyreTemp),
            tyre_temp_imo=np.stack([_arr4(ph.tyreTempI), _arr4(ph.tyreTempM),
                                    _arr4(ph.tyreTempO)], axis=1),
            wheel_angular_speed=_arr4(ph.wheelAngularSpeed),
            spline=float(gr.normalizedCarPosition), lap_count=int(gr.completedLaps),
            lap_time_current=_ms_to_s(gr.iCurrentTime) if gr.iCurrentTime > 0 else 0.0,
            lap_time_last=_ms_to_s(gr.iLastTime), lap_time_best=_ms_to_s(gr.iBestTime),
            tyres_out=int(ph.numberOfTyresOut), in_pit=bool(gr.isInPit),
            in_pitlane=bool(gr.isInPitLane),
            tc_level=float(ph.tc), abs_level=float(ph.abs),
            # Only the explicit 1.14+ flags: the meaning of the older float
            # 'tc'/'abs' (setting vs intervention) is not documented reliably.
            tc_active=bool(ph.tcinAction),
            abs_active=bool(ph.absInAction),
            kers_charge=float(ph.kersCharge), kers_input=float(ph.kersInput),
            kers_current_kj=float(ph.kersCurrentKJ),
            ers_recovery_level=int(ph.ersRecoveryLevel), ers_power_level=int(ph.ersPowerLevel),
            ers_heat_charging=bool(ph.ersHeatCharging), ers_is_charging=bool(ph.ersIsCharging),
            car_damage=np.array(list(ph.carDamage), dtype=np.float64),
            is_ai_controlled=bool(ph.isAIControlled),
            car_model=st.carModel, track=st.track, track_config=st.trackConfiguration,
            track_length=track_len, plugin=plugin,
        )
