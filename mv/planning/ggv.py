"""The car's performance envelope: the "g-g-v diagram".

What it is
----------
At any speed ``v`` a car can only produce a limited combination of sideways
acceleration ``ay`` (cornering) and forward/backward acceleration ``ax``
(driving/braking).  Plot all combinations it *can* reach and you get a blob in
the (ay, ax) plane - the "g-g diagram".  The blob changes with speed (more
downforce -> more grip, more drag and less engine force -> less forward
acceleration), so stacking the blobs for all speeds gives the g-g-v diagram.

A lap-time calculation only needs the *edge* of the blob:

* ``ay_max(v)``            the most cornering the car can do at speed v;
* ``ax_max(v, ay)``        the hardest it can accelerate while cornering at ay;
* ``ax_min(v, ay)``        the hardest it can brake while cornering at ay.

How it is computed here
-----------------------
We use the same physics as the training simulator (``mv/sim/vehicle_sim.py``)
but solve it for a *steady state* instead of integrating it in time:

1. Pick a speed ``v``, a lateral acceleration ``ay`` and a pedal position ``u``
   (``u > 0`` throttle, ``u < 0`` brake).
2. Guess ``ax``.  Compute the load on each of the four tyres: static weight,
   downforce, longitudinal load transfer (from ``ax``) and lateral load
   transfer (from ``ay``).
3. From the loads get every tyre's peak grip (load-sensitive friction) and the
   longitudinal slip the pedal asks of it (engine torque through the gearbox,
   brakes with their front/rear bias, rolling resistance; traction control
   and ABS cap the slip).  The combined-slip tyre model then gives the
   longitudinal and the sideways force of each tyre for a given slip angle.
4. The sideways force the front and the rear axle must deliver follows from
   ``m * ay`` and the requirement that the car does not start to spin (zero
   yaw moment).  If either axle needs more than it can deliver at the slip
   angle of peak grip, this (v, ay, u) is not achievable.
5. Sum the forces along the direction of travel -> a new ``ax``.  Repeat 2-5
   until ``ax`` stops changing (a fixed-point iteration: the loads depend on
   the acceleration and the acceleration depends on the loads).

Searching over ``u`` and ``ay`` (by bisection) gives the edge of the envelope.

Deliberate simplifications (see docs/08-reference-lap.md for their effect)
---------------------------------------------------------------------------
* Steady state: no yaw dynamics, no actuator lag, load transfer is instant.
* One pedal at a time (the simulator would allow throttle and brake together).
* Both tyres of an axle work at the same slip angle, and never beyond the slip
  angle of peak cornering grip (no drifting).
* The achievable pedal range is assumed to be one piece: full pedal is used
  when it is achievable, otherwise the limit is found by bisection outwards
  from the pure-cornering pedal position.  (Near the corners of the envelope
  an axle can sit within 1-3 % of its limit over a wide pedal range, so the
  exact position of the envelope's "cliff" is uncertain by about that much.)
* The gear is the one the simulator's automatic gearbox selects while
  accelerating through speed ``v``; shift time is handled in the speed profile.
* The battery is assumed to stay above its minimum state of charge.

Validation against the simulator (see tests/test_planning.py and the doc):
one simulator step started from the predicted steady state reproduces the
predicted acceleration within ~0.05 m/s^2 with ~zero yaw acceleration; the
steady cornering limit agrees with a simulated skid-pad within 0.3 %;
0-200 km/h and 200-0 km/h agree within 1-2 %.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mv.vehicle.params import G, VehicleParams

__all__ = ["GGV", "build_ggv", "PowertrainMap", "QuasiSteadyCar"]

RPM_PER_RADS = 30.0 / math.pi
_TABLE = 256                 # samples per torque table (same resolution as the simulator)
_FZ_MIN = 50.0               # the simulator never lets a tyre load drop below this (N)
_SLOPE_FLOOR = 0.15          # see QuasiSteadyCar.solve (damped Newton step for the axle slips)
# Damping of the fixed-point iteration: bold first steps, cautious later ones (operating points
# right at the grip limit react very strongly to small changes and need the small steps).
_RELAX = tuple(max(0.9 * 0.95 ** k, 0.15) for k in range(200))
_KAPPA_MAX = 0.15            # tightest path curvature considered (1/m); only matters below ~30 km/h


def _pacejka_bc(peak_slip_deg: float, falloff: float) -> tuple[float, float]:
    """Magic-formula shape factors (B, C) from peak slip angle and sliding grip level.

    F = D * sin(C * atan(B * slip)).  Same definition as the simulator.
    """
    c = (2.0 / math.pi) * (math.pi - math.asin(min(max(falloff, 0.3), 0.9999)))
    b = math.tan(math.pi / (2.0 * c)) / math.radians(peak_slip_deg)
    return b, c


# ==========================================================================
# Powertrain: axle drive force as a function of speed and throttle
# ==========================================================================
class PowertrainMap:
    """Drive force at the front and rear axle versus speed and throttle.

    Follows the simulator's rules: torque tables with power cap and rev
    limiter, a launch rpm for combustion engines (slipping clutch), engine
    braking at closed throttle, the battery's discharge power limit, gearbox
    efficiency, and the automatic gearbox's shift speeds ("change up as soon
    as the next gear pulls at least as hard, and never later than 99 % of the
    rev limit").
    """

    def __init__(self, vp: VehicleParams):
        pt = vp.powertrain
        self.vp = vp
        self.gb_names = list(pt.gearboxes)
        self.r_front, self.r_rear = vp.tire_front.radius_m, vp.tire_rear.radius_m
        self.units = []
        for u in pt.units:
            top = u.rpm_max * 1.02
            grid = np.linspace(0.0, top, _TABLE)
            tq = np.interp(grid, u.rpm, u.torque_nm)
            if u.max_power_kw > 0:
                tq = np.minimum(tq, u.max_power_kw * 1000.0 / np.maximum(grid / RPM_PER_RADS, 1e-3))
            tq[grid > u.rpm_max] = 0.0
            gb = pt.gearboxes[u.gearbox]
            self.units.append(dict(
                name=u.name, grid=grid, table=tq, top=top, gb=self.gb_names.index(u.gearbox),
                front=u.front_frac, motor=(u.kind == "motor"), eff=u.efficiency, coast=u.coast_torque_nm,
                rad=u.front_frac * self.r_front + (1.0 - u.front_frac) * self.r_rear,
                launch=0.0 if u.kind == "motor" else u.rpm_idle + 0.35 * (u.rpm_max - u.rpm_idle),
                gb_eff=gb.efficiency, rpm_max=u.rpm_max))
        self.ratios = [np.asarray(pt.gearboxes[n].ratios, float) * pt.gearboxes[n].final_drive for n in self.gb_names]
        self.shift_time_up = [pt.gearboxes[n].upshift_time_s for n in self.gb_names]
        self.has_motor = any(u["motor"] for u in self.units)
        bat = pt.battery
        self.bat_discharge_w = bat.max_discharge_kw * 1000.0 if bat else 0.0
        self.bat_regen_w = bat.max_regen_kw * 1000.0 if bat else 0.0
        self.bat_usable_j = (bat.soc_init - bat.soc_min) * bat.capacity_kwh * 3.6e6 if bat else 0.0
        self.upshift_v = [self._upshift_speeds(i) for i in range(len(self.gb_names))]

    # ---- gear selection ---------------------------------------------------
    def _gear_force(self, gb: int, gear: int, v: np.ndarray) -> np.ndarray:
        """Full-throttle wheel force of all units on gearbox ``gb`` in ``gear`` (no battery limit)."""
        total = np.zeros_like(v)
        for u in self.units:
            if u["gb"] != gb:
                continue
            ratio = self.ratios[gb][gear]
            rpm = np.maximum(v / u["rad"] * ratio * RPM_PER_RADS, u["launch"])
            total += np.interp(rpm, u["grid"], u["table"]) * ratio / u["rad"]
        return total

    def _upshift_speeds(self, gb: int) -> np.ndarray:
        """Speed (m/s) at which the automatic gearbox leaves each gear (last gear: never)."""
        ratios = self.ratios[gb]
        v = np.arange(0.5, 160.0, 0.1)
        up = np.full(len(ratios), np.inf)
        for g in range(len(ratios) - 1):
            better = (self._gear_force(gb, g + 1, v) >= 0.999 * self._gear_force(gb, g, v)) & (v > 2.0)
            if better.any():
                up[g] = v[np.argmax(better)]
            v_limiter = min(0.99 * u["rpm_max"] / (ratios[g] * RPM_PER_RADS) * u["rad"]
                            for u in self.units if u["gb"] == gb)
            up[g] = min(up[g], v_limiter)
        return up

    def gear(self, gb: int, v: np.ndarray) -> np.ndarray:
        """Gear index the gearbox is in while accelerating through speed ``v``."""
        up = self.upshift_v[gb][:-1]
        return (np.asarray(v)[..., None] > up).sum(-1)

    # ---- forces -------------------------------------------------------------
    def _unit_torques(self, v: np.ndarray, thr: np.ndarray):
        """Per-unit (crank torque, total ratio, angular speed, full-throttle torque) at speed v."""
        out = []
        for u in self.units:
            ratio = self.ratios[u["gb"]][self.gear(u["gb"], v)]
            rpm = v / u["rad"] * ratio * RPM_PER_RADS
            full = np.interp(np.maximum(rpm, u["launch"]), u["grid"], u["table"])
            torque = full * thr
            if not u["motor"]:      # engine braking with the throttle closed
                torque = torque - u["coast"] * np.clip(rpm / u["top"], 0.0, 1.0) * (1.0 - thr) ** 2
            out.append((torque, ratio, rpm / RPM_PER_RADS, full))
        return out

    def _battery_scale(self, per_unit) -> tuple[np.ndarray, np.ndarray]:
        """Factor (<= 1) applied to motor torque so the battery power limit is respected."""
        p_elec = 0.0
        for u, (torque, _ratio, omega, _full) in zip(self.units, per_unit):
            if u["motor"]:
                p_elec = p_elec + np.clip(torque * omega, 0.0, None) / u["eff"]
        scale = np.minimum(self.bat_discharge_w / np.maximum(p_elec, 1.0), 1.0)
        return scale, p_elec * scale

    def axle_forces(self, v, thr) -> tuple[np.ndarray, np.ndarray]:
        """(front axle, rear axle) drive force in N; negative = engine braking."""
        v, thr = np.broadcast_arrays(np.asarray(v, float), np.asarray(thr, float))
        per_unit = self._unit_torques(v, thr)
        scale = self._battery_scale(per_unit)[0] if self.has_motor else 1.0
        f_front, f_rear = np.zeros(v.shape), np.zeros(v.shape)
        for u, (torque, ratio, _omega, _full) in zip(self.units, per_unit):
            wheel_torque = torque * (scale if u["motor"] else 1.0) * ratio * u["gb_eff"]
            f_front = f_front + wheel_torque * u["front"] / self.r_front
            f_rear = f_rear + wheel_torque * (1.0 - u["front"]) / self.r_rear
        return f_front, f_rear

    def gearbox_force(self, gb: int, v) -> np.ndarray:
        """Full-throttle drive force (N) of everything connected to gearbox ``gb`` (for shift losses)."""
        v = np.asarray(v, float)
        per_unit = self._unit_torques(v, np.ones_like(v))
        scale = self._battery_scale(per_unit)[0] if self.has_motor else 1.0
        total = np.zeros(v.shape)
        for u, (torque, ratio, _omega, _full) in zip(self.units, per_unit):
            if u["gb"] == gb:
                total = total + torque * (scale if u["motor"] else 1.0) * ratio * u["gb_eff"] / (
                    u["front"] * self.r_front + (1.0 - u["front"]) * self.r_rear)
        return total

    def electric_power(self, v, thr) -> np.ndarray:
        """Battery discharge power (W) at speed v and throttle thr (0 for cars without motors)."""
        v, thr = np.broadcast_arrays(np.asarray(v, float), np.asarray(thr, float))
        if not self.has_motor:
            return np.zeros(v.shape)
        return self._battery_scale(self._unit_torques(v, thr))[1]

    def regen_capacity(self, v) -> np.ndarray:
        """Power (W) the motors could turn back into electricity at full brake pedal (before the battery limit)."""
        v = np.asarray(v, float)
        if not self.has_motor:
            return np.zeros(v.shape)
        cap = np.zeros(v.shape)
        for u, (_t, _ratio, omega, full) in zip(self.units, self._unit_torques(v, np.ones_like(v))):
            if u["motor"]:
                cap = cap + full * omega * u["eff"]
        return cap


# ==========================================================================
# Quasi-steady-state four-wheel force balance
# ==========================================================================
class QuasiSteadyCar:
    """Steady-state force balance of the four-wheel car (NumPy, broadcasting).

    Wheel order is the simulator's: front-left, front-right, rear-left,
    rear-right.  The car is assumed to turn LEFT (``ay >= 0``); it is
    left/right symmetric, so right-hand corners use the same numbers.

    Tyre model (identical to the simulator's "similarity" combined-slip model)
    ---------------------------------------------------------------------------
    Each tyre has one grip budget.  Two normalised slips say how hard it works:
    ``kappa_n`` for traction/braking and ``alpha_n`` for cornering, both scaled
    so that 1.0 is the slip of peak grip.  With the combined slip
    ``sigma = sqrt(kappa_n^2 + alpha_n^2)`` and the magic-formula curve
    ``P(s) = sin(C * atan(Bn * s))`` (``P(1) = 1``)::

        Fx = Dx * P(sigma) * kappa_n / sigma        Fy = Dy * P(sigma) * alpha_n / sigma

    ``Dx``, ``Dy`` are the load-dependent peak forces.  ``kappa_n`` follows
    from the pedal: it is the slip that would deliver the *requested* force in
    a straight line, capped by traction control / ABS.  ``alpha_n`` follows
    from the slip angle and is the same for both tyres of an axle.

    ``slip_cap`` is the largest ``alpha_n`` the reference is allowed to use.
    At 1.0 (default) no tyre is driven past the slip angle of peak cornering
    grip - the car is never "drifted".
    """

    def __init__(self, vp: VehicleParams, grip: float = 1.0, slip_cap: float = 1.0):
        c, tf, tr = vp.chassis, vp.tire_front, vp.tire_rear
        self.vp, self.grip, self.mass, self.slip_cap = vp, float(grip), c.mass_kg, float(slip_cap)
        self.pt = PowertrainMap(vp)
        self.a, self.b, self.wheelbase = c.a, c.b, c.wheelbase_m
        self.wheel_y = np.array([c.track_front_m, -c.track_front_m, c.track_rear_m, -c.track_rear_m]) / 2.0
        self.fz_static = np.array([vp.static_fz_front] * 2 + [vp.static_fz_rear] * 2)
        self.aero_fz = 0.5 * vp.aero.air_density * np.array([vp.aero.cla_front_m2 / 2] * 2 + [vp.aero.cla_rear_m2 / 2] * 2)
        self.drag_k = 0.5 * vp.aero.air_density * vp.aero.cda_m2
        # load transfer per m/s^2: longitudinal (front loses, rear gains under acceleration) ...
        k_long = c.mass_kg * c.cg_height_m / c.wheelbase_m / 2.0
        self.k_long = np.array([-k_long, -k_long, k_long, k_long])
        # ... and lateral, shared between the axles by roll stiffness (left turn loads the right wheels)
        kf = c.mass_kg * c.cg_height_m / c.track_front_m * c.roll_stiffness_front_frac
        kr = c.mass_kg * c.cg_height_m / c.track_rear_m * (1.0 - c.roll_stiffness_front_frac)
        self.k_lat = np.array([-kf, kf, -kr, kr])
        self.mu_lat = np.array([tf.mu_lat] * 2 + [tr.mu_lat] * 2)
        self.mu_long = np.array([tf.mu_long] * 2 + [tr.mu_long] * 2)
        self.fz_ref = np.array([tf.fz_ref_n] * 2 + [tr.fz_ref_n] * 2)
        self.exp_lat = np.array([tf.load_exp_lat - 1] * 2 + [tr.load_exp_lat - 1] * 2)
        self.exp_long = np.array([tf.load_exp_long - 1] * 2 + [tr.load_exp_long - 1] * 2)
        self.roll_res = np.array([tf.rolling_resistance] * 2 + [tr.rolling_resistance] * 2)
        radius = np.array([tf.radius_m] * 2 + [tr.radius_m] * 2)
        bk = vp.brakes
        self.brake_force = np.array([bk.max_torque_nm * bk.front_share / 2] * 2
                                    + [bk.max_torque_nm * (1 - bk.front_share) / 2] * 2) / radius
        el = vp.electronics
        self.tc_limit = el.tc_slip_limit if el.traction_control else None
        self.abs_limit = el.abs_slip_limit if el.abs else None
        cf, cr = (_pacejka_bc(t.peak_slip_angle_deg, t.falloff_level)[1] for t in (tf, tr))
        self.tire_c = np.array([cf, cf, cr, cr])
        self.tire_bn = np.tan(np.pi / (2.0 * self.tire_c))          # stiffness for slip normalised to 1 at the peak
        self.tan_peak_f = math.tan(math.radians(tf.peak_slip_angle_deg))
        self.tan_peak_r = math.tan(math.radians(tr.peak_slip_angle_deg))

    # ------------------------------------------------------------------------
    def tyre_loads(self, v, ax, ay) -> np.ndarray:
        """Vertical load (N) on each tyre, shape (..., 4), for BODY-frame accelerations."""
        v, ax, ay = (np.asarray(q, float)[..., None] for q in (v, ax, ay))
        fz = self.fz_static + self.aero_fz * v * v + self.k_long * ax + self.k_lat * ay
        return np.maximum(fz, _FZ_MIN)

    def _grip_curve(self, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``P(sigma) / sigma`` and its derivative for each wheel (last axis = 4 wheels)."""
        bs = self.tire_bn * sigma
        ang = self.tire_c * np.arctan(bs)
        p = np.sin(ang)
        dp = self.tire_c * self.tire_bn * np.cos(ang) / (1.0 + bs * bs)
        return p / sigma, (dp * sigma - p) / (sigma * sigma)

    def solve(self, v, ay, u, n_iter: int = 40, relax=_RELAX, tol: float = 1e-7) -> dict:
        """Force balance at speed ``v``, lateral acceleration ``ay`` and pedal ``u``.

        ``ay`` is the centripetal acceleration ``v^2 / R`` of the path, ``u`` in
        [-1, 1] the pedal (positive = throttle, negative = brake).  All inputs
        broadcast against each other.  Returns a dict with

        ``ax``        acceleration along the direction of travel (m/s^2);
        ``util``      the larger of the two axles' lateral utilisation: the side
                      force the axle must deliver divided by what it could
                      deliver at the largest allowed slip angle.  The operating
                      point is achievable iff ``util <= 1``;
        ``util_front``, ``util_rear``, ``fz``, ``fx``, ``fy``, ``slip_front``,
        ``slip_rear`` (slip angles, rad), ``beta`` (body slip angle), ``steer``
        (road-wheel angle), ``ax_body``, ``ay_body`` and ``residual`` (size of
        the last update, in m/s^2: a convergence check).

        Unknowns of the iteration: the body-frame acceleration ``ax_body`` (it
        sets the tyre loads, which set the forces, which set the acceleration)
        and the two axle slips (found with a damped Newton step per round so
        that each axle delivers exactly the side force the balance requires).

        Why the body frame matters: a car at the limit does not point where it
        is going.  The rear tyres need a slip angle to make force, so the nose
        points a few degrees into the corner (body slip angle ``beta``).  The
        simulator computes load transfer from the accelerations along the
        *body* axes; seen from the body, part of the centripetal acceleration
        points forwards, which unloads the front tyres.  For an understeering
        car that costs a few per cent of cornering grip, so it is modelled.
        """
        v, ay, u = np.broadcast_arrays(np.asarray(v, float), np.asarray(ay, float), np.asarray(u, float))
        thr, brk = np.clip(u, 0.0, 1.0), np.clip(-u, 0.0, 1.0)
        f_front, f_rear = self.pt.axle_forces(v, thr)
        fx_pedal = np.stack([f_front, f_front, f_rear, f_rear], axis=-1) * 0.5 - brk[..., None] * self.brake_force
        fz_fixed = self.fz_static + self.aero_fz * (v * v)[..., None]
        drag = self.drag_k * v * v
        kappa = np.minimum(ay / np.maximum(v, 1.0) ** 2, _KAPPA_MAX)       # path curvature
        m, a, b, wb, cap = self.mass, self.a, self.b, self.wheelbase, self.slip_cap
        ax_b = np.zeros(v.shape)                    # acceleration along the body x axis (iterated)
        an_f, an_r = np.zeros(v.shape), np.zeros(v.shape)     # normalised axle slips alpha_n (iterated)
        residual = np.full(v.shape, np.inf)
        for it in range(n_iter):
            slip_f, slip_r = np.arctan(an_f * self.tan_peak_f), np.arctan(an_r * self.tan_peak_r)
            beta = b * kappa - slip_r               # body slip angle: velocity direction relative to the nose
            steer = slip_f + beta + a * kappa       # front road-wheel angle that produces slip_f
            cos_b, sin_b = np.cos(beta), np.sin(beta)
            ay_b = (ay + ax_b * sin_b) / cos_b      # centripetal acceleration seen along the body y axis
            fz = np.maximum(fz_fixed + self.k_long * ax_b[..., None] + self.k_lat * ay_b[..., None], _FZ_MIN)
            load = fz / self.fz_ref
            d_lat = self.mu_lat * load ** self.exp_lat * fz * self.grip
            d_long = self.mu_long * load ** self.exp_long * fz * self.grip
            # ---- longitudinal slip each wheel runs at (from the requested force) ----
            req = (fx_pedal - self.roll_res * fz) / d_long
            req_abs = np.abs(req)
            kap = np.tan(np.arcsin(np.minimum(req_abs, 1.0)) / self.tire_c) / self.tire_bn
            kap = kap + 6.0 * np.maximum(req_abs - 1.0, 0.0)               # wheelspin / lock-up beyond the peak
            if self.tc_limit is not None:
                kap = np.where(req > 0.0, np.minimum(kap, self.tc_limit), kap)
            if self.abs_limit is not None:
                kap = np.where(req < 0.0, np.minimum(kap, self.abs_limit), kap)
            # ---- tyre forces at the current axle slips, and at the largest allowed slip ----
            an = np.stack([an_f, an_f, an_r, an_r], axis=-1)
            sigma = np.maximum(np.sqrt(kap * kap + an * an), 1e-6)
            g, dg = self._grip_curve(sigma)
            fx = d_long * g * kap * np.sign(req)
            fy = d_lat * g * an
            dfy = d_lat * (g + dg * an * an / sigma)                       # d fy / d alpha_n
            sig_cap = np.sqrt(kap * kap + cap * cap)
            fy_cap = d_lat * np.sin(self.tire_c * np.arctan(self.tire_bn * sig_cap)) / sig_cap * cap
            # ---- what the balance of forces and moments asks of each axle ----
            cos_d, sin_d = np.cos(steer), np.sin(steer)
            # Forces along the body x axis.  The front tyres are steered, so their side
            # force points partly backwards ("cornering drag").
            fxb = fx.copy()
            fxb[..., :2] = fx[..., :2] * cos_d[..., None] - fy[..., :2] * sin_d[..., None]
            # Unequal left/right longitudinal forces twist the car; the axles' side
            # forces must cancel that moment as well as provide m * ay.
            mz = -(self.wheel_y * fxb).sum(-1)
            side_f, side_r = (m * ay_b * b - mz) / wb, (m * ay_b * a + mz) / wb   # along body y
            need_f = (side_f - (fx[..., 0] + fx[..., 1]) * sin_d) / cos_d         # in the tyres' own frame
            need_r = side_r
            have_f, have_r = fy[..., 0] + fy[..., 1], fy[..., 2] + fy[..., 3]
            cap_f, cap_r = fy_cap[..., 0] + fy_cap[..., 1], fy_cap[..., 2] + fy_cap[..., 3]
            util_f, util_r = need_f / cap_f, need_r / cap_r
            # ---- update: Newton step for the slips (slope floored: the force curve is flat
            # near its peak and an undamped step would overshoot wildly), then damping ----
            slope_f = np.maximum(dfy[..., 0] + dfy[..., 1], _SLOPE_FLOOR * cap_f / cap)
            slope_r = np.maximum(dfy[..., 2] + dfy[..., 3], _SLOPE_FLOOR * cap_r / cap)
            an_f_new = np.clip(an_f + (need_f - have_f) / slope_f, -cap, cap)
            an_r_new = np.clip(an_r + (need_r - have_r) / slope_r, -cap, cap)
            ax_new = (fxb.sum(-1) - drag) / m
            residual = np.maximum(np.abs(ax_new - ax_b), np.maximum(np.abs(an_f_new - an_f), np.abs(an_r_new - an_r)))
            w = relax[min(it, len(relax) - 1)] if not np.isscalar(relax) else relax
            ax_b = ax_b + w * (ax_new - ax_b)
            an_f = an_f + w * (an_f_new - an_f)
            an_r = an_r + w * (an_r_new - an_r)
            if residual.size == 0 or residual.max() < tol:
                break
        ax_path = ax_b * cos_b + ay_b * sin_b       # rate of change of speed
        return dict(ax=ax_path, util=np.maximum(np.abs(util_f), np.abs(util_r)), util_front=util_f, util_rear=util_r,
                    fz=fz, fx=fx, fy=fy, slip_front=slip_f, slip_rear=slip_r, beta=beta, steer=steer,
                    ax_body=ax_b, ay_body=ay_b, residual=residual)

    def max_lateral(self, v, u, ay_hi: float = 6.0 * G, n_bisect: int = 17, **kw) -> np.ndarray:
        """Largest achievable ``ay`` at speed ``v`` with the pedal held at ``u`` (bisection)."""
        v, u = np.broadcast_arrays(np.asarray(v, float), np.asarray(u, float))
        lo, hi = np.zeros(v.shape), np.full(v.shape, ay_hi)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            ok = self.solve(v, mid, u, **kw)["util"] <= 1.0
            lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
        return lo


# ==========================================================================
# The tabulated envelope
# ==========================================================================
@dataclass
class GGV:
    """Tabulated g-g-v envelope.

    ``ax_max`` and ``ax_min`` are stored on a grid of speed ``v`` and of the
    angle ``theta = asin(ay / ay_max(v))``.  Using the angle instead of the
    ratio puts more samples where the envelope bends sharply (close to the
    cornering limit ax ~ cos(theta)), so linear interpolation stays accurate.
    """
    v: np.ndarray             # (Nv,) speed grid, uniform, starts at 0
    ay_max: np.ndarray        # (Nv,) maximum lateral acceleration
    theta: np.ndarray         # (Nt,) uniform grid from 0 to pi/2
    ax_max: np.ndarray        # (Nv, Nt) maximum forward acceleration
    ax_min: np.ndarray        # (Nv, Nt) maximum braking (negative numbers)
    u_apex: np.ndarray        # (Nv,) pedal position at which ay_max is reached
    mass: float = 0.0
    name: str = ""
    meta: dict = field(default_factory=dict)

    # ---- look-ups -------------------------------------------------------------
    def ay_limit(self, v) -> np.ndarray:
        return np.interp(v, self.v, self.ay_max)

    def _lookup(self, table: np.ndarray, v, ay) -> np.ndarray:
        v = np.clip(np.asarray(v, float), self.v[0], self.v[-1])
        rho = np.clip(np.abs(ay) / np.interp(v, self.v, self.ay_max), 0.0, 1.0)
        fv = (v - self.v[0]) / (self.v[1] - self.v[0])
        ft = np.arcsin(rho) / (self.theta[1] - self.theta[0])
        i = np.minimum(fv.astype(int), len(self.v) - 2)
        j = np.minimum(ft.astype(int), len(self.theta) - 2)
        fv, ft = fv - i, ft - j
        return ((1 - fv) * (1 - ft) * table[i, j] + fv * (1 - ft) * table[i + 1, j]
                + (1 - fv) * ft * table[i, j + 1] + fv * ft * table[i + 1, j + 1])

    def ax_limits(self, v, ay) -> tuple[np.ndarray, np.ndarray]:
        """(strongest braking, strongest acceleration) available at speed v while cornering at ay."""
        return self._lookup(self.ax_min, v, ay), self._lookup(self.ax_max, v, ay)

    def corner_speed(self, kappa) -> np.ndarray:
        """Highest speed at which a corner of curvature ``kappa`` (1/m) can be taken.

        Solves ``v^2 * |kappa| = ay_max(v)``.  Because downforce makes ay_max
        grow with speed this is not simply sqrt(ay_max / kappa); we invert the
        curve "tightest drivable curvature versus speed" instead.
        """
        v = np.linspace(self.v[0], self.v[-1], 20 * (len(self.v) - 1) + 1)[1:]
        k_max = np.interp(v, self.v, self.ay_max) / (v * v)
        k_max = np.minimum.accumulate(k_max)                # guard: must fall with speed
        kappa = np.abs(np.asarray(kappa, float))
        # np.interp needs increasing x: flip (k_max falls with v)
        out = np.interp(kappa, k_max[::-1], v[::-1], left=v[-1], right=v[0])
        return out

    def ay_steady(self, v) -> np.ndarray:
        """Largest lateral acceleration that can be held at CONSTANT speed.

        ``ay_max`` is usually reached with the car slowing down slightly: a
        tyre at its limit drags, and the throttle needed to cancel that drag
        would use up grip on the driven axle.  A car circling at steady speed
        therefore corners a few per cent below ``ay_max``; this is the number a
        skid-pad test measures.
        """
        can_hold = (self.ax_max >= 0.0) & (self.ax_min <= 0.0)                 # (Nv, Nt)
        last = len(self.theta) - 1 - np.argmax(can_hold[:, ::-1], axis=1)       # last angle that can hold speed
        rows = np.arange(len(self.v))
        nxt = np.minimum(last + 1, len(self.theta) - 1)
        # interpolate to the zero crossing of whichever limit ends the constant-speed range
        a0, a1 = self.ax_max[rows, last], self.ax_max[rows, nxt]
        b0, b1 = self.ax_min[rows, last], self.ax_min[rows, nxt]
        frac = np.where(a1 < 0.0, a0 / np.maximum(a0 - a1, 1e-12),
                        np.where(b1 > 0.0, -b0 / np.maximum(b1 - b0, 1e-12), 1.0))
        frac = np.where(nxt == last, 0.0, np.clip(frac, 0.0, 1.0))
        theta = self.theta[last] + frac * (self.theta[nxt] - self.theta[last])
        return np.interp(v, self.v, np.sin(theta) * self.ay_max)

    def top_speed(self) -> float:
        """Speed where full-throttle acceleration in a straight line drops to zero."""
        a = self.ax_max[:, 0]
        pos = np.nonzero(a > 0)[0]
        if len(pos) == 0:
            return 0.0
        i = pos[-1]
        if i == len(a) - 1:
            return float(self.v[-1])
        return float(self.v[i] + (self.v[i + 1] - self.v[i]) * a[i] / (a[i] - a[i + 1]))

    def gg_slice(self, v: float, n: int = 41) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Edge of the g-g diagram at speed v: (ay, ax_max, ax_min) arrays for plotting."""
        ay = np.sin(np.linspace(0.0, np.pi / 2, n)) * float(self.ay_limit(v))
        lo, hi = self.ax_limits(np.full(n, v), ay)
        return ay, hi, lo


def build_ggv(vp: VehicleParams, v_max: float = 116.0, dv: float = 2.0, n_theta: int = 33,
              grip: float = 1.0, n_pedal: int = 33, slip_cap: float = 1.0, **kw) -> GGV:
    """Compute the g-g-v envelope of a car.

    Two stages:

    1. ``ay_max(v)``: for every speed and a grid of pedal positions find the
       largest sustainable ``ay`` (bisection); the best pedal position gives
       the cornering limit.  It is usually a whisker of throttle - enough to
       cancel drag without using up much of the driven tyres' grip.
    2. For a fan of lateral accelerations ``ay = sin(theta) * ay_max`` find the
       pedal range that is still achievable (bisection from the cornering
       pedal towards full throttle and full brake) and record the largest and
       smallest ``ax`` inside it.
    """
    car = QuasiSteadyCar(vp, grip, slip_cap)
    v = np.arange(0.0, v_max + 0.5 * dv, dv)
    nv = len(v)

    # ---- stage 1: cornering limit --------------------------------------------
    pedal = np.linspace(-1.0, 1.0, n_pedal)
    ay_cap = car.max_lateral(v[:, None], pedal[None, :], **kw)                      # (Nv, Np)
    best = ay_cap.argmax(1)
    step = pedal[1] - pedal[0]
    fine = pedal[best][:, None] + np.linspace(-step, step, 13)[None, :]       # refine around the best pedal
    fine = np.clip(fine, -1.0, 1.0)
    ay_fine = car.max_lateral(v[:, None], fine, **kw)
    best_fine = ay_fine.argmax(1)
    rows = np.arange(nv)
    ay_max, u_apex = ay_fine[rows, best_fine], fine[rows, best_fine]

    # ---- stage 2: acceleration and braking limits versus ay -----------------------
    theta = np.linspace(0.0, np.pi / 2, n_theta)
    ay = np.sin(theta)[None, :] * ay_max[:, None] * (1.0 - 1e-6)             # (Nv, Nt)
    vv = np.broadcast_to(v[:, None], ay.shape)
    uc = np.broadcast_to(u_apex[:, None], ay.shape)

    def pedal_limit(end: float) -> np.ndarray:
        """Pedal closest to ``end`` (+1 or -1) that is still achievable at each (v, ay)."""
        ok_end = car.solve(vv, ay, np.full(ay.shape, end), **kw)["util"] <= 1.0
        good, bad = uc.copy(), np.full(ay.shape, end)
        for _ in range(14):
            mid = 0.5 * (good + bad)
            ok = car.solve(vv, ay, mid, **kw)["util"] <= 1.0
            good, bad = np.where(ok, mid, good), np.where(ok, bad, mid)
        return np.where(ok_end, end, good)

    u_hi, u_lo = pedal_limit(1.0), pedal_limit(-1.0)
    # ax normally grows with the pedal, so the extremes sit at the ends of the
    # achievable range.  Without TC/ABS an over-driven tyre delivers *less*, so
    # sample the whole range and take the true extremes.
    frac = np.linspace(0.0, 1.0, 9)[:, None, None]
    ax_hi = car.solve(vv[None], ay[None], uc[None] + frac * (u_hi - uc)[None], **kw)
    ax_lo = car.solve(vv[None], ay[None], uc[None] + frac * (u_lo - uc)[None], **kw)
    k_hi, k_lo = ax_hi["ax"].argmax(0)[None], ax_lo["ax"].argmin(0)[None]
    ax_max, ax_min = np.take_along_axis(ax_hi["ax"], k_hi, 0)[0], np.take_along_axis(ax_lo["ax"], k_lo, 0)[0]
    residual = np.concatenate([np.take_along_axis(ax_hi["residual"], k_hi, 0).ravel(),
                               np.take_along_axis(ax_lo["residual"], k_lo, 0).ravel()])

    g = GGV(v=v, ay_max=ay_max, theta=theta, ax_max=ax_max, ax_min=ax_min, u_apex=u_apex,
            mass=vp.chassis.mass_kg, name=vp.name,
            meta=dict(grip=grip, slip_cap=slip_cap, residual_median=float(np.median(residual)), residual_p99=float(np.percentile(residual, 99)),
                      upshift_speeds=[u.tolist() for u in car.pt.upshift_v]))
    g.meta["top_speed"] = g.top_speed()
    return g
