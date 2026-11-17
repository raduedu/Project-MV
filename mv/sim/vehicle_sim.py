"""Batched vehicle dynamics ("the physics") for thousands of cars at once.

Model tier 1: planar four-wheel model
-------------------------------------
Each car is a rigid body moving in the plane with three degrees of freedom:
forward speed ``vx``, sideways speed ``vy`` and yaw rate ``r``.  Four tyres
generate the forces.  What is modelled:

* Tyre forces from slip angle (Pacejka magic formula) with load sensitivity
  and a combined-slip model that shares one grip budget between cornering and
  traction/braking.
* Vertical load on each tyre: static weight, aerodynamic downforce, and
  load transfer under acceleration, braking and cornering.
* Aerodynamic drag, rolling resistance.
* Powertrain: any number of engines/motors, multi-speed gearboxes with shift
  time, rev limiter, engine braking, battery power limits and state of charge.
* Brakes with front/rear bias, ABS and traction control as *force limits*.
* Actuator delays: steering rate limit, throttle and brake lag.

What is NOT modelled yet (planned "tier 2", see docs/02-roadmap.md): wheel
rotation as its own state (so wheelspin/lock-up are approximated by the force
limits), suspension travel, tyre temperature and wear, differential locking,
road slope and banking.

Conventions: body frame x forward, y left; yaw and steering positive to the
left (counter-clockwise).  All tensors have one entry per simulated car; wheel
tensors have shape (N, 4) in the order front-left, front-right, rear-left,
rear-right.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from mv.vehicle.params import G, VehicleParams

RPM_PER_RADS = 30.0 / math.pi
_TABLE = 256                       # samples per torque lookup table


def _pacejka_bc(peak_slip_deg: float, falloff: float) -> tuple[float, float]:
    """Magic-formula shape factors B and C from two intuitive numbers.

    F = D * sin(C * atan(B * slip)).  For large slip F -> D * sin(C*pi/2), which
    must equal ``falloff * D``; the peak (F = D) is where C*atan(B*slip) = pi/2.
    """
    c = (2.0 / math.pi) * (math.pi - math.asin(min(max(falloff, 0.3), 0.9999)))
    b = math.tan(math.pi / (2.0 * c)) / math.radians(peak_slip_deg)
    return b, c


class VehicleSim:
    """State and physics of ``num_envs`` identical cars (with per-car variations)."""

    def __init__(self, vp: VehicleParams, num_envs: int, device: torch.device | str = "cpu", dt: float = 0.004):
        self.vp, self.n, self.device, self.dt = vp, num_envs, torch.device(device), dt
        dev, f32 = self.device, torch.float32
        c = vp.chassis

        def t(a):
            return torch.as_tensor(np.asarray(a, dtype=np.float32), device=dev)

        # ---- geometry: wheel positions in the body frame ---------------------
        self.wheel_x = t([c.a, c.a, -c.b, -c.b])
        self.wheel_y = t([c.track_front_m / 2, -c.track_front_m / 2, c.track_rear_m / 2, -c.track_rear_m / 2])
        self.is_front = t([1, 1, 0, 0])
        self.side = t([1, -1, 1, -1])          # +1 left wheel, -1 right wheel
        self.radius = t([vp.tire_front.radius_m] * 2 + [vp.tire_rear.radius_m] * 2)
        self.fz_static = t([vp.static_fz_front] * 2 + [vp.static_fz_rear] * 2)

        # ---- tyres -----------------------------------------------------------
        tf, tr = vp.tire_front, vp.tire_rear
        bf, cf = _pacejka_bc(tf.peak_slip_angle_deg, tf.falloff_level)
        br, cr = _pacejka_bc(tr.peak_slip_angle_deg, tr.falloff_level)
        self.tire_b, self.tire_c = t([bf, bf, br, br]), t([cf, cf, cr, cr])
        self.slip_peak = torch.deg2rad(t([tf.peak_slip_angle_deg] * 2 + [tr.peak_slip_angle_deg] * 2))
        self.tan_slip_peak = torch.tan(self.slip_peak)
        # magic-formula stiffness for slip normalised to 1.0 at the peak
        self.tire_bn = torch.tan(math.pi / (2.0 * self.tire_c))
        self.mu_lat = t([tf.mu_lat] * 2 + [tr.mu_lat] * 2)
        self.mu_long = t([tf.mu_long] * 2 + [tr.mu_long] * 2)
        self.fz_ref = t([tf.fz_ref_n] * 2 + [tr.fz_ref_n] * 2)
        self.exp_lat = t([tf.load_exp_lat - 1] * 2 + [tr.load_exp_lat - 1] * 2)
        self.exp_long = t([tf.load_exp_long - 1] * 2 + [tr.load_exp_long - 1] * 2)
        self.roll_res = t([tf.rolling_resistance] * 2 + [tr.rolling_resistance] * 2)

        # ---- load transfer coefficients ---------------------------------------
        # longitudinal: m*ax*h/L moves from the front axle to the rear axle
        self.k_long = c.mass_kg * c.cg_height_m / c.wheelbase_m / 2.0          # per wheel, per m/s^2
        # lateral: m*ay*h/track, shared between the axles by roll stiffness
        kf = c.mass_kg * c.cg_height_m / c.track_front_m * c.roll_stiffness_front_frac
        kr = c.mass_kg * c.cg_height_m / c.track_rear_m * (1.0 - c.roll_stiffness_front_frac)
        self.k_lat = t([-kf, kf, -kr, kr])     # turning left (ay > 0) loads the right wheels
        self.aero_fz = 0.5 * vp.aero.air_density * t([vp.aero.cla_front_m2 / 2] * 2 + [vp.aero.cla_rear_m2 / 2] * 2)
        self.drag_k = 0.5 * vp.aero.air_density * vp.aero.cda_m2

        # ---- brakes, steering, electronics -------------------------------------
        b = vp.brakes
        self.brake_torque = t([b.max_torque_nm * b.front_share / 2] * 2 + [b.max_torque_nm * (1 - b.front_share) / 2] * 2)
        self.steer_max = math.radians(vp.steering.max_wheel_angle_deg)
        self.steer_rate = math.radians(vp.steering.max_rate_deg_s)
        self.tc_limit = vp.electronics.tc_slip_limit if vp.electronics.traction_control else None
        self.abs_limit = vp.electronics.abs_slip_limit if vp.electronics.abs else None
        self.k_thr = 1.0 - math.exp(-dt / max(vp.powertrain.throttle_response_time_s, 1e-3))
        self.k_brk = 1.0 - math.exp(-dt / max(b.response_time_s, 1e-3))
        self.k_acc = 1.0 - math.exp(-dt / 0.06)     # how fast the body settles into load transfer

        self._compile_powertrain()

        # ---- state -------------------------------------------------------------
        z = lambda: torch.zeros(num_envs, device=dev, dtype=f32)  # noqa: E731
        self.x, self.y, self.yaw = z(), z(), z()
        self.vx, self.vy, self.r = z(), z(), z()
        self.steer, self.thr, self.brk = z(), z(), z()
        self.ax, self.ay = z(), z()                   # filtered body accelerations
        self.gear = torch.zeros(num_envs, self.num_gb, device=dev, dtype=torch.long)
        self.shift_timer = torch.zeros(num_envs, self.num_gb, device=dev, dtype=f32)
        self.soc = torch.full((num_envs,), self.soc_init, device=dev, dtype=f32)
        # per-car variation ("domain randomisation"), 1.0 = nominal
        self.grip = torch.ones(num_envs, device=dev)
        self.mass_scale = torch.ones(num_envs, device=dev)
        self.power_scale = torch.ones(num_envs, device=dev)
        self.aero_scale = torch.ones(num_envs, device=dev)
        # diagnostics refreshed every step
        self.tire_util = torch.zeros(num_envs, 4, device=dev)    # combined slip of each tyre; 1.0 = peak grip
        self.slip = torch.zeros(num_envs, 4, device=dev)         # slip angle of each tyre (rad)
        self.rpm_frac = z()
        self.power_elec = z()

    # ------------------------------------------------------------------------
    def _compile_powertrain(self) -> None:
        """Turn the powertrain description into lookup tensors."""
        vp, dev = self.vp, self.device
        pt = vp.powertrain
        gb_names = list(pt.gearboxes)
        self.num_gb = len(gb_names)
        self.num_units = len(pt.units)
        gmax = max(len(g.ratios) for g in pt.gearboxes.values())
        ratio = np.zeros((self.num_gb, gmax), np.float32)
        ngears = np.zeros(self.num_gb, np.int64)
        for i, name in enumerate(gb_names):
            g = pt.gearboxes[name]
            tot = np.asarray(g.ratios, np.float32) * g.final_drive
            ratio[i, : len(tot)] = tot
            ratio[i, len(tot):] = tot[-1]
            ngears[i] = len(tot)
        rf, rr = vp.tire_front.radius_m, vp.tire_rear.radius_m
        table = np.zeros((self.num_units, _TABLE), np.float32)
        u_top, u_gb, u_front, u_motor, u_eff, u_pmax, u_coast, u_rad, u_launch, u_gbeff = ([] for _ in range(10))
        for k, u in enumerate(pt.units):
            top = u.rpm_max * 1.02
            grid = np.linspace(0.0, top, _TABLE)
            tq = np.interp(grid, u.rpm, u.torque_nm)
            if u.max_power_kw > 0:
                tq = np.minimum(tq, u.max_power_kw * 1000.0 / np.maximum(grid / RPM_PER_RADS, 1e-3))
            tq[grid > u.rpm_max] = 0.0                       # rev limiter / motor top speed
            table[k] = tq
            gb = pt.gearboxes[u.gearbox]
            u_top.append(top); u_gb.append(gb_names.index(u.gearbox)); u_front.append(u.front_frac)
            u_motor.append(1.0 if u.kind == "motor" else 0.0); u_eff.append(u.efficiency)
            u_pmax.append(u.max_power_kw * 1000.0); u_coast.append(u.coast_torque_nm)
            u_rad.append(u.front_frac * rf + (1 - u.front_frac) * rr)
            # A combustion engine cannot run below idle: the clutch slips at launch and
            # the engine sits at a useful rpm.  Motors make torque from zero rpm.
            u_launch.append(0.0 if u.kind == "motor" else u.rpm_idle + 0.35 * (u.rpm_max - u.rpm_idle))
            u_gbeff.append(gb.efficiency)

        # Ideal shift speeds: change up as soon as the next gear pulls at least as hard.
        v_grid = np.arange(0.5, 160.0, 0.1)
        up_v = np.full((self.num_gb, gmax), 1e9, np.float32)
        for i in range(self.num_gb):
            units = [k for k in range(self.num_units) if u_gb[k] == i]

            def force(gi: int) -> np.ndarray:
                tot = np.zeros_like(v_grid)
                for k in units:
                    rpm = v_grid / u_rad[k] * ratio[i, gi] * RPM_PER_RADS
                    rpm_eff = np.maximum(rpm, u_launch[k])
                    tq = np.interp(rpm_eff, np.linspace(0, u_top[k], _TABLE), table[k])
                    tot += tq * ratio[i, gi] / u_rad[k]
                return tot

            for gi in range(int(ngears[i]) - 1):
                better = force(gi + 1) >= force(gi) * 0.999
                better &= v_grid > 2.0
                if better.any():
                    up_v[i, gi] = v_grid[np.argmax(better)]
                # Never wait past the rev limit: torque collapses there within a few
                # rpm, and a car that sits just below the "next gear is better" speed
                # would hang on the limiter forever.
                v_limit = min(0.99 * pt.units[k].rpm_max / (ratio[i, gi] * RPM_PER_RADS) * u_rad[k] for k in units)
                up_v[i, gi] = min(up_v[i, gi], v_limit)
        dn_v = np.zeros_like(up_v)
        dn_v[:, 1:] = up_v[:, :-1] * 0.93                    # hysteresis so the box does not hunt

        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), device=dev, dtype=dt)  # noqa: E731
        self.gb_ratio, self.gb_ngears = t(ratio), t(ngears, torch.long)
        self.gb_up_v, self.gb_dn_v = t(up_v), t(dn_v)
        self.gb_up_time = t([pt.gearboxes[n].upshift_time_s for n in gb_names])
        self.gb_dn_time = t([pt.gearboxes[n].downshift_time_s for n in gb_names])
        self.gb_index = torch.arange(self.num_gb, device=dev)
        self.u_table, self.u_top = t(table), t(u_top)
        self.u_gb, self.u_front = t(u_gb, torch.long), t(u_front)
        self.u_motor, self.u_eff, self.u_coast = t(u_motor), t(u_eff), t(u_coast)
        self.u_rad, self.u_launch, self.u_gbeff = t(u_rad), t(u_launch), t(u_gbeff)
        self.u_index = torch.arange(self.num_units, device=dev)
        self.has_motor = bool(sum(u_motor) > 0)
        bat = pt.battery
        self.soc_init = bat.soc_init if bat else 1.0
        self.soc_min = bat.soc_min if bat else 0.0
        self.bat_joule = (bat.capacity_kwh * 3.6e6) if bat else 1.0
        self.bat_discharge = (bat.max_discharge_kw * 1000.0) if bat else 0.0
        self.bat_regen = (bat.max_regen_kw * 1000.0) if bat else 0.0

    # ------------------------------------------------------------------------
    def reset(self, ids: torch.Tensor, x: torch.Tensor, y: torch.Tensor, yaw: torch.Tensor, vx: torch.Tensor) -> None:
        """Place cars ``ids`` at a pose with a forward speed and clear their dynamic state."""
        self.x[ids], self.y[ids], self.yaw[ids], self.vx[ids] = x, y, yaw, vx
        for a in (self.vy, self.r, self.steer, self.thr, self.brk, self.ax, self.ay):
            a[ids] = 0.0
        # start in the right gear for the speed
        g = (vx[:, None, None] > self.gb_up_v[None]).sum(-1)
        self.gear[ids] = torch.minimum(g, self.gb_ngears[None] - 1)
        self.shift_timer[ids] = 0.0
        self.soc[ids] = self.soc_init
        self.slip[ids] = 0.0
        self.tire_util[ids] = 0.0

    # ------------------------------------------------------------------------
    def _powertrain(self, v: torch.Tensor, thr: torch.Tensor, brk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (front axle, rear axle) wheel torque from all power units."""
        dt = self.dt
        # ---- automatic gear selection ---------------------------------------
        gi = self.gb_index[None]
        ready = self.shift_timer <= 0
        up = ready & (v[:, None] > self.gb_up_v[gi, self.gear]) & (self.gear < self.gb_ngears[None] - 1)
        down = ready & (v[:, None] < self.gb_dn_v[gi, self.gear]) & (self.gear > 0)
        self.gear = self.gear + up.long() - down.long()
        self.shift_timer = torch.where(up, self.gb_up_time[None].expand_as(self.shift_timer),
                                       torch.where(down, self.gb_dn_time[None].expand_as(self.shift_timer),
                                                   self.shift_timer - dt))
        engaged = (self.shift_timer <= 0).float()[:, self.u_gb]            # (N, U): torque is cut mid-shift
        ratio = self.gb_ratio[gi, self.gear][:, self.u_gb]                 # (N, U)

        # ---- torque of every unit at the current speed ------------------------
        omega_wheel = v[:, None] / self.u_rad[None]
        rpm = omega_wheel * ratio * RPM_PER_RADS
        rpm_eff = torch.maximum(rpm, self.u_launch[None])
        pos = (rpm_eff / self.u_top[None]).clamp(0.0, 1.0) * (_TABLE - 1)
        i0 = pos.floor().long().clamp(max=_TABLE - 2)
        frac = pos - i0
        ui = self.u_index[None]
        full = self.u_table[ui, i0] * (1 - frac) + self.u_table[ui, i0 + 1] * frac
        torque = full * thr[:, None] * self.power_scale[:, None]
        # combustion engines brake the car when the throttle is closed
        coast = self.u_coast[None] * (rpm / self.u_top[None]).clamp(0, 1) * (1 - thr[:, None]) ** 2
        torque = torque - coast * (1 - self.u_motor[None])

        # ---- battery: limit electric power, track state of charge ---------------
        if self.has_motor:
            omega_unit = rpm / RPM_PER_RADS
            p_mech = (torque * omega_unit * self.u_motor[None]).clamp(min=0)
            p_elec = (p_mech / self.u_eff[None]).sum(1)
            allowed = torch.where(self.soc > self.soc_min, torch.full_like(p_elec, self.bat_discharge),
                                  torch.zeros_like(p_elec))
            scale = (allowed / p_elec.clamp(min=1.0)).clamp(max=1.0)
            torque = torch.where(self.u_motor[None] > 0, torque * scale[:, None], torque)
            p_elec = p_elec * scale
            # regenerative braking: the motors take over part of the braking, up to
            # what they and the battery can absorb.  Total brake force is unchanged,
            # so this only affects the state of charge.
            p_cap = (full * omega_unit * self.u_motor[None] * self.u_eff[None]).sum(1)
            p_regen = torch.minimum(brk * p_cap, torch.full_like(p_cap, self.bat_regen))
            p_regen = torch.where(self.soc < 0.98, p_regen, torch.zeros_like(p_regen))
            self.soc = (self.soc + (p_regen - p_elec) * dt / self.bat_joule).clamp(0.0, 1.0)
            self.power_elec = p_elec - p_regen

        wheel_torque = torque * ratio * self.u_gbeff[None] * engaged
        t_front = (wheel_torque * self.u_front[None]).sum(1)
        t_rear = (wheel_torque * (1 - self.u_front[None])).sum(1)
        self.rpm_frac = (rpm_eff[:, 0] / self.u_top[0]).clamp(0, 1)
        return t_front, t_rear

    # ------------------------------------------------------------------------
    def wheel_slip_angles(self) -> torch.Tensor:
        """Slip angle of each tyre from the current speeds, yaw rate and steering angle.

        Pure geometry (no forces), identical to what :meth:`step` uses.  It lets a
        real car or Assetto Corsa fill in the same observation the policy was
        trained on from quantities any car can measure.
        """
        wvx = self.vx[:, None] - self.r[:, None] * self.wheel_y[None]
        wvy = self.vy[:, None] + self.r[:, None] * self.wheel_x[None]
        delta = self.steer[:, None] * self.is_front[None]
        cd, sd = torch.cos(delta), torch.sin(delta)
        v_long = wvx * cd + wvy * sd
        v_lat = -wvx * sd + wvy * cd
        return -torch.atan2(v_lat, v_long.abs().clamp(min=2.0))

    # ------------------------------------------------------------------------
    def step(self, steer_cmd: torch.Tensor, throttle_cmd: torch.Tensor, brake_cmd: torch.Tensor) -> None:
        """Advance every car by one physics step ``dt``.

        ``steer_cmd`` in [-1, 1] (fraction of full lock, + = left),
        ``throttle_cmd`` and ``brake_cmd`` in [0, 1].
        """
        dt = self.dt
        c = self.vp.chassis
        mass = c.mass_kg * self.mass_scale
        inertia = c.yaw_inertia_kgm2 * self.mass_scale

        # ---- actuators ----------------------------------------------------------
        target = steer_cmd.clamp(-1, 1) * self.steer_max
        self.steer = self.steer + (target - self.steer).clamp(-self.steer_rate * dt, self.steer_rate * dt)
        self.thr = self.thr + (throttle_cmd.clamp(0, 1) - self.thr) * self.k_thr
        self.brk = self.brk + (brake_cmd.clamp(0, 1) - self.brk) * self.k_brk

        vx, vy, r = self.vx, self.vy, self.r
        speed2 = vx * vx + vy * vy

        # ---- vertical load on each tyre -------------------------------------------
        fz = (self.fz_static[None] * self.mass_scale[:, None]
              + self.aero_fz[None] * (speed2 * self.aero_scale)[:, None]
              + self.k_long * (self.mass_scale * self.ax)[:, None] * (1 - 2 * self.is_front[None])
              + self.k_lat[None] * (self.mass_scale * self.ay)[:, None])
        fz = fz.clamp(min=50.0)

        # ---- velocity of each wheel hub, in the wheel's own frame -----------------
        wvx = vx[:, None] - r[:, None] * self.wheel_y[None]
        wvy = vy[:, None] + r[:, None] * self.wheel_x[None]
        delta = self.steer[:, None] * self.is_front[None]
        cd, sd = torch.cos(delta), torch.sin(delta)
        v_long = wvx * cd + wvy * sd
        v_lat = -wvx * sd + wvy * cd
        slip = -torch.atan2(v_lat, v_long.abs().clamp(min=2.0))        # slip angle; clamp avoids /0 at standstill
        self.slip = slip

        # ---- grip available right now --------------------------------------------
        load = fz / self.fz_ref[None]
        d_lat = self.mu_lat[None] * load.pow(self.exp_lat[None]) * fz * self.grip[:, None]
        d_long = self.mu_long[None] * load.pow(self.exp_long[None]) * fz * self.grip[:, None]

        # ---- longitudinal force each wheel is asked to deliver ----------------------
        t_front, t_rear = self._powertrain(vx.clamp(min=0.0), self.thr, self.brk)
        drive = torch.stack([t_front, t_front, t_rear, t_rear], dim=1) * 0.5 / self.radius[None]
        rolling = torch.tanh(v_long / 0.5)                               # smooth sign: no chatter at standstill
        brake = self.brk[:, None] * self.brake_torque[None] / self.radius[None]
        fx_req = drive - (brake + self.roll_res[None] * fz) * rolling

        # ---- combined slip ("similarity" tyre model) -----------------------------------
        # A tyre has ONE grip budget shared between pulling/braking and cornering.
        # Both demands are expressed as slip normalised so that 1.0 = the slip of peak
        # grip:  alpha_n for cornering (from the slip angle) and kappa_n for
        # traction/braking.  The tyre force follows the magic-formula curve P() of the
        # combined slip sigma = sqrt(kappa_n^2 + alpha_n^2) and points along the slip:
        #     Fx = Dx * P(sigma) * kappa_n / sigma      Fy = Dy * P(sigma) * alpha_n / sigma
        # We do not simulate wheel rotation, so kappa_n is the slip a wheel would need
        # to deliver the requested force in a straight line (the inverse of P).  A
        # request beyond the peak means wheelspin / lock-up: slip runs away, force
        # drops to the sliding level and cornering grip collapses - unless traction
        # control / ABS hold the slip at the peak, which is exactly what they do.
        u = fx_req / d_long
        ua = u.abs()
        kappa_n = torch.tan(torch.asin(ua.clamp(max=1.0)) / self.tire_c[None]) / self.tire_bn[None]
        kappa_n = kappa_n + 6.0 * (ua - 1.0).clamp(min=0.0)             # runaway slip beyond the peak
        if self.tc_limit is not None:
            kappa_n = torch.where(u > 0, kappa_n.clamp(max=self.tc_limit), kappa_n)
        if self.abs_limit is not None:
            kappa_n = torch.where(u < 0, kappa_n.clamp(max=self.abs_limit), kappa_n)
        alpha_n = torch.tan(slip) / self.tan_slip_peak[None]
        sigma = torch.sqrt(kappa_n * kappa_n + alpha_n * alpha_n).clamp(min=1e-6)
        grip_curve = torch.sin(self.tire_c[None] * torch.atan(self.tire_bn[None] * sigma)) / sigma
        fx = d_long * grip_curve * kappa_n * torch.sign(u)
        fy = d_lat * grip_curve * alpha_n
        self.tire_util = sigma

        # ---- sum forces and moments in the body frame --------------------------------
        fxb = fx * cd - fy * sd
        fyb = fx * sd + fy * cd
        drag = self.drag_k * self.aero_scale * vx * vx.abs()
        ax_body = (fxb.sum(1) - drag) / mass
        ay_body = fyb.sum(1) / mass
        mz = (self.wheel_x[None] * fyb - self.wheel_y[None] * fxb).sum(1)

        # ---- integrate ------------------------------------------------------------------
        self.vx = vx + (ax_body + vy * r) * dt
        self.vy = vy + (ay_body - vx * r) * dt
        self.r = r + mz / inertia * dt
        cy, sy = torch.cos(self.yaw), torch.sin(self.yaw)
        self.x = self.x + (self.vx * cy - self.vy * sy) * dt
        self.y = self.y + (self.vx * sy + self.vy * cy) * dt
        self.yaw = self.yaw + self.r * dt
        self.ax = self.ax + (ax_body - self.ax) * self.k_acc
        self.ay = self.ay + (ay_body - self.ay) * self.k_acc
