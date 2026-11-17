"""Back-of-the-envelope numbers for a vehicle file.

These are sanity checks a human can compare with a spec sheet: if the peak
power, top speed or cornering grip printed here look wrong, the parameter file
is wrong, and no amount of training will fix that.
"""

from __future__ import annotations

import math

import numpy as np

from mv.vehicle.params import G, VehicleParams

RPM_TO_RADS = math.pi / 30.0


def unit_peak_power_kw(u) -> float:
    rpm, tq = np.asarray(u.rpm, float), np.asarray(u.torque_nm, float)
    grid = np.linspace(max(rpm.min(), 0.0), min(rpm.max(), u.rpm_max), 400)
    p = np.interp(grid, rpm, tq) * grid * RPM_TO_RADS / 1000.0
    peak = float(p.max())
    return min(peak, u.max_power_kw) if u.max_power_kw > 0 else peak


def drive_force_at_speed(vp: VehicleParams, v: float) -> float:
    """Best total drive force (N) available at speed ``v`` with ideal gear choice."""
    total = 0.0
    by_gb: dict[str, list] = {}
    for u in vp.powertrain.units:
        by_gb.setdefault(u.gearbox, []).append(u)
    for gb_name, units in by_gb.items():
        gb = vp.powertrain.gearboxes[gb_name]
        best = 0.0
        for ratio in gb.ratios:
            f_gear = 0.0
            for u in units:
                radius = u.front_frac * vp.tire_front.radius_m + (1 - u.front_frac) * vp.tire_rear.radius_m
                tot = ratio * gb.final_drive
                rpm = v / radius * tot / RPM_TO_RADS
                if rpm > u.rpm_max:
                    continue
                tq = float(np.interp(max(rpm, u.rpm[0]), u.rpm, u.torque_nm))
                if u.max_power_kw > 0 and rpm > 1:
                    tq = min(tq, u.max_power_kw * 1000.0 / (rpm * RPM_TO_RADS))
                f_gear += tq * tot * gb.efficiency / radius
            best = max(best, f_gear)
        total += best
    return total


def top_speed(vp: VehicleParams) -> float:
    """Speed (m/s) where drive force equals drag + rolling resistance."""
    best = 0.0
    for v in np.arange(5.0, 140.0, 0.25):
        drag = 0.5 * vp.aero.air_density * vp.aero.cda_m2 * v * v
        roll = vp.tire_rear.rolling_resistance * vp.chassis.mass_kg * G
        if drive_force_at_speed(vp, float(v)) >= drag + roll:
            best = float(v)
    return best


def describe(vp: VehicleParams) -> str:
    c = vp.chassis
    power = sum(unit_peak_power_kw(u) for u in vp.powertrain.units)
    if vp.powertrain.battery and vp.powertrain.type == "ev":
        power = min(power, vp.powertrain.battery.max_discharge_kw)
    vmax = top_speed(vp)
    mu = c.front_weight_frac * vp.tire_front.mu_lat + (1 - c.front_weight_frac) * vp.tire_rear.mu_lat
    q200 = 0.5 * vp.aero.air_density * (200 / 3.6) ** 2
    df200 = q200 * (vp.aero.cla_front_m2 + vp.aero.cla_rear_m2)
    brake_f = vp.brakes.max_torque_nm * (vp.brakes.front_share / vp.tire_front.radius_m
                                         + (1 - vp.brakes.front_share) / vp.tire_rear.radius_m)
    lines = [
        f"{vp.name}  [{vp.powertrain.type}]",
        f"  mass {c.mass_kg:.0f} kg, {c.front_weight_frac*100:.0f}% front, wheelbase {c.wheelbase_m:.3f} m, "
        f"CG height {c.cg_height_m:.3f} m, yaw inertia {c.yaw_inertia_kgm2:.0f} kg m^2",
        f"  peak power {power:.0f} kW ({power*1.36:.0f} PS), {power/c.mass_kg*1000:.0f} W/kg",
        f"  top speed ~{vmax*3.6:.0f} km/h, CdA {vp.aero.cda_m2:.2f} m^2",
        f"  downforce at 200 km/h: {df200:.0f} N ({df200/G:.0f} kg), "
        f"{vp.aero.cla_front_m2/max(vp.aero.cla_front_m2+vp.aero.cla_rear_m2,1e-9)*100:.0f}% front",
        f"  tyre grip ~{mu:.2f} g lateral (static load), max steer {vp.steering.max_wheel_angle_deg:.1f} deg",
        f"  brakes: {brake_f/(c.mass_kg*G):.2f} g of hydraulic capacity, {vp.brakes.front_share*100:.0f}% front",
        f"  gearboxes: " + ", ".join(f"{k}({len(g.ratios)})" for k, g in vp.powertrain.gearboxes.items())
        + "; units: " + ", ".join(f"{u.name}:{unit_peak_power_kw(u):.0f}kW" for u in vp.powertrain.units),
    ]
    return "\n".join(lines)
