"""Convert an Assetto Corsa car into a Project MV :class:`VehicleParams` file.

Assetto Corsa describes a car with ~15 INI files and lookup tables.  Our
simulator needs far fewer numbers (see ``mv/vehicle/params.py``).  This module
reads the game files and condenses them.  Every reduction step is written out
below so the numbers can be audited.

The output is written to ``vehicles/local/`` which is git-ignored: the numbers
are derived from Kunos' data and must not be redistributed.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from mv.ac.acd import load_car_data
from mv.vehicle.params import (
    Aero, Battery, Brakes, Chassis, Electronics, Gearbox, PowerUnit, Powertrain,
    Steering, TireAxle, VehicleParams,
)

FUEL_DENSITY = 0.74      # kg per litre of petrol
RPM_TO_RADS = math.pi / 30.0


# --------------------------------------------------------------------------
# Tolerant parsers for Kunos' INI and LUT formats
# --------------------------------------------------------------------------
def parse_ini(text: str) -> dict[str, dict[str, str]]:
    """Parse an AC ini file into {SECTION: {KEY: value}} (keys upper-cased)."""
    out: dict[str, dict[str, str]] = {}
    section = None
    for line in text.splitlines():
        line = re.split(r"[;]|//", line, maxsplit=1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and "]" in line:
            section = line[1 : line.index("]")].strip().upper()
            out.setdefault(section, {})
        elif "=" in line and section is not None:
            k, v = line.split("=", 1)
            out[section][k.strip().upper()] = v.strip()
    return out


# A number at the start of a text, the way C's ``strtod`` reads it: optional
# sign, digits with at most one decimal point, optional exponent.
_LEADING_NUMBER = re.compile(r"\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def _leading_float(text: str) -> float | None:
    """Read the number at the start of ``text`` and ignore whatever follows.

    Why: Kunos' files contain the odd typo, e.g. ``8|0.0.91`` (two decimal
    points) in the Cayman GT4's and the Praga R1's wing tables.  Python's
    ``float()`` rejects that; the game (written in C++) reads numbers like C's
    ``strtod``, which stops at the first character that cannot continue the
    number, so it sees ``0.0``.  Reading it the same way keeps our car identical
    to the game's.  Returns None when the text does not start with a number.
    """
    m = _LEADING_NUMBER.match(text)
    return float(m.group(0)) if m else None


def parse_lut(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a ``x|y`` lookup table into two sorted arrays.

    A row is kept only when *both* of its numbers can be read, so x and y always
    stay paired.  (The first version appended x before reading y; one bad y then
    left x one entry longer than y, and sorting crashed with an IndexError -
    engineering log, car catalogue work package.)
    """
    xs, ys = [], []
    for line in text.splitlines():
        line = re.split(r"[;]|//", line, maxsplit=1)[0].strip()
        if "|" not in line:
            continue
        a, b = line.split("|")[:2]
        x, y = _leading_float(a), _leading_float(b)
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    order = np.argsort(xs, kind="stable")
    return np.asarray(xs, float)[order], np.asarray(ys, float)[order]


def _f(sec: dict[str, str], key: str, default: float | None = None) -> float:
    if key not in sec or sec[key] == "":
        if default is None:
            raise KeyError(key)
        return default
    return float(sec[key].split(",")[0])


def _vec(sec: dict[str, str], key: str) -> list[float]:
    return [float(x) for x in sec[key].split(",")]


def _lut(files: dict[str, str], name: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    """A lookup table named in an ini file: a file name, or written inline.

    Inline form (used by AC's controllers): ``(|0=0|1000=0.5|7000=1.3|)`` -
    pairs ``x=y`` separated by ``|``.
    """
    if not name:
        return None
    name = name.strip()
    if name.startswith("("):
        rows = [p.replace("=", "|", 1) for p in name.strip("()").split("|") if "=" in p]
        return parse_lut("\n".join(rows)) if rows else None
    t = files.get(name.lower())
    return parse_lut(t) if t else None


def _default_tyre_sections(tyres: dict[str, dict[str, str]]) -> tuple[dict, dict, str]:
    """Pick the front/rear sections of the car's default tyre compound."""
    idx = int(_f(tyres.get("COMPOUND_DEFAULT", {}), "INDEX", 0))
    suffix = "" if idx == 0 else f"_{idx}"
    front, rear = tyres.get(f"FRONT{suffix}"), tyres.get(f"REAR{suffix}")
    if front is None or rear is None:
        front, rear = tyres["FRONT"], tyres["REAR"]
    return front, rear, front.get("NAME", "?")


# --------------------------------------------------------------------------
# Individual reductions
# --------------------------------------------------------------------------
def _tire(sec: dict[str, str]) -> TireAxle:
    """AC tyre -> TireAxle.

    AC (tyre model v10) uses  D(Fz) = D_REF * (Fz/FZ0)^LS_EXP  for peak force,
    i.e. mu(Fz) = D_REF * (Fz/FZ0)^(LS_EXP-1).  Older tyres only have the
    linear form mu = D0 + D1 * Fz[kN]; we convert that to an equivalent
    exponent around FZ0.
    """
    fz0 = _f(sec, "FZ0", 3000.0)
    if "DY_REF" in sec and "LS_EXPY" in sec:
        mu_y, ey = _f(sec, "DY_REF"), _f(sec, "LS_EXPY")
        mu_x, ex = _f(sec, "DX_REF", mu_y), _f(sec, "LS_EXPX", ey)
    else:
        dy0, dy1 = _f(sec, "DY0"), _f(sec, "DY1", 0.0)
        dx0, dx1 = _f(sec, "DX0", dy0), _f(sec, "DX1", dy1)
        mu_y, mu_x = dy0 + dy1 * fz0 / 1000.0, dx0 + dx1 * fz0 / 1000.0
        ey = 1.0 + dy1 * (fz0 / 1000.0) / mu_y      # d ln(mu)/d ln(Fz) at FZ0, plus 1
        ex = 1.0 + dx1 * (fz0 / 1000.0) / mu_x
    return TireAxle(
        radius_m=_f(sec, "RADIUS"),
        mu_lat=mu_y, mu_long=mu_x, fz_ref_n=fz0,
        load_exp_lat=float(np.clip(ey, 0.6, 1.0)), load_exp_long=float(np.clip(ex, 0.6, 1.0)),
        peak_slip_angle_deg=_f(sec, "FRICTION_LIMIT_ANGLE", 8.0),
        falloff_level=_f(sec, "FALLOFF_LEVEL", 0.87),
        rolling_resistance=0.012,
    )


def _aero(files: dict[str, str], a: float, b: float, ref_speed_kmh: float = 180.0) -> Aero:
    """Sum AC's wing elements into drag area + front/rear downforce areas.

    AC: force = 0.5*rho*v^2 * C(aoa) * gain * chord * span, applied at the
    wing's POSITION (z forward of the CG).  The share carried by the front
    axle follows from a moment balance:  front_share = (z + b) / wheelbase.
    Speed-controlled wings (e.g. the 918's active rear wing) are evaluated at
    ``ref_speed_kmh``.  Ride-height tables are evaluated at their plateau.
    """
    ini = parse_ini(files["aero.ini"])
    wheelbase = a + b
    # speed-dependent wing angle offsets
    extra: dict[int, float] = {}
    for name, sec in ini.items():
        if name.startswith("DYNAMIC_CONTROLLER") and sec.get("INPUT", "").upper() == "SPEED_KMH":
            lut = _lut(files, sec.get("LUT"))
            if lut is not None and sec.get("COMBINATOR", "ADD").upper() == "ADD":
                val = float(np.interp(ref_speed_kmh, *lut))
                val = min(max(val, _f(sec, "DOWN_LIMIT", -1e9)), _f(sec, "UP_LIMIT", 1e9))
                extra[int(_f(sec, "WING"))] = extra.get(int(_f(sec, "WING")), 0.0) + val
    cda = cla_f = cla_r = 0.0
    for name, sec in ini.items():
        if not name.startswith("WING_"):
            continue
        idx = int(name.split("_")[1])
        area = _f(sec, "CHORD") * _f(sec, "SPAN")
        angle = _f(sec, "ANGLE", 0.0) + extra.get(idx, 0.0)
        cl_lut, cd_lut = _lut(files, sec.get("LUT_AOA_CL")), _lut(files, sec.get("LUT_AOA_CD"))
        gh_cl, gh_cd = _lut(files, sec.get("LUT_GH_CL")), _lut(files, sec.get("LUT_GH_CD"))
        cl = float(np.interp(angle, *cl_lut)) * _f(sec, "CL_GAIN", 1.0) if cl_lut else 0.0
        cd = float(np.interp(angle, *cd_lut)) * _f(sec, "CD_GAIN", 1.0) if cd_lut else 0.0
        if gh_cl is not None:
            cl *= float(np.median(gh_cl[1][gh_cl[1] > 0.5 * gh_cl[1].max()]))
        if gh_cd is not None and gh_cd[1].max() > 0:
            cd *= float(np.median(gh_cd[1][gh_cd[1] > 0.5 * gh_cd[1].max()]))
        z = _vec(sec, "POSITION")[2]
        front_share = (z + b) / wheelbase
        cda += cd * area
        cla_f += cl * area * front_share
        cla_r += cl * area * (1.0 - front_share)
    return Aero(cda_m2=cda, cla_front_m2=cla_f, cla_rear_m2=cla_r)


def _engine_curve(files: dict[str, str], eng: dict[str, dict[str, str]],
                  n_gears: int = 1) -> tuple[list[float], list[float], float]:
    """Full-throttle torque curve including steady-state turbo boost.

    AC: torque = power.lut(rpm) * (1 + boost), with each turbo producing
    boost = min(MAX_BOOST * (rpm/REFERENCE_RPM)^GAMMA, WASTEGATE).
    If the car has a boost controller for a turbo (``ctrl_turbo<i>.ini``), the
    controller's limit takes the place of WASTEGATE (see
    :func:`_turbo_controller_limit`).
    """
    rpm, tq = parse_lut(files[eng["HEADER"].get("POWER_CURVE", "power.lut").lower()])
    limiter = _f(eng["ENGINE_DATA"], "LIMITER", float(rpm.max()))
    if limiter <= 0:
        limiter = float(rpm[tq > 0].max())
    grid = np.unique(np.concatenate([rpm[(rpm >= 0) & (rpm <= limiter)], [limiter]]))
    torque = np.interp(grid, rpm, tq)
    boost = np.zeros_like(grid)
    for name, sec in eng.items():
        if name.startswith("TURBO_"):
            ref = max(_f(sec, "REFERENCE_RPM", 3000.0), 1.0)
            b = _f(sec, "MAX_BOOST", 0.0) * np.minimum(1.0, grid / ref) ** _f(sec, "GAMMA", 1.0)
            ctrl = _turbo_controller_limit(files, int(name.split("_")[1]), grid, n_gears)
            if ctrl is not None:
                # a boost controller replaces the fixed wastegate (log: car catalogue WP)
                boost += np.minimum(b, np.maximum(ctrl, 0.0))
            else:
                wg = _f(sec, "WASTEGATE", 0.0)
                boost += np.minimum(b, wg) if wg > 0 else b
    torque = torque * (1.0 + boost)
    return grid.tolist(), torque.tolist(), limiter


def _turbo_controller_limit(files: dict[str, str], index: int, rpm: np.ndarray, n_gears: int) -> np.ndarray | None:
    """Boost limit of turbo ``index`` set by its controller file ``ctrl_turbo<index>.ini``, or None.

    Why: some cars (e.g. the Ferrari 488 GTB, SF70H, Porsche 919, 991 Turbo S)
    do not use the turbo's fixed WASTEGATE; a controller sets the allowed boost
    from engine speed, gear and pedal.  The 488 GTB has ten "turbos" with a
    wastegate of 0.69 each, of which a controller enables only the two that
    belong to the current gear - ignoring the controllers summed all ten and
    gave the car 1400 kW instead of ~490 kW.

    AC combines a controller file's sections top to bottom, starting from 0:
    ``COMBINATOR=ADD`` adds the section's table value, ``MULT`` multiplies by
    it; the running result is then clamped to the section's ``[DOWN_LIMIT, UP_LIMIT]``.  Inputs: engine
    speed (RPMS), gear (GEAR: 1 = first), pedal (GAS: full throttle = 1).
    Inputs that describe sliding (slip angles, oversteer ...) are taken at
    their straight-line value 0.  Our engine model has one full-throttle curve
    for all gears, so we use the TOP gear's limit - the car's normal full
    power; extra boost allowed only in low gears (overboost) or boost held
    back in low gears (traction management) is not modelled.
    """
    text = files.get(f"ctrl_turbo{index}.ini")
    if not text:
        return None
    sections = [s for k, s in parse_ini(text).items() if k.startswith("CONTROLLER_")]
    if not sections:
        return None
    gear = float(max(n_gears, 1))
    value = np.zeros_like(rpm)
    for sec in sections:
        lut = _lut(files, sec.get("LUT"))
        if lut is None:
            continue
        kind = sec.get("INPUT", "").strip().upper()
        x = rpm if kind == "RPMS" else np.full_like(rpm, {"GEAR": gear, "GAS": 1.0}.get(kind, 0.0))
        v = np.interp(x, *lut)
        value = value * v if sec.get("COMBINATOR", "ADD").strip().upper() == "MULT" else value + v
        # the limits bound the running result, not the table value: the 991 Turbo S multiplies by a
        # pedal factor of 1.0 with UP_LIMIT=0.7 - clamping the factor would cut its boost in half
        value = np.clip(value, _f(sec, "DOWN_LIMIT", -1e9), _f(sec, "UP_LIMIT", 1e9))
    return value


def import_ac_car(car_dir: str | Path, fuel_litres: float | None = None) -> VehicleParams:
    """Build :class:`VehicleParams` from an Assetto Corsa car folder."""
    car_dir = Path(car_dir)
    files = load_car_data(car_dir)
    car = parse_ini(files["car.ini"])
    susp = parse_ini(files["suspensions.ini"])
    tyres = parse_ini(files["tyres.ini"])
    eng = parse_ini(files["engine.ini"])
    drv = parse_ini(files["drivetrain.ini"])
    brk = parse_ini(files["brakes.ini"])
    elec = parse_ini(files.get("electronics.ini", ""))
    notes: list[str] = []

    # ---- tyres (default compound) ----------------------------------------
    tf_sec, tr_sec, compound = _default_tyre_sections(tyres)
    tire_f, tire_r = _tire(tf_sec), _tire(tr_sec)
    notes.append(f"tyre compound: {compound}")

    # ---- chassis ---------------------------------------------------------
    fuel = _f(car.get("FUEL", {}), "FUEL", 0.0) if fuel_litres is None else fuel_litres
    mass = _f(car["BASIC"], "TOTALMASS") + fuel * FUEL_DENSITY          # TOTALMASS includes the driver
    w, _h, length = _vec(car["BASIC"], "INERTIA")                       # box dimensions, not inertias
    yaw_inertia = mass * (w * w + length * length) / 12.0               # solid box about the vertical axis
    wheelbase = _f(susp["BASIC"], "WHEELBASE")
    front_frac = _f(susp["BASIC"], "CG_LOCATION")
    # BASEY is the wheel-centre height relative to the CG (negative = below it),
    # so CG height above ground at each axle = tyre radius - BASEY.
    hf = tire_f.radius_m - _f(susp["FRONT"], "BASEY")
    hr = tire_r.radius_m - _f(susp["REAR"], "BASEY")
    cg_h = front_frac * hf + (1.0 - front_frac) * hr
    tf, tr = _f(susp["FRONT"], "TRACK"), _f(susp["REAR"], "TRACK")
    # Roll stiffness of an axle ~ (wheel rate + anti-roll bar rate) * track^2 / 2.
    arb = susp.get("ARB", {})
    kf = (_f(susp["FRONT"], "SPRING_RATE", 5e4) + _f(arb, "FRONT", 0.0)) * tf * tf
    kr = (_f(susp["REAR"], "SPRING_RATE", 5e4) + _f(arb, "REAR", 0.0)) * tr * tr
    chassis = Chassis(
        mass_kg=mass, yaw_inertia_kgm2=yaw_inertia, wheelbase_m=wheelbase,
        front_weight_frac=front_frac, cg_height_m=cg_h, track_front_m=tf, track_rear_m=tr,
        roll_stiffness_front_frac=kf / (kf + kr),
        width_m=max(tf, tr) + 0.5 * (_f(tf_sec, "WIDTH", 0.25) + _f(tr_sec, "WIDTH", 0.3)),
        length_m=wheelbase + 1.9,
    )
    aero = _aero(files, chassis.a, chassis.b)

    # ---- brakes: MAX_TORQUE is per wheel before the bias split -----------
    brakes = Brakes(max_torque_nm=2.0 * _f(brk["DATA"], "MAX_TORQUE"),
                    front_share=_f(brk["DATA"], "FRONT_SHARE", 0.6))

    # ---- steering --------------------------------------------------------
    lock, ratio = _f(car["CONTROLS"], "STEER_LOCK"), _f(car["CONTROLS"], "STEER_RATIO")
    steering = Steering(max_wheel_angle_deg=lock / abs(ratio), steering_ratio=abs(ratio), lock_deg=lock)

    # ---- combustion engine + gearbox -------------------------------------
    n_gears = int(_f(drv["GEARS"], "COUNT"))
    rpm, torque, limiter = _engine_curve(files, eng, n_gears)
    if any(k.startswith("ctrl_turbo") for k in files):
        notes.append("turbo boost from the boost controllers (ctrl_turbo*.ini) in top gear; "
                     "gear-dependent overboost/torque limits not modelled")
    ratios = [_f(drv["GEARS"], f"GEAR_{i}") for i in range(1, n_gears + 1)]
    auto = drv.get("AUTO_SHIFTER", {})
    gb = drv.get("GEARBOX", {})
    main_gb = Gearbox(
        ratios=ratios, final_drive=_f(drv["GEARS"], "FINAL"),
        upshift_rpm=_f(auto, "UP", limiter * 0.97), downshift_rpm=_f(auto, "DOWN", limiter * 0.6),
        upshift_time_s=max(_f(gb, "CHANGE_UP_TIME", 100.0), _f(gb, "AUTO_CUTOFF_TIME", 0.0)) / 1000.0,
        downshift_time_s=_f(gb, "CHANGE_DN_TIME", 150.0) / 1000.0,
    )
    traction = drv["TRACTION"].get("TYPE", "RWD").upper()
    if traction == "FWD":
        ice_front = 1.0
    elif traction in ("AWD", "AWD2"):
        awd = drv.get("AWD", {})
        ice_front = _f(awd, "FRONT_SHARE", 50.0) / 100.0
        notes.append("AWD: fixed front torque share taken from drivetrain.ini [AWD] FRONT_SHARE")
    else:
        ice_front = 0.0
    coast = eng.get("COAST_REF", {})
    units = [PowerUnit(
        name="engine", kind="ice", gearbox="main", front_frac=ice_front,
        rpm=rpm, torque_nm=torque, rpm_max=limiter,
        rpm_idle=_f(eng["ENGINE_DATA"], "MINIMUM", 1000.0),
        coast_torque_nm=_f(coast, "TORQUE", 0.0) * limiter / max(_f(coast, "RPM", limiter), 1.0),
    )]
    gearboxes = {"main": main_gb}
    battery = None
    ptype = "ice"

    # ---- hybrid system (ers.ini) -----------------------------------------
    if "ers.ini" in files:
        ers = parse_ini(files["ers.ini"])
        peak_kw = 0.0
        kin = _lut(files, ers.get("KINETIC", {}).get("TORQUE_CURVE"))
        if kin is not None and kin[1].max() > 0:
            # Rear motor sits on the engine side of the gearbox: curve is vs engine rpm.
            m = kin[0] >= 0
            r, t = kin[0][m], kin[1][m]
            top = float(r[t > 0].max()) if (t > 0).any() else limiter
            units.append(PowerUnit(name="motor_kinetic", kind="motor", gearbox="main", front_frac=ice_front,
                                   rpm=r.tolist(), torque_nm=t.tolist(), rpm_max=min(top * 1.02, 1e5),
                                   efficiency=0.92))
            peak_kw += float((t * r * RPM_TO_RADS).max()) / 1000.0
        frm = _lut(files, ers.get("FRONT_MOTORS", {}).get("TORQUE_CURVE"))
        if frm is not None and frm[1].max() > 0:
            # Front motors: AC's table is torque at the wheels vs wheel rpm -> 1:1 "gearbox".
            m = frm[0] >= 0
            r, t = frm[0][m], frm[1][m]
            top = float(r[t > 0].max())
            keep = r <= top
            units.append(PowerUnit(name="motor_front", kind="motor", gearbox="front_direct", front_frac=1.0,
                                   rpm=r[keep].tolist(), torque_nm=t[keep].tolist(), rpm_max=top * 1.05,
                                   efficiency=0.92))
            gearboxes["front_direct"] = Gearbox(ratios=[1.0], final_drive=1.0)
            peak_kw += float((t * r * RPM_TO_RADS).max()) / 1000.0
        if len(units) > 1:
            ptype = "hybrid"
            battery = Battery(capacity_kwh=6.8, max_discharge_kw=round(peak_kw, 1),
                              max_regen_kw=round(0.6 * peak_kw, 1), soc_init=0.9, soc_min=0.1)
            notes.append("ESTIMATE: battery capacity/limits are placeholders (AC's ERS model has no "
                         "direct equivalent); edit the YAML with real values for the car")
    elif "kers.ini" in files:
        # Older hybrid system (kers.ini, e.g. LaFerrari, F138): one motor with a torque curve.
        # Ignoring it cost the LaFerrari ~120 kW (log: car catalogue WP).
        kers = parse_ini(files["kers.ini"]).get("KERS", {})
        kc = _lut(files, kers.get("TORQUE_CURVE"))
        if kc is not None and kc[1].max() > 0:
            m = kc[0] >= 0
            r, t = kc[0][m], kc[1][m]
            top = float(r[t > 0].max())
            keep = r <= top
            on_wheels = kers.get("ATTACH", "ENGINE").strip().upper() == "WHEELS"
            gb_name = "kers_direct" if on_wheels else "main"      # WHEELS: curve is vs wheel rpm
            if on_wheels:
                gearboxes["kers_direct"] = Gearbox(ratios=[1.0], final_drive=1.0)
            units.append(PowerUnit(name="motor_kers", kind="motor", gearbox=gb_name, front_frac=ice_front,
                                   rpm=r[keep].tolist(), torque_nm=t[keep].tolist(),
                                   rpm_max=min(top * 1.02, 1e5), efficiency=0.92))
            peak_kw = float((t * r * RPM_TO_RADS).max()) / 1000.0
            # DISCHARGE_TIME = how long a full store lasts at full power; our battery runs
            # from soc_init 0.9 down to soc_min 0.1, i.e. 80 % of its capacity.
            discharge_s = _f(kers, "DISCHARGE_TIME", 30000.0) / 1000.0
            ptype = "hybrid"
            battery = Battery(capacity_kwh=round(peak_kw * discharge_s / 3600.0 / 0.8, 3),
                              max_discharge_kw=round(peak_kw, 1), max_regen_kw=round(0.6 * peak_kw, 1),
                              soc_init=0.9, soc_min=0.1)
            notes.append("KERS (kers.ini): motor on the engine side, store sized from DISCHARGE_TIME; "
                         "ESTIMATE: regen limit is a placeholder; AC's KERS controller (gear/pedal) not modelled")

    electronics = Electronics(
        traction_control=int(_f(elec.get("TRACTION_CONTROL", {}), "PRESENT", 0)) == 1,
        abs=int(_f(elec.get("ABS", {}), "PRESENT", 0)) == 1,
    )
    vp = VehicleParams(
        name=car_dir.name,
        chassis=chassis, aero=aero, tire_front=tire_f, tire_rear=tire_r, brakes=brakes,
        steering=steering,
        powertrain=Powertrain(type=ptype, units=units, gearboxes=gearboxes, battery=battery),
        electronics=electronics,
        source=f"Imported from Assetto Corsa car '{car_dir.name}' ({car.get('INFO', {}).get('SCREEN_NAME', '')}). "
               "Kunos data - local use only, do not redistribute.",
        notes="; ".join(notes),
    )
    vp.validate()
    return vp
