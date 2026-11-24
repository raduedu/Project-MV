"""Vehicle parameter schema.

This file is the *contract* between a car and everything else in Project MV.
The simulator, the lap-time estimator and the RL agent never know "it is a
Porsche"; they only ever see a :class:`VehicleParams`.  Supporting a new car
means writing one YAML file that fills in this schema, nothing more.

Conventions
-----------
* SI units everywhere (m, kg, s, N, Nm, rad) unless the field name says
  otherwise (``_deg``, ``_kw``, ``_kwh``, ``_rpm``).
* Body frame: x forward, y left, z up.  Positive steering turns left.
* "front"/"rear" always refer to axles.

Powertrain model
----------------
A powertrain is a list of *power units* (combustion engines and/or electric
motors).  Each unit is connected to a *gearbox* (a list of ratios times a final
drive; an EV's single reduction gear is simply a one-ratio gearbox) and sends
its torque to the front axle, the rear axle, or a fixed split of both.  A
battery is present whenever there is at least one motor.

* ICE car      -> one ``ice`` unit, one multi-ratio gearbox, no battery.
* EV           -> one or more ``motor`` units, each on its own gearbox, battery.
* Hybrid       -> ``ice`` + ``motor`` units (motors may share the ICE gearbox,
                  like the 918's rear motor, or drive an axle directly, like
                  the 918's front motor), battery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Chassis", "Aero", "TireAxle", "Brakes", "Steering", "Gearbox", "PowerUnit",
    "Battery", "Electronics", "Powertrain", "VehicleParams", "load_vehicle",
    "save_vehicle", "find_vehicle",
]

G = 9.81


@dataclass
class Chassis:
    mass_kg: float                    # total running mass: car + driver + fuel
    yaw_inertia_kgm2: float           # moment of inertia about the vertical axis
    wheelbase_m: float
    front_weight_frac: float          # static share of weight on the front axle (0-1)
    cg_height_m: float
    track_front_m: float              # distance between left and right wheel centres
    track_rear_m: float
    roll_stiffness_front_frac: float = 0.55   # share of lateral load transfer taken by the front axle
    width_m: float = 1.9              # overall body width, used for track-limit checks
    length_m: float = 4.5

    @property
    def a(self) -> float:
        """Distance from centre of gravity to front axle."""
        return self.wheelbase_m * (1.0 - self.front_weight_frac)

    @property
    def b(self) -> float:
        """Distance from centre of gravity to rear axle."""
        return self.wheelbase_m * self.front_weight_frac


@dataclass
class Aero:
    """Lumped aerodynamics.  Force = 0.5 * rho * v^2 * (coefficient * area)."""
    cda_m2: float                     # drag area
    cla_front_m2: float = 0.0         # downforce area acting on the front axle (positive = downforce)
    cla_rear_m2: float = 0.0
    air_density: float = 1.225


@dataclass
class TireAxle:
    """Tyre behaviour for one axle (both tyres on an axle are identical).

    Peak friction falls with load:  mu(Fz) = mu_ref * (Fz / fz_ref) ** (load_exp - 1).
    The force/slip-angle curve is a Pacejka "magic formula" whose shape is set
    by two intuitive numbers: the slip angle at which grip peaks and the
    fraction of peak grip left when the tyre is sliding (``falloff_level``).
    """
    radius_m: float
    mu_lat: float                     # peak lateral friction coefficient at fz_ref
    mu_long: float                    # peak longitudinal friction coefficient at fz_ref
    fz_ref_n: float
    load_exp_lat: float = 0.85        # < 1 means grip grows slower than load (load sensitivity)
    load_exp_long: float = 0.90
    peak_slip_angle_deg: float = 7.0
    falloff_level: float = 0.87
    rolling_resistance: float = 0.012 # rolling drag as a fraction of vertical load


@dataclass
class Brakes:
    max_torque_nm: float              # total brake torque of all four wheels at full pedal
    front_share: float = 0.6          # brake bias
    response_time_s: float = 0.05     # first-order lag of the hydraulic system


@dataclass
class Steering:
    max_wheel_angle_deg: float        # maximum road-wheel (not steering-wheel) angle
    max_rate_deg_s: float = 60.0      # how fast the road wheels can be turned
    steering_ratio: float = 12.0      # steering-wheel angle / road-wheel angle
    lock_deg: float = 450.0           # steering-wheel angle from centre to full lock


@dataclass
class Gearbox:
    ratios: list[float]               # forward gear ratios, first gear first
    final_drive: float = 1.0
    upshift_rpm: float = 1e9          # rpm of the *reference unit* that triggers an upshift
    downshift_rpm: float = 0.0        # ...and the rpm in the current gear that triggers a downshift
    upshift_time_s: float = 0.05      # torque is cut for this long while shifting
    downshift_time_s: float = 0.15
    efficiency: float = 1.0           # extra losses if the torque curve is at the crank


@dataclass
class PowerUnit:
    name: str
    kind: str                         # "ice" or "motor"
    gearbox: str                      # key into Powertrain.gearboxes
    front_frac: float = 0.0           # share of this unit's torque that goes to the front axle
    rpm: list[float] = field(default_factory=list)        # torque curve, x axis
    torque_nm: list[float] = field(default_factory=list)  # torque curve, y axis (full throttle)
    rpm_max: float = 9000.0           # limiter; torque is zero above this
    rpm_idle: float = 0.0
    coast_torque_nm: float = 0.0      # ICE: engine braking torque at rpm_max with closed throttle
    max_power_kw: float = 0.0         # motor: mechanical power cap (0 = use curve only)
    efficiency: float = 1.0           # motor: electrical <-> mechanical efficiency
    regen_torque_frac: float = 1.0    # motor: share of drive torque available for regeneration


@dataclass
class Battery:
    capacity_kwh: float               # usable energy
    max_discharge_kw: float
    max_regen_kw: float = 0.0
    soc_init: float = 0.9
    soc_min: float = 0.05             # motors stop assisting below this state of charge


@dataclass
class Electronics:
    traction_control: bool = True
    abs: bool = True
    tc_slip_limit: float = 0.95       # share of peak longitudinal grip TC lets you use
    abs_slip_limit: float = 0.97


@dataclass
class Powertrain:
    type: str                         # "ice", "ev" or "hybrid" (informational; the unit list is what matters)
    units: list[PowerUnit]
    gearboxes: dict[str, Gearbox]
    battery: Battery | None = None
    throttle_response_time_s: float = 0.05


@dataclass
class VehicleParams:
    name: str
    chassis: Chassis
    aero: Aero
    tire_front: TireAxle
    tire_rear: TireAxle
    brakes: Brakes
    steering: Steering
    powertrain: Powertrain
    electronics: Electronics = field(default_factory=Electronics)
    source: str = ""                  # where the numbers came from (for traceability)
    notes: str = ""

    # ---- derived quantities used all over the code base -------------------
    @property
    def static_fz_front(self) -> float:
        """Static vertical load on ONE front tyre."""
        return 0.5 * self.chassis.mass_kg * G * self.chassis.front_weight_frac

    @property
    def static_fz_rear(self) -> float:
        return 0.5 * self.chassis.mass_kg * G * (1.0 - self.chassis.front_weight_frac)

    def validate(self) -> None:
        c = self.chassis
        assert c.mass_kg > 0 and c.wheelbase_m > 0 and c.yaw_inertia_kgm2 > 0
        assert 0.2 < c.front_weight_frac < 0.8, "front_weight_frac looks wrong"
        assert 0.0 < c.cg_height_m < 1.5
        assert self.powertrain.units, "a car needs at least one power unit"
        for u in self.powertrain.units:
            assert u.kind in ("ice", "motor"), f"unknown unit kind {u.kind}"
            assert u.gearbox in self.powertrain.gearboxes, f"unit {u.name}: unknown gearbox {u.gearbox}"
            assert len(u.rpm) == len(u.torque_nm) >= 2, f"unit {u.name}: bad torque curve"
            assert 0.0 <= u.front_frac <= 1.0
            if u.kind == "motor":
                assert self.powertrain.battery is not None, "motors need a battery"
        for t in (self.tire_front, self.tire_rear):
            assert 0.2 < t.radius_m < 0.6 and t.mu_lat > 0 and t.mu_long > 0
            assert 0.3 < t.falloff_level <= 1.0


# --------------------------------------------------------------------------
# YAML <-> dataclass conversion
# --------------------------------------------------------------------------
def _build(cls: type, data: Any) -> Any:
    """Recursively construct dataclass ``cls`` from plain YAML data."""
    if data is None or not is_dataclass(cls):
        return data
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown field(s) {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        kwargs[key] = _convert(cls, key, value)
    return cls(**kwargs)


_NESTED = {
    (VehicleParams, "chassis"): Chassis, (VehicleParams, "aero"): Aero,
    (VehicleParams, "tire_front"): TireAxle, (VehicleParams, "tire_rear"): TireAxle,
    (VehicleParams, "brakes"): Brakes, (VehicleParams, "steering"): Steering,
    (VehicleParams, "powertrain"): Powertrain, (VehicleParams, "electronics"): Electronics,
    (Powertrain, "battery"): Battery,
}


def _convert(cls: type, key: str, value: Any) -> Any:
    if (cls, key) in _NESTED:
        return _build(_NESTED[(cls, key)], value)
    if cls is Powertrain and key == "units":
        return [_build(PowerUnit, u) for u in value]
    if cls is Powertrain and key == "gearboxes":
        return {k: _build(Gearbox, g) for k, g in value.items()}
    return value


def load_vehicle(path: str | Path) -> VehicleParams:
    """Load and validate a vehicle YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    vp = _build(VehicleParams, data)
    vp.validate()
    return vp


def save_vehicle(vp: VehicleParams, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _round(o: Any) -> Any:
        if isinstance(o, float):
            return float(f"{o:.6g}")
        if isinstance(o, dict):
            return {k: _round(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_round(v) for v in o]
        return o

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_round(asdict(vp)), f, sort_keys=False, default_flow_style=None, width=110)


def find_vehicle(name: str, root: str | Path | None = None) -> Path:
    """Find ``<name>.yaml`` in ``vehicles/local`` (preferred) or ``vehicles/library``."""
    root = Path(root) if root else Path(__file__).resolve().parents[2] / "vehicles"
    if Path(name).is_file():
        return Path(name)
    if (root.parent / name).is_file():           # a path relative to the project, e.g. a lap's car_file
        return root.parent / name
    if name.endswith(".yaml"):
        name = Path(name).stem
    for sub in ("local", "library"):
        p = root / sub / f"{name}.yaml"
        if p.is_file():
            return p
    raise FileNotFoundError(f"vehicle '{name}' not found under {root}")
