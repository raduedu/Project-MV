"""The car catalogue: every car we can drive, sorted into racing categories.

Why this exists
---------------
Assetto Corsa ships ~180 cars.  The owner wants to train on many of them,
racing cars first (Formula, prototypes, GT3, GT2/GTE, GT4, touring/cup), with
a short list of *favourites* shown first everywhere a car is picked.  This
module answers four questions:

1. **Which category is a car?**  Decided from the car's own description file
   ``ui/ui_car.json`` (its ``class`` "race"/"street", its ``tags`` such as
   ``#GT3`` or ``singleseater``, and its name).  The rules are a data table
   (:data:`RULES`) evaluated top to bottom; the first match wins.  A small
   override table (:data:`OVERRIDES`) fixes the few cars the rules get wrong,
   each with the reason.
2. **Which cars are favourites?**  ``vehicles/favourites.yaml`` - a short,
   hand-edited, committed list of car ids (ids only, no game data).
3. **Is an imported car physically plausible?**  "Imported without an error"
   is not "physically right".  :func:`plausibility` compares our power, mass
   and top speed with the figures the game itself advertises for the car
   (``ui_car.json`` "specs") and drives the car for a few seconds in our
   simulator (full throttle, then full braking) to make sure nothing blows up.
4. **What does the whole list look like?**  :func:`list_cars` merges the cars
   installed in the game, the imported files (``vehicles/local``) and our own
   library (``vehicles/library``).

The results of the (slow) plausibility checks are kept in
``vehicles/local/_catalog.json``.  That folder is git-ignored on purpose: car
names and spec figures come from the game and must not be published.

Nothing here is specific to a car or a track: the rules read the game's own
labels, and the thresholds are broad physical sanity bounds.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[2]
VEHICLES = ROOT / "vehicles"
LOCAL = VEHICLES / "local"
LIBRARY = VEHICLES / "library"
FAVOURITES_FILE = VEHICLES / "favourites.yaml"
REPORT_FILE = LOCAL / "_catalog.json"          # git-ignored: holds names/specs taken from the game

HP_TO_KW = 0.7457                               # 1 mechanical horsepower (bhp) = 745.7 W

# --------------------------------------------------------------------------
# Categories
# --------------------------------------------------------------------------
#: category key -> label shown to people.  Order = order in menus.
CATEGORIES: dict[str, str] = {
    "formula": "Formula (single-seaters)",
    "prototype": "Prototypes (LMP, Group C)",
    "gt3": "GT3",
    "gt2_gte": "GT2 / GTE / GT1",
    "gt4": "GT4",
    "touring_cup": "Touring, cup & other race cars",
    "road": "Road cars",
    "vintage": "Vintage",
    "drift": "Drift",
}
#: The categories the owner calls "racing-style cars".
RACING: tuple[str, ...] = ("formula", "prototype", "gt3", "gt2_gte", "gt4", "touring_cup")


@dataclass(frozen=True)
class Rule:
    """One line of the classification table.

    A car matches when *any* of ``tags`` matches one of its tags (whole tag,
    case-insensitive, leading ``#`` removed, regular expressions allowed) or
    *any* of ``name`` is found in its display name - and, if ``race_only``, the
    car is a race car (see :func:`is_race_car`).  A rule with no tags and no
    names matches every car that passes ``race_only``.
    """
    category: str
    tags: tuple[str, ...] = ()
    name: tuple[str, ...] = ()
    race_only: bool = False


#: Evaluated top to bottom; the first rule that matches decides.  The order
#: matters: e.g. the Praga R1 carries both "lmp3" and "singleseater", and is
#: a prototype; the "Porsche 911 GT3 Cup" has GT3 in its name but is a cup car,
#: so cup names are tested before GT names (GT *tags* are reliable and come first).
RULES: tuple[Rule, ...] = (
    Rule("drift", tags=("drift",), name=(r"\bdrift\b",)),
    Rule("prototype", tags=(r"lmp\d*", r"prototype( c)?", r"proto( c)?", r"group c", r"gr\.? ?c"),
         name=(r"\bLMP\d?\b",)),
    Rule("formula", tags=(r"single ?-?seaters?", r"formula.*", r"open ?-?wheel(er)?", r"indy(car)?", r"f[1-4]"),
         name=(r"\bIndy", r"\bFormula\b")),
    Rule("gt3", tags=("gt3",), race_only=True),
    Rule("gt2_gte", tags=("gt2", "gte", "gt1", "gtlm", "lmgte"), race_only=True),
    Rule("gt4", tags=("gt4",), race_only=True),
    Rule("touring_cup", tags=("touring", "tcr", "dtm", "btcc", r".*\bcup\b.*", r".*challenge.*", r".*trofeo.*", "rally"),
         name=(r"\bCup\b", r"Trofeo", r"Challenge", r"\bDTM\b", r"\bTCR\b"), race_only=True),
    Rule("gt3", name=(r"\bGT3\b",), race_only=True),
    Rule("gt2_gte", name=(r"\bGT[12]\b", r"\bGTE\b", r"\bGT1-", r"\bGTLM\b"), race_only=True),
    Rule("gt4", name=(r"\bGT4\b",), race_only=True),
    Rule("vintage", tags=("vintage", "historic")),
    Rule("touring_cup", race_only=True),            # any other race car: one-make, track-only, rally ...
    Rule("road"),
)

#: car id -> (category, reason).  Only for cars the rules above get wrong.
OVERRIDES: dict[str, tuple[str, str]] = {
    "ks_lamborghini_huracan_st": ("touring_cup", "Super Trofeo one-make car; the game groups it with GT3 but it is not one"),
    "ktm_xbow_r": ("road", "road car; the game also tags it 'race' and 'gt4'"),
    "ks_mclaren_f1_gtr": ("gt2_gte", "1995-97 GT1 race car; the game files it under vintage GT"),
    "ks_mclaren_p1_gtr": ("touring_cup", "track-only car; the game lists it as a street car"),
    # DLC folders without any files (not owned): nothing to read, so say what they are
    "ks_ferrari_488_gt3_2020": ("gt3", "DLC car without files in this install"),
    "ks_ferrari_488_challenge_evo": ("touring_cup", "DLC one-make car without files in this install"),
    # our own library files have no ui_car.json
    "fs_generic_ev": ("formula", "Formula Student car (project library)"),
    "porsche_taycan_turbo_gt_weissach": ("road", "project library"),
    "porsche_911_gt3_rs_992": ("road", "project library"),
    "porsche_918_spyder": ("road", "project library"),
}

#: The tuned "Step 1/2/3" versions the game ships for many road cars.
_TUNED = re.compile(r"(_s\d$)|(\b(step|stage) ?\d\b)|\btuned\b|time attack", re.I)


def _norm_tag(t: str) -> str:
    return str(t).strip().lstrip("#").strip().lower()


def is_race_car(ui: dict) -> bool:
    """A race car per the game's description.

    The ``class`` field says "race" for most race cars; a few (e.g. Lotus Evora
    GX/GTC) have class "street" but a "race" tag and no "street" tag.
    """
    tags = {_norm_tag(t) for t in ui.get("tags", []) or []}
    return str(ui.get("class", "")).lower() == "race" or ("race" in tags and "street" not in tags)


def _rule_matches(rule: Rule, ui: dict) -> bool:
    if rule.race_only and not is_race_car(ui):
        return False
    if not rule.tags and not rule.name:
        return True
    tags = [_norm_tag(t) for t in ui.get("tags", []) or []]
    for pat in rule.tags:
        if any(re.fullmatch(pat, t) for t in tags):
            return True
    name = str(ui.get("name", ""))
    return any(re.search(pat, name, re.I) for pat in rule.name)


def classify(car_id: str, ui: dict | None) -> tuple[str, list[str]]:
    """Category and flags of a car from its ``ui_car.json`` content.

    Returns ``(category, flags)``.  Flags are extra facts that do not decide the
    category: ``tuned`` (a game "Step"/"Stage" variant), ``vintage``, ``rally``,
    ``track-only``, ``override`` (category set by hand, see :data:`OVERRIDES`).
    """
    ui = ui or {}
    tags = {_norm_tag(t) for t in ui.get("tags", []) or []}
    flags: list[str] = []
    if _TUNED.search(car_id) or _TUNED.search(str(ui.get("name", ""))):
        flags.append("tuned")
    if "vintage" in tags or "historic" in tags:
        flags.append("vintage")
    if "rally" in tags:
        flags.append("rally")
    if "hypercars r" in tags:
        flags.append("track-only")
    if car_id in OVERRIDES:
        return OVERRIDES[car_id][0], flags + ["override"]
    for rule in RULES:
        if _rule_matches(rule, ui):
            if rule.category == "vintage":
                flags = [f for f in flags if f != "vintage"]
            return rule.category, flags
    return "road", flags


# --------------------------------------------------------------------------
# Reading the game's description of a car
# --------------------------------------------------------------------------
def read_ui_car(car_dir: str | Path) -> dict:
    """``ui/ui_car.json`` of an AC car folder, or {} if missing/unreadable.

    Kunos' files are not always clean JSON: some are not UTF-8 (Windows-1252
    characters such as the trademark sign), some contain raw tab/newline
    characters inside strings.  We read them leniently.
    """
    p = Path(car_dir) / "ui" / "ui_car.json"
    if not p.is_file():
        return {}
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        data = json.loads(text, strict=False)
        return data if isinstance(data, dict) else {}
    except ValueError:
        # last resort: pull out the few fields we need
        out: dict = {}
        for key in ("name", "brand", "class"):
            m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
            if m:
                out[key] = m.group(1)
        m = re.search(r'"tags"\s*:\s*\[([^\]]*)\]', text)
        if m:
            out["tags"] = re.findall(r'"([^"]*)"', m.group(1))
        return out


_NUM = re.compile(r"(\d+(?:\.\d+)?)")


def _spec_number(text) -> tuple[float | None, bool]:
    """First number in a spec string like ``"250+km/h"`` or ``"1,495kg"``.

    Returns ``(value, at_least)``; ``at_least`` is True for figures written
    as a lower bound ("250+", ">300").  ``"--kg"``, ``"N/A"`` -> ``(None, False)``.
    """
    if text is None:
        return None, False
    s = str(text).replace(",", "")
    m = _NUM.search(s)
    if not m:
        return None, False
    return float(m.group(1)), ("+" in s or ">" in s)


def parse_specs(specs: dict | None) -> dict:
    """The game's advertised figures in SI-ish units.

    Keys (all optional): ``power_kw`` (+ ``power_at_least``, ``power_at_wheels``),
    ``mass_kg``, ``top_kmh`` (+ ``top_at_least``), ``t_0_100_s``.
    ``bhp`` is horsepower: x 0.7457 = kW.  "whp" means measured at the wheels
    (after drivetrain losses), which is less than engine power.
    """
    specs = specs or {}
    out: dict = {}
    hp, plus = _spec_number(specs.get("bhp"))
    if hp:
        out["power_kw"] = hp * HP_TO_KW
        out["power_at_least"] = plus
        out["power_at_wheels"] = "whp" in str(specs.get("bhp", "")).lower()
    kg, _ = _spec_number(specs.get("weight"))
    if kg:
        out["mass_kg"] = kg
    v, plus = _spec_number(specs.get("topspeed"))
    if v:
        out["top_kmh"] = v
        out["top_at_least"] = plus
    acc = specs.get("acceleration")
    if acc and "0-100" in str(acc):
        t, _ = _spec_number(str(acc).split("0-100")[0])
        if t and "<" not in str(acc):
            out["t_0_100_s"] = t
    return out


# --------------------------------------------------------------------------
# Favourites (vehicles/favourites.yaml, committed, ids only)
# --------------------------------------------------------------------------
_FAV_HEADER = """\
# Favourite cars - shown first (starred) wherever the control panel asks for a car.
# Car ids only (the file name of vehicles/local/<id>.yaml or vehicles/library/<id>.yaml).
# Edit by hand or with the star in the control panel. No Assetto Corsa data in this file.
"""


def load_favourites(path: str | Path | None = None) -> list[str]:
    """Favourite car ids, in file order (missing file -> empty list)."""
    import yaml

    p = Path(path) if path else FAVOURITES_FILE
    if not p.is_file():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    ids = data.get("favourites", []) if isinstance(data, dict) else data
    seen: list[str] = []
    for i in ids or []:
        i = str(i).strip()
        if i and i not in seen:
            seen.append(i)
    return seen


def save_favourites(ids: Iterable[str], path: str | Path | None = None) -> None:
    """Write the favourites file (comment header + one id per line)."""
    p = Path(path) if path else FAVOURITES_FILE
    clean: list[str] = []
    for i in ids:
        i = str(i).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,80}", i):
            raise ValueError(f"not a car id: {i!r}")
        if i not in clean:
            clean.append(i)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_FAV_HEADER + "favourites:\n" + "".join(f"  - {i}\n" for i in clean), encoding="utf-8")


def set_favourite(car_id: str, on: bool, path: str | Path | None = None) -> list[str]:
    """Star (``on=True``) or unstar a car; returns the new list."""
    ids = load_favourites(path)
    if on and car_id not in ids:
        ids.append(car_id)
    if not on:
        ids = [i for i in ids if i != car_id]
    save_favourites(ids, path)
    return ids


def favourites(path: str | Path | None = None) -> list[str]:
    """The owner's favourite car ids (same as :func:`load_favourites`)."""
    return load_favourites(path)


# --------------------------------------------------------------------------
# Plausibility check of an imported car
# --------------------------------------------------------------------------
#: Sanity bounds.  Broad on purpose: they catch import mistakes (a factor 2,
#: a wrong unit, a missing table), not small modelling differences.
BOUNDS = {
    # our peak power / advertised power.  Ours is the engine (+ motors) at the
    # crank; the advertised figure is rounded and sometimes metric PS.
    "power": (0.80, 1.25),
    # our mass / advertised mass.  Ours includes the driver (~75 kg) and the
    # default fuel load; the advertised figure usually does not.
    "mass": (0.90, 1.40),
    # our top speed / advertised top speed.  Ours is the gear- or drag-limited
    # speed on a level road (mv/vehicle/summary.py::top_speed).
    "top": (0.85, 1.20),
    # our simulated 0-100 km/h / advertised.  Our tyre model has no wheel spin
    # state yet, so launches can be a little optimistic.
    "t_0_100": (0.55, 1.80),
    "brake_100_0_m": (15.0, 60.0),     # stopping distance from 100 km/h, metres
    "grip_g": (0.7, 2.5),              # static-load tyre friction, in g
}


def sim_check(vp, seconds_throttle: float = 8.0, seconds_brake: float = 5.0, device: str = "cpu") -> dict:
    """Drive the car in our simulator: does it accelerate, stop, and stay finite?

    Cost: ~3000 simulation steps of 3 cars, a few seconds on one CPU core.

    Three copies of the car run together:

    * row 0: from rest, full throttle straight ahead, then full braking;
    * row 1: the same with a little steering, so the cornering forces and load
      transfer are exercised too;
    * row 2: starts at 100 km/h and brakes to a stop (braking distance).

    Returns 0-100 km/h time (s, None if never reached), speed reached (km/h),
    100-0 km/h braking distance (m) and ``finite`` (no NaN/inf anywhere).
    """
    import torch

    from mv.sim.vehicle_sim import VehicleSim

    sim = VehicleSim(vp, 3, device)
    dev = sim.device
    sim.reset(torch.arange(3, device=dev), torch.zeros(3, device=dev), torch.zeros(3, device=dev),
              torch.zeros(3, device=dev), torch.tensor([0.0, 0.0, 100 / 3.6], device=dev))
    steer = torch.tensor([0.0, 0.08, 0.0], device=dev)
    n_thr, n_brk = int(seconds_throttle / sim.dt), int(seconds_brake / sim.dt)
    t_100, v_max, brake_dist = None, 0.0, None
    for i in range(n_thr + n_brk):
        throttling = i < n_thr
        thr = torch.tensor([1.0 if throttling else 0.0] * 2 + [0.0], device=dev)
        brk = torch.tensor([0.0 if throttling else 1.0] * 2 + [1.0], device=dev)
        sim.step(steer, thr, brk)
        if i % 25 == 0 or throttling:
            vx = sim.vx.tolist()
            if t_100 is None and vx[0] >= 100 / 3.6:
                t_100 = (i + 1) * sim.dt
            v_max = max(v_max, vx[0])
            if brake_dist is None and vx[2] < 0.5:
                brake_dist = float(sim.x[2])
    state = torch.stack([sim.x, sim.y, sim.yaw, sim.vx, sim.vy, sim.r])
    finite = bool(torch.isfinite(state).all()) and float(state[3:5].abs().max()) < 200.0
    return {"t_0_100_s": t_100, "v_max_kmh": v_max * 3.6, "brake_100_0_m": brake_dist, "finite": finite}


def our_numbers(vp) -> dict:
    """Back-of-the-envelope numbers of a vehicle file (no simulation)."""
    from mv.vehicle.params import G
    from mv.vehicle.summary import top_speed, unit_peak_power_kw

    c = vp.chassis
    power = sum(unit_peak_power_kw(u) for u in vp.powertrain.units)
    if vp.powertrain.battery and vp.powertrain.type == "ev":
        power = min(power, vp.powertrain.battery.max_discharge_kw)
    grip = c.front_weight_frac * vp.tire_front.mu_lat + (1 - c.front_weight_frac) * vp.tire_rear.mu_lat
    brake_f = vp.brakes.max_torque_nm * (vp.brakes.front_share / vp.tire_front.radius_m
                                         + (1 - vp.brakes.front_share) / vp.tire_rear.radius_m)
    q200 = 0.5 * vp.aero.air_density * (200 / 3.6) ** 2
    return {
        "power_kw": power, "mass_kg": c.mass_kg, "top_kmh": top_speed(vp) * 3.6, "grip_g": grip,
        "brake_capacity_g": brake_f / (c.mass_kg * G), "type": vp.powertrain.type,
        "downforce_200_kg": q200 * (vp.aero.cla_front_m2 + vp.aero.cla_rear_m2) / G,
    }


def plausibility(vp, spec: dict | None = None, simulate: bool = True) -> tuple[str, list[str], dict]:
    """Judge an imported car: ``("ok" | "suspicious" | "failed", reasons, numbers)``.

    ``spec`` is :func:`parse_specs` of the game's ui_car.json (may be empty, e.g.
    for our library cars).  "failed" = the simulator produced NaN/inf (unusable);
    "suspicious" = usable but at least one number is outside :data:`BOUNDS`.
    """
    spec = spec or {}
    ours = our_numbers(vp)
    reasons: list[str] = []

    def ratio_check(key: str, our: float | None, theirs: float | None, unit: str, at_least: bool = False,
                    bounds: tuple[float, float] | None = None, label: str = ""):
        if our is None or not theirs:
            return
        lo, hi = bounds or BOUNDS[key]
        r = our / theirs
        if r < lo or (r > hi and not at_least):
            d = 1 if unit == "s" else 0                 # seconds with one decimal, the rest whole numbers
            reasons.append(f"{label or key} {our:.{d}f} {unit} vs game spec {'>=' if at_least else ''}{theirs:.{d}f} {unit} "
                           f"({(r - 1) * 100:+.0f} %)")

    p_bounds = (0.90, 1.60) if spec.get("power_at_wheels") else None    # wheel figure < engine figure
    ratio_check("power", ours["power_kw"], spec.get("power_kw"), "kW", spec.get("power_at_least", False), p_bounds)
    ratio_check("mass", ours["mass_kg"], spec.get("mass_kg"), "kg")
    if reasons and reasons[-1].startswith("mass"):
        reasons[-1] += " - ours includes driver and the game's default fuel load"
    ratio_check("top", ours["top_kmh"], spec.get("top_kmh"), "km/h", spec.get("top_at_least", False),
                label="top speed")
    lo, hi = BOUNDS["grip_g"]
    if not lo <= ours["grip_g"] <= hi:
        reasons.append(f"tyre grip {ours['grip_g']:.2f} g outside {lo}-{hi} g")
    if ours["brake_capacity_g"] < 0.8:
        reasons.append(f"brakes can only produce {ours['brake_capacity_g']:.2f} g")
    status = "ok"
    if simulate:
        try:
            s = sim_check(vp)
        except Exception as e:                       # the simulator itself refused the car
            return "failed", [f"simulator error: {type(e).__name__}: {e}"], ours
        ours.update(s)
        if not s["finite"]:
            return "failed", ["simulator produced NaN/inf or absurd speeds"], ours
        if s["t_0_100_s"] is None:
            # Only a fault if the car should manage it: the game says so, or (no figure) it has
            # more than 100 W/kg - slow cars (a 1960s Fiat 500 needs ~26 s) are fine.
            expected = spec.get("t_0_100_s")
            strong = ours["power_kw"] * 1000.0 / ours["mass_kg"] > 100.0
            if (expected is not None and expected < 8.0) or (expected is None and strong):
                reasons.append(f"does not reach 100 km/h in 8 s of full throttle (got {s['v_max_kmh']:.0f} km/h)")
        elif spec.get("t_0_100_s"):
            ratio_check("t_0_100", s["t_0_100_s"], spec["t_0_100_s"], "s", label="0-100 km/h")
        lo, hi = BOUNDS["brake_100_0_m"]
        if s["brake_100_0_m"] is None:
            reasons.append("does not stop from 100 km/h within 5 s")
        elif not lo <= s["brake_100_0_m"] <= hi:
            reasons.append(f"100-0 km/h braking distance {s['brake_100_0_m']:.0f} m outside {lo:.0f}-{hi:.0f} m")
    if reasons:
        status = "suspicious"
    return status, reasons, ours


# --------------------------------------------------------------------------
# The report (vehicles/local/_catalog.json, git-ignored)
# --------------------------------------------------------------------------
def load_report(path: str | Path | None = None) -> dict:
    p = Path(path) if path else REPORT_FILE
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def save_report(report: dict, path: str | Path | None = None) -> None:
    p = Path(path) if path else REPORT_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


# --------------------------------------------------------------------------
# The merged list
# --------------------------------------------------------------------------
@dataclass
class CarEntry:
    """One row of the catalogue."""
    id: str
    title: str
    category: str
    flags: list[str] = field(default_factory=list)
    brand: str = ""
    favourite: bool = False
    installed: bool = False        # a folder of this name exists in the game
    available: bool = False        # physics data present (game) or our own library file
    imported: bool = False         # a vehicle file exists (vehicles/local or vehicles/library)
    where: str = ""                # "local" | "library" | "" (not imported)
    status: str = "unchecked"      # ok | suspicious | failed | unavailable | unchecked
    reasons: list[str] = field(default_factory=list)
    ours: dict = field(default_factory=dict)
    spec: dict = field(default_factory=dict)

    @property
    def racing(self) -> bool:
        return self.category in RACING

    def to_dict(self) -> dict:
        d = asdict(self)
        d["racing"] = self.racing
        d["category_label"] = CATEGORIES.get(self.category, self.category)
        return d


def _ac_cars_dir(ac_root: str | Path | None) -> Path | None:
    if ac_root is False:
        return None
    try:
        from mv.ac.paths import find_ac_root
        d = find_ac_root(ac_root) / "content" / "cars"
        return d if d.is_dir() else None
    except Exception:
        return None


def list_cars(ac_root: str | Path | None = None, vehicles_dir: str | Path | None = None,
              favourites_path: str | Path | None = None, report: dict | None = None) -> list[CarEntry]:
    """Every car we know of: installed in the game, imported, or in our library.

    Sorted favourites first (in favourites-file order), then by category
    (order of :data:`CATEGORIES`), then by title.  ``ac_root=False`` skips the
    game folder (used by tests).
    """
    from mv.ac.acd import car_has_data

    vdir = Path(vehicles_dir) if vehicles_dir else VEHICLES
    favs = load_favourites(favourites_path or (vdir / "favourites.yaml"))
    rep = load_report(vdir / "local" / "_catalog.json") if report is None else report
    entries: dict[str, CarEntry] = {}

    cars_dir = _ac_cars_dir(ac_root)
    if cars_dir:
        for d in sorted(cars_dir.iterdir()):
            if not d.is_dir():
                continue
            ui = read_ui_car(d)
            cat, flags = classify(d.name, ui)
            available = car_has_data(d)
            entries[d.name] = CarEntry(
                id=d.name, title=str(ui.get("name") or d.name), brand=str(ui.get("brand", "")), category=cat,
                flags=flags, installed=True, available=available, spec=parse_specs(ui.get("specs")),
                status="unchecked" if available else "unavailable",
                reasons=[] if available else ["not installed (no physics data - e.g. a DLC that is not owned)"])
    for sub in ("local", "library"):
        for p in sorted((vdir / sub).glob("*.yaml")):
            e = entries.get(p.stem)
            if e is None:
                r = rep.get(p.stem, {})
                cat, flags = (r["category"], r.get("flags", [])) if "category" in r else classify(p.stem, {})
                e = entries[p.stem] = CarEntry(id=p.stem, title=r.get("title") or _yaml_title(p) or p.stem,
                                               category=cat, flags=flags, spec=r.get("spec", {}))
            if not e.imported:                   # vehicles/local wins over the library (find_vehicle's order)
                e.imported, e.where, e.available = True, sub, True
    for cid, e in entries.items():
        r = rep.get(cid)
        if r and e.imported and r.get("status"):
            e.status, e.reasons, e.ours = r["status"], list(r.get("reasons", [])), dict(r.get("ours", {}))
        elif r and not e.imported and r.get("status") == "failed":
            e.status, e.reasons = "failed", list(r.get("reasons", []))
        e.favourite = cid in favs
    order = {c: i for i, c in enumerate(CATEGORIES)}
    fav_pos = {c: i for i, c in enumerate(favs)}
    return sorted(entries.values(), key=lambda e: (0 if e.favourite else 1, fav_pos.get(e.id, 0),
                                                    order.get(e.category, 99), e.title.lower()))


def _yaml_title(p: Path) -> str:
    """The ``name:`` line of a vehicle file (cheap: no full YAML parse)."""
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.startswith("name:"):
                    return line.split(":", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return ""


def racing_cars(cars: list[CarEntry] | None = None, usable_only: bool = False) -> list[CarEntry]:
    """Cars in the racing categories (:data:`RACING`).

    ``usable_only``: only imported cars whose check did not fail.
    """
    cars = list_cars() if cars is None else cars
    out = [c for c in cars if c.racing]
    if usable_only:
        out = [c for c in out if c.imported and c.status in ("ok", "suspicious", "unchecked")]
    return out


def select_ids(spec: str, cars: list[CarEntry]) -> list[str]:
    """Car ids for a selection word: ``all``, ``racing``, ``favourites``, a
    category key (``gt3``), or several of these separated by commas."""
    out: list[str] = []
    for word in [w.strip().lower() for w in spec.split(",") if w.strip()]:
        if word == "all":
            pick = cars
        elif word == "racing":
            pick = [c for c in cars if c.racing]
        elif word in ("favourites", "favorites", "fav"):
            pick = [c for c in cars if c.favourite]
        elif word in CATEGORIES:
            pick = [c for c in cars if c.category == word]
        else:
            raise ValueError(f"unknown selection '{word}': use all, racing, favourites or one of {', '.join(CATEGORIES)}")
        out += [c.id for c in pick if c.id not in out]
    return out


# --------------------------------------------------------------------------
# Batch import from the game
# --------------------------------------------------------------------------
def import_cars(car_ids: Iterable[str], cars_dir: str | Path, out_dir: str | Path | None = None,
                check: bool = True, simulate: bool = True, fuel_litres: float | None = None,
                report_path: str | Path | None = None, log: Callable[[str], None] = print) -> list[dict]:
    """Import many game cars; never stop at a bad one.

    For each car: no physics data -> "unavailable" (skipped); an exception
    while reading -> "failed" with the error; otherwise the vehicle file is
    written to ``out_dir/<id>.yaml`` and, if ``check``, judged by
    :func:`plausibility`.  Results are merged into the report file after every
    car (so an interrupted batch keeps what it did) and returned.
    """
    from mv.ac.acd import CarNotAvailable, car_has_data
    from mv.ac.car_import import import_ac_car
    from mv.vehicle.params import save_vehicle

    cars_dir = Path(cars_dir)
    out_dir = Path(out_dir) if out_dir else LOCAL
    rpath = Path(report_path) if report_path else out_dir / "_catalog.json"
    report = load_report(rpath)
    results = []
    ids = list(car_ids)
    for k, cid in enumerate(ids, 1):
        d = cars_dir / cid
        ui = read_ui_car(d)
        cat, flags = classify(cid, ui)
        row = {"id": cid, "title": str(ui.get("name") or cid), "category": cat, "flags": flags,
               "spec": parse_specs(ui.get("specs")), "checked": time.strftime("%Y-%m-%d %H:%M")}
        t0 = time.time()
        try:
            if not d.is_dir() or not car_has_data(d):
                raise CarNotAvailable(f"{cid}: no physics data")
            vp = import_ac_car(d, fuel_litres=fuel_litres)
            save_vehicle(vp, out_dir / f"{cid}.yaml")
            row["status"], row["reasons"] = "unchecked", []
            if check:
                row["status"], row["reasons"], row["ours"] = plausibility(vp, row["spec"], simulate=simulate)
        except CarNotAvailable:
            row["status"], row["reasons"] = "unavailable", ["not installed (no physics data - e.g. a DLC that is not owned)"]
        except Exception as e:                      # one broken car must not stop the batch
            row["status"], row["reasons"] = "failed", [f"{type(e).__name__}: {e}"]
        row["ours"] = _round(row.get("ours", {}))
        row["spec"] = _round(row["spec"])
        report[cid] = row
        save_report(report, rpath)
        results.append(row)
        log(f"[{k}/{len(ids)}] {cid:<34} {row['status']:<11} {time.time() - t0:5.1f} s"
            + (f"  - {'; '.join(row['reasons'])}" if row["reasons"] else ""))
    return results


def check_vehicle_file(car_id: str, report_path: str | Path | None = None, simulate: bool = True) -> dict:
    """Plausibility check of an existing vehicle file (e.g. a library car); updates the report."""
    from mv.vehicle.params import find_vehicle, load_vehicle

    rpath = Path(report_path) if report_path else REPORT_FILE
    report = load_report(rpath)
    p = find_vehicle(car_id)
    cat, flags = classify(car_id, {})
    old = report.get(car_id, {})
    row = {"id": car_id, "title": old.get("title") or _yaml_title(p) or car_id,
           "category": old.get("category", cat), "flags": old.get("flags", flags), "spec": old.get("spec", {}),
           "checked": time.strftime("%Y-%m-%d %H:%M")}
    try:
        vp = load_vehicle(p)
        row["status"], row["reasons"], ours = plausibility(vp, row["spec"], simulate=simulate)
        row["ours"] = _round(ours)
    except Exception as e:
        row["status"], row["reasons"] = "failed", [f"{type(e).__name__}: {e}"]
    report[car_id] = row
    save_report(report, rpath)
    return row


def _round(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            v = None if not math.isfinite(v) else round(v, 3)
        out[k] = v
    return out


# --------------------------------------------------------------------------
# Text table (for the terminal and the engineering log)
# --------------------------------------------------------------------------
def table(cars: list[CarEntry], markdown: bool = True) -> str:
    """Per-car table: id, name, category, our power/mass/top speed vs the game's spec, status."""
    def num(v, fmt="{:.0f}"):
        return fmt.format(v) if isinstance(v, (int, float)) and v is not None else "-"

    head = ["", "id", "name", "category", "kW ours/spec", "kg ours/spec", "km/h ours/spec", "0-100 s", "status"]
    rows = []
    for c in cars:
        o, s = c.ours, c.spec
        status = c.status + (f": {'; '.join(c.reasons)}" if c.reasons and c.status != "ok" else "")
        rows.append(["*" if c.favourite else "", c.id, c.title, c.category,
                     f"{num(o.get('power_kw'))}/{num(s.get('power_kw'))}{'+' if s.get('power_at_least') else ''}",
                     f"{num(o.get('mass_kg'))}/{num(s.get('mass_kg'))}",
                     f"{num(o.get('top_kmh'))}/{num(s.get('top_kmh'))}{'+' if s.get('top_at_least') else ''}",
                     f"{num(o.get('t_0_100_s'), '{:.1f}')}/{num(s.get('t_0_100_s'), '{:.1f}')}", status])
    if markdown:
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        lines += ["| " + " | ".join(str(x).replace("|", "/") for x in r) + " |" for r in rows]
        return "\n".join(lines)
    return "\n".join("  ".join(str(x) for x in r) for r in rows)
