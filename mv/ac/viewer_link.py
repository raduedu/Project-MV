"""Show a recorded lap inside Assetto Corsa, with the ghost (engineering log WP32).

Why
---
Lap replays in the web viewer are drawings from above. For presentations - and
to watch *how* the driver handles a corner - it is much clearer to see the
lap in the game: the real car model, the real circuit, any camera (chase,
TV cameras, onboard), and the ideal lap driving alongside as a "ghost".

How
---
Python never touches the game. It writes the lap into a small shared-memory
block (``Local\\mv_viewer_v1``, same mechanism as the bridge, chapter 07).
The CSP Lua app "MV Viewer" (``mv/ac/plugin/apps/lua/mv_viewer``, installed
by the owner with ``install_plugin.ps1``) reads it and, every frame:

* places the player's car where the recorded car was at that moment
  (``physics.setCarPosition`` - CSP puts it on the track surface, so heights
  come from the game) with the recorded speed, steering and pedals, so wheels,
  steering wheel and brake lights move;
* shows the ghost - the physics reference lap for the same car and circuit -
  either as a second car (if the session has one; collisions off) or as a
  marker, plus the ghost's racing line ahead and the time gap.

So the lap you see is *our simulator's* lap re-played on AC's visuals (not
AC's physics). To watch the driver on AC's physics, drive live with
``scripts/ac_drive.py``; the viewer can then still show the ghost.

Coordinates: lap files are in the map frame (map_x = AC x, map_y = -AC z,
chapter 07 section 2); the Lua side converts back.  Only laps on circuits
imported from Assetto Corsa can be shown (generated circuits do not exist in
the game).
"""

from __future__ import annotations

import ctypes
from pathlib import Path
import mmap
from ctypes import c_float, c_int32, c_uint8, c_uint32

import numpy as np

__all__ = ["VIEWER_MAP", "VIEWER_STATE_MAP", "MAX_SAMPLES", "ViewerData", "ViewerState", "ViewerLink",
           "viewer_layout", "lap_arrays", "find_ghost", "viewer_payload", "engine_rpm", "Command"]

VIEWER_MAP = "mv_viewer_v1"
VIEWER_STATE_MAP = "mv_viewer_state_v1"
VIEWER_MAGIC = 0x3157564D          # "MVW1"
VIEWER_STATE_MAGIC = 0x3153564D    # "MVS1"
VIEWER_VERSION = 3                  # 2: engine rpm, QUIT (WP33); 3: ghost model, mass/inertia for the force follower (WP34)
MAX_SAMPLES = 12000                 # 8 minutes at 25 Hz - longer than any lap here
LABEL_LEN = 96
MODEL_LEN = 160                     # path of the car model (.kn5) for the see-through ghost, relative to the game folder


class Command:
    """What the in-game app should do (``ViewerData.command``)."""
    NONE, PLAY, PAUSE, STOP, RESTART, QUIT = 0, 1, 2, 3, 4, 5      # QUIT: close the game (to restart it)


_CAR_FIELDS = ("t", "x", "y", "yaw", "speed", "steer", "thr", "brk", "gear", "rpm")
_GHOST_FIELDS = ("t", "x", "y", "yaw", "speed")


class ViewerData(ctypes.Structure):
    """Python -> Lua: the lap to show and the playback command."""
    _pack_ = 4
    _fields_ = [
        ("magic", c_uint32), ("version", c_uint32), ("seq", c_uint32),
        ("lap_id", c_uint32),               # changes whenever a new lap is loaded
        ("command_seq", c_uint32), ("command", c_int32),
        ("speed", c_float),                 # playback speed, 1 = real time
        ("loop", c_int32),
        ("n_car", c_int32), ("n_ghost", c_int32),
        ("car_lap_time", c_float), ("ghost_lap_time", c_float),
        ("label", c_uint8 * LABEL_LEN),
        ("model", c_uint8 * MODEL_LEN),
        ("mass_kg", c_float), ("yaw_inertia_kgm2", c_float),   # for the force follower (mv_follow.lua)
        *[(f"car_{f}", c_float * MAX_SAMPLES) for f in _CAR_FIELDS],
        *[(f"ghost_{f}", c_float * MAX_SAMPLES) for f in _GHOST_FIELDS],
    ]


class ViewerState(ctypes.Structure):
    """Lua -> Python: what the app is doing (so the UI can say "AC is not running")."""
    _pack_ = 4
    _fields_ = [
        ("magic", c_uint32), ("version", c_uint32), ("seq", c_uint32), ("frame", c_uint32),
        ("lap_id_loaded", c_uint32), ("command_seq_seen", c_uint32),
        ("playing", c_int32), ("physics_allowed", c_int32), ("cars_count", c_int32),
        ("ghost_is_car", c_int32), ("play_time", c_float), ("gap_s", c_float),
    ]


_C_NAMES = {c_uint32: "uint32_t", c_int32: "int32_t", c_float: "float", c_uint8: "uint8_t"}


def viewer_layout(struct: type[ctypes.Structure]) -> str:
    """C struct body for ``ac.writeMemoryMappedFile`` (must equal the Lua file's block; tested)."""
    lines = []
    for name, typ in struct._fields_:
        if hasattr(typ, "_length_"):
            lines.append(f"{_C_NAMES[typ._type_]} {name}[{typ._length_}];")
        else:
            lines.append(f"{_C_NAMES[typ]} {name};")
    return "\n".join(lines)


# ---------------------------------------------------------------------- data preparation
def lap_arrays(lap: dict, fields=_CAR_FIELDS) -> dict[str, np.ndarray]:
    """Sample arrays of a lap file, resampled to at most MAX_SAMPLES, missing fields as zeros."""
    s = lap["samples"]
    n = len(s["t"])
    idx = np.arange(n) if n <= MAX_SAMPLES else np.linspace(0, n - 1, MAX_SAMPLES).round().astype(int)
    out = {}
    for f in fields:
        v = np.asarray(s.get(f, np.zeros(n)), np.float64)
        out[f] = v[idx]
    if "yaw" in out and not np.any(out["yaw"]):                    # older laps: heading from the path
        out["yaw"] = np.unwrap(np.arctan2(np.gradient(out["y"]), np.gradient(out["x"])))
    return out


def engine_rpm(speed, gear, vp) -> np.ndarray:
    """Engine speed (rpm) along a lap, as our simulator computes it (mv/sim/vehicle_sim.py).

    rpm = wheel speed (rad/s) x gear ratio x final drive x 30/pi, never below the
    launch rpm of a combustion engine (the clutch slips there), never above the
    limiter. The recorded ``gear`` is 1-based and belongs to the first gearbox,
    whose driving unit (an engine if there is one) sets the sound in the game.
    """
    pt = vp.powertrain
    gb_key = next(iter(pt.gearboxes))
    gb = pt.gearboxes[gb_key]
    units = [u for u in pt.units if u.gearbox == gb_key]
    u = next((u for u in units if u.kind == "ice"), units[0] if units else pt.units[0])
    radius = u.front_frac * vp.tire_front.radius_m + (1 - u.front_frac) * vp.tire_rear.radius_m
    g = np.clip(np.round(np.asarray(gear, float)).astype(int), 1, len(gb.ratios)) - 1
    ratio = np.asarray(gb.ratios, float)[g] * gb.final_drive
    rpm = np.asarray(speed, float) / radius * ratio * 30.0 / np.pi
    if u.kind == "ice":
        rpm = np.maximum(rpm, u.rpm_idle + 0.35 * (u.rpm_max - u.rpm_idle))
    return np.minimum(rpm, u.rpm_max)


def _stand_in_body(meta: dict, vp):
    """Car file of the game car standing in for a non-game car, or None (normal case)."""
    try:
        from mv.ac.launch import ac_ids_for_lap, car_ids_of, stand_in_for
        from mv.vehicle.params import find_vehicle, load_vehicle
        sid = next((s for s in (stand_in_for(c) for c in car_ids_of(meta, vp)) if s), None)
        if not sid or ac_ids_for_lap(meta)[0] != sid:
            return None
        return load_vehicle(find_vehicle(sid))
    except Exception:
        return None


def stand_in_gears(t, speed, vp, hold_s: float = 0.6, rev_frac: float = 0.92) -> np.ndarray:
    """Gears (1-based) a car would use along a speed trace: the lowest gear that keeps the engine
    below ``rev_frac`` of its limiter (= high revs, strong pull), held at least ``hold_s`` seconds
    between changes so the box does not hunt. Used when the game shows a stand-in body whose
    gearbox differs from the car that drove the lap."""
    pt = vp.powertrain
    gb_key = next(iter(pt.gearboxes))
    gb = pt.gearboxes[gb_key]
    units = [u for u in pt.units if u.gearbox == gb_key]
    u = next((u for u in units if u.kind == "ice"), units[0] if units else pt.units[0])
    radius = u.front_frac * vp.tire_front.radius_m + (1 - u.front_frac) * vp.tire_rear.radius_m
    ratios = np.asarray(gb.ratios, float) * gb.final_drive
    t, v = np.asarray(t, float), np.asarray(speed, float)
    rpm = v[:, None] / radius * ratios[None, :] * 30.0 / np.pi          # (samples, gears)
    ok = rpm <= rev_frac * u.rpm_max
    want = np.where(ok.any(1), ok.argmax(1), len(ratios) - 1) + 1
    out = want.copy()
    last_t = -1e9
    for i in range(1, len(out)):
        out[i] = out[i - 1]
        if want[i] != out[i - 1] and t[i] - last_t >= hold_s:
            out[i] = out[i - 1] + np.sign(want[i] - out[i - 1])        # one gear at a time, like a paddle shift
            last_t = t[i]
    return out.astype(float)


def find_ghost(lap_meta: dict, laps: list[dict]) -> dict | None:
    """Meta of the physics reference lap for the same car and circuit (newest), or None."""
    refs = [m for m in laps if m.get("kind") == "reference" and m.get("car") == lap_meta.get("car")
            and m.get("track") == lap_meta.get("track") and (lap_meta.get("id") is None or m.get("id") != lap_meta.get("id"))]
    return max(refs, key=lambda m: m.get("created", "")) if refs else None


def viewer_payload(key: str, ghost_key: str | None = "auto") -> dict:
    """Everything the viewer needs for one lap: car arrays, ghost arrays, label.

    ``ghost_key``: "auto" = reference lap of the same car and circuit, None = no ghost.
    Raises ValueError for laps on circuits that do not exist in Assetto Corsa.
    """
    from mv.laps import ROOT as PROJECT_ROOT, list_laps, load_lap
    from mv.track.track import Track

    lap = load_lap(key)
    meta = lap["meta"]
    track_file = PROJECT_ROOT / "tracks" / "local" / f"{meta['track']}.npz"
    if not track_file.is_file() or Track.load(track_file).meta.get("source") != "assetto_corsa":
        raise ValueError(f"'{meta['track']}' is not a circuit imported from Assetto Corsa - "
                         "only laps on imported circuits can be shown in the game")
    ghost = None
    if ghost_key == "auto":
        gm = find_ghost(meta, list_laps(True))
        ghost = load_lap(gm["key"]) if gm else None
    elif ghost_key:
        ghost = load_lap(ghost_key)
    car = lap_arrays(lap)
    vp = _vehicle_for(meta)
    if vp is not None and np.any(car["gear"]):
        car["rpm"] = engine_rpm(car["speed"], car["gear"], vp)
    if vp is not None and "steer" not in lap["samples"]:
        # a real lap without a steering channel (F1 timing data, WP43): show the steering a car on
        # that path needs at low slip - wheelbase x path curvature - so the wheels point the right way
        d1x, d1y = np.gradient(car["x"]), np.gradient(car["y"])
        d2x, d2y = np.gradient(d1x), np.gradient(d1y)
        kappa = (d1x * d2y - d1y * d2x) / np.maximum((d1x ** 2 + d1y ** 2) ** 1.5, 1e-6)
        kappa = np.convolve(kappa, np.ones(9) / 9, "same")
        car["steer"] = np.clip(np.arctan(vp.chassis.wheelbase_m * kappa) / np.radians(vp.steering.max_wheel_angle_deg), -1, 1)
    model = ghost_model_path(meta)
    label = f"{meta['driver']} | {meta['car']} @ {meta['track']} | {meta['lap_time']:.3f} s"
    body = _stand_in_body(meta, vp)
    if body is not None:
        # The game shows another car's body (vehicles/ac_stand_ins.yaml, WP46): its own gearbox and
        # engine run in the game, so choose gears for THAT gearbox, and push it with ITS mass.
        car["gear"] = stand_in_gears(car["t"], car["speed"], body)
        car["rpm"] = engine_rpm(car["speed"], car["gear"], body)
        label += f" | shown with the {body.name} body"
        vp = body
        from mv.ac.launch import car_ids_of, stand_in_model
        drawn = next((d for d in (stand_in_model(c) for c in car_ids_of(meta)) if d), None)
        if drawn:                               # the viewer hides the stand-in and draws our own model (WP47)
            model = f"mv:{drawn}"
            label = label.replace(f"shown with the {body.name} body", f"MV {drawn} model (moved as {body.name})")
    if ghost:
        label += f" | ghost {ghost['meta']['lap_time']:.3f} s"
    return {"meta": meta, "label": label, "car": car, "model": model,
            "mass_kg": float(vp.chassis.mass_kg) if vp else 1400.0,
            "yaw_inertia_kgm2": float(vp.chassis.yaw_inertia_kgm2) if vp else 2200.0,
            "ghost": lap_arrays(ghost, _GHOST_FIELDS) if ghost else None,
            "ghost_meta": ghost["meta"] if ghost else None}


def ghost_model_path(meta: dict, ac_root=None) -> str:
    """The car's main 3D model in the game ("content/cars/<id>/<name>.kn5"), or "" if unknown.

    The main model is the largest .kn5 in the car folder that is not a lower-detail
    copy ("_lod_") or the collision shape ("collider")."""
    try:
        from mv.ac.launch import ac_ids_for_lap
        from mv.ac.paths import find_ac_root
        car_id = ac_ids_for_lap(meta)[0]
        root = Path(ac_root) if ac_root else find_ac_root()
        files = [f for f in (root / "content" / "cars" / car_id).glob("*.kn5")
                 if "_lod_" not in f.name.lower() and f.stem.lower() != "collider"]
        if not files:
            return ""
        best = max(files, key=lambda f: f.stat().st_size)
        return f"content/cars/{car_id}/{best.name}"
    except Exception:
        return ""


def _vehicle_for(meta: dict):
    """The car file a lap was driven with (None if it cannot be found)."""
    from mv.vehicle.params import find_vehicle, load_vehicle
    for name in (meta.get("car_file"), meta.get("car")):
        try:
            if name:
                return load_vehicle(find_vehicle(name))
        except FileNotFoundError:
            continue
    return None


# ---------------------------------------------------------------------- shared memory
class ViewerLink:
    """Owns the viewer mappings on the Python side (creates them or opens existing ones).

    Keep one object alive while the game should be able to pick the lap up
    (the web UI keeps it for its lifetime; ``scripts/ac_view.py`` waits until
    the game confirms it has loaded the lap - from then on the game holds the
    mapping open itself).
    """

    def __init__(self, data_name: str = VIEWER_MAP, state_name: str = VIEWER_STATE_MAP):
        self._data_mm = mmap.mmap(-1, ctypes.sizeof(ViewerData), tagname=data_name)
        self._state_mm = mmap.mmap(-1, ctypes.sizeof(ViewerState), tagname=state_name)
        self.data = ViewerData.from_buffer(self._data_mm)
        self._state = ViewerState.from_buffer(self._state_mm)

    def _begin(self) -> None:
        d = self.data
        seq = int(d.seq)
        d.seq = (seq + 1 if seq % 2 == 0 else seq + 2) & 0xFFFFFFFF   # odd: writing
        d.magic, d.version = VIEWER_MAGIC, VIEWER_VERSION

    def _end(self) -> None:
        self.data.seq = (int(self.data.seq) + 1) & 0xFFFFFFFF          # even: consistent

    def load(self, payload: dict, speed: float = 1.0, loop: bool = True, play: bool = True) -> int:
        """Write a lap (from :func:`viewer_payload`) and optionally start it. Returns its lap id."""
        d = self.data
        self._begin()
        car, ghost = payload["car"], payload["ghost"]
        n = len(car["t"])
        for f in _CAR_FIELDS:
            arr = getattr(d, f"car_{f}")
            ctypes.memmove(arr, np.ascontiguousarray(car[f], np.float32).ctypes.data, 4 * n)
        d.n_car = n
        d.car_lap_time = float(payload["meta"]["lap_time"])
        if ghost is not None:
            m = len(ghost["t"])
            for f in _GHOST_FIELDS:
                arr = getattr(d, f"ghost_{f}")
                ctypes.memmove(arr, np.ascontiguousarray(ghost[f], np.float32).ctypes.data, 4 * m)
            d.n_ghost = m
            d.ghost_lap_time = float(payload["ghost_meta"]["lap_time"])
        else:
            d.n_ghost, d.ghost_lap_time = 0, 0.0
        for field, text, size in (("label", payload["label"], LABEL_LEN), ("model", payload.get("model", ""), MODEL_LEN)):
            raw = text.encode("utf-8")[:size - 1]
            buf = getattr(d, field)
            ctypes.memset(buf, 0, size)
            ctypes.memmove(buf, raw, len(raw))
        d.mass_kg = float(payload.get("mass_kg", 1400.0))
        d.yaw_inertia_kgm2 = float(payload.get("yaw_inertia_kgm2", 2200.0))
        d.speed, d.loop = float(speed), int(bool(loop))
        d.lap_id = (int(d.lap_id) + 1) & 0xFFFFFFFF
        d.command = Command.PLAY if play else Command.PAUSE
        d.command_seq = (int(d.command_seq) + 1) & 0xFFFFFFFF
        self._end()
        return int(d.lap_id)

    def command(self, cmd: int, speed: float | None = None) -> None:
        d = self.data
        self._begin()
        d.command = int(cmd)
        if speed is not None:
            d.speed = float(speed)
        d.command_seq = (int(d.command_seq) + 1) & 0xFFFFFFFF
        self._end()

    def state(self) -> dict | None:
        """What the game reports, or None if the MV Viewer app has never written."""
        s = self._state
        if s.magic != VIEWER_STATE_MAGIC:
            return None
        return {"frame": int(s.frame), "lap_id_loaded": int(s.lap_id_loaded), "playing": bool(s.playing),
                "physics_allowed": bool(s.physics_allowed), "cars_count": int(s.cars_count),
                "ghost_is_car": bool(s.ghost_is_car), "play_time": float(s.play_time), "gap_s": float(s.gap_s)}

    def close(self) -> None:
        """Release the mappings (the game keeps its own handle, so a loaded lap stays playable)."""
        del self.data, self._state
        for mm in (self._data_mm, self._state_mm):
            try:
                mm.close()
            except BufferError:          # a caller still holds a view of the struct: freed with it
                pass
