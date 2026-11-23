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
import mmap
from ctypes import c_float, c_int32, c_uint8, c_uint32

import numpy as np

__all__ = ["VIEWER_MAP", "VIEWER_STATE_MAP", "MAX_SAMPLES", "ViewerData", "ViewerState", "ViewerLink",
           "viewer_layout", "lap_arrays", "find_ghost", "viewer_payload", "Command"]

VIEWER_MAP = "mv_viewer_v1"
VIEWER_STATE_MAP = "mv_viewer_state_v1"
VIEWER_MAGIC = 0x3157564D          # "MVW1"
VIEWER_STATE_MAGIC = 0x3153564D    # "MVS1"
VIEWER_VERSION = 1
MAX_SAMPLES = 12000                 # 8 minutes at 25 Hz - longer than any lap here
LABEL_LEN = 96


class Command:
    """What the in-game app should do (``ViewerData.command``)."""
    NONE, PLAY, PAUSE, STOP, RESTART = 0, 1, 2, 3, 4


_CAR_FIELDS = ("t", "x", "y", "yaw", "speed", "steer", "thr", "brk", "gear")
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


def find_ghost(lap_meta: dict, laps: list[dict]) -> dict | None:
    """Meta of the physics reference lap for the same car and circuit (newest), or None."""
    refs = [m for m in laps if m.get("kind") == "reference" and m.get("car") == lap_meta.get("car")
            and m.get("track") == lap_meta.get("track")]
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
    label = f"{meta['driver']} | {meta['car']} @ {meta['track']} | {meta['lap_time']:.3f} s"
    if ghost:
        label += f" | ghost {ghost['meta']['lap_time']:.3f} s"
    return {"meta": meta, "label": label, "car": lap_arrays(lap),
            "ghost": lap_arrays(ghost, _GHOST_FIELDS) if ghost else None,
            "ghost_meta": ghost["meta"] if ghost else None}


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
        raw = payload["label"].encode("utf-8")[:LABEL_LEN - 1]
        ctypes.memset(d.label, 0, LABEL_LEN)
        ctypes.memmove(d.label, raw, len(raw))
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
