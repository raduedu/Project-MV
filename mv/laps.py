"""Recorded laps: full telemetry of a lap, saved so it can be replayed and checked.

Why this exists
---------------
A lap time on its own is just a number.  A recorded lap lets anyone *watch* the
car go round - where it brakes, how much it steers, how fast it carries
through each corner - and download the raw data.  Every lap file also says
where it came from (which driver, which training update, which car file,
when), so a number can always be traced back to how it was produced.

What a lap file contains (JSON, one file per lap)
-------------------------------------------------
``meta``      car, track, driver, kind ("learned", "rule-based", "reference"),
              lap time, track length, date, source (e.g. "training validation,
              update 1200"), checkpoint, notes.
``track``     the track edges (every ~3 m) and the start line, so the file can
              be drawn on its own - no other file needed.
``samples``   25 per second: t [s], s [m along the lap], x, y [m], yaw [rad],
              speed [m/s], steer [-1..1 of full lock, + = left], thr, brk [0..1],
              gear.

Where they live
---------------
``runs/laps/<car>/<track>/<id>.json`` (+ a small ``<id>.meta.json`` for fast
listing).  ``runs/`` is not published.  A few example laps on tracks generated
by the project itself live in ``docs/laps/`` and are published; laps on tracks
imported from Assetto Corsa never are (they contain the track's shape).
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np

from mv.track.track import Track

ROOT = Path(__file__).resolve().parents[1]
LAP_DIR = ROOT / "runs" / "laps"
EXAMPLE_DIR = ROOT / "docs" / "laps"

__all__ = ["make_lap", "laps_from_result", "save_lap", "list_laps", "load_lap", "reference_lap_file",
           "lap_to_csv", "LAP_DIR", "EXAMPLE_DIR"]

_ROUND = {"t": 3, "s": 2, "x": 2, "y": 2, "yaw": 4, "speed": 2, "steer": 3, "thr": 3, "brk": 3, "gear": 0}


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", text).strip("_")[:60] or "lap"


def _track_geometry(track: Track, step_m: float = 3.0) -> dict:
    k = max(int(round(step_m / track.ds)), 1)
    r = lambda a: np.round(np.asarray(a)[::k], 2).tolist()  # noqa: E731
    return {"name": track.name, "length_m": round(track.length, 1), "left": r(track.left_xy), "right": r(track.right_xy),
            "centre": r(track.xy), "start": [round(float(track.xy[0, 0]), 2), round(float(track.xy[0, 1]), 2),
                                             round(float(track.psi[0]), 4)],
            "direction": track.turning_direction()}


def make_lap(car: str, track: Track, driver: str, kind: str, lap_time: float, samples: dict[str, np.ndarray],
             source: str, checkpoint: str = "", notes: str = "", car_file: str = "", track_recipe: dict | None = None) -> dict:
    """Build a lap dictionary from sample arrays (times from 0, distance from the start line)."""
    out = {}
    for f, nd in _ROUND.items():
        if f in samples:
            v = np.asarray(samples[f], float)
            out[f] = np.round(v, nd).astype(int).tolist() if nd == 0 else np.round(v, nd).tolist()
    meta = {"car": car, "track": track.name, "driver": driver, "kind": kind, "lap_time": round(float(lap_time), 3),
            "track_length_m": round(track.length, 1), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source, "checkpoint": checkpoint, "notes": notes,
            # what is needed to drive this lap again (scripts/reproduce_lap.py)
            "car_file": car_file, "track_recipe": track_recipe or track.meta.get("recipe"), "eval": "per-car",
            "vmax_kmh": round(float(np.max(samples["speed"])) * 3.6, 1),
            "avg_kmh": round(track.length / float(lap_time) * 3.6, 1), "samples": len(out.get("t", [])),
            # how busy the steering is: total wheel travel per second, in fractions of full lock (jitter shows up here)
            "steer_activity": round(float(np.abs(np.diff(np.asarray(samples["steer"], float))).sum()) / float(lap_time), 3)
            if "steer" in samples and len(samples["steer"]) > 1 else None}
    return {"meta": meta, "track": _track_geometry(track), "samples": out}


def laps_from_result(res, track: Track, car: str, driver: str, kind: str, source: str, checkpoint: str = "",
                     flying_only: bool = True, car_file: str = "") -> list[dict]:
    """Cut the recorded trace of an evaluation run (``mv.rl.evaluate``) into individual laps.

    Lap 1 starts from a standstill on the line; later laps are "flying".  By
    default only flying laps are returned.
    """
    tr = res.trace
    if not tr or not res.lap_times:
        return []
    t, s = np.asarray(tr["t"]), np.asarray(tr["s"])
    ends = np.cumsum(res.lap_times)
    starts = np.concatenate([[0.0], ends[:-1]])
    L = track.length
    laps = []
    for k, (t0, t1) in enumerate(zip(starts, ends)):
        if flying_only and k == 0:
            continue
        m = (t > t0 + 1e-6) & (t <= t1 + 1e-6)
        if m.sum() < 10:
            continue
        smp = {f: np.asarray(tr[f])[m] for f in _ROUND if f in tr}
        smp["t"] = t[m] - t0
        smp["s"] = np.clip(s[m] - k * L, 0.0, L)
        # start the lap exactly on the line: prepend the interpolated crossing point
        laps.append(make_lap(car, track, driver, kind, float(res.lap_times[k]), smp, source + f", lap {k + 1}",
                             checkpoint, car_file=car_file))
    return laps


def save_lap(lap: dict, root: Path | None = None) -> Path:
    """Write a lap file (+ its small meta file).  Returns the path."""
    meta = lap["meta"]
    root = root or LAP_DIR
    d = root / _slug(meta["car"]) / _slug(meta["track"])
    d.mkdir(parents=True, exist_ok=True)
    lap_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}_{_slug(meta['driver'])}"
    meta["id"] = f"{d.relative_to(root).as_posix()}/{lap_id}"
    path = d / f"{lap_id}.json"
    path.write_text(json.dumps(lap, separators=(",", ":")))
    (d / f"{lap_id}.meta.json").write_text(json.dumps(meta))
    return path


def list_laps(include_examples: bool = True) -> list[dict]:
    """Meta data of every saved lap (fast: reads only the small meta files)."""
    out = []
    for root, origin in ((LAP_DIR, "local"), (EXAMPLE_DIR, "example")):
        if origin == "example" and not include_examples:
            continue
        if not root.is_dir():
            continue
        for f in root.rglob("*.meta.json"):
            try:
                m = json.loads(f.read_text())
            except Exception:
                continue
            m["origin"] = origin
            m["key"] = f"{origin}:{m['id']}"
            out.append(m)
    out.sort(key=lambda m: (m["track"], m["car"], m["lap_time"]))
    return out


def load_lap(key: str) -> dict:
    """Load a lap by the ``key`` given by :func:`list_laps` (``local:...`` or ``example:...``)."""
    origin, lap_id = key.split(":", 1)
    root = EXAMPLE_DIR if origin == "example" else LAP_DIR
    path = (root / f"{lap_id}.json").resolve()
    if not str(path).startswith(str(root.resolve())) or not path.is_file():
        raise FileNotFoundError(key)
    return json.loads(path.read_text())


def best_lap_time(car: str, track: str, driver: str) -> float:
    """Fastest saved lap of this driver on this car/track (inf if none)."""
    times = [m["lap_time"] for m in list_laps(False) if m["car"] == car and m["track"] == track and m["driver"] == driver]
    return min(times) if times else math.inf


def reference_lap_file(car: str, track: Track, ref_npz: Path, wheelbase_m: float, max_steer_rad: float) -> dict:
    """Turn a physics reference lap (``scripts/reference_lap.py``) into a lap file.

    The reference is a speed profile along a line, not a driven lap, so heading,
    steering and pedals are derived from it: heading from the line direction,
    steering from curvature (wheelbase x curvature), throttle/brake from the
    longitudinal acceleration.  It is the "ideal" ghost to race against.
    """
    z = np.load(ref_npz, allow_pickle=False)
    xy, v, tt, s = z["xy"], z["speed"], z["time"], z["s"]
    ax, kappa = z["ax"], z["kappa"]
    d = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    yaw = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    steer = np.clip(np.arctan(wheelbase_m * kappa) / max_steer_rad, -1, 1)
    a_max = max(float(np.max(ax)), 1e-3)
    a_min = min(float(np.min(ax)), -1e-3)
    thr = np.clip(ax / a_max, 0, 1)
    brk = np.clip(ax / a_min, 0, 1)
    t_s = np.arange(0.0, float(z["lap_time"]), 0.04)                 # resample to 25 per second, like driven laps
    smp = {"t": t_s, "s": np.interp(t_s, tt, s), "x": np.interp(t_s, tt, xy[:, 0]), "y": np.interp(t_s, tt, xy[:, 1]),
           "yaw": np.interp(t_s, tt, yaw), "speed": np.interp(t_s, tt, v), "steer": np.interp(t_s, tt, steer),
           "thr": np.interp(t_s, tt, thr), "brk": np.interp(t_s, tt, brk), "gear": np.zeros_like(t_s)}
    return make_lap(car, track, "physics reference", "reference", float(z["lap_time"]), smp,
                    f"scripts/reference_lap.py ({ref_npz.name})",
                    notes="Ideal lap computed from physics (racing line + speed profile); pedals and steering derived.")


def lap_to_csv(lap: dict) -> str:
    s = lap["samples"]
    cols = [c for c in ("t", "s", "x", "y", "yaw", "speed", "steer", "thr", "brk", "gear") if c in s]
    head = ["t_s", "distance_m", "x_m", "y_m", "yaw_rad", "speed_m_s", "steer_frac", "throttle", "brake", "gear"]
    lines = [",".join(h for h, c in zip(head, ("t", "s", "x", "y", "yaw", "speed", "steer", "thr", "brk", "gear")) if c in s)]
    for i in range(len(s["t"])):
        lines.append(",".join(str(s[c][i]) for c in cols))
    m = lap["meta"]
    header = (f"# Project MV lap: {m['driver']} | {m['car']} | {m['track']} | {m['lap_time']} s | {m['created']} | "
              f"source: {m['source']}\n")
    return header + "\n".join(lines) + "\n"
