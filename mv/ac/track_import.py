"""Import an Assetto Corsa circuit as a Project MV :class:`Track`.

AC stores the line its AI drivers follow in ``<track>/ai/fast_lane.ai`` (or
``<track>/<layout>/ai/fast_lane.ai``).  Besides the line itself the file holds,
for every point, the distance to the left and right edge of the track.  That is
everything we need.

Binary layout (little endian)
-----------------------------
::

    int32 version (7)   int32 point_count   int32 lap_time   int32 sample_count
    point_count  x { float x, y, z, distance_from_start ; int32 id }        (20 bytes)
    int32 extra_count
    extra_count  x { 18 floats: speed, gas, brake, obsolete_lat_g, radius,
                     side_left, side_right, camber, direction,
                     normal xyz, length, forward xyz, tag, grade }           (72 bytes)

AC's world has y pointing up; the ground plane is (x, z).  Seen from above, AC's
+z axis points *down* the map, so we use  map_x = x,  map_y = -z  to get a
right-handed top view in which "left" really is left.  The bridge that reads
live car positions from the game must apply the same transform.

The racing line is not the centre line, so the centre line is rebuilt from the
two edges:  centre = line + normal * (side_left - side_right) / 2.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from mv.track.track import Track

__all__ = ["read_ai_spline", "import_ac_track", "ac_world_to_map"]


def ac_world_to_map(x, z):
    """Convert AC world ground-plane coordinates to Project MV map coordinates."""
    return np.asarray(x, float), -np.asarray(z, float)


def read_ai_spline(path: str | Path) -> dict[str, np.ndarray]:
    raw = Path(path).read_bytes()
    version, count, _lap_time, _samples = struct.unpack_from("<4i", raw, 0)
    if version != 7:
        raise ValueError(f"unsupported AI spline version {version} in {path}")
    pts = np.frombuffer(raw, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("dist", "<f4"), ("id", "<i4")]),
                        count=count, offset=16)
    off = 16 + 20 * count
    (extra_count,) = struct.unpack_from("<i", raw, off)
    off += 4
    if extra_count != count:
        raise ValueError(f"AI spline has {count} points but {extra_count} detail records")
    extra = np.frombuffer(raw, dtype="<f4", count=18 * count, offset=off).reshape(count, 18)
    return {
        "x": pts["x"].astype(float), "y": pts["y"].astype(float), "z": pts["z"].astype(float),
        "dist": pts["dist"].astype(float),
        "speed": extra[:, 0].astype(float), "side_left": extra[:, 5].astype(float),
        "side_right": extra[:, 6].astype(float),
    }


def import_ac_track(ac_root: str | Path, track: str, layout: str | None = None, ds: float = 1.0,
                    smooth_m: float = 3.0) -> Track:
    """Load ``track`` (and optional ``layout``) from an AC install."""
    base = Path(ac_root) / "content" / "tracks" / track
    if layout:
        base = base / layout
    ai = base / "ai" / "fast_lane.ai"
    if not ai.is_file():
        layouts = [p.name for p in (Path(ac_root) / "content" / "tracks" / track).iterdir()
                   if (p / "ai" / "fast_lane.ai").is_file()]
        raise FileNotFoundError(f"{ai} not found. Available layouts: {layouts}")
    d = read_ai_spline(ai)
    mx, my = ac_world_to_map(d["x"], d["z"])
    line = np.stack([mx, my], axis=1)
    # direction of travel from neighbouring points (the line is closed)
    tangent = np.roll(line, -1, axis=0) - np.roll(line, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)          # points left
    sl, sr = d["side_left"], d["side_right"]
    centre = line + normal * ((sl - sr) / 2.0)[:, None]
    half = (sl + sr) / 2.0
    name = track if not layout else f"{track}__{layout}"
    return Track.from_centerline(
        centre, half, half, ds=ds, name=name, smooth_m=smooth_m,
        meta={"source": "assetto_corsa", "track": track, "layout": layout or "",
              "ai_line_length_m": float(d["dist"].max())},
    )
