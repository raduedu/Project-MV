"""Axis conversions between Assetto Corsa and Project MV.

AC frames (verified, see telemetry.py): right-handed, Y up.
  * car-local: X = left, Y = up, Z = forward
  * world:     X, Y = up, Z; seen from above with +X to the right, +Z points
    *down* the screen.

Project frames:
  * body (car-local): x forward, y left, z up (``mv.vehicle.params``)
  * map (world): the frame of ``mv.ac.track_import.ac_world_to_map``:
    ``map_x = X``, ``map_y = -Z``, ``map_z = Y`` — x right, y up on a top-down
    map, z out of the map; yaw counter-clockwise positive, yaw = 0 along +map_x.

Both conversions are proper rotations (det = +1), so cross products and
angular velocities keep their sense.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ac_local_to_body", "ac_world_to_map3"]


def ac_local_to_body(v) -> np.ndarray:
    """AC car-local (X left, Y up, Z fwd) -> body (x fwd, y left, z up)."""
    return np.array([v[2], v[0], v[1]], dtype=np.float64)


def ac_world_to_map3(v) -> np.ndarray:
    """AC world (X, Y up, Z) -> map (X, -Z, Y); same as track_import.ac_world_to_map + height."""
    return np.array([v[0], -v[2], v[1]], dtype=np.float64)
