"""Compute physics reference laps for a car on every imported circuit and save them as replayable laps.

    python scripts/record_reference_laps.py ks_porsche_918_spyder porsche_taycan_turbo_gt_weissach

For each car and each track in ``tracks/local``: compute the reference
(``runs/reference/<car>__<track>.npz``, skipped if it exists) and save it as a
lap file in ``runs/laps`` (the "ideal ghost" in the web interface).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mv.laps import list_laps, reference_lap_file, save_lap  # noqa: E402
from mv.planning.reference import reference_lap  # noqa: E402
from mv.track.track import Track  # noqa: E402
from mv.vehicle.params import find_vehicle, load_vehicle  # noqa: E402


def main() -> None:
    for car in sys.argv[1:]:
        vp = load_vehicle(find_vehicle(car))
        have = {(m["car"], m["track"]) for m in list_laps(False) if m["kind"] == "reference"}
        for npz in sorted((ROOT / "tracks" / "local").glob("*.npz")):
            track = Track.load(npz)
            ref = ROOT / "runs" / "reference" / f"{vp.name}__{track.name}.npz"
            if not ref.is_file():
                print(f"{vp.name} @ {track.name}: computing reference ...", flush=True)
                reference_lap(track, vp).save(ref)
            if (vp.name, track.name) in have:
                continue
            lap = reference_lap_file(vp.name, track, ref, vp.chassis.wheelbase_m, math.radians(vp.steering.max_wheel_angle_deg))
            save_lap(lap)
            print(f"{vp.name} @ {track.name}: reference lap {lap['meta']['lap_time']:.3f} s saved", flush=True)


if __name__ == "__main__":
    main()
