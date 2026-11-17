"""Physics reference lap: what lap time is achievable, and along which line?

Computes, for one car on one track, the lap time on the centre line, on the
minimum-curvature line and on the minimum-time line (see
docs/08-reference-lap.md for the method), prints them, and saves

    runs/reference/<vehicle name>__<track name>.npz     the fastest lap (line, speed, lap_time, ...)
    runs/reference/<vehicle name>__<track name>.png     map coloured by speed + speed trace

``scripts/evaluate.py`` picks the ``.npz`` up automatically and prints how far a
driver is from the reference.

Usage::

    python scripts/reference_lap.py --vehicle ks_porsche_911_gt3_rs --track imola
    python scripts/reference_lap.py --vehicle ks_porsche_911_gt3_rs --track ks_nurburgring --layout layout_gp_a
    python scripts/reference_lap.py --vehicle porsche_taycan_turbo_gt_weissach --random 3 --preset mixed

``--track NAME`` loads ``tracks/local/NAME.npz`` if it exists, otherwise it
imports the Assetto Corsa track of that name.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mv.planning.ggv import build_ggv  # noqa: E402
from mv.planning.reference import ReferenceLap, reference_lines  # noqa: E402
from mv.track.generator import random_track  # noqa: E402
from mv.track.track import Track  # noqa: E402
from mv.vehicle.params import find_vehicle, load_vehicle  # noqa: E402

LABEL = {"centre": "centre line", "min_curvature": "minimum curvature", "min_time": "minimum time"}


def fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:06.3f}"


def load_track(args) -> Track:
    if args.random is not None:
        track = random_track(args.random, args.preset)
        track.name = f"random_{args.preset}_{args.random}"
        return track
    local = ROOT / "tracks" / "local" / f"{args.track}.npz"
    if local.is_file() and not args.layout:
        return Track.load(local)
    from mv.ac.paths import find_ac_root
    from mv.ac.track_import import import_ac_track
    return import_ac_track(find_ac_root(args.ac_root), args.track, args.layout)


def plot(track: Track, laps: dict[str, ReferenceLap], best: str, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    lap = laps[best]
    fig, (ax_map, ax_v) = plt.subplots(2, 1, figsize=(13, 13.5), gridspec_kw={"height_ratios": [1.9, 1.0]})

    # ---- map: track edges and the line coloured by speed ----------------------
    for edge in (track.left_xy, track.right_xy):
        closed = np.vstack([edge, edge[:1]])
        ax_map.plot(closed[:, 0], closed[:, 1], color="0.45", lw=0.8)
    pts = np.vstack([lap.xy, lap.xy[:1]])
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segments, cmap="viridis", lw=2.2)
    lc.set_array(lap.speed * 3.6)
    ax_map.add_collection(lc)
    ax_map.plot(*lap.xy[0], marker="o", color="black", ms=5)
    ax_map.annotate("start", lap.xy[0], textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax_map.set_aspect("equal")
    ax_map.autoscale()
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")
    ax_map.set_title(f"{LABEL[best]} line, coloured by speed")
    fig.colorbar(lc, ax=ax_map, label="speed (km/h)", shrink=0.7, pad=0.02)

    # ---- speed against distance for every line -----------------------------------
    style = {"centre": dict(color="0.6", lw=1.0), "min_curvature": dict(color="#d95f02", lw=1.2),
             "min_time": dict(color="#1b3a8c", lw=1.6)}
    for name, other in laps.items():
        ax_v.plot(other.s, other.speed * 3.6, label=f"{LABEL[name]}  {fmt(other.lap_time)}", **style[name])
    ax_v.set_xlabel("distance along the centre line (m)")
    ax_v.set_ylabel("speed (km/h)")
    ax_v.set_xlim(0, track.length)
    ax_v.set_ylim(bottom=0)
    ax_v.grid(alpha=0.3)
    ax_v.legend(loc="lower right", framealpha=0.9)
    ax_v.set_title("speed profile")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vehicle", required=True, help="vehicle name (vehicles/local or vehicles/library) or YAML path")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--track", help="track name: tracks/local/<name>.npz or an Assetto Corsa track")
    src.add_argument("--random", type=int, metavar="SEED", help="generate a random track with this seed")
    ap.add_argument("--layout", default=None, help="Assetto Corsa layout (for tracks that have several)")
    ap.add_argument("--preset", default="mixed", help="random-track preset (circuit, fs, mixed)")
    ap.add_argument("--ac-root", default=None, help="Assetto Corsa install folder (auto-detected by default)")
    ap.add_argument("--safety", type=float, default=0.2, help="extra distance kept from the track edge, m")
    ap.add_argument("--no-min-time", action="store_true", help="stop after the minimum-curvature line")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "reference")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    vp = load_vehicle(find_vehicle(args.vehicle))
    track = load_track(args)
    print(f"{vp.name} on {track.name}: {track.length:.0f} m, width {2 * track.w_left.min():.1f}-"
          f"{2 * track.w_left.max():.1f} m, {track.turning_direction()}")

    t0 = time.perf_counter()
    ggv = build_ggv(vp)
    t_ggv = time.perf_counter() - t0
    print(f"  performance envelope ({t_ggv:.1f} s): top speed {ggv.top_speed() * 3.6:.0f} km/h, cornering "
          f"{ggv.ay_steady(30.0) / 9.81:.2f} g at 108 km/h, {ggv.ay_steady(60.0) / 9.81:.2f} g at 216 km/h, "
          f"braking {-ggv.ax_min[np.searchsorted(ggv.v, 50.0), 0] / 9.81:.2f} g at 180 km/h")

    methods = ("centre", "min_curvature") if args.no_min_time else ("centre", "min_curvature", "min_time")
    laps = reference_lines(track, vp, methods, ggv=ggv, safety_m=args.safety)
    best = min(laps, key=lambda k: laps[k].lap_time)
    print(f"  {'line':<20}{'lap time':>10}{'vs centre':>11}{'top':>9}{'slowest':>9}{'length':>9}{'compute':>9}")
    for name, lap in laps.items():
        gain = 100.0 * (lap.lap_time / laps["centre"].lap_time - 1.0)
        print(f"  {LABEL[name]:<20}{fmt(lap.lap_time):>10}{gain:>+10.2f}%{lap.speed.max() * 3.6:>6.0f} km/h"
              f"{lap.speed.min() * 3.6:>5.0f} km/h{lap.meta['line_length']:>7.0f} m{lap.meta['compute_s']:>7.1f} s")
    if "min_time" in laps:
        mc, mt = laps["min_curvature"].lap_time, laps["min_time"].lap_time
        print(f"  minimum time beats minimum curvature by {mc - mt:.3f} s ({100 * (1 - mt / mc):.2f} %)")
    ref = laps[best]
    if "battery_lasts_the_lap" in ref.meta:
        print(f"  battery: {ref.meta['battery_used_kwh']:.2f} kWh net of {ref.meta['battery_usable_kwh']:.2f} kWh usable"
              + ("" if ref.meta["battery_lasts_the_lap"] else "  -> NOT enough for a full lap: the reference is optimistic"))

    ref.meta["lap_times"] = {name: lap.lap_time for name, lap in laps.items()}
    stem = f"{vp.name}__{track.name}"
    ref.save(args.out / f"{stem}.npz")
    print(f"  reference ({LABEL[best]}): {fmt(ref.lap_time)}  ->  {args.out / (stem + '.npz')}")
    if not args.no_plot:
        plot(track, laps, best, args.out / f"{stem}.png",
             f"{vp.name} on {track.name}: reference lap {fmt(ref.lap_time)}")
        print(f"  plot: {args.out / (stem + '.png')}")


if __name__ == "__main__":
    main()
