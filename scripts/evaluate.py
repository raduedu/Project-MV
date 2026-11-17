"""Time a driver over full laps and draw what it did.

Examples (from the project root)::

    # trained driver on the imported real circuits + 4 unseen random circuits
    python scripts/evaluate.py --vehicle ks_porsche_911_gt3_rs --policy runs/gt3rs_v1/best.pt

    # the rule-based baseline on Imola only
    python scripts/evaluate.py --vehicle ks_porsche_911_gt3_rs --scripted --tracks imola

    # compare both, mimic a blind first lap, on a dusty track
    python scripts/evaluate.py --vehicle ks_porsche_911_gt3_rs --policy runs/gt3rs_v1/best.pt --scripted --blind --grip 0.9

For every track it prints the standing-start lap, the flying lap, top speed and
how often the car put any part of its body over the track edge, and saves a
picture (driven line coloured by speed + speed trace) to ``runs/eval/``.
If a physics reference exists for the track (see scripts/reference_lap.py) the
gap to it is printed too.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mv.control.scripted import ScriptedDriver  # noqa: E402
from mv.envs.racing_env import EnvConfig, RacingEnv  # noqa: E402
from mv.rl.evaluate import LapResult, eval_config, evaluate  # noqa: E402
from mv.rl.ppo import load_policy  # noqa: E402
from mv.track.generator import random_tracks  # noqa: E402
from mv.track.track import Track, TrackBatch  # noqa: E402
from mv.vehicle.params import find_vehicle, load_vehicle  # noqa: E402


def fmt(t: float) -> str:
    if not np.isfinite(t):
        return "   --   "
    return f"{int(t // 60)}:{t % 60:06.3f}"


def plot(track: Track, results: dict[str, LapResult], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axs = plt.subplots(1, 2, figsize=(18, 7.5), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    left, right = track.left_xy, track.right_xy
    ax.plot(*np.vstack([left, left[:1]]).T, color="0.35", lw=0.8)
    ax.plot(*np.vstack([right, right[:1]]).T, color="0.35", lw=0.8)
    name, res = next(iter(results.items()))
    tr = res.trace
    if len(tr.get("x", [])) > 2:
        pts = np.stack([tr["x"], tr["y"]], 1)
        seg = np.stack([pts[:-1], pts[1:]], 1)
        jump = np.linalg.norm(pts[1:] - pts[:-1], axis=1) < 10.0
        lc = LineCollection(seg[jump], cmap="turbo", array=tr["speed"][:-1][jump] * 3.6, lw=1.6)
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, label="speed (km/h)", shrink=0.8)
    ax.plot(*track.xy[0], "k^", ms=8)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{track.name}  ({track.length:.0f} m)  line of: {name}")
    ax = axs[1]
    for name, res in results.items():
        tr = res.trace
        if len(tr.get("t", [])) < 2:
            continue
        dist = np.concatenate([[0.0], np.cumsum(tr["speed"][1:] * np.diff(tr["t"]))])
        label = f"{name}: flying {fmt(res.flying_lap).strip()}" if res.finished else f"{name}: {res.end_reason} at {res.distance_m:.0f} m"
        ax.plot(dist, tr["speed"] * 3.6, lw=1.1, label=label)
    ax.set_xlabel("distance driven (m)"); ax.set_ylabel("speed (km/h)"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vehicle", required=True)
    ap.add_argument("--policy", type=Path, default=None, help="trained driver checkpoint")
    ap.add_argument("--scripted", action="store_true", help="also/only run the rule-based baseline")
    ap.add_argument("--tracks", nargs="*", default=None,
                    help="names in tracks/local (default: all of them) - e.g. imola monza")
    ap.add_argument("--random", type=int, default=4, help="number of unseen random circuits to add")
    ap.add_argument("--preset", default="mixed")
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--blind", action="store_true", help="short track preview, as on a first lap of an unknown track")
    ap.add_argument("--grip", type=float, default=1.0, help="tyre grip multiplier")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "eval")
    args = ap.parse_args()
    if args.policy is None and not args.scripted:
        ap.error("give --policy and/or --scripted")

    vp = load_vehicle(find_vehicle(args.vehicle))
    local = ROOT / "tracks" / "local"
    names = args.tracks if args.tracks is not None else [p.stem for p in sorted(local.glob("*.npz"))]
    tracks = [Track.load(local / f"{n}.npz") for n in names]
    extra = random_tracks(args.random, seed=424_242, cfg=args.preset) if args.random > 0 else []
    for i, t in enumerate(extra):
        t.name = f"unseen_random_{i}"
    tracks += extra
    cfg = EnvConfig()
    drivers: dict[str, object] = {}
    if args.policy is not None:
        cfg_file = args.policy.parent / "config.json"
        if cfg_file.is_file():
            raw = json.loads(cfg_file.read_text())["env"]
            cfg = EnvConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
        drivers["learned"] = load_policy(args.policy, "cpu")
    if args.scripted:
        probe = RacingEnv(vp, TrackBatch(tracks[:1], ds=tracks[0].ds), 1, eval_config(cfg), device="cpu")
        drivers["scripted"] = ScriptedDriver(probe)

    all_res: dict[str, list[LapResult]] = {}
    for name, drv in drivers.items():
        print(f"running {name} driver on {len(tracks)} tracks ...", flush=True)
        all_res[name] = evaluate(drv, vp, tracks, "cpu", laps=args.laps, cfg=cfg, record=not args.no_plots,
                                 blind=args.blind, grip=args.grip)

    print(f"\n{vp.name}" + ("  [blind preview]" if args.blind else "") + (f"  [grip x{args.grip}]" if args.grip != 1 else ""))
    header = f"{'track':<24}{'length':>8}  " + "  ".join(f"{n + ' standing':>18}{n + ' flying':>16}{'vmax':>6}{'off':>5}" for n in drivers)
    print(header + f"{'reference':>12}")
    for i, t in enumerate(tracks):
        row = f"{t.name:<24}{t.length:7.0f}m  "
        for n in drivers:
            r = all_res[n][i]
            if r.finished:
                row += f"{fmt(r.lap_times[0]):>18}{fmt(r.flying_lap):>16}{r.max_speed * 3.6:6.0f}{r.off_track_steps:5d}  "
            else:
                row += f"{r.end_reason + ' @' + str(int(r.distance_m)) + 'm':>34}{r.max_speed * 3.6:6.0f}{r.off_track_steps:5d}  "
        ref = ROOT / "runs" / "reference" / f"{vp.name}__{t.name}.npz"
        if ref.is_file():
            lap = float(np.load(ref)["lap_time"])
            row += f"{fmt(lap):>10}"
            for n in drivers:
                r = all_res[n][i]
                if r.finished:
                    row += f"  {n} {100 * (r.flying_lap / lap - 1):+.1f}%"
        print(row)
        if not args.no_plots:
            plot(t, {n: all_res[n][i] for n in drivers}, args.out / f"{vp.name}__{t.name}.png")
    done = {n: sum(r.finished for r in rs) for n, rs in all_res.items()}
    print("\ncompleted: " + ", ".join(f"{n} {k}/{len(tracks)}" for n, k in done.items()))
    if not args.no_plots:
        print(f"plots: {args.out}")


if __name__ == "__main__":
    main()
