"""Tune a vehicle file so the simulator reproduces recorded driving.

1. Record in the game (see ``scripts/ac_drive.py --record``), e.g. your own laps::

       python scripts/ac_drive.py --vehicle ks_porsche_911_gt3_rs --track imola --scripted --dry-run --laps 3 --record runs/sysid/imola_me.npz

   (with ``--dry-run`` YOU drive; the more of the car's limits the laps use -
   hard braking, fast corners, high speed - the more factors can be determined)

2. Fit::

       python scripts/sysid_fit.py --vehicle ks_porsche_911_gt3_rs runs/sysid/imola_me.npz runs/sysid/monza_me.npz

   This prints, for every correction factor, the fitted value and whether the
   recordings actually determine it, and writes ``vehicles/local/<car>_tuned.yaml``
   with ONLY the well-determined factors applied.

3. Check and use: ``python scripts/vehicle_check.py <car>_tuned``, then train or
   fine-tune with ``--vehicle <car>_tuned``.

Self-test without the game: ``--synthetic`` records a deliberately altered car
in the simulator and checks that the fit finds the alterations.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mv.sysid.replay import FACTORS, Recording, apply_factors, fit, replay_error  # noqa: E402
from mv.vehicle.params import find_vehicle, load_vehicle, save_vehicle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="*", type=Path)
    ap.add_argument("--vehicle", required=True)
    ap.add_argument("--factors", default=",".join(FACTORS), help=f"comma list from: {', '.join(FACTORS)}")
    ap.add_argument("--out", type=Path, default=None, help="tuned vehicle file (default vehicles/local/<car>_tuned.yaml)")
    ap.add_argument("--synthetic", action="store_true", help="self-test on a deliberately altered simulated car")
    ap.add_argument("--quick", action="store_true", help="smaller search (for a first look)")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(2)
    vp = load_vehicle(find_vehicle(args.vehicle))
    names = tuple(n.strip() for n in args.factors.split(",") if n.strip())
    truth = None
    if args.synthetic:
        from mv.sysid.synthetic import record_in_simulator
        from mv.track.track import Track
        truth = {"grip": 0.96, "grip_balance": 1.087, "brakes": 0.90, "power": 1.0, "downforce": 1.0, "drag": 1.0}
        altered = apply_factors(vp, truth)
        tracks = [Track.load(p) for p in sorted((ROOT / "tracks" / "local").glob("*.npz"))[:2]]
        recs = [record_in_simulator(altered, t, seconds=70) for t in tracks]
        print(f"synthetic: recorded {[len(r) for r in recs]} ticks of a car altered by {truth}")
    else:
        if not args.recordings:
            ap.error("give recordings, or --synthetic")
        recs = [Recording.load(p) for p in args.recordings]
    print(f"{vp.name}: {sum(len(r) for r in recs)} ticks in {len(recs)} recording(s); fitting {', '.join(names)}")
    t0 = time.time()
    kw = dict(global_generations=6, global_popsize=6, rounds=2) if args.quick else {}
    res = fit(vp, recs, names=names, **kw)
    ok = res.well_determined()
    print(f"\n{'factor':<14}{'fitted':>8}{'determined':>12}   meaning")
    for n in names:
        extra = f"   (true {truth[n]:.3f})" if truth and n in truth else ""
        print(f"{n:<14}{res.factors[n]:8.3f}{('yes' if ok[n] else 'NO'):>12}   {FACTORS[n]}{extra}")
    print(f"\ntypical replay error per 1.5 s slice, before -> after:")
    for c in res.per_channel_before:
        print(f"  {c:<3} {res.per_channel_before[c]:7.3f} -> {res.per_channel_after[c]:7.3f}")
    print(f"({time.time() - t0:.0f} s)")
    if args.synthetic:
        return
    applied = {n: v for n, v in res.factors.items() if ok[n]}
    tuned = apply_factors(vp, applied)
    tuned.name = f"{vp.name}_tuned"
    tuned.source = (vp.source + f" | sysid-tuned from {', '.join(p.name for p in args.recordings)}: "
                    + ", ".join(f"{k} x{v:.3f}" for k, v in applied.items()))
    out = args.out or ROOT / "vehicles" / "local" / f"{tuned.name}.yaml"
    save_vehicle(tuned, out)
    skipped = [n for n in names if not ok[n]]
    print(f"\nwrote {out} (applied: {', '.join(applied) or 'nothing'}"
          + (f"; NOT applied, recordings do not determine them: {', '.join(skipped)}" if skipped else "") + ")")


if __name__ == "__main__":
    main()
