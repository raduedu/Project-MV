"""Real F1 qualifying laps as benchmarks, with an "equal machinery" car for each.

    python scripts/f1_benchmark.py                  # every circuit in tracks/f1_circuits.yaml, driver 1
    python scripts/f1_benchmark.py --only Monza Spa-Francorchamps
    python scripts/f1_benchmark.py --list 2023      # what OpenF1 has that year
    python scripts/f1_benchmark.py --race runs/multi_v5_zigzag/best.pt   # our driver vs him, same car

For each circuit (engineering log WP43):

1. download the driver's fastest qualifying lap from OpenF1 (cached in .cache/openf1);
2. lay it onto the circuit imported from Assetto Corsa (stretches where the two
   circuits differ are detected and left out);
3. fit "his car": the template car (default: the Ferrari SF70H from the owner's
   game) at the regulation mass, with power, drag, downforce and tyre grip
   scaled until our physics matches his corner speeds, straight-end speeds and
   acceleration (mv/benchmark/car_fit.py). Saved as
   ``vehicles/local/f1_<driver>_<year>_<track>.yaml`` - git-ignored, because
   it is derived from the game's car;
4. save his lap as a lap file (kind "benchmark", car = that car file) and the
   physics-perfect lap of that car, so the control panel can replay any of our
   drivers in that car against him;
5. write ``runs/benchmark/summary.json``.

The F1 data is used privately only - never committed or published (D-038).
Default driver number 1 = Max Verstappen (world champion's number in 2023-2024).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from mv.benchmark.car_fit import fit_car, lap_target, with_mass  # noqa: E402
from mv.benchmark.openf1 import fastest_lap, fetch, to_lap_file  # noqa: E402
from mv.laps import LAP_DIR, list_laps, reference_lap_file, save_lap  # noqa: E402
from mv.planning.reference import reference_lap  # noqa: E402
from mv.track.track import Track  # noqa: E402
from mv.vehicle.params import find_vehicle, load_vehicle, save_vehicle  # noqa: E402

OUT = ROOT / "runs" / "benchmark"


def _remove_old(car: str, track: str, kind: str) -> None:
    """Delete earlier lap files of this kind for this car and circuit (re-running replaces them)."""
    for m in list_laps(False):
        if m["car"] == car and m["track"] == track and m["kind"] == kind:
            for suffix in (".json", ".meta.json"):
                (LAP_DIR / f"{m['id']}{suffix}").unlink(missing_ok=True)


def run_one(entry: dict, args) -> dict:
    t0 = time.time()
    track = Track.load(ROOT / "tracks" / "local" / f"{entry['track']}.npz")
    lap = fastest_lap(int(entry["year"]), entry["circuit"], args.driver, args.session)
    drv = fetch("drivers", session_key=lap.meta["session_key"], driver_number=args.driver)
    acr = (drv[0].get("name_acronym") if drv else None) or f"car{args.driver}"
    full = (drv[0].get("full_name") if drv else None) or acr
    driver = " ".join(w.capitalize() for w in full.split())
    car_id = f"f1_{acr.lower()}_{entry['year']}_{entry['track']}"
    lap_file, al = to_lap_file(lap, track, car_id, driver=driver, notes=entry.get("layout_note", ""))
    bad = al.s_m[al.residual_m > args.layout_tol]
    print(f"{entry['circuit']} {entry['year']}: {driver} {lap.lap_time:.3f} s (lap {lap.lap_number}); "
          f"alignment {al.summary()}", flush=True)

    template = with_mass(load_vehicle(find_vehicle(args.template)), args.mass)
    tg = lap_target(track, np.array(lap_file["samples"]["s"]), np.array(lap_file["samples"]["speed"]),
                    lap.lap_time, template, bad_s=bad)
    print(f"  matching {len(tg.apex_s)} corner minimum speeds, {len(tg.top_s)} straight ends, "
          f"{len(tg.acc_s)} points part-way down straights; {tg.excluded_m:.0f} m left out (layout differs)", flush=True)
    fit = fit_car(template, tg, printer=print if args.verbose else None)
    vp = fit.vehicle
    vp.name = car_id
    vp.source = (f"Assetto Corsa car '{args.template}' (its 3-D model is used in the game) at {args.mass:.0f} kg, scaled to match {driver}'s {entry['year']} {entry['circuit']} "
                 f"{args.session} lap (OpenF1); factors {json.dumps({k: round(v, 3) for k, v in fit.factors.items()})}")
    vp.notes = ("Equal-machinery benchmark car (mv/benchmark/car_fit.py, engineering log WP43). Derived from an "
                "Assetto Corsa car - local only, never commit.")
    car_path = ROOT / "vehicles" / "local" / f"{car_id}.yaml"
    save_vehicle(vp, car_path)

    _remove_old(car_id, track.name, "benchmark")
    lap_file["meta"]["car_file"] = str(car_path.relative_to(ROOT)).replace("\\", "/")
    lap_file["meta"]["car_fit"] = fit.summary()
    save_lap(lap_file)

    ref_npz = ROOT / "runs" / "reference" / f"{car_id}__{track.name}.npz"
    ref = reference_lap(track, vp)
    ref.save(ref_npz)
    _remove_old(car_id, track.name, "reference")
    save_lap(reference_lap_file(car_id, track, ref_npz, vp.chassis.wheelbase_m,
                                math.radians(vp.steering.max_wheel_angle_deg)))
    row = {"circuit": entry["circuit"], "year": entry["year"], "track": track.name, "driver": driver,
           "car": car_id, "real_lap_s": lap.lap_time, "reference_lap_s": round(ref.lap_time, 3),
           "reference_faster_by_pct": round(100 * (1 - ref.lap_time / lap.lap_time), 2),
           "alignment": al.summary(), "layout_note": entry.get("layout_note", ""),
           "excluded_m": round(tg.excluded_m), "fit": fit.summary(), "rain": lap.rainfall,
           "seconds": round(time.time() - t0, 1)}
    print(f"  car factors {row['fit']['factors']}, fit error {fit.rms_kmh:.1f} km/h rms; physics-perfect lap of that "
          f"car {ref.lap_time:.3f} s vs his {lap.lap_time:.3f} s ({row['reference_faster_by_pct']:+.2f} %)", flush=True)
    return row


def race(policy: Path, args) -> None:
    """Drive our driver in each fitted car on its circuit (strict, 2 laps) and compare with his lap."""
    import torch
    from mv.envs.racing_env import EnvConfig
    from mv.laps import laps_from_result
    from mv.rl.evaluate import evaluate
    from mv.rl.ppo import load_policy
    torch.set_num_threads(2)
    net = load_policy(policy)
    raw = json.loads((policy.parent / "config.json").read_text())["env"]
    cfg = EnvConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("f1_*.json"))]
    if args.only:
        rows = [r for r in rows if r["circuit"] in args.only]
    run = policy.parent.name
    print(f"{run} ({policy.name}) against the real laps, in the fitted car of each weekend:")
    out = []
    for r in rows:
        vp = load_vehicle(find_vehicle(r["car"]))
        track = Track.load(ROOT / "tracks" / "local" / f"{r['track']}.npz")
        res = evaluate(net, vp, [track], "cpu", laps=2, cfg=cfg, record=True)[0]
        lap = res.flying_lap if res.finished else None
        line = f"  {r['circuit']:18s} his {r['real_lap_s']:8.3f}  ideal {r['reference_lap_s']:8.3f}  ours "
        if lap is None:
            line += f"  -- ({res.end_reason} at {res.distance_m:.0f} m)"
        else:
            line += f"{lap:8.3f}  gap to him {lap - r['real_lap_s']:+7.3f} s ({100 * (lap / r['real_lap_s'] - 1):+.2f} %)"
            laps = laps_from_result(res, track, vp.name, run, "learned", f"f1_benchmark --race ({policy.name})",
                                    str(policy.relative_to(ROOT)).replace("\\", "/") if policy.is_relative_to(ROOT) else str(policy),
                                    car_file=f"vehicles/local/{vp.name}.yaml")
            if laps:
                save_lap(min(laps, key=lambda l: l["meta"]["lap_time"]))
        if r.get("layout_note"):
            line += "   [layout differs: not comparable]"
        print(line, flush=True)
        out.append({**{k: r[k] for k in ("circuit", "track", "car", "real_lap_s", "reference_lap_s")},
                    "ours_s": lap, "end": res.end_reason})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"race_{run}_{policy.stem}.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="OpenF1 circuit names from tracks/f1_circuits.yaml")
    ap.add_argument("--driver", type=int, default=1, help="car number (1 = Max Verstappen in 2023-2024)")
    ap.add_argument("--session", default="Qualifying")
    ap.add_argument("--template", default="ks_ferrari_sf70h", help="car file the fit starts from")
    ap.add_argument("--mass", type=float, default=805.0,
                    help="car + driver mass, kg (default: 2023 minimum 798 kg, FIA F1 Technical Regulations "
                         "art. 4.1, plus a few kg of qualifying fuel)")
    ap.add_argument("--layout-tol", type=float, default=12.0,
                    help="metres: where his path is further than this from our circuit, the layouts differ")
    ap.add_argument("--list", type=int, metavar="YEAR", help="list OpenF1's qualifying sessions of a year")
    ap.add_argument("--race", type=Path, metavar="CHECKPOINT",
                    help="instead of fitting: drive this trained driver in each fitted car and compare with him")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.race:
        race(args.race.resolve(), args)
        return
    if args.list:
        for r in fetch("sessions", year=args.list, session_name=args.session):
            print(f"{r['circuit_short_name']:22s} {r['date_start'][:10]}")
        return
    entries = yaml.safe_load((ROOT / "tracks" / "f1_circuits.yaml").read_text(encoding="utf-8"))
    if args.only:
        entries = [e for e in entries if e["circuit"] in args.only]
    rows = []
    for e in entries:
        if not (ROOT / "tracks" / "local" / f"{e['track']}.npz").is_file():
            print(f"{e['circuit']}: circuit {e['track']} not imported (scripts/import_ac_track.py) - skipped")
            continue
        try:
            rows.append(run_one(e, args))
        except Exception as ex:                                # one bad circuit must not stop the others
            print(f"{e['circuit']}: FAILED - {type(ex).__name__}: {ex}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for r in rows:                                  # one file per car: several runs may work in parallel
        (OUT / f"{r['car']}.json").write_text(json.dumps(r, indent=1))
    allrows = [json.loads(p.read_text()) for p in sorted(OUT.glob("f1_*.json"))]
    f = OUT / "summary.json"
    f.write_text(json.dumps(allrows, indent=1))
    print(f"\nsummary of {len(allrows)} circuits: {f}")


if __name__ == "__main__":
    main()
