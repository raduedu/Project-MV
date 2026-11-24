"""Import Assetto Corsa cars into ``vehicles/local/<car_id>.yaml``.

Usage (from the project root)::

    python scripts/import_ac_car.py ks_porsche_911_gt3_rs ks_porsche_918_spyder
    python scripts/import_ac_car.py --list porsche          # search installed cars
    python scripts/import_ac_car.py --all                   # every installed car
    python scripts/import_ac_car.py --category racing       # formula, prototype, gt3, gt2_gte, gt4, touring_cup
    python scripts/import_ac_car.py --category gt3,gt4      # one or more categories (or: favourites)
    python scripts/import_ac_car.py --table                 # print the catalogue table, import nothing

Every imported car is checked for plausibility (our power, mass and top speed
against the figures the game advertises, plus a few seconds of full throttle
and full braking in our simulator - see ``mv/vehicle/catalog.py``).  Cars that
are not installed (empty DLC folders) are reported as "unavailable" and
skipped; a car that cannot be read is reported as "failed" and the batch goes
on.  The results are kept in ``vehicles/local/_catalog.json`` (git-ignored)
and shown in the control panel.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mv.ac.paths import find_ac_root  # noqa: E402
from mv.vehicle import catalog  # noqa: E402
from mv.vehicle.params import find_vehicle, load_vehicle  # noqa: E402
from mv.vehicle.summary import describe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cars", nargs="*", help="car folder names, e.g. ks_porsche_911_gt3_rs")
    ap.add_argument("--ac-root", default=None, help="Assetto Corsa install folder (auto-detected if omitted)")
    ap.add_argument("--list", metavar="TEXT", help="list installed cars whose id contains TEXT and exit")
    ap.add_argument("--all", action="store_true", help="import every installed car")
    ap.add_argument("--category", metavar="SEL",
                    help="import a selection: racing, favourites, all, or categories "
                         f"({', '.join(catalog.CATEGORIES)}), comma-separated")
    ap.add_argument("--fuel", type=float, default=None, help="fuel load in litres (default: car's default)")
    ap.add_argument("--no-check", action="store_true", help="skip the plausibility check")
    ap.add_argument("--no-sim", action="store_true", help="plausibility check without the simulator run (faster)")
    ap.add_argument("--table", action="store_true", help="print the catalogue table (markdown) and exit")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads for the simulator check (default 2)")
    args = ap.parse_args()

    root = find_ac_root(args.ac_root)
    cars_dir = root / "content" / "cars"
    if args.list is not None:
        for p in sorted(cars_dir.iterdir()):
            if args.list.lower() in p.name.lower():
                print(p.name)
        return
    if args.table:
        print(catalog.table(catalog.list_cars(root)))
        return

    import torch
    torch.set_num_threads(max(1, args.threads))     # the GPU trains; stay polite on the CPU

    batch = args.all or args.category
    if batch:
        entries = catalog.list_cars(root)
        sel = "all" if args.all else args.category
        ids = [i for i in catalog.select_ids(sel, entries) if (cars_dir / i).is_dir()]
        library_ids = [e.id for e in entries if e.where == "library" and e.id in catalog.select_ids(sel, entries)]
    else:
        ids, library_ids = list(args.cars), []
    if not ids and not library_ids:
        ap.error("name at least one car, or use --all / --category")

    print(f"importing {len(ids)} car(s) from {cars_dir}")
    results = catalog.import_cars(ids, cars_dir, ROOT / "vehicles" / "local", check=not args.no_check,
                                  simulate=not args.no_sim, fuel_litres=args.fuel)
    for cid in library_ids:                           # our own files: check only (nothing to import)
        row = catalog.check_vehicle_file(cid, simulate=not args.no_sim)
        results.append(row)
        print(f"[library] {cid:<34} {row['status']:<11}" + (f"  - {'; '.join(row['reasons'])}" if row["reasons"] else ""))

    if not batch:                                     # single cars: show the familiar summary
        for r in results:
            if r["status"] not in ("failed", "unavailable"):
                p = find_vehicle(r["id"])
                print(f"\nwrote {p}")
                print(describe(load_vehicle(p)))
    counts = Counter(r["status"] for r in results)
    print("\nsummary: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    for r in results:
        if r["status"] != "ok":
            print(f"  {r['id']:<34} {r['status']}: {'; '.join(r['reasons'])}")
    if counts.get("failed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
