"""Import one or more Assetto Corsa cars into ``vehicles/local/<car_id>.yaml``.

Usage (from the project root)::

    python scripts/import_ac_car.py ks_porsche_911_gt3_rs ks_porsche_918_spyder
    python scripts/import_ac_car.py --ac-root "E:\\SteamLibrary\\steamapps\\common\\assettocorsa" --list porsche
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mv.ac.car_import import import_ac_car  # noqa: E402
from mv.ac.paths import find_ac_root  # noqa: E402
from mv.vehicle.params import save_vehicle  # noqa: E402
from mv.vehicle.summary import describe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cars", nargs="*", help="car folder names, e.g. ks_porsche_911_gt3_rs")
    ap.add_argument("--ac-root", default=None, help="Assetto Corsa install folder (auto-detected if omitted)")
    ap.add_argument("--list", metavar="TEXT", help="list installed cars whose id contains TEXT and exit")
    ap.add_argument("--fuel", type=float, default=None, help="fuel load in litres (default: car's default)")
    args = ap.parse_args()

    cars_dir = find_ac_root(args.ac_root) / "content" / "cars"
    if args.list is not None:
        for p in sorted(cars_dir.iterdir()):
            if args.list.lower() in p.name.lower():
                print(p.name)
        return
    for car in args.cars:
        vp = import_ac_car(cars_dir / car, fuel_litres=args.fuel)
        out = ROOT / "vehicles" / "local" / f"{car}.yaml"
        save_vehicle(vp, out)
        print(f"\nwrote {out}")
        print(describe(vp))


if __name__ == "__main__":
    main()
