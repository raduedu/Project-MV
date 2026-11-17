"""Import Assetto Corsa circuits into ``tracks/local/<name>.npz``.

Usage (from the project root)::

    python scripts/import_ac_track.py imola monza spa mugello
    python scripts/import_ac_track.py ks_nurburgring:layout_gp_a
    python scripts/import_ac_track.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mv.ac.paths import find_ac_root  # noqa: E402
from mv.ac.track_import import import_ac_track  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks", nargs="*", help="track or track:layout")
    ap.add_argument("--ac-root", default=None)
    ap.add_argument("--list", action="store_true", help="list importable tracks and layouts")
    ap.add_argument("--ds", type=float, default=1.0, help="sample spacing in metres")
    args = ap.parse_args()
    ac = find_ac_root(args.ac_root)
    if args.list:
        for p in sorted((ac / "content" / "tracks").iterdir()):
            if (p / "ai" / "fast_lane.ai").is_file():
                print(p.name)
            for q in sorted(p.iterdir()) if p.is_dir() else []:
                if (q / "ai" / "fast_lane.ai").is_file():
                    print(f"{p.name}:{q.name}")
        return
    for spec in args.tracks:
        name, _, layout = spec.partition(":")
        t = import_ac_track(ac, name, layout or None, ds=args.ds)
        out = ROOT / "tracks" / "local" / f"{t.name}.npz"
        t.save(out)
        print(f"{t.name}: {t.length:.0f} m, {t.turning_direction()}, mean width {2 * t.w_left.mean():.1f} m, "
              f"tightest corner {t.min_radius():.0f} m radius -> {out}")


if __name__ == "__main__":
    main()
