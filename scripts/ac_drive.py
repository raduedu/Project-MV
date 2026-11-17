"""Drive the car in a running Assetto Corsa session with a Project MV driver.

Prerequisites: see docs/07-assetto-corsa-bridge.md (install the MV Bridge app,
offline Practice session, car on track, hands and feet off the controls).

Recommended order for the very first time::

    # 1. You drive, the script only watches: do position / lateral offset / heading make sense?
    python scripts/ac_drive.py --vehicle ks_porsche_911_gt3_rs --track imola --scripted --dry-run

    # 2. Rule-based driver, speed capped at 80 km/h
    python scripts/ac_drive.py --vehicle ks_porsche_911_gt3_rs --track imola --scripted --max-kmh 80

    # 3. Rule-based driver, no cap
    python scripts/ac_drive.py --vehicle ks_porsche_911_gt3_rs --track imola --scripted --laps 2

    # 4. Trained driver
    python scripts/ac_drive.py --vehicle ks_porsche_911_gt3_rs --track imola --policy runs/gt3rs_v1/best.pt --laps 3

Stop at any time with Ctrl+C: the car brakes to a halt and control returns to you.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mv.ac.bridge import ACRawEnv  # noqa: E402
from mv.ac.driver import ObservationAdapter, drive  # noqa: E402
from mv.ac.paths import find_ac_root  # noqa: E402
from mv.ac.track_import import import_ac_track  # noqa: E402
from mv.control.scripted import ScriptedDriver  # noqa: E402
from mv.envs.racing_env import EnvConfig  # noqa: E402
from mv.rl.ppo import load_policy  # noqa: E402
from mv.vehicle.params import find_vehicle, load_vehicle  # noqa: E402


def env_config_for(policy_path: Path | None) -> EnvConfig:
    """Use the exact sensor configuration the driver was trained with."""
    if policy_path is not None:
        cfg_file = policy_path.parent / "config.json"
        if cfg_file.is_file():
            raw = json.loads(cfg_file.read_text())["env"]
            return EnvConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
        print(f"warning: {cfg_file} not found, assuming the default sensor configuration")
    return EnvConfig()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vehicle", required=True, help="vehicle file name (must describe the car loaded in the game)")
    ap.add_argument("--track", required=True, help="AC track folder, optionally track:layout (must be the loaded track)")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--policy", type=Path, help="trained driver checkpoint (.pt)")
    who.add_argument("--scripted", action="store_true", help="use the rule-based driver")
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--max-kmh", type=float, default=None, help="safety speed cap")
    ap.add_argument("--max-time", type=float, default=900.0)
    ap.add_argument("--dry-run", action="store_true", help="send no controls; just show what the driver would do")
    ap.add_argument("--blind", action="store_true", help="short track preview (first-lap mode)")
    ap.add_argument("--backend", default="csp", help="csp (default), vjoy or vgamepad")
    ap.add_argument("--ac-root", default=None)
    args = ap.parse_args()

    vp = load_vehicle(find_vehicle(args.vehicle))
    name, _, layout = args.track.partition(":")
    track = import_ac_track(find_ac_root(args.ac_root), name, layout or None)
    cfg = env_config_for(args.policy)
    adapter = ObservationAdapter(vp, track, cfg, blind=args.blind)
    if args.scripted:
        driver = ScriptedDriver(adapter.env)
    else:
        driver = load_policy(args.policy, "cpu")
        if driver.obs_norm.mean.shape[0] != adapter.num_obs:
            raise SystemExit(f"this driver expects {driver.obs_norm.mean.shape[0]} observation values but the "
                             f"current sensor configuration produces {adapter.num_obs}")
    print(f"car {vp.name} | track {track.name} ({track.length:.0f} m) | driver "
          f"{'scripted' if args.scripted else args.policy} | {'DRY RUN' if args.dry_run else 'LIVE CONTROL'}")
    backend = "null" if args.dry_run else args.backend
    with ACRawEnv.make(backend, rate_hz=1.0 / cfg.control_dt, wait_for_game_s=30.0, reset_mode="none") as raw:
        tel = raw.session.observe(timeout_s=10.0)
        if tel.car_model and vp.name.startswith("ks_") and tel.car_model != vp.name:
            print(f"warning: the game reports car '{tel.car_model}' but the vehicle file is '{vp.name}'")
        try:
            res = drive(raw, driver, adapter, vp, laps=args.laps, max_time_s=args.max_time,
                        max_speed=None if args.max_kmh is None else args.max_kmh / 3.6, dry_run=args.dry_run)
        except KeyboardInterrupt:
            print("stopped by user")
            return
        print(f"\nfinished: {res.reason}; laps {[round(t, 3) for t in res.laps]}; {res.distance_m:.0f} m; "
              f"top speed {res.max_speed * 3.6:.0f} km/h; ticks over the track edge: {res.off_track_ticks}")
        print(raw.session.stats.format() if hasattr(raw.session, "stats") else "")


if __name__ == "__main__":
    main()
