"""Race the Project MV race bot in Assetto Corsa: you drive car 0, the bot drives car 1 (race bot stage 5).

Once (owner): install the apps - ``powershell -ExecutionPolicy Bypass -File mv\\ac\\plugin\\install_plugin.ps1``
- and enable "MV Race Bot" in the game's Lua apps.

Then::

    # write a 2-car race (you + the bot, same car, standing start) and start the game
    python scripts/race_bot.py --policy runs/race_v3_racing/best.pt --track ks_nordschleife__nordschleife ^
        --vehicle ks_porsche_911_gt3_rs --start-game --laps 3

    # or: start a race with one AI car of your car yourself, then just run
    python scripts/race_bot.py --policy runs/race_v3_racing/best.pt

The bot waits on the grid until the start, sees only ``--view`` metres ahead (sensors; default 120 m) and
knows nothing about the circuit beyond what it has driven itself (own map).  Ctrl+C stops it and gives
the car back to the game's AI.  ``--dry-run`` sends nothing: the AI keeps driving car 1 and the bot only
prints what it would do - the safe first check that positions, track and observation line up.

Safety: if the game cannot hand the pedals over completely (the AI would still press them), the bot
stops and says so; the app brakes the car when Python stops sending, and hands it back to the AI after
10 s.  The game is started only with --start-game (D-033: the owner's action).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mv.ac import launch  # noqa: E402
from mv.ac.paths import find_ac_root  # noqa: E402
from mv.ac.racebot import RaceBotAdapter, race  # noqa: E402
from mv.ac.racebot_link import RaceBotLink  # noqa: E402
from mv.envs.racing_env import EnvConfig  # noqa: E402
from mv.rl.ppo import load_policy  # noqa: E402
from mv.track.track import Track  # noqa: E402
from mv.vehicle.params import find_vehicle, load_vehicle  # noqa: E402


def run_config(policy: Path) -> EnvConfig:
    raw = json.loads((policy.parent / "config.json").read_text())["env"]
    return EnvConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()
                        if k in EnvConfig.__dataclass_fields__})


def track_name(track: str, layout: str) -> str:
    return f"{track}__{layout}" if layout else track


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=Path, required=True, help="trained driver (.pt)")
    ap.add_argument("--track", default=None, help="imported circuit (tracks/local/<name>.npz); default: the game's")
    ap.add_argument("--vehicle", default=None, help="car file / game car id; default: the game's")
    ap.add_argument("--view", type=float, default=120.0, help="sensor range in metres (default 120)")
    ap.add_argument("--bot-car", type=int, default=1, help="0-based index of the AI car the bot drives")
    ap.add_argument("--dry-run", action="store_true", help="send nothing; print what the bot would do")
    ap.add_argument("--start-game", action="store_true",
                    help="OWNER: write a 2-car race (you + bot, same car) to race.ini and start the game")
    ap.add_argument("--laps", type=int, default=3, help="race length with --start-game")
    ap.add_argument("--ac-root", default=None)
    args = ap.parse_args()

    session = None if args.start_game else launch.running_session()
    car_id = args.vehicle or (session or {}).get("car")
    name = args.track or (track_name(session["track"], session["layout"]) if session else None)
    if not car_id or not name:
        sys.exit("give --vehicle and --track (or start the game first so they can be read from it)")
    track_file = ROOT / "tracks" / "local" / f"{name}.npz"
    if not track_file.is_file():
        sys.exit(f"{track_file} not found - import the circuit first (scripts/import_ac_track.py)")
    vp = load_vehicle(find_vehicle(car_id))
    if args.start_game:
        ac_root = find_ac_root(args.ac_root)
        track, _, layout = name.partition("__")
        old = launch.race_ini_path().read_text(encoding="utf-8") if launch.race_ini_path().is_file() else ""
        skins = launch.car_skins(ac_root, car_id)
        skin = skins[0] if skins else ""
        text = launch.build_race_ini(old, car_id, skin, track, layout, ghost_car=True,
                                     ghost_skin=skins[1] if len(skins) > 1 else skin, race_laps=args.laps,
                                     ghost_name="MV Race Bot")
        backup = launch.write_race_ini(text)
        print(f"race.ini: {args.laps}-lap race at {name}, you vs the MV Race Bot, both in {car_id} "
              f"(previous file kept as {backup})")
        launch.start_game(ac_root)
        print("game starting - enable the 'MV Race Bot' app once you are on the grid")

    cfg = run_config(args.policy)
    net = load_policy(args.policy, "cpu")
    adapter = RaceBotAdapter(vp, Track.load(track_file), cfg, view_m=args.view)
    link = RaceBotLink(bot_index=args.bot_car)
    print(f"bot: {args.policy} ({'sees other cars' if cfg.opponents else 'CANNOT see other cars'}, sensors "
          f"{args.view:.0f} m, own map {'on' if cfg.own_map else 'off'}) in car {args.bot_car}; waiting for the game ...")
    deadline = time.time() + 600
    while link.read() is None:
        if time.time() > deadline:
            sys.exit("no data from the MV Race Bot app - is it installed and enabled in the game?")
        time.sleep(0.5)
    st = link.read()
    if st.cars_count <= args.bot_car:
        sys.exit(f"the session has {st.cars_count} car(s): add an AI opponent (car {args.bot_car}) of your car")
    reason = race(link, net, adapter, vp, dry_run=args.dry_run)
    link.close()
    print(f"race bot stopped: {reason}")


if __name__ == "__main__":
    main()
