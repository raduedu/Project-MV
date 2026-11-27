"""Show a recorded lap inside Assetto Corsa, with the ideal-lap ghost (engineering log WP32).

Before: install the in-game apps once (``mv\\ac\\plugin\\install_plugin.ps1``),
This command starts the game on the lap's circuit and car (offline Practice;
your weather, time of day and driver name are kept). Then::

    python scripts/ac_view.py --list                     # laps that can be shown (imported circuits)
    python scripts/ac_view.py --best ks_porsche_911_gt3_rs imola
    python scripts/ac_view.py "local:ks_porsche_911_gt3_rs/imola/2026..._multi_v3_cars" --speed 0.5
    python scripts/ac_view.py --pause | --play | --restart | --stop | --quit

The same is one click in the control panel: lap replay -> "View in Assetto Corsa".
The command waits until the game confirms it has the lap (the game then keeps it).
To film: Windows Game Bar (Win+Alt+R) or OBS; AC's own replay also records it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mv.ac.viewer_link import Command, ViewerLink, viewer_payload  # noqa: E402
from mv.laps import list_laps  # noqa: E402


def viewable_laps() -> list[dict]:
    """Driven laps (not ghosts) on circuits imported from Assetto Corsa."""
    tracks = {p.stem for p in (ROOT / "tracks" / "local").glob("*.npz")}
    return [m for m in list_laps(True) if m["kind"] != "reference" and m["track"] in tracks]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("key", nargs="?", help="lap key (see --list)")
    ap.add_argument("--best", nargs=2, metavar=("CAR", "TRACK"), help="fastest saved lap of this car on this circuit")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--no-ghost", action="store_true")
    ap.add_argument("--keep-off-kerbs", action="store_true",
                    help="demos: where the car's centre ran past the track edge, show it at the edge (WP59)")
    ap.add_argument("--no-loop", action="store_true")
    ap.add_argument("--wait", type=float, default=600.0, help="seconds to wait for the game (default 600)")
    ap.add_argument("--restart-game", action="store_true",
                    help="if the game runs another car/circuit: close it and start it in the right session")
    ap.add_argument("--no-launch", action="store_true", help="only send the lap; do not start the game")
    for c in ("play", "pause", "restart", "stop", "quit"):
        ap.add_argument(f"--{c}", action="store_true")
    a = ap.parse_args()

    if a.list:
        for m in sorted(viewable_laps(), key=lambda m: (m["car"], m["track"], m["lap_time"])):
            print(f"{m['lap_time']:9.3f} s  {m['car']:34s} {m['track']:28s} {m['key']}")
        return 0
    link = ViewerLink()
    for name, cmd in (("play", Command.PLAY), ("pause", Command.PAUSE), ("restart", Command.RESTART),
                      ("stop", Command.STOP), ("quit", Command.QUIT)):
        if getattr(a, name):
            link.command(cmd, speed=a.speed)
            print(f"sent {name}")
            return 0
    key = a.key
    if a.best:
        car, track = a.best
        cands = [m for m in viewable_laps() if track == m["track"] and car in (m["car"], m.get("car_file"), Path(m.get("car_file") or "").stem)]
        if not cands:
            print(f"no saved lap of {car} at {track}")
            return 1
        key = min(cands, key=lambda m: m["lap_time"])["key"]
    if not key:
        ap.error("give a lap key, --best CAR TRACK, --list or a playback command")
    payload = viewer_payload(key, None if a.no_ghost else "auto", keep_off_kerbs=a.keep_off_kerbs)
    if a.no_launch:
        lap_id = link.load(payload, speed=a.speed, loop=not a.no_loop)
    else:
        from mv.ac.launch import open_lap_in_game
        res = open_lap_in_game(payload, link, restart=a.restart_game, speed=a.speed)
        print(res["msg"])
        if not res["ok"] or res.get("needs_restart"):
            if res.get("needs_restart"):
                print("  -> run again with --restart-game to close it and start it in the right session")
            return 1
        lap_id = int(link.data.lap_id)
    print(f"sent: {payload['label']}")
    if payload["ghost"] is None:
        print("  (no ghost: no physics reference lap for this car and circuit - see scripts/reference_lap.py)")
    print("waiting for Assetto Corsa (MV Viewer app) to pick it up ... (Ctrl+C to give up)")
    t0 = time.time()
    try:
        while time.time() - t0 < a.wait:
            st = link.state()
            if st and st["lap_id_loaded"] == lap_id:
                print("the game has the lap." + ("" if st["physics_allowed"] else
                      "  WARNING: no physics access in this session - use offline Practice."))
                return 0
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    print("the game did not pick it up: is AC running a session with the MV Viewer app installed?")
    return 2


if __name__ == "__main__":
    sys.exit(main())
