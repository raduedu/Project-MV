"""Start Assetto Corsa directly in the right session for a lap (engineering log WP33).

Why
---
"View in Assetto Corsa" should be one click: the game opens on the lap's
circuit, with the lap's car, in a session where the MV Viewer app may move
cars - no clicking through Content Manager first.

How
---
Assetto Corsa starts a session from ``Documents\\Assetto Corsa\\cfg\\race.ini``
(Content Manager writes the same file before it starts the game). We

1. read the owner's current ``race.ini`` and keep everything personal
   (weather, time of day, temperatures, driver name, nationality) - only the
   car, skin, circuit, number of cars and the session are changed: offline
   **Practice**, unlimited time; with ``ghost_car`` one AI car of the same
   model is added (the MV Viewer turns it into the ghost);
2. copy the old file to ``runs/ac_launch/`` first (last 10 kept);
3. start ``acs.exe`` from the game folder as a separate process, so closing
   the control panel never closes the game.

Nothing is written into the game folder. If the game is already running with
another car or circuit, the MV Viewer app can close it (on request) and it is
started again with the new session.
"""

from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

__all__ = ["documents_dir", "race_ini_path", "ac_ids_for_lap", "stand_in_for", "stand_in_model", "car_skins", "build_race_ini", "write_race_ini",
           "game_running", "steam_running", "running_session", "start_game", "open_lap_in_game", "watch_game", "BACKUP_DIR"]

ROOT = Path(__file__).resolve().parents[2]
BACKUP_DIR = ROOT / "runs" / "ac_launch"
_DOCUMENTS = uuid.UUID("FDD39AD0-238F-46AF-ADB4-6C85480369C7")      # FOLDERID_Documents


def documents_dir() -> Path:
    """The user's Documents folder as Windows knows it (follows OneDrive redirection)."""
    class GUID(ctypes.Structure):
        _fields_ = [("a", ctypes.c_ulong), ("b", ctypes.c_ushort), ("c", ctypes.c_ushort), ("d", ctypes.c_ubyte * 8)]
    u = _DOCUMENTS
    g = GUID(u.fields[0], u.fields[1], u.fields[2], (ctypes.c_ubyte * 8)(*u.bytes[8:]))
    p = ctypes.c_wchar_p()
    if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(p)) != 0:
        return Path.home() / "Documents"
    try:
        return Path(p.value)
    finally:
        ctypes.windll.ole32.CoTaskMemFree(p)


def race_ini_path() -> Path:
    return documents_dir() / "Assetto Corsa" / "cfg" / "race.ini"


# ---------------------------------------------------------------------- which car / circuit in the game
def ac_ids_for_lap(meta: dict) -> tuple[str, str, str]:
    """(AC car folder, AC track folder, layout) for a lap; ValueError if the game does not have them."""
    from mv.track.track import Track
    from mv.vehicle.params import find_vehicle, load_vehicle

    track_file = ROOT / "tracks" / "local" / f"{meta['track']}.npz"
    if not track_file.is_file():
        raise ValueError(f"'{meta['track']}' is not a circuit imported from Assetto Corsa")
    tm = Track.load(track_file).meta
    if tm.get("source") != "assetto_corsa":
        raise ValueError(f"'{meta['track']}' is not a circuit imported from Assetto Corsa")
    vp = None
    for name in (meta.get("car_file"), meta["car"]):
        try:
            vp = load_vehicle(find_vehicle(name)) if name else None
        except FileNotFoundError:
            vp = None
        if vp:
            break
    m = re.search(r"Assetto Corsa car '([^']+)'", (vp.source if vp else "") or "")
    car_id = m.group(1) if m else None
    for cid in ([] if car_id else car_ids_of(meta, vp)):     # file id first: a car's display name may differ
        car_id = stand_in_for(cid, ac_root=None)
        if car_id:
            break
    if not car_id:
        raise ValueError(f"'{meta['car']}' is not a car from Assetto Corsa (its file was built from public data) "
                         "and vehicles/ac_stand_ins.yaml names no installed game car to show it with")
    return car_id, str(tm.get("track") or meta["track"]), str(tm.get("layout") or "")


def car_ids_of(meta: dict, vp=None) -> list[str]:
    """Names a lap's car may be listed under: its car file's id, the lap's car name, the file's name field."""
    out = []
    if meta.get("car_file"):
        out.append(Path(str(meta["car_file"])).stem)
    out += [n for n in (meta.get("car"), getattr(vp, "name", None)) if n and n not in out]
    return out


def stand_in_repaint(car: str) -> dict[str, str]:
    """{material: hex colour or "hide"} to paint a stand-in plain (no names on it), or {} (WP52)."""
    import yaml
    f = ROOT / "vehicles" / "ac_stand_ins.yaml"
    if not f.is_file():
        return {}
    entry = ((yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("stand_ins") or {}).get(car)
    rp = entry.get("repaint") if isinstance(entry, dict) else None
    return {str(k): str(v) for k, v in (rp or {}).items()}


def stand_in_model(car: str) -> str | None:
    """Our own model the viewer draws instead of the stand-in's body ("fs_generic"), or None."""
    import yaml
    f = ROOT / "vehicles" / "ac_stand_ins.yaml"
    if not f.is_file():
        return None
    entry = ((yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("stand_ins") or {}).get(car)
    return entry.get("draw") if isinstance(entry, dict) else None


def stand_in_for(car: str, ac_root=None) -> str | None:
    """The installed game car that shows a car file the game does not have (vehicles/ac_stand_ins.yaml).

    Example: our Formula Student electric car has no counterpart in Assetto Corsa; the
    Tatuus FA1 single-seater body is shown instead (WP46). None if nothing is listed or
    the listed car is not installed.
    """
    import yaml
    f = ROOT / "vehicles" / "ac_stand_ins.yaml"
    if not f.is_file():
        return None
    table = (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("stand_ins") or {}
    sid = table.get(car)
    if isinstance(sid, dict):                # {car: <game car>, draw: <our model>}
        sid = sid.get("car")
    if not sid:
        return None
    try:
        from mv.ac.paths import find_ac_root
        root = Path(ac_root) if ac_root else find_ac_root()
    except Exception:
        return sid                       # no game found here: let the caller fail later with a clear message
    return sid if (root / "content" / "cars" / sid).is_dir() else None


def car_skins(ac_root: Path, car_id: str) -> list[str]:
    d = Path(ac_root) / "content" / "cars" / car_id / "skins"
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []


# ---------------------------------------------------------------------- race.ini
def _parse_ini(text: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """Sections in file order, keys in file order, values as raw text (AC's INI dialect: no quoting)."""
    out: list[tuple[str, list[tuple[str, str]]]] = []
    cur: list[tuple[str, str]] | None = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            cur = []
            out.append((s[1:-1], cur))
        elif "=" in s and cur is not None and not s.startswith((";", "#")):
            k, v = s.split("=", 1)
            cur.append((k.strip(), v.strip()))
    return out


def _dump_ini(sections) -> str:
    return "\n".join(f"[{name}]\n" + "".join(f"{k}={v}\n" for k, v in keys) for name, keys in sections)


def build_race_ini(old_text: str, car: str, skin: str, track: str, layout: str,
                   ghost_car: bool = False, ghost_skin: str = "", ideal: bool = True) -> str:
    """New race.ini text: the owner's file with car, circuit, session and cars replaced.

    ``ideal`` (default, WP49): conditions for a demo that loops for hours - full and constant track
    grip (no rubbering-in, no randomness), 22 C air / 30 C road, no wind. The owner's own assist
    settings (damage, wear, fuel use) live in assists.ini and are left alone; the MV Viewer app keeps
    the demo car fresh instead (tyre wear, tyre temperature, fuel, damage).
    """
    secs = _parse_ini(old_text)
    order = [n for n, _ in secs]
    d = {n: dict(keys) for n, keys in secs}

    def put(section: str, **kv):
        if section not in d:
            d[section] = {}
            order.append(section)
        d[section].update({k: str(v) for k, v in kv.items()})

    put("RACE", TRACK=track, CONFIG_TRACK=layout, MODEL=car, MODEL_CONFIG="", CARS=2 if ghost_car else 1,
        SKIN=skin, DRIFT_MODE=0, JUMP_START_PENALTY=0)
    for sec in ("REMOTE", "REPLAY", "BENCHMARK", "RESTART", "__PREVIEW_GENERATION"):
        put(sec, ACTIVE=0)
    put("GHOST_CAR", RECORDING=0, PLAYING=0, LOAD=0, ENABLED=0)
    if ideal:
        put("DYNAMIC_TRACK", SESSION_START=100, SESSION_TRANSFER=100, RANDOMNESS=0, LAP_GAIN=0)
        put("TEMPERATURE", AMBIENT=22, ROAD=30)
        put("WIND", SPEED_KMH_MIN=0, SPEED_KMH_MAX=0)
    # one Practice session, no time limit; drop any other sessions of the last race weekend
    for n in [n for n in order if re.fullmatch(r"SESSION_\d+", n) and n != "SESSION_0"]:
        order.remove(n)
        d.pop(n)
    d["SESSION_0"] = {"NAME": "Practice", "TYPE": "1", "DURATION_MINUTES": "0", "SPAWN_SET": "PIT"}
    if "SESSION_0" not in order:
        order.append("SESSION_0")
    me = d.get("CAR_0", {})
    put("CAR_0", MODEL="-", MODEL_CONFIG="", SKIN=skin, DRIVER_NAME=me.get("DRIVER_NAME", "MV"),
        NATIONALITY=me.get("NATIONALITY", ""), NATION_CODE=me.get("NATION_CODE", ""))
    for n in [n for n in order if re.fullmatch(r"CAR_\d+", n) and n != "CAR_0"]:
        order.remove(n)
        d.pop(n)
    if ghost_car:
        put("CAR_1", MODEL=car, MODEL_CONFIG="", SKIN=ghost_skin or skin, DRIVER_NAME="Ghost (ideal lap)",
            NATIONALITY="", NATION_CODE="", SETUP="", AI_LEVEL=100, AI_AGGRESSION=0, BALLAST=0, RESTRICTOR=0)
    return _dump_ini([(n, list(d[n].items())) for n in order])


def write_race_ini(text: str, path: Path | None = None) -> Path | None:
    """Back up the current race.ini (runs/ac_launch, last 10) and write the new one. Returns the backup."""
    path = path or race_ini_path()
    backup = None
    if path.is_file():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup = BACKUP_DIR / f"race_{time.strftime('%Y%m%d_%H%M%S')}.ini"
        shutil.copy2(path, backup)
        for old in sorted(BACKUP_DIR.glob("race_*.ini"))[:-10]:
            old.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return backup


# ---------------------------------------------------------------------- the game process
def _running(image: str) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {image}", "/NH"], capture_output=True, text=True,
                         creationflags=0x08000000).stdout
    return image.lower() in out.lower()


def game_running() -> bool:
    return _running("acs.exe")


def steam_running() -> bool:
    return _running("steam.exe")


def running_session() -> dict | None:
    """Car and circuit of the session the game has loaded, from its shared memory (None if unknown)."""
    try:
        from mv.ac.bridge.reader import ACReader
        with ACReader() as r:
            s = r.refresh_static()
            car, track, layout = str(s.carModel), str(s.track), str(s.trackConfiguration)
    except Exception:
        return None
    return {"car": car, "track": track, "layout": layout} if car else None


def start_game(ac_root: Path) -> subprocess.Popen:
    """Start acs.exe from the game folder as an independent process (survives the control panel)."""
    exe = Path(ac_root) / "acs.exe"
    if not exe.is_file():
        raise FileNotFoundError(exe)
    flags = 0x00000008 | 0x00000200                      # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    since = time.time()
    proc = subprocess.Popen([str(exe)], cwd=str(ac_root), creationflags=flags, close_fds=True)
    _start_watchdog(proc.pid, since)                     # ends it if it hangs after you quit (watch_game)
    return proc


# ---------------------------------------------------------------------- one call for the UI and the command line
def _same_session(sess: dict | None, car: str, track: str, layout: str) -> bool:
    return bool(sess) and sess["car"] == car and sess["track"] == track and (sess["layout"] or "") == layout


def open_lap_in_game(payload: dict, link, ghost_car: bool = False, restart: bool = False,
                     speed: float = 1.0, ac_root: Path | None = None) -> dict:
    """Show a lap (``viewer_payload``) in Assetto Corsa, starting the game if needed.

    Returns ``{"ok", "msg", "needs_restart"?, "started"?}``:
    * game not running -> race.ini written (backup kept), game started, lap queued;
    * game running in the lap's car and circuit -> the lap plays at once;
    * game running anything else -> ``needs_restart`` unless ``restart``: then
      the MV Viewer app closes the game and it is started in the right session.
    """
    from mv.ac.paths import find_ac_root
    from mv.ac.viewer_link import Command

    car, track, layout = ac_ids_for_lap(payload["meta"])
    where = f"{track}{' / ' + layout if layout else ''} with {car}"
    if game_running():
        sess = running_session()
        if _same_session(sess, car, track, layout):
            link.load(payload, speed=speed, loop=True)
            return {"ok": True, "msg": f"Playing in the running game: {payload['label']}"}
        if not restart:
            now = f"{sess['track']} with {sess['car']}" if sess else "another session (or its menu)"
            return {"ok": True, "needs_restart": True,
                    "msg": f"Assetto Corsa is running {now}. Restart it on {where}?"}
        link.command(Command.QUIT)                        # the MV Viewer app closes the game
        t0 = time.time()
        while game_running() and time.time() - t0 < 25:
            time.sleep(0.5)
        if game_running():
            return {"ok": False, "msg": "The game did not close (is the MV Viewer app installed and in a session?). "
                                       "Quit Assetto Corsa yourself and click again."}
        time.sleep(1.0)
    root = Path(ac_root) if ac_root else find_ac_root()
    skins = car_skins(root, car)
    path = race_ini_path()
    old = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    prev = dict(next((keys for n, keys in _parse_ini(old) if n == "RACE"), []))
    skin = prev.get("SKIN") if prev.get("MODEL") == car and prev.get("SKIN") in skins else (skins[0] if skins else "")
    ghost_skin = next((s for s in skins if s != skin), skin)
    write_race_ini(build_race_ini(old, car, skin, track, layout, ghost_car=ghost_car, ghost_skin=ghost_skin), path)
    start_game(root)
    link.load(payload, speed=speed, loop=True)
    warn = "" if steam_running() else " Steam does not seem to be running - the game needs it."
    return {"ok": True, "started": True,
            "msg": f"Starting Assetto Corsa on {where}{' (+ ghost car)' if ghost_car else ''}. "
                   f"The lap starts by itself when the session has loaded.{warn}"}


# ---------------------------------------------------------------------- exit watchdog
def game_log_path() -> Path:
    return documents_dir() / "Assetto Corsa" / "logs" / "log.txt"


def _alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True,
                         creationflags=0x08000000).stdout
    return str(pid) in out


def _clean_exit_logged(log: Path, since: float) -> bool:
    try:
        if log.stat().st_mtime < since:
            return False
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            return b"CLEAN EXIT" in f.read()
    except OSError:
        return False


def watch_game(pid: int, since: float, log: Path | None = None, grace_s: float = 8.0, poll_s: float = 1.0,
               max_hours: float = 24.0, kill=None) -> str:
    """End a game process that hangs after a clean exit (owner's report, 2026-11-23; log WP33).

    Assetto Corsa writes "CLEAN EXIT" to its log when the player quits. Started
    outside Steam's own launcher, ``acs.exe`` sometimes stays alive afterwards
    without a window, and Steam then shows the game as running/"stopping"
    forever. Content Manager handles this the same way: if the process is still
    there ``grace_s`` seconds after a clean exit was logged, it is terminated.
    A game that has not logged its exit is never touched. Returns what happened.
    """
    log = log or game_log_path()
    kill = kill or (lambda p: subprocess.run(["taskkill", "/PID", str(p), "/F"], capture_output=True,
                                             creationflags=0x08000000))
    t_end = time.time() + max_hours * 3600
    exit_seen = None
    while time.time() < t_end:
        if not _alive(pid):
            return "exited"
        if exit_seen is None and _clean_exit_logged(log, since):
            exit_seen = time.time()
        if exit_seen is not None and time.time() - exit_seen > grace_s:
            kill(pid)
            return "ended after clean exit"
        time.sleep(poll_s)
    return "gave up"


def _start_watchdog(pid: int, since: float) -> None:
    """Run :func:`watch_game` in its own windowless process (outlives the control panel)."""
    import sys
    pyw = Path(sys.executable).with_name("pythonw.exe")
    exe = str(pyw if pyw.is_file() else sys.executable)
    flags = 0x00000008 | 0x00000200 | 0x08000000
    subprocess.Popen([exe, "-m", "mv.ac.launch", "watch", str(pid), str(since)], cwd=str(ROOT),
                     creationflags=flags, close_fds=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "watch":
        watch_game(int(sys.argv[2]), float(sys.argv[3]))
