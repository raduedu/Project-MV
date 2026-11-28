"""Project MV web interface - a small local control panel.

    python scripts/webui.py start     # start in the background and open http://127.0.0.1:8765
    python scripts/webui.py stop      # stop it (training is NOT affected)
    python scripts/webui.py serve     # run in this window (for debugging)

(or ``mvctl ui`` / ``mvctl ui-stop``)

Design notes
------------
* Only the Python standard library: nothing to install.
* Listens on 127.0.0.1 only - nothing outside this laptop can reach it.
* The server runs as its own background process: closing the browser tab
  changes nothing.  Training does not depend on the server at all (it is run
  by the plan supervisor, ``scripts/train_plan.py``), so stopping the server
  never stops training.
* Long actions (imports, evaluations, checks, reference laps) run as
  background tasks; their output is kept in ``runs/ui_tasks/``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_plan as tp  # noqa: E402

HOST, PORT = "127.0.0.1", 8765
UI_FILE = ROOT / "mv" / "ui" / "index.html"
TASKS = ROOT / "runs" / "ui_tasks"
PID_FILE = ROOT / "runs" / "webui.pid"
PY = sys.executable
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,60}$")
PRESETS = ["complex+complex+gp", "complex", "mixed+gp", "mixed", "gp", "circuit", "fs+fs_segments", "fs_segments", "fs"]

_tasks: dict[str, dict] = {}            # task id -> {"proc": Popen, ...}  (this server's tasks)
_lock = threading.Lock()
_ac_cache: dict = {}


# --------------------------------------------------------------------------
# data helpers
# --------------------------------------------------------------------------
def vehicles() -> list[dict]:
    """Every vehicle file we can train/evaluate with, with its catalogue entry.

    Order: favourites first, then by category (mv/vehicle/catalog.py).  Each
    item keeps the original ``name``/``where`` keys and adds ``title``,
    ``category``, ``favourite``, ``status`` and ``reasons`` for the car pickers.
    """
    out, seen = [], set()
    try:
        from mv.vehicle.catalog import list_cars
        for c in list_cars(ac_root()):
            if c.imported:
                seen.add(c.id)
                out.append({"name": c.id, "where": c.where, "title": c.title, "category": c.category,
                            "favourite": c.favourite, "status": c.status, "reasons": c.reasons, "flags": c.flags})
    except Exception as e:                          # the catalogue must never hide the plain file list
        print(f"catalogue unavailable: {e}", file=sys.stderr)
    for sub in ("local", "library"):
        for p in sorted((ROOT / "vehicles" / sub).glob("*.yaml")):
            if p.stem not in seen:
                seen.add(p.stem)
                out.append({"name": p.stem, "where": sub, "title": p.stem, "category": "road",
                            "favourite": False, "status": "unchecked", "reasons": [], "flags": []})
    return out


def catalogue_cars() -> list[dict]:
    """All cars installed in Assetto Corsa (and our library) for the Import tab."""
    from mv.vehicle.catalog import list_cars
    return [{"id": c.id, "title": c.title, "category": c.category, "favourite": c.favourite,
             "installed": c.installed, "available": c.available, "imported": c.imported, "where": c.where,
             "status": c.status, "reasons": c.reasons, "flags": c.flags} for c in list_cars(ac_root())]


def categories() -> list[dict]:
    from mv.vehicle.catalog import CATEGORIES, RACING
    return [{"key": k, "label": v, "racing": k in RACING} for k, v in CATEGORIES.items()]


_viewer = None


def _viewer_link():
    """The shared-memory link to the in-game MV Viewer app, opened once per server (mv/ac/viewer_link.py)."""
    global _viewer
    if _viewer is None:
        from mv.ac.viewer_link import ViewerLink
        _viewer = ViewerLink()
    return _viewer


def local_tracks() -> list[str]:
    return sorted(p.stem for p in (ROOT / "tracks" / "local").glob("*.npz") if not p.stem.startswith("_"))


def checkpoints() -> list[str]:
    out = []
    for run in sorted((ROOT / "runs").iterdir()):
        if run.is_dir():
            for f in ("best.pt", "latest.pt"):
                if (run / f).is_file():
                    out.append(f"runs/{run.name}/{f}")
    return out


def ac_root() -> Path | None:
    try:
        from mv.ac.paths import find_ac_root
        return find_ac_root(None)
    except Exception:
        return None


def ac_lists() -> dict:
    """Cars and tracks installed in Assetto Corsa (cached)."""
    if _ac_cache:
        return _ac_cache
    root = ac_root()
    cars, tracks = [], []
    if root:
        for d in sorted((root / "content" / "cars").iterdir()):
            if not d.is_dir():
                continue
            title = d.name
            ui = d / "ui" / "ui_car.json"
            if ui.is_file():
                m = re.search(r'"name"\s*:\s*"([^"]+)"', ui.read_text(encoding="utf-8", errors="ignore"))
                if m:
                    title = m.group(1)
            cars.append({"id": d.name, "title": title})
        for d in sorted((root / "content" / "tracks").iterdir()):
            if not d.is_dir():
                continue
            if (d / "ai" / "fast_lane.ai").is_file():
                tracks.append(d.name)
            for lay in sorted(x for x in d.iterdir() if x.is_dir()):
                if (lay / "ai" / "fast_lane.ai").is_file():
                    tracks.append(f"{d.name}:{lay.name}")
    _ac_cache.update(root=str(root) if root else None, cars=cars, tracks=tracks)
    return _ac_cache


def run_details(name: str) -> dict:
    """Curves and validation history of one run (for the charts)."""
    run = ROOT / "runs" / name
    out: dict = {"name": name, "curves": {}, "validations": [], "log_tail": "", "files": []}
    csv = run / "log.csv"
    if csv.is_file():
        rows = csv.read_text().strip().splitlines()
        head = rows[0].split(",")
        cols = {h: [] for h in head}
        for r in rows[1:]:
            for h, v in zip(head, r.split(",")):
                try:
                    cols[h].append(float(v))
                except ValueError:
                    cols[h].append(None)
        step = max(1, len(rows) // 400)          # at most ~400 points per curve
        for key in ("hours", "pace_kmh", "crashes_per_100km", "mean_speed_kmh", "val_finished", "std"):
            if key in cols:
                out["curves"][key] = cols[key][::step]
    log = ROOT / "runs" / f"{name}.log"
    if log.is_file():
        txt = log.read_text(encoding="utf-8", errors="replace")
        for block in txt.split("  validation:")[1:]:
            m_upd = re.search(r"\nupdate\s+(\d+)", block)
            tracks = {}
            for mm in re.finditer(r"\n\s+(\S+)\s+flying lap\s+([^\n]*)", block.split("\nupdate")[0]):
                v = mm.group(2).strip()
                tracks[mm.group(1)] = float(v.split()[0]) if v[:1].isdigit() else None
            out["validations"].append({"update": int(m_upd.group(1)) if m_upd else None, "tracks": tracks})
        out["log_tail"] = "\n".join(txt.strip().splitlines()[-25:])
    if run.is_dir():
        out["files"] = sorted(p.name for p in run.iterdir() if p.is_file() and not p.name.startswith("events."))
    cfg = run / "config.json"
    if cfg.is_file():
        try:
            out["config"] = json.loads(cfg.read_text())["args"]
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------
# background tasks
# --------------------------------------------------------------------------
def start_task(kind: str, title: str, args: list[str], out_dir: str | None = None) -> str:
    TASKS.mkdir(parents=True, exist_ok=True)
    # unique even for several tasks started within the same second
    tid = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}_{kind}"
    while (TASKS / f"{tid}.json").exists():
        tid += "x"
    log = TASKS / f"{tid}.log"
    meta = {"id": tid, "kind": kind, "title": title, "cmd": args, "started": time.strftime("%H:%M:%S"),
            "out_dir": out_dir}
    (TASKS / f"{tid}.json").write_text(json.dumps(meta))
    f = open(log, "w", encoding="utf-8")
    proc = subprocess.Popen([PY, "-u", *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                            creationflags=0x08000000, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    with _lock:
        _tasks[tid] = {**meta, "proc": proc}
    return tid


def task_list() -> list[dict]:
    out = []
    for meta_f in sorted(TASKS.glob("*.json"), reverse=True)[:30] if TASKS.is_dir() else []:
        meta = json.loads(meta_f.read_text())
        tid = meta["id"]
        with _lock:
            live = _tasks.get(tid)
        if live:
            rc = live["proc"].poll()
            meta["status"] = "running" if rc is None else ("done" if rc == 0 else f"failed ({rc})")
        else:
            meta["status"] = "ended"
        log = TASKS / f"{tid}.log"
        txt = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        meta["output"] = "\n".join(txt.strip().splitlines()[-60:])
        if meta.get("out_dir"):
            d = ROOT / meta["out_dir"]
            meta["images"] = [f"{meta['out_dir']}/{p.name}" for p in sorted(d.glob("*.png"))] if d.is_dir() else []
        out.append(meta)
    return out


# --------------------------------------------------------------------------
# actions
# --------------------------------------------------------------------------
def build_job(f: dict) -> dict:
    name = str(f.get("name", "")).strip()
    if not NAME_RE.match(name):
        raise ValueError("name: letters, digits, _ and - only")
    if any(j["name"] == name for lane in tp.load_plan() for j in lane) or (ROOT / "runs" / name).exists():
        raise ValueError(f"a run called '{name}' already exists - choose another name")
    cars = [c for c in f.get("vehicles", []) if c]
    known = {v["name"] for v in vehicles()}
    if not cars or any(c not in known for c in cars):
        raise ValueError("choose at least one known vehicle")
    preset = f.get("preset", "mixed+gp")
    if preset not in PRESETS:
        raise ValueError("unknown track preset")
    hours = float(f.get("hours", 2))
    if not 0.05 <= hours <= 48:
        raise ValueError("hours must be between 0.05 and 48")
    args = ["--vehicle", ",".join(cars), "--preset", preset, "--val-preset", preset.split("+")[0],
            "--envs", str(int(f.get("envs", 16384))), "--tracks", "512", "--minibatches", "8",
            "--eval-every", "100", "--hours", f"{hours:g}"]
    init = f.get("init_from") or ""
    if init:
        if init not in checkpoints():
            raise ValueError("unknown checkpoint")
        args += ["--init-from", init]
    elif f.get("warm_start"):
        args += ["--warm-start"]
    if int(f.get("history", 0) or 0) > 0:
        args += ["--history", str(int(f["history"]))]
    if f.get("limit_seeking"):
        args += ["--limit-seeking"]
    lo, hi = f.get("grip_low"), f.get("grip_high")
    if lo and hi:
        args += ["--grip-range", f"{float(lo):g}", f"{float(hi):g}"]
    if f.get("reset_noise"):
        args += ["--reset-noise", f"{float(f['reset_noise']):g}"]
    return {"name": name, "args": args}


def act(path: str, body: dict) -> dict:
    if path == "/api/pause":
        threading.Thread(target=tp.pause, kwargs={"quiet": True}, daemon=True).start()
        return {"ok": True, "msg": "Pausing: every training saves its driver and stops (a few seconds)."}
    if path == "/api/resume":
        started = tp.start_supervisor()
        return {"ok": True, "msg": "Training resumed." if started else "The plan was already running."}
    if path == "/api/plan/add":
        job = build_job(body)
        lanes = tp.load_plan()
        lane = int(body.get("lane", 0))
        if lane >= len(lanes):
            lanes.append([])
            lane = len(lanes) - 1
        lanes[lane].append(job)
        tp.save_plan(lanes)
        return {"ok": True, "msg": f"Added {job['name']} to lane {lane + 1}. It starts when its turn comes"
                                   + (" (the plan is paused - resume to run it)." if tp.PAUSE_FLAG.exists() else ".")}
    if path == "/api/plan/remove":
        name = body.get("name")
        lanes = tp.load_plan()
        running = {p["job"] for p in tp.training_processes()}
        if name in running:
            raise ValueError("this job is running - pause first")
        lanes = [[j for j in lane if j["name"] != name] for lane in lanes]
        tp.save_plan([lane for lane in lanes if lane])
        return {"ok": True, "msg": f"Removed {name} from the plan (its files in runs/ are kept)."}
    if path == "/api/plan/move":
        name, d = body.get("name"), int(body.get("dir", 0))
        lanes = tp.load_plan()
        for lane in lanes:
            for i, j in enumerate(lane):
                if j["name"] == name and 0 <= i + d < len(lane):
                    lane[i], lane[i + d] = lane[i + d], lane[i]
                    tp.save_plan(lanes)
                    return {"ok": True, "msg": "Moved."}
        return {"ok": False, "msg": "Cannot move further."}
    if path == "/api/import/car":
        car = body.get("car", "")
        if car not in {c["id"] for c in ac_lists()["cars"]}:
            raise ValueError("unknown car")
        tid = start_task("import", f"Import car {car}", ["scripts/import_ac_car.py", car])
        return {"ok": True, "task": tid, "msg": f"Importing {car} ..."}
    if path == "/api/import/many":
        # several cars in one background task: "racing", "all", "favourites" or a category key
        from mv.vehicle.catalog import CATEGORIES
        sel = str(body.get("selection", "racing"))
        if sel not in ("racing", "all", "favourites", *CATEGORIES):
            raise ValueError("unknown selection")
        tid = start_task("import", f"Import {sel} cars", ["scripts/import_ac_car.py", "--category", sel])
        return {"ok": True, "task": tid, "msg": f"Importing {sel} cars - progress under Tools > Tasks (a few s per car)."}
    if path == "/api/favourite":
        from mv.vehicle.catalog import set_favourite
        car = str(body.get("car", ""))
        known = {c["id"] for c in catalogue_cars()}
        if car not in known:
            raise ValueError("unknown car")
        on = bool(body.get("on"))
        set_favourite(car, on)
        return {"ok": True, "msg": f"{car} {'added to' if on else 'removed from'} favourites (vehicles/favourites.yaml)."}
    if path == "/api/import/track":
        track = body.get("track", "")
        if track not in ac_lists()["tracks"]:
            raise ValueError("unknown track")
        tid = start_task("import", f"Import track {track}", ["scripts/import_ac_track.py", track])
        return {"ok": True, "task": tid, "msg": f"Importing {track} ..."}
    if path == "/api/ac_view":
        # Show a lap in Assetto Corsa: starts the game on the lap's circuit and car if needed
        # (mv/ac/launch.py, log WP33). The shared-memory link stays open for the server's lifetime.
        from mv.ac.launch import open_lap_in_game
        from mv.ac.viewer_link import viewer_payload
        # every lap passes the playback stage (mv/ac/playback.py, WP70): unsafe laps are repaired or refused
        payload = viewer_payload(body.get("key", ""), body.get("ghost", "auto") or None)
        res = open_lap_in_game(payload, _viewer_link(), ghost_car=bool(body.get("ghost_car")),
                               restart=bool(body.get("restart")), speed=float(body.get("speed", 1.0)))
        if payload.get("repairs") and isinstance(res, dict):
            res["msg"] = f"{res.get('msg', '')} - playback repaired: " + "; ".join(payload["repairs"])
        return res
    if path == "/api/ac_view_cmd":
        from mv.ac.viewer_link import Command
        cmd = {"play": Command.PLAY, "pause": Command.PAUSE, "restart": Command.RESTART,
               "stop": Command.STOP, "quit": Command.QUIT}[body.get("cmd", "")]
        _viewer_link().command(cmd, speed=body.get("speed"))
        return {"ok": True, "msg": f"Assetto Corsa viewer: {body.get('cmd')}", "state": _viewer_link().state()}
    if path == "/api/reproduce":
        from mv.laps import load_lap
        key = body.get("key", "")
        meta = load_lap(key)["meta"]
        tid = start_task("reproduce", f"Reproduce {meta['driver']} @ {meta['track']} ({meta['lap_time']:.3f} s)",
                         ["scripts/reproduce_lap.py", key])
        return {"ok": True, "task": tid, "msg": "Driving the lap again from the saved driver - see Tools > Tasks."}
    if path == "/api/check":
        v = body.get("vehicle", "")
        if v not in {x["name"] for x in vehicles()}:
            raise ValueError("unknown vehicle")
        tid = start_task("check", f"Vehicle check {v}", ["scripts/vehicle_check.py", v])
        return {"ok": True, "task": tid, "msg": f"Checking {v} ..."}
    if path == "/api/reference":
        v, t = body.get("vehicle", ""), body.get("track", "")
        if v not in {x["name"] for x in vehicles()} or t not in local_tracks():
            raise ValueError("choose a vehicle and an imported track")
        tid = start_task("reference", f"Reference lap {v} @ {t}", ["scripts/reference_lap.py", "--vehicle", v, "--track", t])
        return {"ok": True, "task": tid, "msg": "Computing the reference lap ..."}
    if path == "/api/evaluate":
        v = body.get("vehicle", "")
        if v not in {x["name"] for x in vehicles()}:
            raise ValueError("unknown vehicle")
        tracks = [t for t in body.get("tracks", []) if t in local_tracks()]
        tid_dir = f"runs/eval_ui/{time.strftime('%Y%m%d_%H%M%S')}"
        args = ["scripts/evaluate.py", "--vehicle", v, "--tracks", *tracks, "--random", str(int(body.get("random", 2))),
                "--laps", str(int(body.get("laps", 2))), "--out", tid_dir]
        pol = body.get("policy") or ""
        if pol:
            if pol not in checkpoints():
                raise ValueError("unknown checkpoint")
            args += ["--policy", pol]
        if body.get("scripted") or not pol:
            args += ["--scripted"]
        if body.get("blind"):
            args += ["--blind"]
        if body.get("grip") and float(body["grip"]) != 1.0:
            args += ["--grip", f"{float(body['grip']):g}"]
        title = f"Evaluate {pol or 'rule-based driver'} ({v})"
        tid = start_task("evaluate", title, args, out_dir=tid_dir)
        return {"ok": True, "task": tid, "msg": "Evaluation started - results appear under Tasks."}
    raise ValueError(f"unknown action {path}")


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "ProjectMV/1"

    def log_message(self, *a):                  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        try:
            if p in ("/", "/index.html"):
                return self._send(200, UI_FILE.read_bytes(), "text/html; charset=utf-8")
            if p == "/replay.js":
                return self._send(200, (UI_FILE.parent / "replay.js").read_bytes(), "text/javascript; charset=utf-8")
            if p == "/api/state":
                return self._json(tp.snapshot(max_age_s=3.0))
            if p == "/api/options":
                return self._json({"vehicles": vehicles(), "tracks": local_tracks(), "checkpoints": checkpoints(),
                                   "presets": PRESETS, "lanes": len(tp.load_plan()), "categories": categories()})
            if p == "/api/ac":
                # tracks are cached; cars come fresh from the catalogue (favourites/imports change)
                return self._json({**ac_lists(), "cars": catalogue_cars()})
            if p == "/api/runs":
                runs = sorted((d.name for d in (ROOT / "runs").iterdir() if d.is_dir() and (d / "log.csv").is_file()),
                              key=lambda n: (ROOT / "runs" / n / "log.csv").stat().st_mtime, reverse=True)
                return self._json(runs)
            if p == "/api/run":
                name = parse_qs(u.query).get("name", [""])[0]
                if not NAME_RE.match(name):
                    raise ValueError("bad run name")
                return self._json(run_details(name))
            if p == "/api/tasks":
                return self._json(task_list())
            if p == "/api/laps":
                from mv.laps import list_laps
                return self._json(list_laps())
            if p in ("/api/lap", "/api/lap.csv"):
                from mv.laps import lap_to_csv, load_lap
                key = parse_qs(u.query).get("key", [""])[0]
                lap = load_lap(key)
                if p == "/api/lap":
                    return self._json(lap)
                body = lap_to_csv(lap).encode()
                fname = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{lap['meta']['track']}_{lap['meta']['driver']}_{lap['meta']['lap_time']}.csv")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if p.startswith("/files/"):
                rel = unquote(p[len("/files/"):])
                f = (ROOT / rel).resolve()
                allowed = (ROOT / "runs").resolve(), (ROOT / "docs").resolve()
                if not any(str(f).startswith(str(a)) for a in allowed) or not f.is_file():
                    return self._send(404, b"not found", "text/plain")
                ctype = {".png": "image/png", ".gif": "image/gif", ".csv": "text/csv", ".json": "application/json",
                         ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8"}.get(
                    f.suffix, "text/plain; charset=utf-8")
                return self._send(200, f.read_bytes(), ctype)
            self._send(404, b"not found", "text/plain")
        except Exception as e:
            self._json({"ok": False, "msg": str(e)}, 400)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            self._json(act(urlparse(self.path).path, body))
        except Exception as e:
            self._json({"ok": False, "msg": str(e)}, 400)


def serve() -> None:
    PID_FILE.write_text(str(__import__("os").getpid()))
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        httpd.serve_forever()
    finally:
        PID_FILE.unlink(missing_ok=True)


def running_pid() -> int | None:
    if not PID_FILE.is_file():
        return None
    pid = int(PID_FILE.read_text().strip() or 0)
    for p in tp.python_processes():
        if p["ProcessId"] == pid and "webui.py" in (p.get("CommandLine") or ""):
            return pid
    return None


def start(open_browser: bool = True) -> None:
    url = f"http://{HOST}:{PORT}"
    if running_pid():
        print(f"already running: {url}")
    else:
        flags = 0x00000008 | 0x00000200 | 0x08000000      # detached, own process group, no window
        log = open(ROOT / "runs" / "webui.log", "a")
        subprocess.Popen([PY, "-u", str(Path(__file__).resolve()), "serve"], cwd=ROOT, stdout=log, stderr=log,
                         creationflags=flags, close_fds=True)
        time.sleep(2)
        print(f"web interface running: {url}   (stop with: mvctl ui-stop - training is not affected)")
    if open_browser:                                  # not for "mvctl remote": nobody is at the laptop
        webbrowser.open(url)


def stop() -> None:
    pid = running_pid()
    if pid:
        # only the server itself - NOT its process tree: a training supervisor started with
        # the Resume button is a child of the server and must keep running
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, creationflags=0x08000000)
        PID_FILE.unlink(missing_ok=True)
        print("web interface stopped (training continues)")
    else:
        print("web interface is not running")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "start"
    if cmd == "start":
        start(open_browser="--no-browser" not in sys.argv)
    else:
        {"stop": stop, "serve": serve}.get(cmd, start)()
