"""Run a plan of training jobs unattended, with PAUSE and RESUME.

The plan (``runs/plan.json``) is a list of *lanes*; each lane is a list of
training jobs run one after another.  Lanes run side by side (two lanes fit on
the laptop GPU).  Each job is an ordinary ``scripts/train.py`` command::

    {"lanes": [
        [{"name": "gt3rs_v4_memory", "args": ["--vehicle", "...", "--hours", "2.5"]},
         {"name": "gt3rs_limit_v1",  "args": [...]}],
        [{"name": "fs_v1", "args": [...]}]
    ]}

Commands (from the project root - or simply ``mvctl <command>``, or the .cmd files)::

    python scripts/train_plan.py status     # what is running, done, paused, waiting
    python scripts/train_plan.py watch      # the same, refreshed every 5 s with progress bars (Ctrl+C to leave)
    python scripts/train_plan.py pause      # save every running driver and stop; nothing new starts
    python scripts/train_plan.py resume     # continue exactly where it stopped, then the rest of the plan

How it knows where it is (no hidden state - it is all in ``runs/``):
  * a job is DONE when ``runs/<name>/FINISHED`` exists (or its log ends with "done:");
  * a job is RESUMABLE when ``runs/<name>/latest.pt`` exists - it continues with
    ``--resume`` and the same time budget (``--hours`` counts the whole run);
  * otherwise it has not started.

The plan file may be edited while the plan runs (e.g. from the web UI): the
supervisor re-reads it and starts new jobs and lanes when their turn comes.

Pausing: ``pause`` creates ``runs/PAUSE``.  Every trainer checks for it after
each update (a few seconds), saves ``latest.pt`` and exits.  A trainer that does
not react within the waiting time (for example one started before this
mechanism existed) is stopped; its ``latest.pt`` is then at most 25 updates
(~1-2 minutes of training) old.  ``runs/PAUSE`` stays until ``resume``, so
nothing can start in between.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PLAN = RUNS / "plan.json"
PAUSE_FLAG = RUNS / "PAUSE"
SUPERVISOR_PID = RUNS / "plan_supervisor.pid"
LOG = RUNS / "plan.log"
EXIT_PAUSED = 3
_PLAN_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def say(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


_proc_cache: tuple[float, list[dict]] = (0.0, [])


def python_processes(max_age_s: float = 0.0) -> list[dict]:
    """All python.exe processes with their command lines (Windows)."""
    global _proc_cache
    if max_age_s > 0 and time.time() - _proc_cache[0] < max_age_s:
        return _proc_cache[1]
    cmd = ["powershell", "-NoProfile", "-Command",
           "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
           "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"]
    out = subprocess.run(cmd, capture_output=True, text=True, creationflags=0x08000000).stdout.strip()
    data = json.loads(out) if out else []
    data = data if isinstance(data, list) else [data]
    _proc_cache = (time.time(), data)
    return data


def training_processes(name: str | None = None, max_age_s: float = 0.0) -> list[dict]:
    """Running train.py processes of this project (optionally only the job ``name``)."""
    out = []
    for p in python_processes(max_age_s):
        cl = p.get("CommandLine") or ""
        if "train.py" not in cl or "--name" not in cl:
            continue
        job = cl.split("--name", 1)[1].split()[0].strip('"')
        if name is None or job == name:
            out.append({**p, "job": job})
    return out


def kill_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=0x08000000)


def load_plan() -> list[list[dict]]:
    if not PLAN.is_file():
        return []
    with _PLAN_LOCK:
        return json.loads(PLAN.read_text())["lanes"]


def save_plan(lanes: list[list[dict]]) -> None:
    with _PLAN_LOCK:
        tmp = PLAN.with_suffix(".tmp")
        tmp.write_text(json.dumps({"lanes": lanes}, indent=2))
        os.replace(tmp, PLAN)


def job_state(job: dict) -> str:
    run = RUNS / job["name"]
    if (run / "FINISHED").is_file():
        return "done"
    if (run / "FAILED").is_file():
        return "failed"
    log = RUNS / f"{job['name']}.log"
    if log.is_file():
        tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:]
        if tail and tail[0].startswith("done:"):
            return "done"
    if (run / "latest.pt").is_file():
        return "resumable"
    return "not started"


def _arg(job: dict, flag: str, default=None):
    a = job.get("args", [])
    return next((a[i + 1] for i, x in enumerate(a) if x == flag and i + 1 < len(a)), default)


def job_info(job: dict, running: dict[str, int] | None = None) -> dict:
    """Everything known about one job, for the status views and the web UI."""
    name = job["name"]
    running = running if running is not None else {p["job"]: p["ProcessId"] for p in training_processes()}
    info = {"name": name, "args": job.get("args", []), "vehicle": _arg(job, "--vehicle", "?"),
            "state": "running" if name in running else job_state(job),
            "hours_goal": float(_arg(job, "--hours", 0) or 0), "hours": 0.0, "update": 0, "decisions": 0,
            "pace_kmh": None, "crashes_100km": None, "speed": None, "validation": None}
    csv = RUNS / name / "log.csv"
    if csv.is_file():
        rows = csv.read_text().strip().splitlines()
        if len(rows) > 1:
            head, last = rows[0].split(","), rows[-1].split(",")
            row = dict(zip(head, last))
            info.update(update=int(float(row.get("iteration", 0))), decisions=int(float(row.get("decisions", 0))),
                        hours=float(row.get("hours", 0)))
            for k_csv, k_info in (("pace_kmh", "pace_kmh"), ("crashes_per_100km", "crashes_100km")):
                try:
                    info[k_info] = float(row[k_csv])
                except (KeyError, ValueError):
                    pass
    log = RUNS / f"{name}.log"
    if log.is_file():
        txt = log.read_text(encoding="utf-8", errors="replace")
        m = re.findall(r"\| ([\d,]+)/s \|", txt[-4000:])
        if m:
            info["speed"] = int(m[-1].replace(",", ""))
        if "  validation:" in txt:
            block = txt.rsplit("  validation:", 1)[1].split("\nupdate")[0].split("\n  validating")[0]
            lines = block.splitlines()
            tracks = []
            for line in lines[1:]:
                mm = re.match(r"\s+(\S+)\s+flying lap\s+(.*)", line)
                if mm:
                    v = mm.group(2).strip()
                    lap = float(v.split()[0]) if v[:1].isdigit() else None
                    tracks.append({"track": mm.group(1), "lap": lap, "note": "" if lap else v.strip("-() ")})
            pct = re.match(r"\s*(\d+)", lines[0]) if lines else None
            info["validation"] = {"completed_pct": int(pct.group(1)) if pct else None, "tracks": tracks}
    goal = info["hours_goal"]
    info["progress"] = 1.0 if info["state"] == "done" else (min(info["hours"] / goal, 1.0) if goal > 0 else 0.0)
    return info


def supervisor_alive(max_age_s: float = 0.0) -> int | None:
    if SUPERVISOR_PID.is_file():
        try:
            pid = int(SUPERVISOR_PID.read_text().strip())
        except ValueError:
            return None
        for p in python_processes(max_age_s):
            if p["ProcessId"] == pid and "train_plan.py" in (p.get("CommandLine") or ""):
                return pid
    return None


def snapshot(max_age_s: float = 0.0) -> dict:
    """The whole picture: plan, states, progress (used by status, watch and the web UI)."""
    running = {p["job"]: p["ProcessId"] for p in training_processes(max_age_s=max_age_s)}
    lanes = load_plan()
    out_lanes = [[job_info(j, running) for j in jobs] for jobs in lanes]
    planned = {j["name"] for jobs in lanes for j in jobs}
    sup = supervisor_alive(max_age_s)
    if PAUSE_FLAG.exists():
        overall = "pausing" if running else "paused"
    elif running:
        overall = "running"
    elif sup:
        overall = "idle"                     # supervisor alive, nothing to do
    else:
        overall = "stopped"
    log_tail = LOG.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-8:] if LOG.is_file() else []
    return {"overall": overall, "paused": PAUSE_FLAG.exists(), "supervisor": bool(sup), "lanes": out_lanes,
            "others": sorted(set(running) - planned), "log": log_tail, "time": time.strftime("%H:%M:%S")}


# --------------------------------------------------------------------------
# the supervisor (runs detached; started by "resume")
# --------------------------------------------------------------------------
def _next_job(k: int) -> dict | None:
    lanes = load_plan()
    if k >= len(lanes):
        return None
    for job in lanes[k]:
        if job_state(job) not in ("done", "failed"):
            return job
    return None


def run_lane(k: int) -> None:
    while True:
        if PAUSE_FLAG.exists():
            return
        job = _next_job(k)                   # re-read every time: the plan may have been edited
        if job is None:
            return
        name = job["name"]
        running = training_processes(name)
        if running:
            say(f"lane {k + 1}: {name} is already running (pid {running[0]['ProcessId']}); waiting for it")
            while training_processes(name):
                time.sleep(30)
            continue                          # re-check: finished, paused or crashed
        state = job_state(job)
        cmd = [sys.executable, "-u", str(ROOT / "scripts" / "train.py"), "--name", name, *job["args"]]
        if state == "resumable":
            cmd.append("--resume")
        say(f"lane {k + 1}: {'resume' if state == 'resumable' else 'start'} {name}")
        with open(RUNS / f"{name}.log", "a") as out, open(RUNS / f"{name}.err", "a") as err:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=err).returncode
        if rc == EXIT_PAUSED:
            say(f"lane {k + 1}: {name} paused")
            return
        if rc != 0:
            if PAUSE_FLAG.exists():           # stopped by "pause"
                say(f"lane {k + 1}: {name} stopped for pause (exit {rc})")
                return
            (RUNS / name).mkdir(parents=True, exist_ok=True)
            (RUNS / name / "FAILED").write_text(f"exit code {rc} at {time.ctime()}; see runs/{name}.err\n")
            say(f"lane {k + 1}: {name} FAILED (exit {rc}); continuing with the next job")


def supervise() -> None:
    """Keep every lane busy; pick up jobs/lanes added to the plan later; stop when paused."""
    SUPERVISOR_PID.write_text(str(os.getpid()))
    say("supervisor started")
    threads: dict[int, threading.Thread] = {}
    idle_said = False
    try:
        while True:
            if PAUSE_FLAG.exists():
                if not any(t.is_alive() for t in threads.values()):
                    break
            else:
                for k in range(len(load_plan())):
                    if (k not in threads or not threads[k].is_alive()) and _next_job(k) is not None:
                        threads[k] = threading.Thread(target=run_lane, args=(k,), daemon=False)
                        threads[k].start()
                        idle_said = False
                if not any(t.is_alive() for t in threads.values()) and not idle_said:
                    say("all planned jobs are done - waiting for new ones")
                    idle_said = True
            time.sleep(20)
    finally:
        say("supervisor stopped")
        SUPERVISOR_PID.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# user commands
# --------------------------------------------------------------------------
def _bar(frac: float, width: int = 28) -> str:
    n = int(round(max(0.0, min(frac, 1.0)) * width))
    return "█" * n + "░" * (width - n)


def _fmt_lap(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:06.3f}"


def render(snap: dict) -> str:
    head = {"paused": "PAUSED  - resume to continue", "pausing": "PAUSING - saving drivers ...", "running": "RUNNING", "idle": "IDLE - plan finished, waiting for jobs",
            "stopped": "STOPPED - resume to start the plan"}[snap["overall"]]
    lines = [f"Project MV training   {snap['time']}   {head}", ""]
    for k, jobs in enumerate(snap["lanes"]):
        lines.append(f"lane {k + 1}")
        for j in jobs:
            st = j["state"].upper() if j["state"] == "running" else j["state"]
            detail = ""
            if j["update"]:
                detail = f"{j['hours']:.2f}/{j['hours_goal']:g} h  upd {j['update']}"
                if j["state"] == "running":
                    left = max(j["hours_goal"] - j["hours"], 0.0)
                    detail += f"  ~{left * 60:.0f} min left"
                    if j["speed"]:
                        detail += f"  {j['speed'] / 1000:.0f}k/s"
                    if j["pace_kmh"] is not None:
                        detail += f"  pace {j['pace_kmh']:.0f} km/h"
            lines.append(f"  {j['name']:<20} {st:<12} {_bar(j['progress'])} {j['progress'] * 100:3.0f}%  {detail}")
            v = j.get("validation")
            if v and j["state"] in ("running", "resumable"):
                done = [t for t in v["tracks"] if t["lap"]]
                real = ", ".join(f"{t['track']} {_fmt_lap(t['lap'])}" for t in v["tracks"]
                                 if t["lap"] and not t["track"].startswith("val_"))
                lines.append(f"  {'':<20} last check: {len(done)}/{len(v['tracks'])} tracks  {real}")
        lines.append("")
    if snap["others"]:
        lines.append(f"other training processes (not in the plan): {', '.join(snap['others'])}")
    return "\n".join(lines)


def status() -> None:
    print(render(snapshot()))


def watch(every_s: float = 5.0) -> None:
    try:
        while True:
            txt = render(snapshot())
            os.system("cls" if os.name == "nt" else "clear")
            print(txt)
            print(f"(refreshes every {every_s:g} s - Ctrl+C to leave; training is not affected)")
            time.sleep(every_s)
    except KeyboardInterrupt:
        print()


def pause(wait_s: float = 240.0, quiet: bool = False) -> None:
    PAUSE_FLAG.write_text(f"paused at {time.ctime()}\n")
    say("pause requested")
    if not quiet:
        print("pause requested: each training saves its driver after its current update ...")
    t0 = time.time()
    while training_processes() and time.time() - t0 < wait_s:
        time.sleep(5)
    for p in training_processes():
        # did not react (e.g. started before pausing existed): stop it; latest.pt is <= 25 updates old
        if not quiet:
            print(f"  stopping {p['job']} (pid {p['ProcessId']}) - its last save is at most ~1-2 minutes old")
        kill_tree(int(p["ProcessId"]))
    for p in python_processes():             # older queue processes would otherwise start their next job
        if "queue_runs.py" in (p.get("CommandLine") or ""):
            kill_tree(int(p["ProcessId"]))
    sup = supervisor_alive()
    if sup:
        time.sleep(3)
        if supervisor_alive():
            kill_tree(sup)
    SUPERVISOR_PID.unlink(missing_ok=True)
    say("paused")
    if not quiet:
        time.sleep(2)
        if training_processes():
            print("WARNING: some training is still running:", [p["job"] for p in training_processes()])
        else:
            print("\nAll training stopped and saved. It is safe to close the laptop.\n")
        status()


def start_supervisor() -> bool:
    """Start the detached supervisor (removes the pause flag).  False if already running."""
    if supervisor_alive():
        return False
    PAUSE_FLAG.unlink(missing_ok=True)
    flags = 0x00000008 | 0x00000200 | 0x08000000          # DETACHED_PROCESS | NEW_PROCESS_GROUP | NO_WINDOW
    log = open(LOG, "a")
    subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "run"], cwd=ROOT, stdout=log, stderr=log,
                     creationflags=flags, close_fds=True)
    say("resumed by user")
    return True


def resume() -> None:
    if not start_supervisor():
        print("the plan is already running\n")
    else:
        time.sleep(8)
    status()


def main() -> None:
    # progress bars use block characters; make sure any output (console, file, pipe) can show them
    if (sys.stdout.encoding or "").lower().replace("-", "") != "utf8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"status": status, "watch": watch, "pause": pause, "resume": resume, "start": resume, "run": supervise}.get(
        cmd, lambda: print(__doc__))()


if __name__ == "__main__":
    main()
