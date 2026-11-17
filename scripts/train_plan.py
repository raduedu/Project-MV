"""Run a plan of training jobs unattended, with PAUSE and RESUME.

The plan (``runs/plan.json``) is a list of *lanes*; each lane is a list of
training jobs run one after another.  Lanes run side by side (two lanes fit on
the laptop GPU).  Each job is an ordinary ``scripts/train.py`` command::

    {"lanes": [
        [{"name": "gt3rs_v4_memory", "args": ["--vehicle", "...", "--hours", "2.5"]},
         {"name": "gt3rs_limit_v1",  "args": [...]}],
        [{"name": "fs_v1", "args": [...]}]
    ]}

Commands (from the project root; or double-click the .cmd files in the project folder)::

    python scripts/train_plan.py status     # what is running, done, paused, waiting
    python scripts/train_plan.py pause      # save every running driver and stop; nothing new starts
    python scripts/train_plan.py resume     # continue exactly where it stopped, then the rest of the plan

How it knows where it is (no hidden state - it is all in ``runs/``):
  * a job is DONE when ``runs/<name>/FINISHED`` exists (or its log ends with "done:");
  * a job is RESUMABLE when ``runs/<name>/latest.pt`` exists - it continues with
    ``--resume`` and the same time budget (``--hours`` counts the whole run);
  * otherwise it has not started.

Pausing: ``pause`` creates ``runs/PAUSE``.  Every trainer checks for it after
each update (a few seconds), saves ``latest.pt`` and exits.  A trainer that does
not react within the waiting time (for example one started before this
mechanism existed) is stopped; its ``latest.pt`` is then at most 25 updates
(~1-2 minutes of training) old.  ``runs/PAUSE`` stays until ``resume``, so
nothing can start in between.
"""

from __future__ import annotations

import json
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


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def say(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def python_processes() -> list[dict]:
    """All python.exe processes with their command lines (Windows)."""
    cmd = ["powershell", "-NoProfile", "-Command",
           "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
           "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    if not out:
        return []
    data = json.loads(out)
    return data if isinstance(data, list) else [data]


def training_processes(name: str | None = None) -> list[dict]:
    """Running train.py processes of this project (optionally only the job ``name``)."""
    out = []
    for p in python_processes():
        cl = p.get("CommandLine") or ""
        if "train.py" not in cl or "--name" not in cl:
            continue
        job = cl.split("--name", 1)[1].split()[0].strip('"')
        if name is None or job == name:
            out.append({**p, "job": job})
    return out


def pid_alive(pid: int) -> bool:
    return any(p["ProcessId"] == pid for p in python_processes())


def kill_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)


def load_plan() -> list[list[dict]]:
    if not PLAN.is_file():
        raise SystemExit(f"no plan: {PLAN} does not exist")
    return json.loads(PLAN.read_text())["lanes"]


def job_state(job: dict) -> str:
    run = RUNS / job["name"]
    if (run / "FINISHED").is_file() or (run / "FAILED").is_file():
        return "done" if (run / "FINISHED").is_file() else "failed"
    log = RUNS / f"{job['name']}.log"
    if log.is_file():
        tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:]
        if tail and tail[0].startswith("done:"):
            return "done"
    if (run / "latest.pt").is_file():
        return "resumable"
    return "not started"


def job_progress(job: dict) -> str:
    """Last progress numbers of a job, from its log.csv and screen log."""
    parts = []
    csv = RUNS / job["name"] / "log.csv"
    if csv.is_file():
        rows = csv.read_text().strip().splitlines()
        if len(rows) > 1:
            last = rows[-1].split(",")
            hours_goal = next((job["args"][i + 1] for i, a in enumerate(job["args"]) if a == "--hours"), "?")
            parts.append(f"update {last[0]}, {float(last[1]) / 1e6:.0f} M decisions, {float(last[2]):.2f} of {hours_goal} h")
    log = RUNS / f"{job['name']}.log"
    if log.is_file():
        txt = log.read_text(encoding="utf-8", errors="replace")
        if "  validation:" in txt:
            block = txt.rsplit("  validation:", 1)[1].split("\nupdate")[0]
            parts.append("last validation:" + block.splitlines()[0])
    return "; ".join(parts)


def supervisor_alive() -> int | None:
    if SUPERVISOR_PID.is_file():
        try:
            pid = int(SUPERVISOR_PID.read_text().strip())
        except ValueError:
            return None
        for p in python_processes():
            if p["ProcessId"] == pid and "train_plan.py" in (p.get("CommandLine") or ""):
                return pid
    return None


# --------------------------------------------------------------------------
# the supervisor (runs detached; started by "resume")
# --------------------------------------------------------------------------
def run_lane(k: int, jobs: list[dict]) -> None:
    for job in jobs:
        name = job["name"]
        while True:
            if PAUSE_FLAG.exists():
                say(f"lane {k + 1}: paused before {name}")
                return
            state = job_state(job)
            if state in ("done", "failed"):
                break
            # a run of this job may already be going (started by hand or by an older queue)
            running = training_processes(name)
            if running:
                say(f"lane {k + 1}: {name} is already running (pid {running[0]['ProcessId']}); waiting for it")
                while training_processes(name):
                    time.sleep(30)
                continue                                       # re-check: finished, paused or crashed
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
                if PAUSE_FLAG.exists():                        # stopped by "pause"
                    say(f"lane {k + 1}: {name} stopped for pause (exit {rc})")
                    return
                (RUNS / name).mkdir(parents=True, exist_ok=True)
                (RUNS / name / "FAILED").write_text(f"exit code {rc} at {time.ctime()}; see runs/{name}.err\n")
                say(f"lane {k + 1}: {name} FAILED (exit {rc}); continuing with the next job")
            break
    say(f"lane {k + 1}: plan finished")


def supervise() -> None:
    SUPERVISOR_PID.write_text(str(__import__("os").getpid()))
    lanes = load_plan()
    say(f"supervisor started: {len(lanes)} lane(s)")
    threads = [threading.Thread(target=run_lane, args=(k, jobs), daemon=False) for k, jobs in enumerate(lanes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    say("supervisor finished")
    SUPERVISOR_PID.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# user commands
# --------------------------------------------------------------------------
def status() -> None:
    lanes = load_plan()
    sup = supervisor_alive()
    print(f"PAUSED (runs/PAUSE present - run 'resume' to continue)" if PAUSE_FLAG.exists()
          else ("running" if sup else "not running - run 'resume' to start the plan"))
    running = {p["job"]: p["ProcessId"] for p in training_processes()}
    for k, jobs in enumerate(lanes):
        print(f"\nlane {k + 1}:")
        for job in jobs:
            st = "RUNNING" if job["name"] in running else job_state(job)
            prog = job_progress(job)
            print(f"  {job['name']:<22} {st:<12} {prog}")
    others = [j for j in running if not any(j == x["name"] for jobs in lanes for x in jobs)]
    if others:
        print(f"\nother training processes (not in the plan): {others}")


def pause(wait_s: float = 240.0) -> None:
    PAUSE_FLAG.write_text(f"paused at {time.ctime()}\n")
    print("pause requested: each training saves its driver after its current update ...")
    t0 = time.time()
    while training_processes() and time.time() - t0 < wait_s:
        time.sleep(5)
    left = training_processes()
    for p in left:
        # did not react (e.g. started before pausing existed): stop it; latest.pt is <= 25 updates old
        print(f"  stopping {p['job']} (pid {p['ProcessId']}) - its last save is at most ~1-2 minutes old")
        kill_tree(int(p["ParentProcessId"]) if any(q["ProcessId"] == p["ParentProcessId"] for q in python_processes())
                  else int(p["ProcessId"]))
        kill_tree(int(p["ProcessId"]))
    # older queue processes (scripts/queue_runs.py) would otherwise start their next job later
    for p in python_processes():
        if "queue_runs.py" in (p.get("CommandLine") or ""):
            kill_tree(int(p["ProcessId"]))
    sup = supervisor_alive()
    if sup:
        time.sleep(3)
        if supervisor_alive():
            kill_tree(sup)
    SUPERVISOR_PID.unlink(missing_ok=True)
    time.sleep(2)
    if training_processes():
        print("WARNING: some training is still running:", [p["job"] for p in training_processes()])
    else:
        print("\nAll training stopped and saved. It is safe to close the laptop.")
    say("paused by user")
    status()


def resume() -> None:
    if supervisor_alive():
        print("the plan is already running")
        status()
        return
    PAUSE_FLAG.unlink(missing_ok=True)
    flags = 0x00000008 | 0x00000200 | 0x08000000          # DETACHED_PROCESS | NEW_PROCESS_GROUP | NO_WINDOW
    log = open(LOG, "a")
    subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "run"], cwd=ROOT, stdout=log, stderr=log,
                     creationflags=flags, close_fds=True)
    say("resumed by user")
    time.sleep(8)
    status()


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"status": status, "pause": pause, "resume": resume, "start": resume, "run": supervise}.get(
        cmd, lambda: print(__doc__))()


if __name__ == "__main__":
    main()
