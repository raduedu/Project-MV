"""Run training jobs one after another, unattended.

Lets the laptop keep training while nobody is at it: each job starts when the
previous one (or a given running process) has finished.  Every job is an
ordinary ``scripts/train.py`` command; its screen output goes to
``runs/<name>.log``, exactly as when started by hand.

Queue file (JSON)::

    {
      "wait_for_pid": 12345,                  # optional: start only after this process has exited
      "jobs": [
        {"name": "multi_v1", "args": ["--vehicle", "a,b", "--hours", "3"]},
        {"name": "fs_v1",    "args": ["--vehicle", "fs_generic_ev", "--preset", "fs+fs_segments"]}
      ]
    }

Usage::

    python scripts/queue_runs.py runs/queue_lane1.json

Stop the queue by ending this process; the job that is running keeps going
until it finishes (or end its own process too).  Progress of the queue itself
is written to ``runs/<queue file name>.queue.log``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
        return str(pid) in out.stdout
    try:
        import os
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    qfile = Path(sys.argv[1])
    queue = json.loads(qfile.read_text())
    log = open(ROOT / "runs" / f"{qfile.stem}.queue.log", "a", buffering=1)

    def say(msg: str) -> None:
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")

    pid = queue.get("wait_for_pid")
    if pid:
        say(f"waiting for process {pid} to finish")
        while pid_alive(int(pid)):
            time.sleep(60)
    python = sys.executable
    for job in queue["jobs"]:
        name = job["name"]
        cmd = [python, "-u", str(ROOT / "scripts" / "train.py"), "--name", name, *job["args"]]
        say(f"start {name}: {' '.join(cmd[2:])}")
        with open(ROOT / "runs" / f"{name}.log", "a") as out, open(ROOT / "runs" / f"{name}.err", "a") as err:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=err).returncode
        say(f"end {name}: exit code {rc}")
    say("queue finished")


if __name__ == "__main__":
    main()
