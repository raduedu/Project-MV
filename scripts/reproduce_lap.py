"""Drive a recorded lap again from scratch and check that the same lap time comes out.

    python scripts/reproduce_lap.py "local:<lap id>"          # key as shown by the web interface
    python scripts/reproduce_lap.py runs/laps/<car>/<track>/<id>.json

Why: a lap time is only believable if anyone can reproduce it.  The simulator
and the driver are deterministic (same inputs -> same outputs), so loading the
exact driver checkpoint, car file and track that produced a lap and driving it
again must give the same time, to the millisecond.

What is needed: the driver checkpoint named in the lap file (or the rule-based
driver), the car file, and the track (``tracks/local/<track>.npz``, or the
generator recipe stored in the lap file for project-generated tracks).
Physics reference laps are computed, not driven, and are re-checked with
``scripts/reference_lap.py`` instead.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from mv.envs.racing_env import EnvConfig, RacingEnv  # noqa: E402
from mv.laps import load_lap  # noqa: E402
from mv.rl.evaluate import eval_config, evaluate  # noqa: E402
from mv.track.track import Track, TrackBatch  # noqa: E402
from mv.vehicle.params import load_vehicle  # noqa: E402


def find_car_file(meta: dict) -> Path:
    if meta.get("car_file"):
        return ROOT / meta["car_file"]
    for p in sorted((ROOT / "vehicles").glob("*/*.yaml")):         # older laps: match the car's name
        if p.stem == meta["car"]:
            return p
        try:
            if load_vehicle(p).name == meta["car"]:
                return p
        except Exception:
            continue
    raise FileNotFoundError(f"no vehicle file for car '{meta['car']}'")


def find_track(lap: dict) -> Track:
    meta = lap["meta"]
    recipe = meta.get("track_recipe")
    if recipe:
        from mv.track.segments import segment_track
        from mv.track.generator import random_track
        gen = segment_track if recipe["generator"] == "segment_track" else random_track
        return gen(recipe["seed"], recipe["preset"], name=meta["track"])
    p = ROOT / "tracks" / "local" / f"{meta['track']}.npz"
    if p.is_file():
        return Track.load(p)
    raise FileNotFoundError(f"track '{meta['track']}' is not available on this computer")


def main() -> None:
    torch.set_num_threads(1)
    arg = sys.argv[1]
    lap = json.loads(Path(arg).read_text()) if arg.endswith(".json") else load_lap(arg)
    meta = lap["meta"]
    print(f"recorded : {meta['driver']} | {meta['car']} | {meta['track']} | {meta['lap_time']:.3f} s | {meta['created']}")
    print(f"           source: {meta['source']}")
    if meta["kind"] == "reference":
        print("This is a physics reference lap (computed, not driven): re-check it with scripts/reference_lap.py.")
        return
    vp = load_vehicle(find_car_file(meta))
    track = find_track(lap)
    lap_no = int(m.group(1)) if (m := re.search(r"lap (\d+)", meta["source"])) else 2
    cfg = EnvConfig()
    if meta["kind"] == "rule-based" or not meta.get("checkpoint"):
        from mv.control.scripted import ScriptedDriver
        probe = RacingEnv(vp, TrackBatch([track], ds=track.ds), 1, eval_config(cfg), device="cpu")
        driver = ScriptedDriver(probe)
        print("driver   : rule-based driver")
    else:
        from mv.rl.ppo import load_policy
        ck = ROOT / meta["checkpoint"]
        cfg_file = ck.parent / "config.json"
        if cfg_file.is_file():
            raw = json.loads(cfg_file.read_text())["env"]
            cfg = EnvConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
        driver = load_policy(ck, "cpu")
        info = torch.load(ck, map_location="cpu", weights_only=False)
        print(f"driver   : {meta['checkpoint']} (training update {info.get('iteration', '?')})")
    blind = "blind" in meta["source"]
    g = re.search(r"grip x([\d.]+)", meta["source"])
    res = evaluate(driver, vp, [track], "cpu", laps=lap_no, cfg=cfg, blind=blind, grip=float(g.group(1)) if g else 1.0)[0]
    if not res.finished or len(res.lap_times) < lap_no:
        print(f"RESULT   : did NOT reproduce - the run ended ({res.end_reason} at {res.distance_m:.0f} m)")
        sys.exit(1)
    t = res.lap_times[lap_no - 1]
    diff = t - meta["lap_time"]
    print(f"driven   : {t:.3f} s (lap {lap_no})")
    exact = meta.get("eval") == "per-car"
    ok = abs(diff) <= 0.0015
    if not ok and not exact and abs(diff) / meta["lap_time"] < 0.003:
        print(f"RESULT   : REPRODUCED within numerical noise ({diff:+.3f} s, {abs(diff) / meta['lap_time'] * 100:.2f} %).")
        print("           This lap was recorded before evaluation became batch-independent: it was driven in a batch with")
        print("           other tracks, and tiny rounding differences grow over a lap (engineering log WP26).")
        return
    print(f"RESULT   : {'REPRODUCED - same lap time' if ok else 'DIFFERENT'} (difference {diff:+.3f} s)")
    if not ok:
        print("           (a different time usually means the checkpoint file was overwritten by later training,"
              " or the code changed since the lap was recorded)")
        sys.exit(1)


if __name__ == "__main__":
    main()
