"""A race in our simulator: the driver against copies of itself in slightly slower cars (race bot, WP64).

The driver under test starts LAST on a grid of N cars (standing start, staggered left/right, ``--grid-gap``
metres apart).  The rivals are the same driver in the same car with ``--rival-power`` x the engine power,
so the test car is faster and has to get past them - cleanly.  Everyone sees only ``--view`` metres
(sensors), the run-off is grass/gravel up to a barrier (self-recovery), and the own map / lap memory work
as the driver was trained.  A car that hits the barrier, spins for long or stalls is out (DNF) and stays
where it stopped.

Reports per car: finishing place and time, contacts, off-road excursions; for the test car also how many
places it gained and lost on the way.

Example::

    python scripts/race_eval.py --policy runs/race_v3_racing/best.pt --tracks spa,monza --laps 3

A driver trained without the opponents input (``--race`` in train.py) cannot see the other cars: the
race then shows what happens without that sense (the baseline for stage 4).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mv.envs.racing_env import EnvConfig, RacingEnv  # noqa: E402
from mv.rl.evaluate import _act_per_car, eval_config  # noqa: E402
from mv.rl.ppo import load_policy  # noqa: E402
from mv.track.track import Track, TrackBatch  # noqa: E402
from mv.vehicle.params import find_vehicle, load_vehicle  # noqa: E402


def run_config(policy: Path) -> EnvConfig:
    raw = json.loads((policy.parent / "config.json").read_text())["env"]
    return EnvConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()
                        if k in EnvConfig.__dataclass_fields__})


@torch.no_grad()
def race(net, vp, track: Track, cfg: EnvConfig, cars: int = 4, laps: int = 3, view: float = 120.0,
         rival_power: float = 0.95, grid_gap: float = 12.0) -> dict:
    """Run one race; car ``cars - 1`` (last on the grid) is the one under test."""
    base = eval_config(cfg, max_time_s=1e9, strict=False)
    cfg = replace(base, self_recovery=True, race_group=cars, spawn_near_prob=0.0, max_episode_s=1e9)
    env = RacingEnv(vp, TrackBatch([track], ds=track.ds), cars, cfg, device="cpu")
    env.reset = lambda ids: None                 # an eliminated car stays where it stopped (no restart)
    sim, n_pts, ds = env.sim, int(env.tracks.n[0]), float(env.tracks.ds_each[0])
    L = float(env.tracks.length[0])
    test = cars - 1
    for k in range(cars):                        # grid: car k starts k * gap behind the line
        s0 = -k * grid_gap
        i = int(round(s0 / ds)) % n_pts
        lat = 2.0 if k % 2 == 0 else -2.0
        psi, p = env.tracks.psi[0, i], env.tracks.xy[0, i]
        ids = torch.tensor([k])
        sim.reset(ids, (p[0] - torch.sin(psi) * lat)[None], (p[1] + torch.cos(psi) * lat)[None], psi[None],
                  torch.zeros(1))
        env.idx[k], env.along[k], env.s_total[k], env.t_ep[k] = i, 0.0, 0.0, 0.0
    env._locate()
    env.stall_t[:] = 0.0
    sim.power_scale[:] = rival_power
    sim.power_scale[test] = 1.0
    env.set_view(view)
    start = torch.tensor([-k * grid_gap for k in range(cars)])
    out = [False] * cars
    finish = [math.nan] * cars
    gained = lost = 0
    order_prev = None
    contacts = np.zeros(cars, dtype=int)
    excursions = np.zeros(cars, dtype=int)
    was_out = np.zeros(cars, dtype=bool)
    obs = env.observe()
    t = 0.0
    while t < laps * L / 12.0 + 60.0:
        action = _act_per_car(net, obs)
        for k in range(cars):
            if out[k]:
                action[k] = torch.tensor([0.0, -1.0])        # out of the race: brake and stay put
        # (finished cars keep driving, as on a cool-down lap, so they are no parked obstacle)
        obs, _r, term, _tr, info = env.step(action)
        env.set_view(view)
        t += cfg.control_dt
        live = [not out[k] and math.isnan(finish[k]) for k in range(cars)]
        contacts += ((info["contact"] > 0).numpy() & np.array(live))
        now_out = info["wheels_off"].numpy() >= 3
        excursions += (now_out & ~was_out & np.array(live))
        was_out = now_out
        dist = (env.s_total + start).tolist()
        for k in range(cars):
            if live[k] and bool(term[k]):
                out[k] = True
            if live[k] and dist[k] >= laps * L:
                finish[k] = t
        # running order: finished cars by time, then distance; eliminated cars last
        order = sorted(range(cars), key=lambda k: (out[k], math.isnan(finish[k]), finish[k] if not math.isnan(finish[k])
                                                    else -dist[k]))
        place = order.index(test)
        if order_prev is not None and not out[test]:
            prev = order_prev.index(test)
            gained += max(prev - place, 0)
            lost += max(place - prev, 0)
        order_prev = order
        if all(out[k] or not math.isnan(finish[k]) for k in range(cars)):
            break
    return {"track": track.name, "order": order, "test": test, "finish": finish, "out": out,
            "contacts": contacts.tolist(), "excursions": excursions.tolist(), "gained": gained, "lost": lost}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--vehicle", default="ks_porsche_911_gt3_rs")
    ap.add_argument("--tracks", default="spa,monza,ks_brands_hatch__gp,mugello")
    ap.add_argument("--cars", type=int, default=4)
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--view", type=float, default=120.0, help="sensor range, m")
    ap.add_argument("--rival-power", type=float, default=0.95)
    ap.add_argument("--grid-gap", type=float, default=12.0)
    args = ap.parse_args()
    torch.set_num_threads(1)
    net = load_policy(args.policy, "cpu")
    cfg = run_config(args.policy)
    vp = load_vehicle(find_vehicle(args.vehicle))
    fmt = lambda s: "  --  " if math.isnan(s) else f"{int(s // 60)}:{s % 60:06.3f}"  # noqa: E731
    print(f"{args.policy}: {args.cars} cars, {args.laps} laps, sensors {args.view:.0f} m, rivals at "
          f"{args.rival_power:.0%} power; test car starts last; opponents input: {cfg.opponents or 'none'}")
    for name in args.tracks.split(","):
        r = race(net, vp, Track.load(ROOT / "tracks" / "local" / f"{name}.npz"), cfg, args.cars, args.laps, args.view,
                 args.rival_power, args.grid_gap)
        print(f"  {name}: test car {'P' + str(r['order'].index(r['test']) + 1) if not r['out'][r['test']] else 'DNF'}"
              f"  (gained {r['gained']}, lost {r['lost']})")
        for pos, k in enumerate(r["order"], 1):
            who = "TEST " if k == r["test"] else "rival"
            state = "DNF" if r["out"][k] else fmt(r["finish"][k])
            print(f"     {pos}. {who} car {k}  {state:>9}  contacts {r['contacts'][k]}  excursions {r['excursions'][k]}")


if __name__ == "__main__":
    main()
