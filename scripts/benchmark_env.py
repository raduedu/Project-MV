"""Measure simulator throughput (driver decisions per second).

Usage::

    python scripts/benchmark_env.py --vehicle ks_porsche_911_gt3_rs --envs 4096 --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mv.envs.racing_env import RacingEnv  # noqa: E402
from mv.track.generator import random_tracks  # noqa: E402
from mv.track.track import TrackBatch  # noqa: E402
from mv.vehicle.params import find_vehicle, load_vehicle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle", default="ks_porsche_911_gt3_rs")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--tracks", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    vp = load_vehicle(find_vehicle(args.vehicle))
    tracks = TrackBatch(random_tracks(args.tracks, seed=0, cfg="mixed"), device=args.device)
    env = RacingEnv(vp, tracks, args.envs, device=args.device)
    print(f"{args.envs} cars on {args.tracks} tracks, device {args.device}, observation size {env.num_obs}")
    obs = env.observe()
    done_count, t0 = 0, None
    for i in range(args.steps + 20):
        if i == 20:                                    # skip warm-up
            if env.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        a = torch.rand(args.envs, 2, device=env.device) * 2 - 1
        a[:, 1] = a[:, 1] * 0.5 + 0.3
        obs, rew, term, trunc, info = env.step(a)
        done_count += int((term | trunc).sum())
    if env.device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sps = args.steps * args.envs / dt
    print(f"{sps:,.0f} decisions/s  = {sps * env.cfg.control_dt / 3600:,.1f} hours of driving per second "
          f"({sps * env.cfg.control_dt:,.0f}x real time)")
    print(f"observation finite: {bool(torch.isfinite(obs).all())}, episodes ended: {done_count}, "
          f"mean reward {float(rew.mean()):.3f}")
    if env.device.type == "cuda":
        print(f"GPU memory peak {torch.cuda.max_memory_allocated() / 2**20:,.0f} MB")


if __name__ == "__main__":
    main()
