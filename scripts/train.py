"""Train the Project MV driver.

Typical use (from the project root)::

    python scripts/train.py --vehicle ks_porsche_911_gt3_rs --name gt3rs_v1 --hours 2
    python scripts/train.py --vehicle ks_porsche_911_gt3_rs --name gt3rs_v1 --resume      # continue later

What happens
------------
* ``--envs`` cars drive at the same time, each on one of ``--tracks`` random
  circuits.  A share of the circuits is replaced by brand-new ones every few
  updates, so the driver never gets to memorise a layout.
* Every ``--eval-every`` updates the driver is tested, without exploration
  noise, on validation tracks it never trains on: fixed random circuits plus any
  imported real circuits in ``tracks/local`` (see scripts/import_ac_track.py).
* Progress is printed, written to ``runs/<name>/log.csv`` and to TensorBoard
  (``tensorboard --logdir runs``).  ``runs/<name>/latest.pt`` is the newest
  driver, ``best.pt`` the one with the best validation result so far.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# Pausing: scripts/train_plan.py pause creates this file; the trainer saves and exits with
# EXIT_PAUSED after the current update, so the run can later continue with --resume.
PAUSE_FLAG = ROOT / "runs" / "PAUSE"
EXIT_PAUSED = 3
sys.path.insert(0, str(ROOT))

from mv.envs.multi_car import MultiCarEnv  # noqa: E402
from mv.envs.racing_env import EnvConfig, RacingEnv  # noqa: E402
from mv.rl.evaluate import evaluate  # noqa: E402
from mv.rl.ppo import PPO, PPOConfig  # noqa: E402
from mv.rl.pretrain import warm_start  # noqa: E402
from mv.control.scripted import ScriptedDriver  # noqa: E402
from mv.track.generator import max_length_m, random_track, random_tracks  # noqa: E402
from mv.track.track import Track, TrackBatch  # noqa: E402
from mv.vehicle.params import find_vehicle, load_vehicle  # noqa: E402


def validation_tracks(preset: str, count: int) -> list[Track]:
    """Fixed random circuits (seed never used in training) + imported real circuits."""
    tracks = random_tracks(count, seed=987_654, cfg=preset)
    for i, t in enumerate(tracks):
        t.name = f"val_random_{i}"
    local = ROOT / "tracks" / "local"
    if not preset.startswith("fs") and local.is_dir():          # real circuits only for full-size cars
        tracks += [Track.load(p) for p in sorted(local.glob("*.npz"))]
    return tracks


def save_record_laps(res, tracks, vp, run_name: str, it: int, best: dict) -> None:
    """Save the full telemetry of a lap whenever this run beats its own record on a real circuit.

    Only imported circuits (tracks/local) - the random validation circuits change
    meaning between presets.  Lap files go to runs/laps (see mv/laps.py) and can
    be replayed in the web interface.
    """
    from mv.laps import best_lap_time, laps_from_result, save_lap
    for r, t in zip(res, tracks):
        if not r.finished or t.name.startswith("val_random"):
            continue
        key = (vp.name, t.name)
        if key not in best:
            best[key] = best_lap_time(vp.name, t.name, run_name)
        car_files = {load_vehicle(p).name: p for p in (ROOT / "vehicles").glob("*/*.yaml")}
        car_file = str(car_files[vp.name].relative_to(ROOT)).replace(chr(92), "/") if vp.name in car_files else ""
        laps = laps_from_result(r, t, vp.name, run_name, "learned",
                                f"training validation of {run_name}, update {it}", f"runs/{run_name}/eval_{it:05d}.pt",
                                car_file=car_file)
        if laps:
            lap = min(laps, key=lambda l: l["meta"]["lap_time"])
            if lap["meta"]["lap_time"] < best[key] - 1e-3:
                save_lap(lap)
                best[key] = lap["meta"]["lap_time"]
                print(f"    new record lap for {run_name} at {t.name}: {lap['meta']['lap_time']:.3f} s (saved for replay)",
                      flush=True)


def load_grown(net, state: dict) -> None:
    """Load a driver's weights, allowing the new observation to have EXTRA inputs at the end.

    Example: adding a memory block (``--history``) appends inputs.  The first layer
    of both networks gets zero weights for the new inputs, so the new driver
    starts out driving exactly like the old one and learns to use the new
    inputs from there - instead of throwing away billions of decisions of
    experience.  Anything else that does not match is an error.
    """
    own = net.state_dict()
    grown = []
    for key, old in state.items():
        new = own[key]
        if old.shape == new.shape:
            own[key] = old
        elif key.endswith("0.weight") and old.shape[0] == new.shape[0] and old.shape[1] < new.shape[1]:
            w = torch.zeros_like(new)
            w[:, : old.shape[1]] = old
            own[key] = w
            grown.append(key)
        elif key.startswith("obs_norm.") and old.dim() == 1 and old.shape[0] < new.shape[0]:
            v = new.clone()
            v[: old.shape[0]] = old
            own[key] = v
            grown.append(key)
        else:
            raise ValueError(f"cannot load {key}: checkpoint {tuple(old.shape)} vs network {tuple(new.shape)}")
    net.load_state_dict(own)
    if grown:
        print(f"observation grew: new inputs start with zero weight ({', '.join(grown)})", flush=True)


def keep_awake() -> None:
    """Ask Windows not to go to sleep while this process is running.

    The same request a video player makes; it ends automatically when the
    process exits and changes no system setting.  Closing the lid or choosing
    Sleep still suspends the machine.
    """
    if sys.platform == "win32":
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vehicle", default="ks_porsche_911_gt3_rs",
                    help="car file name; several names separated by commas train ONE driver for all of them")
    ap.add_argument("--name", default=None, help="run name (folder under runs/)")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--tracks", type=int, default=256, help="size of the pool of training circuits")
    ap.add_argument("--preset", default="mixed",
                    help="track generator preset(s): circuit, mixed, fs, gp, fs_segments; combine with + (e.g. mixed+gp)")
    ap.add_argument("--val-preset", default=None,
                    help="preset for the random validation tracks (default: the first one in --preset)")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="start from the weights of an existing driver (.pt) instead of from scratch")
    ap.add_argument("--reset-noise", type=float, default=None,
                    help="set the exploration noise to this value at the start (e.g. 0.1 after --init-from)")
    ap.add_argument("--history", type=int, default=0,
                    help="memory: number of past car-state samples in the observation (0 = off; e.g. 8 = last ~1 s)")
    ap.add_argument("--limit-seeking", action="store_true",
                    help="going off costs time (car is put back on) instead of ending the episode, the driver "
                         "remembers last lap per track section, and episodes span several laps (log WP21)")
    ap.add_argument("--episode-s", type=float, default=None, help="episode length in seconds (default 60; 240 with --limit-seeking)")
    ap.add_argument("--grip-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                    help="tyre grip multiplier range for training cars (default 0.88 1.05)")
    ap.add_argument("--vary-cars", action="store_true",
                    help="vary each car (mass, power, downforce, drag, tyre grip) per episode and tell the driver, so it "
                         "learns to read the car description and handles new car files (engineering log WP29)")
    ap.add_argument("--holdout", default="",
                    help="comma-separated car files NOT trained on; validation also drives them (one per check, in turn)")
    ap.add_argument("--steer-filter", type=float, default=None,
                    help="steering smoothing: time constant (s) of a low-pass filter on the steering command, e.g. 0.05")
    ap.add_argument("--steer-penalty", type=float, default=None,
                    help="penalty on steering jerks (change of the steering command per step, squared), e.g. 0.5")
    ap.add_argument("--steer-zigzag", type=float, default=None,
                    help="penalty on steering zig-zag only (direction flips from one decision to the next), e.g. 5; "
                         "fast steering in one direction stays free (engineering log WP35)")
    ap.add_argument("--downshift-frac", type=float, default=None,
                    help="gearbox: change down below this share of the lower gear's upshift speed; 0.98 = always "
                         "(nearly) in the gear that pulls hardest, default 0.93 (engineering log WP36)")
    ap.add_argument("--min-noise", type=float, default=None,
                    help="floor for the exploration noise (default 0.03)")
    ap.add_argument("--refresh-every", type=int, default=10, help="replace circuits every N updates")
    ap.add_argument("--refresh-frac", type=float, default=0.1, help="share of circuits replaced each time")
    ap.add_argument("--hours", type=float, default=None, help="stop after this much wall-clock time")
    ap.add_argument("--steps", type=float, default=None, help="stop after this many driver decisions (e.g. 2e8)")
    ap.add_argument("--iters", type=int, default=None, help="stop after this many updates")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--val-tracks", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-sleep", action="store_true", help="do not keep the computer awake during training")
    ap.add_argument("--warm-start", action="store_true",
                    help="first imitate the rule-based driver, then train (see mv/rl/pretrain.py)")
    ap.add_argument("--warm-steps", type=int, default=600)
    ap.add_argument("--teacher", type=Path, default=None,
                    help="warm start by copying an existing trained driver (.pt) - e.g. to give a BIGGER network "
                         "today's skill before PPO continues (engineering log WP30)")
    ap.add_argument("--kl-stop", type=float, default=0.0,
                    help="end a PPO update early once it changed the driver by more than this many times "
                         "the target (e.g. 4); protects a good driver from one oversized step")
    ap.add_argument("--hidden", default=None,
                    help="network size: neurons per hidden layer, e.g. 1024,1024,512 (default 512,256,128)")
    ap.add_argument("--minibatches", type=int, default=None, help="PPO minibatches per epoch (default 4)")
    ap.add_argument("--rollout", type=int, default=None, help="decisions per car between updates (default 32)")
    args = ap.parse_args()
    if args.hours is None and args.steps is None and args.iters is None:
        args.iters = 300

    if not args.allow_sleep:
        keep_awake()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    vps = [load_vehicle(find_vehicle(v.strip())) for v in args.vehicle.split(",") if v.strip()]
    vp = vps[0]
    name = args.name or f"{vp.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run = ROOT / "runs" / name
    run.mkdir(parents=True, exist_ok=True)

    env_cfg, ppo_cfg = EnvConfig(), PPOConfig()
    if args.minibatches:
        ppo_cfg.minibatches = args.minibatches
    if args.hidden:
        ppo_cfg.hidden = tuple(int(x) for x in args.hidden.split(","))
    if args.kl_stop:
        ppo_cfg.kl_stop = args.kl_stop
    if args.rollout:
        ppo_cfg.rollout_steps = args.rollout
    if args.min_noise:
        ppo_cfg.min_std = args.min_noise
    if args.history:
        env_cfg.history_steps = args.history
    if args.grip_range:
        env_cfg.rand_grip = tuple(args.grip_range)
    if args.vary_cars:
        env_cfg.vary_car = True
    holdout_vps = [load_vehicle(find_vehicle(v.strip())) for v in args.holdout.split(",") if v.strip()]
    if args.steer_filter is not None:
        env_cfg.steer_filter_s = args.steer_filter
    if args.steer_penalty is not None:
        env_cfg.steer_penalty = args.steer_penalty
    if args.steer_zigzag is not None:
        env_cfg.steer_zigzag_penalty = args.steer_zigzag
    if args.downshift_frac is not None:
        env_cfg.downshift_frac = args.downshift_frac
    if args.limit_seeking:
        env_cfg.recover_on_crash = True
        env_cfg.lap_memory = True
        env_cfg.max_episode_s = 240.0
    if args.episode_s:
        env_cfg.max_episode_s = args.episode_s
    print(f"generating {args.tracks} training circuits ({args.preset}) ...", flush=True)
    pool = [random_track(rng, args.preset, name=f"train_{i}") for i in range(args.tracks)]
    ds = pool[0].ds
    # reserve room for the longest circuit the generator can produce, so replacements always fit
    from mv.track.generator import PRESETS
    batch = TrackBatch(pool, device=args.device, ds=ds, min_points=int(max_length_m(args.preset) / ds) + 8)
    if len(vps) == 1:
        env = RacingEnv(vp, batch, args.envs, env_cfg, device=args.device, seed=args.seed)
    else:
        env = MultiCarEnv(vps, batch, args.envs, env_cfg, device=args.device, seed=args.seed)
        args.envs = env.n
    ppo = PPO(env, ppo_cfg)
    val = validation_tracks(args.val_preset or args.preset.split("+")[0], args.val_tracks)
    best_score, it0 = -1e9, 0
    if args.resume and (run / "latest.pt").is_file():
        ck = ppo.load(run / "latest.pt")
        best_score, it0 = ck.get("best_score", -1e9), ck.get("iteration", 0)
        print(f"resumed {run / 'latest.pt'} at update {it0}, {ppo.total_steps:,} decisions")
        # the time budget (--hours) counts the whole run, not only this session
        try:
            last = (run / "log.csv").read_text().strip().splitlines()[-1].split(",")
            hours_before = float(last[2])
        except Exception:
            hours_before = 0.0
        print(f"already trained {hours_before:.2f} h of this run", flush=True)
    else:
        hours_before = 0.0
    if args.init_from is not None and it0 == 0:
        ck = torch.load(args.init_from, map_location=args.device, weights_only=False)
        load_grown(ppo.net, ck["model"])
        ppo.ret_norm.load_state_dict(ck["ret_norm"])
        # continue at the learning rate the driver ended with: a converged driver has very
        # little exploration noise, so a fresh (larger) rate would change it far too abruptly
        ppo.lr = float(ck.get("lr", ppo.lr))
        for g in ppo.opt.param_groups:
            g["lr"] = ppo.lr
        print(f"initialised from {args.init_from} ({ck.get('total_steps', 0):,} decisions of experience)", flush=True)
        args.warm_start = False
    if args.reset_noise is not None and it0 == 0:          # not again when resuming
        import math as _math
        with torch.no_grad():
            ppo.net.log_std.fill_(_math.log(args.reset_noise))
        print(f"exploration noise reset to {args.reset_noise}", flush=True)
    if args.teacher is not None and it0 == 0:
        from mv.rl.ppo import load_policy as _load_policy
        tnet = _load_policy(args.teacher, args.device)

        class _Teacher:                                   # the teacher's best guess, no exploration noise
            obs_norm = tnet.obs_norm                      # same input scaling -> student copies it
            def act(self, obs, deterministic=True):
                return tnet.act(obs, deterministic=True)

        n_params = sum(p.numel() for p in ppo.net.actor.parameters())
        t_std = float(tnet.log_std.detach().exp().mean())
        print(f"warm start: a {n_params:,}-parameter driver copies {args.teacher} "
              f"(teacher noise {t_std:.3f}) ...", flush=True)
        info = warm_start(ppo, _Teacher(), steps=args.warm_steps, noise=0.1, critic_updates=40,
                          start_std=args.reset_noise or t_std, buffer_steps=48, grad_steps=4, lr=5e-4)
        print(f"warm start done: imitation error {info['imitation_error']:.5f}", flush=True)
        # A fresh optimiser's first steps move EVERY weight by the full learning rate; in a wide
        # network that rewrites the copied driver in one update (smoke test: KL 450, 0 % of
        # tracks after update 1).  Start tiny; the KL rule raises it x1.5 per step as allowed.
        ppo.lr = 1e-5
        for g in ppo.opt.param_groups:
            g["lr"] = ppo.lr
    elif args.warm_start and it0 == 0:
        print("warm start: imitating the rule-based driver ...", flush=True)
        info = warm_start(ppo, env.scripted_teacher(), steps=args.warm_steps)
        print(f"warm start done: imitation error {info['imitation_error']:.4f}", flush=True)
    (run / "config.json").write_text(json.dumps({
        "vehicle": args.vehicle, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "env": asdict(env_cfg), "ppo": asdict(ppo_cfg),
        "num_obs": env.num_obs, "num_actions": env.num_actions}, indent=2))
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(run))
    except Exception:                                   # TensorBoard is optional
        tb = None
    log_path = run / "log.csv"
    new_log = not log_path.is_file()
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["iteration", "decisions", "hours", "ep_return", "ep_len_s", "ep_dist_m", "crash_rate",
                      "mean_speed_kmh", "kl", "std", "lr", "value_loss", "val_finished", "val_score",
                      "crashes_per_100km", "pace_kmh", "reward_per_step"])

    print(f"{' + '.join(v.name for v in vps)}: {args.envs} cars, {batch.num_tracks} circuits, obs {env.num_obs}, "
          f"device {args.device}")
    print(f"validation tracks: {[t.name for t in val]}")
    t_start = time.time()
    it = it0
    val_finished, val_score = float("nan"), float("nan")
    record_laps: dict[tuple[str, str], float] = {}     # (car, track) -> this run's fastest saved lap
    n_evals = 0
    paused = False
    hours = hours_before
    try:
        while True:
            it += 1
            stats = ppo.collect()
            upd = ppo.update()

            # ---- keep the circuits fresh -------------------------------------------------
            if args.refresh_every > 0 and it % args.refresh_every == 0:
                k = max(int(batch.num_tracks * args.refresh_frac), 1)
                ids = rng.choice(batch.num_tracks, size=k, replace=False)
                for tid in ids:
                    batch.replace(int(tid), random_track(rng, args.preset, name=f"train_{it}_{tid}"))
                env.tracks_replaced(torch.as_tensor(ids, device=env.device))
                ppo.obs = env.observe()

            # ---- validation -----------------------------------------------------------------
            if it % args.eval_every == 0 or it == it0 + 1:
                val_vp = vps[n_evals % len(vps)]          # with several cars, validate them in turn
                n_evals += 1
                if len(vps) > 1:
                    print(f"  validating with {val_vp.name}", flush=True)
                res = evaluate(ppo.net, val_vp, val, "cpu", laps=2, cfg=env_cfg, record=True)
                save_record_laps(res, val, val_vp, name, it, record_laps)
                val_finished = sum(r.finished for r in res) / len(res)
                # score: average speed over the distance covered, zero credit beyond 2 laps
                val_score = float(np.mean([min(r.distance_m / (2 * r.length_m), 1.0) for r in res]))
                lines = []
                for r in res:
                    lap = f"{r.flying_lap:7.2f} s" if r.finished else f"-- ({r.end_reason} at {r.distance_m:.0f} m)"
                    lines.append(f"    {r.track:<22} flying lap {lap}")
                    if tb and r.finished:
                        tb.add_scalar(f"lap/{r.track}", r.flying_lap, ppo.total_steps)
                print(f"  validation: {val_finished * 100:.0f} % of tracks completed\n" + "\n".join(lines), flush=True)
                # "best" = most validation tracks completed; among equals, the highest average
                # speed over the completed tracks (km/h / 1000 can never outweigh one track)
                speeds = [r.length_m / r.flying_lap * 3.6 for r in res if r.finished]
                score = sum(r.finished for r in res) + (float(np.mean(speeds)) / 1000.0 if speeds else 0.0)
                ppo.save(run / f"eval_{it:05d}.pt", {"iteration": it, "score": score, "vehicle": args.vehicle})
                if holdout_vps:                  # a car file the driver never trained on
                    hv = holdout_vps[(n_evals - 1) % len(holdout_vps)]
                    hres = evaluate(ppo.net, hv, val, "cpu", laps=2, cfg=env_cfg)
                    hdone = sum(r.finished for r in hres)
                    print(f"  held-out car {hv.name}: {hdone}/{len(hres)} completed  " +
                          " ".join(f"{r.track}:{r.flying_lap:.2f}" for r in hres if r.finished and not r.track.startswith("val_")),
                          flush=True)
                    if tb:
                        tb.add_scalar(f"holdout/{hv.name}", hdone / len(hres), ppo.total_steps)
                if score > best_score:
                    best_score = score
                    ppo.save(run / "best.pt", {"iteration": it, "best_score": best_score, "vehicle": args.vehicle})

            # ---- logging ------------------------------------------------------------------------
            hours = hours_before + (time.time() - t_start) / 3600.0
            cat = lambda k: torch.cat(stats[k]).mean().item() if stats[k] else float("nan")  # noqa: E731
            row = [it, ppo.total_steps, round(hours, 4), cat("ep_return"), cat("ep_len") * env_cfg.control_dt, cat("ep_dist"),
                   stats["crashes"] / max(stats["episodes"], 1), stats["speed"] * 3.6, upd["kl"], upd["std"], upd["lr"],
                   upd["value_loss"], val_finished, val_score]
            # Unbiased whole-fleet numbers.  (The per-episode figures above only count
            # episodes that ENDED in this rollout, which over-represents short, crashed ones.)
            km = max(stats["progress_m"], 1.0) / 1000.0
            crashes_100km = stats["crashes"] / km * 100.0
            pace = stats["progress_m"] / (args.envs * ppo_cfg.rollout_steps * env_cfg.control_dt) * 3.6
            row += [crashes_100km, pace, stats["reward"]]
            log.writerow([f"{v:.5g}" if isinstance(v, float) else v for v in row]); log_f.flush()
            if tb:
                for k, v in zip(["ep_return", "ep_len_s", "ep_dist_m", "crash_rate", "mean_speed_kmh", "kl", "std", "lr",
                                 "value_loss", "val_finished", "val_score", "crashes_per_100km", "pace_kmh",
                                 "reward_per_step"], row[3:]):
                    tb.add_scalar(f"train/{k}", v, ppo.total_steps)
            if it % 10 == 0 or it == it0 + 1:
                sps = (it - it0) * ppo_cfg.rollout_steps * args.envs / max(time.time() - t_start, 1e-6)
                print(f"update {it:5d} | {ppo.total_steps / 1e6:7.1f} M decisions | {sps:,.0f}/s | "
                      f"pace {pace:5.1f} km/h | {crashes_100km:6.1f} crashes/100 km | reward/step {stats['reward']:+.3f} | "
                      f"ended episodes: {row[5]:6.0f} m | noise {upd['std']:.2f} | kl {upd['kl']:.4f}", flush=True)
            if it % 25 == 0:
                ppo.save(run / "latest.pt", {"iteration": it, "best_score": best_score, "vehicle": args.vehicle})

            # budgets count the whole run (a resumed run continues towards the same totals)
            if args.iters is not None and it >= args.iters:
                break
            if args.steps is not None and ppo.total_steps >= args.steps:
                break
            if args.hours is not None and hours >= args.hours:
                break
            if PAUSE_FLAG.exists():                      # scripts/train_plan.py pause
                paused = True
                break
    except KeyboardInterrupt:
        print("\nstopped by user - saving the current driver", flush=True)
        paused = True
    ppo.save(run / "latest.pt", {"iteration": it, "best_score": best_score, "vehicle": args.vehicle})
    log_f.close()
    if paused:
        print(f"paused at update {it}: driver saved to {run / 'latest.pt'}; continue with --resume", flush=True)
        sys.exit(EXIT_PAUSED)
    (run / "FINISHED").write_text(f"update {it}, {ppo.total_steps} decisions, {hours:.2f} h\n")
    print(f"done: {run}")


if __name__ == "__main__":
    main()
