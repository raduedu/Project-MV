# Project MV — Training Guide

*MV-GUIDE-001. How to train a driver on the development laptop (RTX 3080
Laptop, 16 GB), what the numbers on screen mean, and what to do when something
looks wrong. All commands are run from the project root.*

## 1. Before the first run (once)

```powershell
.venv\Scripts\python.exe scripts\import_ac_car.py ks_porsche_911_gt3_rs
.venv\Scripts\python.exe scripts\import_ac_track.py imola monza spa mugello
.venv\Scripts\python.exe scripts\vehicle_check.py ks_porsche_911_gt3_rs
.venv\Scripts\python.exe -m pytest tests -q
```

The imported circuits in `tracks/local/` are used **only for validation**; the
driver never trains on them.

## 2. Start a run

```powershell
.venv\Scripts\python.exe -u scripts\train.py --vehicle ks_porsche_911_gt3_rs --name gt3rs_v1 --envs 16384 --tracks 512 --hours 3 --eval-every 100
```

| Option | Meaning | Guidance |
|---|---|---|
| `--vehicle` | Car file name (in `vehicles/local` or `vehicles/library`) | |
| `--name` | Folder under `runs/` | Use a new name per experiment |
| `--envs` | Cars driving in parallel | 8 192–16 384 on this GPU. More cars = more decisions per second at almost no cost |
| `--tracks` | Size of the pool of random circuits | 256–512. 10 % are replaced every 10 updates |
| `--preset` | Track generator(s): `mixed` (default), `circuit`, `gp`, `fs`, `fs_segments`; combine with `+` | **Recommended: `mixed+gp`** — `gp` adds long straights into slow corners and chicanes (engineering log WP13–14). FS scale: `fs+fs_segments` |
| `--val-preset` | Generator for the random validation tracks | Keep it fixed between runs you want to compare |
| `--init-from` | Start from an existing driver's weights | For fine-tuning on new tracks or a similar car. Also works when the new observation has extra inputs at the end (e.g. `--history`) |
| `--history N` | Memory: N past samples of the car state (0.12 s apart) | 8 ≈ the last second. Changes the observation size |
| `--grip-range LOW HIGH` | Hidden tyre-grip variation per episode | Default 0.88 1.05 |
| `--reset-noise X` / `--min-noise X` | Exploration noise at the start / its floor | For fine-tuning a converged driver: e.g. 0.1 / 0.06 |
| `--vehicle a,b,c` | One driver for several cars | Fleet is split evenly between them |
| `--limit-seeking` | Going off costs time (car rejoins) instead of ending the episode; lap memory; 240 s episodes | The owner's "always at the limit" principle (architecture §1b). Changes the observation size |
| `--episode-s S` | Episode length | Default 60 s (240 s with `--limit-seeking`) |
| `--hours` / `--steps` / `--iters` | When to stop | Any one of them |
| `--eval-every` | Updates between validation runs | 100 ≈ every 5–8 minutes |
| `--warm-start` | First imitate the rule-based driver | See engineering log WP7 for the measured effect |
| `--resume` | Continue the run with this `--name` | Loads `runs/<name>/latest.pt` |

Stop with Ctrl+C at any time; continue later with the same command plus
`--resume` (at most the last 25 updates are lost).

Keep the laptop plugged in. While training runs, the script asks Windows not to
idle-sleep (the same request a video player makes; `--allow-sleep` turns this
off). Closing the lid or choosing Sleep still suspends it. Nothing else heavy
should use the GPU.

## 3. What appears on screen

```
update   140 |   73.4 M decisions | 150,000/s | pace 152.1 km/h |   12.3 crashes/100 km | reward/step +0.612 | ended episodes:   1560 m | noise 0.18 | kl 0.0088
```

| Field | Meaning | Healthy trend |
|---|---|---|
| decisions | Total (car × tick) decisions so far | — |
| /s | Simulator + learning throughput | 100 000–250 000 on this GPU |
| pace | Average speed *along the track* of all cars, including the time lost by crashing and restarting | Rises, then flattens |
| crashes/100 km | How often a car leaves the road, with exploration noise on | Falls towards low single digits |
| reward/step | Average reward per decision | Rises |
| ended episodes | Mean distance of episodes that ended in this update (biased towards short ones — crashes end early) | Rises |
| noise | Exploration noise on the actions | Falls from 0.6 to ~0.1 |
| kl | Size of the last policy change | Stays around 0.005–0.02 |

Every `--eval-every` updates:

```
  validation: 67 % of tracks completed
    imola                  flying lap  118.42 s
    monza                  flying lap -- (off at 940 m)
```

This is the driver **without** exploration noise on tracks it has never
trained on, standing start, two laps. It is the number that matters.
`off at 940 m` = left the track 940 m into the run; `spun`, `stalled`, `slow`
are the other ways to fail.

`runs/<name>/best.pt` is the checkpoint that completed the most validation
tracks (ties: the faster one); `latest.pt` is the most recent; `eval_<update>.pt`
is saved at every validation.

Graphs: `.venv\Scripts\tensorboard.exe --logdir runs`, then open the address
it prints. Raw numbers: `runs/<name>/log.csv`.

## 4. How long does it take?

Measured so far on this machine (911 GT3 RS, engineering log WP7): after
~40 million decisions (≈ 10 minutes) the driver completes some unseen circuits
and is already ~11 % faster than the rule-based baseline on those. Expect
hundreds of millions to a few billion decisions for a driver near the limit
everywhere — hours, not days, on this GPU. Train until the validation lap
times stop improving for ~30 minutes.

## 5. Judge the result

```powershell
.venv\Scripts\python.exe scripts\evaluate.py --vehicle ks_porsche_911_gt3_rs --policy runs\gt3rs_v1\best.pt --scripted
.venv\Scripts\python.exe scripts\evaluate.py --vehicle ks_porsche_911_gt3_rs --policy runs\gt3rs_v1\best.pt --blind
.venv\Scripts\python.exe scripts\evaluate.py --vehicle ks_porsche_911_gt3_rs --policy runs\gt3rs_v1\best.pt --grip 0.88
```

* First command: learned vs. rule-based driver on the real circuits and four
  unseen random ones, with plots in `runs/eval/` (line coloured by speed, speed
  trace) and the gap to the physics reference where one exists
  (`scripts\reference_lap.py`).
* `--blind`: track preview cut to 40 m — a first lap on an unknown track.
* `--grip 0.88`: a slippery day the driver is not told about.

Targets are in the roadmap, phase 1.

## 6. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `/s` far below 50 000 | Something else is using the GPU, laptop on battery, or thermal throttling | Close other GPU programs, plug in, check temperatures |
| CUDA out of memory | Too many cars | Halve `--envs` |
| pace stuck near 0, crashes/100 km huge after 50+ updates | Learning has collapsed | Stop; restart with a new `--name`; if it repeats, lower the learning rate (`PPOConfig.lr`) and log it |
| `kl` persistently > 0.05 | Updates too aggressive | Same as above |
| Validation fails at the same spot every time | A feature the driver has not mastered (typically a very heavy braking zone or a chicane) | Normal early on. If it persists after pace has flattened: more training, or raise `blind_prob`/track difficulty; log what you try |
| Validation good on random tracks, bad on the real ones | Real circuits contain shapes the generator rarely makes | Roadmap phase 3: curvature-profile generator |
| Lap times good but "off" count high in `evaluate.py` | Driver abuses the edge | Raise `edge_penalty` in `EnvConfig` |
| Old checkpoint refuses to load / `expects N observation values` | The observation layout changed since it was trained | Checkpoints are tied to the observation contract; retrain |

## 7. Changing settings

Environment settings (sensors, reward, randomisation) are the fields of
`EnvConfig` in `mv/envs/racing_env.py`; learning settings are `PPOConfig` in
`mv/rl/ppo.py`. Every run stores the exact settings it used in
`runs/<name>/config.json`, and the in-game driver reads the sensor settings
from there. Change **one thing at a time** and write the experiment and its
outcome into `engineering-log.md`.

## 8. Training while away (queue)

`scripts\queue_runs.py runs\my_queue.json` runs jobs one after another,
optionally waiting for a running process first. Format and an example are in
the script's docstring; `runs/queue_lane1.json` is a real one. Two queues can run
side by side: one run uses only ~40 % of this GPU.

## 9. Pausing and resuming (taking the laptop away)

Quickest: **`mvctl pause`** / **`mvctl resume`** in a terminal in the project
folder (PowerShell: `.\mvctl pause`), or the Pause / Resume buttons of the web
interface (`mvctl ui`). `mvctl watch` shows live progress bars.

All planned training is described in `runs/plan.json` (lanes of jobs, see
`scripts/train_plan.py`). Three double-click files in the project folder:

| File | What it does |
|---|---|
| `pause_training.cmd` | Every running training saves its driver after its current update (a few seconds) and stops; nothing new starts. Wait for **"It is safe to close the laptop."** |
| `resume_training.cmd` | Continues every interrupted run from its saved driver — with the time already trained counted against its `--hours` — then the rest of the plan |
| `status_training.cmd` | What is running, finished, paused or waiting, with the latest numbers |

Same from a terminal: `.venv\Scripts\python.exe scripts\train_plan.py pause|resume|status`.

Notes:
- The pause lasts until you resume: `runs\PAUSE` exists while paused, and no
  job will start (also not after a reboot).
- Measured: pause took 10 s and lost nothing. A run started before pausing
  existed is stopped instead after a 4-minute wait; it then loses at most the
  last 25 updates (~1–2 minutes).
- Closing the lid *without* pausing suspends Windows mid-training; it usually
  continues after waking, but pausing first is the safe way.
