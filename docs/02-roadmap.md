# Project MV — Roadmap

*MV-PLAN-001. Phases are ordered by dependency. Each has an objective, the work
in it, and a measurable exit criterion ("done when"). Status as of 2026-11-17.*

Legend: ✅ done · 🕓 in progress · 🔲 not started

## Overview

| Phase | Name | Outcome | Status |
|---|---|---|---|
| 0 | Foundations | Simulator, sensors, learning algorithm, AC import/bridge, docs, tests | ✅ |
| 1 | Solo hot-lap driver (one car) | A driver that laps any unseen circuit fast in our simulator | 🕓 next |
| 2 | Into Assetto Corsa | The same driver laps in the real game; simulator tuned to match AC | 🔲 |
| 3 | Simulator tier 2 | Wheel dynamics, elevation, kerbs, tyre temperature; richer track generator | 🔲 |
| 4 | One driver, any car | ICE + hybrid + EV in one network; gear and energy management; adapts lap by lap | 🔲 |
| 5 | Wheel-to-wheel racing | Opponents, overtaking, defending, racing rules | 🔲 |
| 6 | Formula Student Driverless | Cones, missions, AS state machine, safety, on-car computer | 🔲 |

---

## Phase 0 — Foundations ✅

Delivered (details and test evidence in `engineering-log.md`):

* Vehicle parameter schema for ICE / hybrid / EV; importer from Assetto Corsa;
  public-data files for Taycan Turbo GT, 911 GT3 RS (992), 918 Spyder.
* Track format, AC circuit importer, random circuit generator.
* GPU physics simulator (tier 1), verified against road-test figures.
* Racing environment: lidar, track preview, blind first-lap mode, reward,
  track limits, lap timing, randomisation.
* PPO learning algorithm, trainer, evaluator with plots.
* Rule-based baseline driver (completes all test circuits in all three cars).
* Physics reference lap (g-g-v + racing line) — see `08-reference-lap.md`.
* Assetto Corsa live bridge (telemetry, control, resets) and driver adapter.
* Formula Student rules digest.

## Phase 1 — Solo hot-lap driver 🕓

**Objective.** A single network that drives the 911 GT3 RS around circuits it
has never seen, close to the physical limit, without leaving the track.

**Work**

1. Long training run on the RTX 3080 (commands in the README, "Train").
   Budget: start with 2 h; extend while the validation lap times still fall.
2. Track the gap to the physics reference on Imola / Monza / Spa / Mugello and
   on unseen random circuits (`scripts/evaluate.py`).
3. Tune, one change at a time, logging each experiment in the engineering log:
   reward weights (edge penalty, smoothness), exploration noise floor,
   network size, discount, number of cars, track preset mix.
4. Robustness checks: grip ×0.85 and ×1.05, blind preview, standing start.

**Done when**

| Metric | Target |
|---|---|
| Unseen circuits completed (2 laps, no crash) | 100 % of 20 |
| Flying lap vs. rule-based baseline | ≥ 15 % faster on every circuit |
| Flying lap vs. physics reference | within 3 % (median) |
| Decisions with any body part over the edge | < 0.5 % |
| Same at grip ×0.85 (driver not told) | 100 % completed |

## Phase 2 — Into Assetto Corsa 🔲

**Objective.** The driver trained in Phase 1 laps in the real game.

**Work, in this order (each step de-risks the next)**

1. **Bridge check** (needs the owner at the PC): install the MV Bridge app,
   run `scripts/ac_bridge_check.py`; resolve the "not yet verified in-game"
   list in `07-assetto-corsa-bridge.md`.
2. **Dry run**: `scripts/ac_drive.py --scripted --dry-run` while driving by
   hand — do track position, lateral offset and heading on screen agree with
   the car in the game?
3. **Rule-based driver in the game**, speed-capped, then uncapped. This proves
   telemetry → observation → action → control end to end with no learning.
4. **System identification**: record AC telemetry for standard manoeuvres
   (full-throttle run, full braking, steady circles at several speeds, step
   steer) and compare with `scripts/vehicle_check.py`. Tune the reduction rules
   in `mv/ac/car_import.py` (tyre grip, load sensitivity, aero map, roll
   stiffness split, brake torque) until the simulator matches within a few
   percent. Automate as `scripts/ac_sysid.py`.
5. **Learned driver in the game**, zero-shot. Expect it to be drivable but not
   optimal (elevation, kerbs, tyre temperature are not in tier 1).
6. **Fine-tuning in the game**: continue learning from real-time AC experience.
   Sample efficiency matters here (one car, 25 Hz), so use an off-policy method
   (SAC-style) or learn a small *residual* on top of the frozen simulator
   policy. Resets via the bridge's `load_state`.

**Done when** the learned driver completes 10 consecutive clean laps of Imola
in AC and its best lap is within 3 % of a good human alien-level reference lap
for the car (take the reference from an online leaderboard at the time).

## Phase 3 — Simulator tier 2 🔲

Driven by what Phase 2's system identification shows to matter most. Candidates,
in expected order of value:

| Item | Why | Notes |
|---|---|---|
| Elevation and banking | Spa, Mugello, Nordschleife are not flat; changes grip and braking | AC AI line already contains height; add slope/bank to `Track` and gravity terms to `VehicleSim` |
| Wheel rotation as a state | True wheelspin and lock-up, needed for cars without TC/ABS (most FS cars) | Stiff equations: semi-implicit update or more sub-steps |
| Kerbs and run-off | Real lap times use kerbs; run-off has less grip instead of a wall | Per-point surface bands beside the edges |
| Differential model | Power-on rotation of RWD cars | Open / locked / clutch-type with preload |
| Tyre temperature and wear | Grip changes over a stint | Simple thermal mass per tyre |
| Suspension kinematics / ride-height aero | High-downforce cars | Only if sysid shows a gap |
| Curvature-profile track generator | Switchbacks, parallel straights, hairpins the star-shaped generator cannot make | Random straights/arcs/clothoids + loop closure |
| `torch.compile` / fewer operations per step | More decisions per second | Needs a C++ compiler on Windows; measure first |

**Done when** simulator vs. AC lap time for the same recorded control inputs
differs by < 2 % on three circuits.

## Phase 4 — One driver, any car 🔲

**Objective.** The requirement "works with any car, just add its parameters"
at the level of the *trained network*, not only the code.

1. **Multi-car training**: several environments (one per car, sharing the
   learner); the car descriptor in the observation tells the driver what it is
   in. Add parameter *sampling* around each car so the driver interpolates to
   cars it has never seen. Test on a held-out car.
2. **Gear control** as an optional third action (needed for manual-gearbox
   cars and for short-shifting on low grip).
3. **Energy management** as an optional fourth action for hybrids and EVs
   (deploy/recover), with lap-level reward (stint time under an energy budget —
   this is also Formula Student Efficiency).
4. **Memory**: give the driver a short history of its own observations
   (recurrent network or history encoder). This is what allows "greater and
   greater by each lap": it can *infer* grip, tyre state and its own car's
   quirks from how the car responded, and it can use the map built on lap 1.
5. **Lap-over-lap learning on one track**: an explicit mode where the preview
   horizon grows from "sensor range" on lap 1 to "full map" from lap 2, plus a
   learned or optimised reference line from `mv/planning` as extra input.

**Done when** one network drives 911 GT3 RS, 918 Spyder, Taycan Turbo GT and a
generic FS car within 4 % of each car's physics reference on unseen circuits,
and a held-out fifth car within 6 %.

## Phase 5 — Wheel-to-wheel racing 🔲

1. Several cars per track in the simulator; collisions between cars.
2. Opponent observations (relative position/velocity of the nearest cars;
   lidar rays that also hit cars; rear-facing rays).
3. Self-play league training (current driver vs. past versions).
4. Racing rules in the reward: no contact, no blocking moves under braking,
   leave space — the approach of *Gran Turismo Sophy* (Wurman et al., Nature 2022).
5. AC multi-car sessions via the bridge.

**Done when** the driver wins > 50 % of 8-car races against AC's AI at 100 %
strength from mid-grid without causing contact.

## Phase 6 — Formula Student Driverless 🔲

Reference: `fs-rules-dv-summary.md` (rule numbers there).

| Work package | Content |
|---|---|
| FS vehicle file | Team car parameters (≤ 80 kW EV, wheelbase ≥ 1 525 mm, no ABS/TC unless fitted) |
| Cone-level tracks | Generator for trackdrive/autocross (width ≥ 3 m, straights ≤ 80 m, hairpins ≥ 9 m diameter, lap 200–500 m), skidpad figure-8 (D 4), 75 m acceleration (D 5). Cone positions and colours from the FSG handbook |
| Perception interface | Replace "lidar to the edge" by "detected cones → edge estimate" with realistic range (≈ 10–25 m), field of view, noise, missed and false cones. The driver's observation contract stays the same |
| Missions | Acceleration, skidpad (fixed lap sequence), autocross (blind lap), trackdrive (10 laps: lap 1 blind, then map), finish and stop within the stop zone, lap counting |
| AS state machine wrapper | AS Off / Ready / Driving / Finished / Emergency around the driver; RES, EBS supervision (T 14, T 15). **Not learned** — conventional, testable code |
| Safety envelope | Independent checks that override the network: track-limit prediction, max speed, actuator plausibility, watchdog → EBS |
| Penalty-aware reward | Cone hit 2 s, off-course 10 s (D 10) folded into the reward |
| Deployment | Export to ONNX / TensorRT; fixed-rate C++ or Python node; latency budget; logging |
| Validation ladder | Simulator → AC with an FS car mod → hardware-in-the-loop → car on stands → low-speed track → full speed |

**Done when** the car completes all four dynamic missions in simulation and AC
within the rules, and the safety envelope has a written test report.

---

## How to continue after a break

1. Read the last entries of `engineering-log.md` ("Next steps").
2. `.venv\Scripts\python.exe -m pytest tests -q` — everything should pass.
3. Pick the first 🔲 item of the current phase above.
4. Log what you try and what happens in `engineering-log.md` — including what
   did **not** work.
