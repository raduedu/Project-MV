# Project MV — Engineering Log

| | |
|---|---|
| **Document** | MV-LOG-001, running engineering log |
| **Project** | Project MV — modular autonomous racing driver (reinforcement learning) |
| **Owner** | Project owner (sole IP holder; independent of any team) |
| **Started** | 2026-11-16 |
| **Status** | Living document — newest entries at the bottom of each work package |

## How to read this document

It follows the structure of a Formula Student design report / test log. Every
work package (WP) has the same five headings:

1. **Objective** — what we want and why.
2. **Approach** — what was built or tried, and the reasoning.
3. **Verification** — the test that was run and the measured result.
4. **Problems & resolutions** — what went wrong, root cause, fix. Unresolved items stay open.
5. **Next steps**.

Status tags: ✅ verified by test · ⚠️ works with known limitation · ❌ failed / abandoned · 🔲 not started · 🕓 in progress.

Numbers quoted here were measured on the development machine: Ryzen 9 5900HX,
32 GB RAM, RTX 3080 Laptop (16 GB), Windows 11, Python 3.13, PyTorch 2.14 + CUDA 12.6.

---

## WP0 — Requirements and system concept

**Objective.** A driver ("the agent") that can race any car on any track, learned
by reinforcement learning. First car: Porsche 911 GT3 RS in Assetto Corsa (AC).
Must also cover hybrid (918 Spyder) and electric (Taycan Turbo GT) drivetrains.
End goal: a Formula Student Driverless car. The software must be independent of
any specific car: a new car = one parameter file.

**Requirements derived from the brief**

| ID | Requirement | Source |
|---|---|---|
| R1 | Car-agnostic: all car knowledge lives in one parameter file | "modular so it can work with any car" |
| R2 | Supports combustion, hybrid and electric powertrains | "accounts for EV, HY and CV cars" |
| R3 | Track-agnostic: trained on fully randomised tracks, generalises to unseen ones | "tracks ... randomised fully in every aspect" |
| R4 | Sole purpose is lap time / racing performance | "great at racing, that's its entire scope" |
| R5 | Uses track understanding + lidar-type sensing | brief |
| R6 | Validated in Assetto Corsa (original), same sim later used for the FS car | brief |
| R7 | No third-party IP inside the repository (game data stays local) | "completely keep the intellectual property" |
| R8 | Everything documented so each complex part can be explained simply | brief |
| R9 | Improves lap after lap on an unknown track (FS Trackdrive: 10 laps, first lap blind) | "greater and greater by each lap" + FS rules D 8 |

**Concept decision (see `03-decision-log.md` D-001…D-004).** Train in our own
fast simulator (thousands of cars in parallel on the GPU), validate and
fine-tune in Assetto Corsa. Reason: AC runs in real time with one car; RL needs
hundreds of millions of driving steps. At 1× real time 100 M steps would take
~46 days; the in-house simulator targets >100 000 steps/s (≈17 minutes).

---

## WP1 — Development environment ✅

**Approach.** Git repository + Python virtual environment (`.venv`) in the project
folder. PyTorch with CUDA for GPU simulation and learning.

**Verification.** `torch.cuda.is_available()` → `True`, device "RTX 3080 Laptop GPU".

**Problems & resolutions**

| # | Problem | Root cause | Resolution |
|---|---|---|---|
| 1.1 | The rules PDF could not be opened by the built-in reader | `pdftoppm` (poppler) not installed | Text extracted with the `pypdf` library instead |
| 1.2 | A test script crashed with `module 'inspect' has no attribute 'get_annotations'` | A temporary file named `inspect.py` in the same temp folder shadowed Python's standard `inspect` module | Run throw-away scripts with `python -I` (isolated mode). Lesson: never name a script after a standard-library module |
| 1.3 | Multi-line shell snippets containing `'''` failed to parse | Quote clash inside a bash here-document | Scripts are written to files instead of inline |

---

## WP2 — Vehicle parameter system ✅

**Objective (R1, R2).** One schema that can describe any car, and a tool that fills
it from an Assetto Corsa car automatically.

**Approach.**
- `mv/vehicle/params.py` defines the schema (chassis, aero, tyres per axle, brakes,
  steering, powertrain, electronics). The powertrain is a *list of power units*
  (engines and/or motors), each attached to a gearbox and an axle. ICE, EV and
  hybrid are the same structure with different lists — no special cases in code.
- `mv/ac/acd.py` reads the packed `data.acd` archive every Kunos car ships with.
- `mv/ac/car_import.py` condenses ~15 game files into the schema
  (`scripts/import_ac_car.py <car_id>` → `vehicles/local/<car_id>.yaml`).
- `mv/vehicle/summary.py` prints spec-sheet numbers so a human can sanity check.

**Verification.**

| Check | Result | Reference | Verdict |
|---|---|---|---|
| `data.acd` decodes to readable INI text (911 GT3 RS, 918) | 53 and 74 files decoded | — | ✅ |
| 911 GT3 RS peak power | 325 kW / 442 PS (at the wheels) | 500 PS at the crank; ~11 % driveline loss is baked into AC curves | ✅ plausible |
| 911 GT3 RS downforce at 200 km/h | 153 kg | Porsche: ~345 kg at 300 km/h ⇒ ~153 kg at 200 km/h | ✅ |
| 911 GT3 RS top speed (force balance) | ~320 km/h | 310 km/h | ✅ within 3 % |
| 911 GT3 RS CG height | 0.456 m | typical 911: 0.45–0.48 m | ✅ |
| 918 Spyder mass / power | 1731 kg / 691 kW | 1 675 kg + driver / 652 kW | ⚠️ see 2.4 |

**Problems & resolutions**

| # | Problem | Root cause | Resolution |
|---|---|---|---|
| 2.1 | Car physics files are not readable on disk | Kunos pack them into `data.acd` with a key derived from the folder name | Implemented the reader (pure Python). Extracted data is git-ignored (R7) |
| 2.2 | CG height ambiguous: the comment in the game file suggests `radius + BASEY` = 0.26 m, which is physically impossible for a 911 | `BASEY` is the wheel-centre height *relative to the CG* (negative = below) | Use `radius − BASEY` = 0.44–0.46 m. Cross-checked with the ride-height pickup (0.315 m below CG + ~0.12 m ride height ≈ 0.44 m) |
| 2.3 | Brake torque would give only 0.76 g if `MAX_TORQUE` were the total | It is per wheel, before the bias split | Total = 2 × `MAX_TORQUE` → 1.50 g hydraulic capacity |
| 2.4 | 918 front-motor table gives 191 kW; the real motor has 95 kW | AC's table is the envelope before its ERS controllers reduce it | Battery discharge limit caps total electric power; flagged in the YAML notes. Real data file for the 918 is being researched separately (WP7) |
| 2.5 | AC auto-shifter thresholds (up 8850, down 5900 rpm) would shift up and immediately down again in our model | After a 1→2 upshift the engine lands at 5 617 rpm, below the down threshold | Our simulator ignores those and always picks the gear with the highest wheel torque at the current speed, with hysteresis |
| 2.6 | Porsche Taycan Turbo GT does not exist in AC | — | Parameter file built from public data (WP7) |

**Next steps.** Compare imported cars against live AC telemetry (0–200 km/h time,
steady-state cornering g) once the bridge runs, and tune the reduction rules.

---

## WP3 — Track system ✅ / ⚠️

**Objective (R3, R6).** A track format that is independent of where the track
came from; an importer for AC circuits; a generator of unlimited random circuits.

**Approach.**
- `mv/track/track.py`: a track is a closed centre line sampled every metre plus
  left/right edge distances. `TrackBatch` stacks many tracks on the GPU so every
  simulated car can be on a different one.
- `mv/ac/track_import.py`: reads AC's AI line file (`fast_lane.ai`), which also
  stores the distance to both track edges at every point.
- `mv/track/generator.py`: random circuits (styles: `polygon` = straights +
  corners + chicanes, `spline` = flowing). Presets: `circuit`, `fs`, `mixed`.

**Verification.**

| Check | Result | Reference | Verdict |
|---|---|---|---|
| Imola length / direction | 4 900 m, counter-clockwise | 4 909 m, counter-clockwise | ✅ |
| Monza | 5 787 m, clockwise | 5 793 m, clockwise | ✅ |
| Spa | 6 991 m, clockwise | 7 004 m, clockwise | ✅ |
| Mugello | 5 235 m, clockwise | 5 245 m, clockwise | ✅ |
| Generator speed | 60 valid tracks in 2.5 s | needs to be ≪ training time | ✅ |
| Generator variety (60 "mixed" tracks) | length 664–5 895 m, tightest corner 9–304 m radius (median 25 m) | Imola tightest ≈ 10 m | ✅ |

Picture: `docs/img/tracks_overview.png`.

**Problems & resolutions**

| # | Problem | Root cause | Resolution |
|---|---|---|---|
| 3.1 | First generator produced "potato" shapes: tightest corners 83–433 m radius, no straights | A smooth spline through radial control points cannot make sharp corners | Rewrote: polygon (straights) + corner rounding with a smoothing length that varies along the lap, plus chicanes |
| 3.2 | Map orientation of AC tracks (mirror risk: left/right corners swapped) | AC's world has y up; seen from above its z axis points down the map | `map_x = x, map_y = −z`. Verified: Imola comes out counter-clockwise, the other three clockwise, as in reality |
| 3.3 ⚠️ open | Random tracks are star-shaped: no sections that fold back alongside each other (e.g. long hairpins, parallel straights) | Construction scatters points around a centre | Planned: second generator that integrates a random curvature profile (straight / arc / clothoid segments) and closes the loop |
| 3.4 ⚠️ open | Tracks are flat (2-D). AC tracks have elevation, banking and kerbs | Out of scope for the first simulator tier | Planned: per-point slope and banking in the track format and dynamics |

---

## WP4 — Formula Student rules digest ✅

**Objective.** Know exactly what the FS Driverless rules demand so the design
does not paint itself into a corner.

**Approach.** Delegated to a helper agent: full read of *FS Rules 2027 v1.0*, every
statement cited by rule number → `docs/fs-rules-dv-summary.md`.

**Key numbers** (rule numbers in the digest): track width ≥ 3 m, straights ≤ 80 m,
min turning diameter 9 m, lap 200–500 m, Trackdrive = 10 laps and the car must
count laps itself; cone hit 2 s, off-course 10 s; EBS reaction ≤ 200 ms and
> 10 m/s² deceleration; EV power ≤ 80 kW.

**Problems & resolutions**

| # | Problem | Resolution |
|---|---|---|
| 4.1 | Cone colours, sizes and spacing are not in the rules | They are in the FSG Competition Handbook — needed before building the cone-level FS track generator (open) |
| 4.2 | State-machine figure and formulas are vector drawings, not text | The agent rendered the pages to images to read them |

---

## WP5 — Physics simulator (tier 1) ✅ / ⚠️

**Objective.** Simulate thousands of cars in parallel on the GPU, accurately
enough that a driver trained here is useful in Assetto Corsa.

**Approach.** `mv/sim/vehicle_sim.py`: planar four-wheel model — magic-formula
tyres with load sensitivity and combined slip, load transfer, aero, multi-unit
powertrain with gearboxes/battery, ABS/TC, actuator lags. Test tool:
`scripts/vehicle_check.py <car>`.

**Verification** (911 GT3 RS 991 imported from AC vs published road tests)

| Test | Simulated | Real car | Verdict |
|---|---|---|---|
| 0–100 km/h | 3.30 s | 3.3 s | ✅ |
| 0–200 km/h | 10.1 s | 10.9 s | ✅ |
| 400 m | 11.1 s | 11.2 s | ✅ |
| 100–0 km/h | 30.1 m | ~31 m | ✅ |
| Cornering limit, 60–100 km/h | 1.20–1.26 g | 1.1–1.3 g | ✅ |
| Cornering limit, 200 km/h | 1.98 g (downforce) | — | plausible, unverified |
| 918 Spyder 0–100 / 0–200 | 2.43 / 6.1 s | 2.6 / 7.2 s | ⚠️ too strong (see 2.4) |
| Taycan Turbo GT 0–100 / top speed | 2.54 s / 308 km/h | 2.2 s (overboost) / 305 km/h | ✅ (sustained power only) |

Time-step study (rule-based driver, flying laps in seconds, same car):

| Physics rate | Imola | random A | random B | random C | FS-size A | FS-size B |
|---|---|---|---|---|---|---|
| 250 Hz | 146.711 | 51.081 | 96.229 | 87.233 | 18.760 | 27.201 |
| 125 Hz | 146.671 | 51.081 | 96.269 | 87.273 | 18.720 | 27.241 |
| 100 Hz | 146.711 | 51.121 | 96.269 | 87.273 | 18.760 | 27.241 |

→ identical within 0.05 s (one control tick is 0.04 s). **Default lowered from
250 Hz to 125 Hz**, which roughly doubles training throughput.

**Problems & resolutions**

| # | Problem | Root cause | Resolution |
|---|---|---|---|
| 5.1 | 911 topped out at 78 km/h, 400 m in 19.6 s | Car sat on the rev limiter in 1st gear: the "next gear is better" speed lay just *above* the speed where limiter torque equals drag (torque collapses within 36 rpm, narrower than the search grid) | Shift speed capped at 99 % of the rev limit |
| 5.2 | **Cars spun at every heavy braking zone** (found via the rule-based driver, WP6; also where the first RL runs kept crashing: Imola 705 m, Monza 940 m, Spa 400 m) | First tyre model = friction ellipse with a force clamp: a wheel at the ABS/TC limit (97 % longitudinal) kept only √(1−0.97²) ≈ 24 % of its cornering grip. Real tyres / AC keep ≈ 70 % | Replaced by a combined-slip similarity model (decision D-013). Straight-line figures unchanged; baseline driver 3/14 → 14/14 tracks after its own fixes |
| 5.3 ⚠️ open | No wheel-spin state, no suspension, no tyre temperature, flat road, ideal differential | Tier-1 scope | Roadmap phase 3; randomisation of grip/mass/power/aero covers part of the gap |
| 5.4 ⚠️ open | The 911 oversteers under trail braking in the simulator (rear slip angle passes its peak at ~0.6 brake + 1.0 g lateral) | Plausible for a rear-engined car with 54 % front brake bias, but not yet compared with AC | Compare against AC telemetry in phase 2 (system identification) |

## WP6 — Racing environment and baseline driver ✅

**Approach.**
- `mv/envs/racing_env.py`: observation (98 numbers: car state incl. kinematic
  slip angles, position on track, 23-point preview to 340 m, 19-ray lidar, car
  descriptor), action (steer, combined pedal), reward (progress − edge −
  jerk; crash −5), random track/start/speed and ±grip/mass/power/aero per
  episode, 30 % "blind" episodes.
- `mv/control/scripted.py`: rule-based driver that uses only the observation
  (Stanley-type steering on the centre line, backward-pass speed plan, traction
  circle on throttle and brake). Lower yardstick + self-test + first in-game test.
- `mv/rl/evaluate.py`, `scripts/evaluate.py`: timed standing + flying laps,
  no randomness, plots to `runs/eval/`.

**Verification.**

| Check | Result |
|---|---|
| Throughput, 4 096 cars, GPU (250 Hz physics) | 65 600 decisions/s |
| Throughput, 16 384 cars, GPU (250 Hz physics) | 168 000 decisions/s, 800 MB GPU memory |
| Lidar vs. hand calculation on a circular track (left, right, straight ahead) | within 3 % (`test_lidar_on_circle_matches_geometry`) |
| Flying lap time = lap length ÷ mean speed | within 3 % (`test_scripted_driver_laps_a_circle…`) |
| Reward sum = 0.25 × distance progressed | exact to 0.1 % |
| Baseline driver, 10 random + 4 real circuits, 2 laps each | **14/14 completed in all three cars, 0 track-limit violations on the real circuits** |

Baseline flying laps (rule-based, deliberately cautious):

| Car | Imola | Monza | Mugello | Spa |
|---|---|---|---|---|
| 911 GT3 RS (AC import) | 2:26.6 | 2:24.2 | 2:33.0 | 3:17.0 |
| Taycan Turbo GT Weissach | 2:19.7 | 2:16.8 | 2:26.4 | 3:08.2 |
| 918 Spyder (AC import) | 2:12.6 | 2:11.3 | 2:18.9 | 3:01.5 |

**Problems & resolutions**

| # | Problem | Root cause | Resolution |
|---|---|---|---|
| 6.1 | Environment crashed with a single car | Observation size was probed with two car indices | Probe with one |
| 6.2 | Baseline driver spun under power out of corners | Full throttle while the rear tyres were at their cornering limit (rear-wheel drive, 440+ PS) | Throttle capped by the share of grip not used laterally; cut when rear slip angle passes its peak |
| 6.3 | Baseline driver spun under braking | (a) tyre model, see 5.2; (b) heavy trail braking | (a) new tyre model; (b) brake capped by traction circle |
| 6.4 | Baseline driver ran wide in corners that tighten gradually | Speed plan assumed full braking is available while cornering | Backward pass over the preview points in which braking capacity shrinks with lateral grip in use |
| 6.5 | Validation during training took minutes | A handful of cars on the GPU costs ~54 ms per step regardless of count | Validation on one CPU thread (11 ms per step); runs abandoned as "slow" if average speed < 8 m/s after 20 s |
| 6.6 ⚠️ open | Track limits are strict (no body part over the edge). AC tolerates kerbs / two wheels | Simplest safe rule first | Make the overhang allowance a parameter when kerbs are modelled |

## WP7 — Learning algorithm ✅ (works; tuning in progress)

**Approach.** `mv/rl/ppo.py` (own PPO), `scripts/train.py` (track pool with
continuous replacement, validation on unseen tracks, CSV + TensorBoard logs,
resume), `mv/rl/pretrain.py` (optional warm start by imitating the baseline).

**Experiments** (911 GT3 RS, 8 192 cars, 256 circuits, "mixed" preset; "distance"
= mean metres driven per episode during training, with exploration noise)

| Run | Change under test | Decisions | Distance | Crash rate | Validation (6 unseen tracks, 2 laps) |
|---|---|---|---|---|---|
| verify1 | first working version (old tyre model, 92-value observation) | 47 M | 1 463 m | 66 % | 0/6; always off at the first heavy braking zone |
| verify2 | new tyre model + slip-angle observation + 340 m preview | 29 M (stopped) | 1 202 m | 77 % | 0/6 at 15 M |
| verify3 | + feasible starting speeds (D-014) | 39 M | 1 545 m | 59 % | **2/6: Mugello 2:16.5, random 1:47.2** |
| verify4 | + warm start from the baseline driver (D-015) | 48 M (9 M of them warm start) | 1 908 m | 31 % | **3/6: Imola 2:12.1, Mugello 2:21.5, random 1:45.4** |

Warm-start detail: straight after imitation (before any PPO) the network completed 3/6 tracks at the
teacher's pace (Imola 2:20.6). After 75 PPO updates validation had *dropped* to 1/6 while training pace rose —
the driver was trading safety for speed — and by update 150 it was back to 3/6 and faster (Imola 2:12.1).
Conclusion: warm start is kept (D-015 active); judge runs by validation over time, not by a single check.

Reading: learning works and each change helped. After only ~10–15 minutes the
driver already beats the rule-based baseline on the tracks it completes (Imola
2:12.1 vs 2:26.6, −10 %; Mugello 2:16.5 vs 2:33.0, −11 %), but it does not yet
finish every track — it needs far more than 50 M decisions, as expected.

Throughput after the 125 Hz change (simulation only): 147 000 decisions/s with
8 192 cars, 263 000 with 16 384, 440 000 with 32 768 (1.6 GB of GPU memory).

**Problems & resolutions**

| # | Problem | Root cause | Resolution |
|---|---|---|---|
| 7.1 | First training script appeared to hang | Validation of an untrained driver crawled for the full 900 s limit on the GPU | 6.5 |
| 7.2 | Many early episodes were unwinnable | Random start speed up to 90 % of top speed with a hairpin just beyond the 160 m check | D-014: start at 30–100 % of the highest *feasible* speed for the next 450 m |
| 7.3 ⚠️ open | Hyper-parameters are first guesses (rollout 32, γ 0.995, 5 epochs, entropy 0.002, network 512-256-128) | No time yet for a sweep | Phase 1, one change at a time |

## WP8 — Assetto Corsa link 🕓 (built and unit-tested; never run against the game)

**Approach.**
- `mv/ac/bridge/` (helper agent): shared-memory telemetry reader with exact
  struct layouts, SI/project-frame conversion, control back-ends (CSP Lua app
  preferred; vJoy and ViGEm fall-backs), resets, fixed-rate loop with fail-safe.
  Setup and checklist: `docs/07-assetto-corsa-bridge.md`.
- `mv/ac/driver.py` (lead): `ObservationAdapter` — a one-car copy of the
  training environment whose state is overwritten from telemetry, so in-game
  observations are produced by the training code itself; `ActuatorModel`
  reproduces the simulator's steering-rate limit and pedal lag; `drive()` loop
  with dry-run mode and a speed governor. CLI: `scripts/ac_drive.py`.

**Verification (offline).**

| Check | Result |
|---|---|
| Bridge unit tests (struct sizes, decoding, axis conventions, fail-safe) | 54 passed |
| Adapter observation vs. training observation, 250 steps, 3 cars | max difference < 2 × 10⁻³ |
| Adapter re-locates the car after a teleport | ✅ |
| Actuator model vs. simulator actuators, 60 random commands | equal to 10⁻⁴ |

**Problems & resolutions**

| # | Problem | Root cause | Resolution |
|---|---|---|---|
| 8.1 | Adapter and environment disagreed by 5 × 10⁻³ | Environment used slip angles computed one physics step earlier | Both now compute slip angles from the current state (`VehicleSim.wheel_slip_angles`) |
| 8.2 ⚠️ open | Nothing here has touched the running game | Needs the owner at the PC | Roadmap phase 2, steps 1–3. The "not yet verified in-game" list is in `07-assetto-corsa-bridge.md` |
| 8.3 ⚠️ open | Dry-run mode uses the null back-end, so yaw comes from velocity/heading instead of the Lua app | The CSP back-end's watchdog could brake the car while the owner drives | Acceptable for a dry run; revisit after the first in-game session |

## WP9 — Delegated work packages

| Package | Helper | Status |
|---|---|---|
| FS rules digest → `docs/fs-rules-dv-summary.md` | Opus | ✅ delivered |
| Public-data car files → `vehicles/library/`, `docs/vehicle-data-sources.md` | Sonnet | ✅ delivered; all three load, simulate and complete laps. Estimates are marked per line; Taycan gearing/motor split and 992 gear ratios are low-confidence |
| Live Assetto Corsa bridge → `mv/ac/bridge/`, `docs/07-…` | Opus | ✅ delivered after one interruption (usage limit) |
| Reference lap (g-g-v + racing line) → `mv/planning/`, `docs/08-…` | Fable | ✅ delivered after three interruptions (usage limit); 20 tests pass. Results and method in `08-reference-lap.md`; summary in WP11 |

Lesson: helpers share the 5-hour usage limit and are cut off without warning.
Give them exclusive files, ask them to save after every verified step, and
resume them with a message rather than starting over.

## WP10 — Generic Formula Student car ✅

**Objective.** Exercise the FS-scale path (small, light, high-downforce car
without ABS on 3–5 m wide, 200–500 m tracks) before real car data exists.

**Approach.** `vehicles/library/fs_generic_ev.yaml`: typical class values —
265 kg with driver, 80 kW rear motor (rules limit), wheelbase 1.55 m, ClA 3.2,
tyre μ 1.55, traction control but **no ABS**. Not any real team's car; every
value is an estimate and marked as such.

**Verification.**

| Check | Result | Expectation |
|---|---|---|
| 0–100 km/h | 2.72 s | FS EVs: 2.5–3.5 s ✅ |
| Top speed | 138 km/h (gearing) | ✅ |
| Cornering at 60 km/h | 1.59 g | 1.5–2.0 g ✅ |
| Baseline driver on 10 random FS-size tracks, 2 laps | 10/10 completed, 55–89 km/h average | ✅ runs; averages are high for FS (see 10.2) |
| Locked front wheels cannot steer (no ABS) | yaw gained under full braking < 60 % of that under gentle braking | ✅ `test_car_without_abs_loses_steering_when_brakes_lock` |

**Problems & resolutions**

| # | Problem | Root cause | Resolution |
|---|---|---|---|
| 10.1 | Full braking in a corner *spun* the car instead of making it run straight on | First guess of 66 % front brake bias: at 1.5–1.9 g about 72–75 % of the load is on the front axle, so the rear wheels locked first | Bias set to 75 % front. This is the classic brake-balance design rule (fronts must lock first) appearing by itself in the simulator — a good sign for the tyre model |
| 10.2 ⚠️ open | Generated "fs" tracks are faster and more flowing than real FS tracks, and have no cones | Same generator as for circuits, only scaled | Roadmap phase 6: cone-level generator with slaloms, hairpins (9 m min. diameter), ≤ 80 m straights, plus skidpad and acceleration layouts |

## WP11 — Physics reference lap ✅ (helper-built; not yet independently reviewed)

**Objective.** An upper yardstick: the lap time the car could theoretically
achieve on a track, from physics alone, to grade the learned driver.

**Approach** (`mv/planning`, details in `08-reference-lap.md`): g-g-v envelope
from a steady-state four-wheel model that uses the simulator's own tyre, aero,
load-transfer and powertrain equations → minimum-curvature racing line (QP) →
minimum-time refinement → forward–backward speed profile on the closed lap.
CLI: `scripts/reference_lap.py`; results in `runs/reference/`.

**Verification (reported by the helper; spot checks by the lead still to do).**
Envelope vs. `VehicleSim`: steady cornering limit within 0.1–0.3 % at
60–200 km/h for all three cars; 0–200 km/h 10.01 s vs 10.10 s; 200–0 km/h
112.3 m vs 113.7 m. 20 unit tests pass.

**Results — flying lap, 911 GT3 RS (AC import), flat 2-D tracks**

| Track | Rule-based baseline | Reference on the centre line | **Reference (best line)** | Baseline gap to reference |
|---|---|---|---|---|
| Imola | 2:26.6 | 2:12.4 | **1:52.0** | +31 % |
| Monza | 2:24.2 | 2:11.0 | **1:57.9** | +22 % |
| Spa | 3:17.0 | 2:57.0 | **2:35.8** | +26 % |
| Mugello | 2:33.0 | 2:16.4 | **2:02.7** | +25 % |

918 Spyder: 1:42.2 / 1:47.2 / 2:23.8 / 1:54.4. Taycan Turbo GT: 1:48.3 /
1:53.2 / 2:31.3 / 1:59.6 (same track order).

Reading: the racing line alone is worth 9–15 % over the centre line — the
learned driver must leave the centre line to be competitive. After 15 minutes
of training (verify4) it was at 2:12.1 on Imola: level with "perfect driving on
the centre line", 18 % off the reference.

**Problems & open points**

| # | Item | Status |
|---|---|---|
| 11.1 | The reference could not be *driven* in the simulator by a simple tracking controller (lost the car on corner entry: the reference trail-brakes with the rear axle at its limit) | ⚠️ open — the reference is probably 1–2 % optimistic on corner entry; treat "within 3 %" as the realistic target |
| 11.2 | Direct lap-time optimisation of the line (L-BFGS on a spline) failed: track-edge clipping makes the objective noisy | Replaced by a curvature QP re-weighted by measured lap-time sensitivity (gain over minimum curvature: 0.7–2.3 %) |
| 11.3 | Minimum-curvature QP diverged at Spa's La Source (track wider than the corner radius) | Line limited to 70 % of the local radius on the inside |
| 11.5 | Independent code review (Opus helper, 2026-11-17) | ✅ One bug fixed: at low speed the envelope ignored steering lock (it capped path curvature at 0.15 1/m instead of checking lock), so it allowed corners the car cannot steer — FS car skid-pad at 18 km/h: planner 1.36 g vs simulator 0.69 g (+98 %); after the fix 0.77 g. Road-car reference lap times unchanged (all corners above 48 km/h); FS references 0.02–0.58 % slower. All other checks agree with the simulator within ~0.5–1 % (skid-pad, 0–200, 200–0, shift model); stored laps reproduce to the millisecond. Verdict: road-car references 0.5–2 % optimistic; FS references ±1 %, unreliable below ~25 km/h (use `step_m=1.5` for FS tracks). 22 planning tests pass |
| 11.4 | Lead review of `mv/planning` code | ⚠️ partly done (superseded by 11.5). Integration checked: `scripts/evaluate.py` reads the reference files and prints the gap (rule-based driver +30.8 % at Imola, +22.3 % at Monza). Consistency cross-check against the independently trained RL driver (WP13): it never beats the reference (Mugello 2:07.9 vs 2:02.7, Spa 2:42.0 vs 2:35.8) but does beat the reference's *centre-line* time (2:07.9 vs 2:16.4), exactly as expected for a driver that uses a racing line. A line-by-line code review is still open |

## WP12 — One driver for several cars ✅ (mechanism only; no multi-car driver trained yet)

**Approach.** `mv/envs/multi_car.py`: several single-car environments side by
side behind one interface; the fleet is split evenly between the car models
and all share the track pool. `scripts/train.py --vehicle a,b,c` trains one
network on all of them; validation cycles through the cars; the warm start
uses one rule-based teacher per car.

**Verification.** `test_one_driver_can_train_on_several_cars` (3 library cars +
FS car, warm start, PPO update, track replacement) passes; a 3-car end-to-end
run of `train.py` on the CPU completes.

**Not done.** No real multi-car training run yet — roadmap phase 4.

## WP13 — First long training run 🕓 (started 2026-11-17, running detached)

```
scripts\train.py --vehicle ks_porsche_911_gt3_rs --name gt3rs_v1 --envs 16384 --tracks 512 --warm-start --minibatches 8 --eval-every 100 --hours 3
```

Output: `runs/gt3rs_v1.log` (screen output), `runs/gt3rs_v1/` (checkpoints,
`log.csv`, TensorBoard). Process id in `runs/gt3rs_v1.pid`. To stop it early
without losing the driver: nothing special is needed — `latest.pt` is written
every 25 updates and `best.pt` at every validation improvement; end the Python
process from Task Manager, then `--resume` later.

| Milestone | Value |
|---|---|
| Straight after warm start (no PPO yet) | **9 of 10 validation tracks completed** |
| Update 40 (39 M decisions) | pace 124.9 km/h, 13.7 crashes per 100 km (was 977 at update 1), ~96 000 decisions/s including learning |

**Result (finished after 3 h: 3 000 updates, 1.59 billion decisions, ~145 000 decisions/s including learning).**

Training fleet: pace 103 → 161 km/h along the track; crashes 977 → 3–6 per 100 km;
exploration noise 0.25 → 0.03.

Validation (deterministic driver, standing start, 2 laps; flying lap shown):

| Track | Update 1 (cloned) | Update 1 000 | Update 2 000 | Update 3 000 | Physics reference | Gap at 3 000 |
|---|---|---|---|---|---|---|
| Mugello | 2:31.9 | off @ 785 m | 2:09.4 | **2:07.9** | 2:02.7 | **+4.3 %** |
| Spa | 3:12.2 | off @ 414 m | 2:46.4 | **2:42.0** | 2:35.8 | **+4.0 %** |
| Imola | 2:20.3 | off @ 1 388 m | off @ 3 364 m | off @ 3 364 m | 1:52.0 | — |
| Monza | 2:23.1 | off @ 943 m | off @ 944 m | off @ 943 m | 1:57.9 | — |
| 6 unseen random | 5/6 | 5/6 | 6/6 | 5/6 (one off at 298 m) | — | lap times 8–14 % faster than at update 1 |

Reading:
- Where it finishes, the driver is **within ~4 % of the physics reference** after
  3 hours — the reference is itself 1–2 % optimistic (11.1), so the true gap is
  smaller. It is 16–18 % faster than the rule-based driver on the same tracks.
- **It never again completed Imola or Monza after update ~100.** It leaves the
  track at the same metre every time: Monza 943 m (first chicane, braking from
  245 km/h) and Imola 3 364 m. The speed trace
  (`runs/eval_gt3rs_v1/…monza.png`) shows braking starting ~180 m before the
  chicane at only ~1.1 g, while the car can brake at ~1.45 g. **Root cause
  (hypothesis, strongly supported):** the random generator almost never builds
  "long straight → tight chicane/corner", so the driver had nearly no practice
  at braking from top speed into a slow corner (item 3.3). Fixed by WP14.
- Validation results fluctuate between evaluations (one random track flips
  between finished and off at ~300 m); the driver is near its limit there.
- `best.pt` of this run is the update-1 checkpoint: the original selection
  score rewarded completion only. Selection rule changed (D-020); every
  validation now also saves `eval_<update>.pt`.

## WP14 — Segment-based track generator ✅

**Objective.** Close the gap found in WP13: circuits with long straights that
end in slow corners, hairpins and real chicanes.

**Approach.** `mv/track/segments.py`: draw a sequence of features (straights,
corners, hairpins, chicanes; 30 % of features preceded by a 400–1 000 m
"heavy braking" straight), scale the turning to exactly 360°, write the
curvature with softened corner entries, then **close the loop exactly** by
changing only straight lengths (bounded least squares: straights may grow, never
shrink below a minimum). Presets `gp` (circuits) and `fs_segments` (FS scale:
width 3–4.5 m, straights ≤ 78 m, hairpins ≥ 4.6 m radius, lap ≤ 600 m). Presets
can be mixed: `--preset mixed+gp` picks one generator per circuit.

**Verification.**

| Check | Result |
|---|---|
| 40 `gp` circuits | 3.8 s; length 1.2–5.5 km; tightest corner median 16.5 m; longest straight median 749 m, corner after it median radius **27 m** |
| 20 `fs_segments` circuits | 0.2 s; 307–578 m; longest straight median 65 m, corner after it median radius 6 m |
| Closure, spacing, radius, self-clearance, braking zones | `test_segment_tracks_are_closed_valid_and_have_braking_zones` |
| Picture | `docs/img/tracks_segments.png` |

**Problems & resolutions**

| # | Problem | Root cause | Resolution |
|---|---|---|---|
| 14.1 | No circuit closed (400/400 rejected) | Minimum-norm length correction needed negative straight lengths: a random lap misses its start by ~1.8 km | Bounded least squares (straights only lengthen, never below minimum) |
| 14.2 | Closure needed many passes / laps of 9.7 km | Straight direction read at the straight's start, still bent by the corner exit; too many long straights | Direction read at the middle of each straight; feature count and long-straight lengths reduced; laps > 7.5 km rejected |

## WP15 — Fine-tuning on the new circuits 🕓 (running)

```
scripts\train.py --vehicle ks_porsche_911_gt3_rs --name gt3rs_v2 --init-from runs\gt3rs_v1\latest.pt --preset mixed+gp --val-preset mixed --envs 16384 --tracks 512 --minibatches 8 --eval-every 100 --hours 2.5
```

Same validation set as WP13 (so results compare directly). Question under
test: does practice on heavy braking zones fix Imola and Monza without losing
Mugello/Spa pace? Output `runs/gt3rs_v2.log`, `runs/gt3rs_v2/`.

First data point (update 100, 52 M decisions, ~114 000 decisions/s): all 6
random validation circuits completed (5/6 before), Mugello 2:07.5, Spa 2:41.4
(both marginally faster), **Imola and Monza still off at the same metre
(3 368 m, 946 m)**. Training stable (KL ≈ 0.012, noise 0.03, 6.8 crashes per
100 km on the harder track mix). Too early to judge; the deciding numbers are
the later validation blocks in `runs/gt3rs_v2.log`.

If Imola/Monza still fail by the end: the likely reason is that exploration
noise is already at its floor (0.03), so the driver hardly tries braking
earlier. Next experiment would be the same fine-tune with the noise reset to
~0.1 (`PPOConfig.min_std` / `log_std`), logged as a separate run.

**Final (stopped at update ~650 / 341 M decisions — question answered).**
Imola and Monza still failed at the same metre in every validation; random
circuits 6/6, Mugello 2:06.8, Spa 2:41.2. `gt3rs_v3_noise` (more exploration)
was stopped at update ~500 with the same result. **Neither more track variety
nor more exploration fixed it → the cause had to be elsewhere (WP20).**

A second, parallel experiment tests the competing explanation (too little
exploration): `gt3rs_v3_noise` — identical, but exploration noise reset to 0.12
with a floor of 0.06 (`--reset-noise 0.12 --min-noise 0.06`). Running two
trainings at once works: with one run the GPU was only ~40 % busy (the loop is
limited by Python overhead), with two it is ~100 % busy and the first run kept
its ~120 000 decisions/s.

## WP16 — Robustness of the 3-hour driver ⚠️ (important findings)

`runs/gt3rs_v1/latest.pt`, Mugello + Spa + 4 unseen random circuits, 2 laps:

| Condition | Completed | Notes |
|---|---|---|
| Normal | 5/6 | Mugello +4.2 %, Spa +4.0 % vs reference; one short random circuit off at 598 m |
| Blind: track preview cut to 40 m (lidar unchanged, 150 m) | 3/6 | still ~220 km/h where it can only see 40 m — it has not learned "slower when you cannot see" |
| Grip × 0.88, driver not told | **1/6** | drives at the limit of *nominal* grip; cannot notice the car responding less |
| Track-edge use (decisions with any body part over the edge, per run) | 59–99 on 4 of 5 completed tracks | the driver uses the edge routinely; the penalty is too weak relative to the time gained |

Reading:
1. **Grip:** training already randomised grip between ×0.88 and ×1.05, yet
   the driver learned to drive the *average*. With only the present instant to
   look at, it has no way to tell a slippery day from a normal one. Fix: give
   it a short memory of how the car responded (WP17) — this is also the basis
   for "better lap after lap".
2. **Blind laps:** the "blind" training episodes cut the preview but left the
   150 m lidar, which still shows the road far ahead — so blind training was
   never really blind, while a real FS car perceives cones to ~20–25 m. Fix
   planned: cut the lidar range together with the preview in blind episodes.
3. **Edge use:** acceptable for Assetto Corsa (kerbs), not for Formula Student
   (cones). Make the edge penalty a per-target setting (open item 6.6).

## WP17 — Memory and truly blind laps ✅ (built and tested; training run queued)

**Approach.**
- **Memory** (`EnvConfig.history_steps`, `train.py --history N`): the driver
  also receives the car-state block and its own actions at N earlier moments,
  `history_every` decisions apart (default 3 → 0.12 s; N = 8 covers ~1 s). Off
  by default so existing drivers keep working. Appended at the *end* of the
  observation.
- **Starting a memory driver from an existing one** (`--init-from` with a
  larger observation): the first layer gets zero weights for the new inputs,
  so the new driver initially drives *exactly* like the old one and learns to
  use the memory from there. Measured: identical actions and values (difference 0.0).
- **Truly blind episodes** (`EnvConfig.blind_lidar`, default on): when the
  preview is cut to h metres, the lidar range is cut to h as well.
  `env.set_view(h)` does both (used by `evaluate.py --blind` and the in-game
  adapter). Note: `--blind` results from before this change are not
  comparable with later ones.
- `train.py --grip-range LOW HIGH` widens the hidden grip variation so that
  sensing grip pays off.

**Verification.** `test_memory_holds_past_car_states` (sample *j* equals the
state *j·every* decisions earlier), `test_blind_view_limits_preview_and_lidar`,
and the in-game adapter reproduces training observations with memory on
(`test_adapter_reproduces_training_observation[...-memory]`). All tests pass.

## WP18 — Unattended training queue 🕓

`scripts/queue_runs.py <queue.json>` runs training jobs one after another,
optionally after a given process has ended. Two lanes run in parallel on the
one GPU (see WP15 note on GPU use):

| Lane | Now | Then | Then |
|---|---|---|---|
| 1 (`runs/queue_lane1.json`) | `gt3rs_v2` (fine-tune on mixed+gp) | `gt3rs_v4_memory`: from v2, `--history 8 --grip-range 0.82 1.08 --reset-noise 0.1`, 2.5 h | `multi_v1`: one driver for 911 + 918 + Taycan, from v2, 3 h |
| 2 (`runs/queue_lane2.json`) | `gt3rs_v3_noise` (same as v2, more exploration) | `fs_v1`: generic FS car on `fs+fs_segments` tracks, warm start, 2.5 h | — |

Queue progress: `runs/queue_lane*.queue.log`; each job's screen output:
`runs/<name>.log`.

Questions each run answers:
- v2 vs v3: is the Imola/Monza failure fixed by the new tracks alone, or does it need more exploration?
- v4: does memory + wider grip variation make the driver robust to an unknown grip level (WP16: 1/6 at grip ×0.88)?
- multi_v1: can one network drive a combustion, a hybrid and an electric car?
- fs_v1: first driver at Formula Student scale.

## Next steps (in order)

1. **Read the WP15 result** (`runs/gt3rs_v2.log`, validation blocks). If Imola
   and Monza now complete, evaluate fully:
   `scripts\evaluate.py --vehicle ks_porsche_911_gt3_rs --policy runs\gt3rs_v2\best.pt --scripted`,
   then `--blind` and `--grip 0.88`; append to WP15. If not, look at the speed
   trace at the failure point and log what it shows.
2. **Lead review of `mv/planning`** (11.4) and a check of 11.1.
3. **First in-game session** (needs the owner): `docs/07-assetto-corsa-bridge.md`
   checklist → `scripts/ac_bridge_check.py` → `scripts/ac_drive.py --scripted
   --dry-run` → scripted driver with `--max-kmh 80` → trained driver
   (`--policy runs\gt3rs_v2\best.pt`).
4. **System identification** against AC telemetry and tuning of the importer
   (roadmap phase 2, step 4).
5. Hyper-parameter study, one change at a time (7.3). Candidates suggested by
   WP13: exploration noise floor (0.03 reached by update ~2 000 — possibly too
   little exploration to escape the late-braking habit), larger network.
6. Open items flagged ⚠️ above: 3.4 (flat tracks), 5.3/5.4 (tier-2 physics,
   trail-braking balance), 6.6 (track-limit rule), 8.2/8.3 (in-game
   verification), 10.2 (FS cone tracks — `fs_segments` is the starting point).

## WP19 — System identification (simulator ↔ Assetto Corsa) ✅ (tested on synthetic data; no game data yet)

**Objective.** Measure how closely the simulator reproduces Assetto Corsa and
correct the vehicle file where it does not (roadmap phase 2, step 4) — built
now so the first in-game session already produces useful data.

**Approach.**
- **Record** (`scripts/ac_drive.py --record file.npz`): every 40 ms tick, the
  car's motion (vx, vy, yaw rate, accelerations smoothed like the simulator's)
  and the inputs it actually received. Works in `--dry-run`, i.e. **recording
  the owner's own laps**.
- **Replay** (`mv/sysid/replay.py`): cut the recording into 1.5 s slices; start
  the simulator in each slice's measured state and feed the recorded inputs
  (mid-tick values); compare. Restarting every slice keeps small errors from
  compounding. All slices run as one batch (~0.5 s per evaluation on the CPU).
- **Fit** (`scripts/sysid_fit.py`): six correction factors — overall grip,
  front/rear grip balance, downforce, drag, power, brakes — by a global search
  (differential evolution) followed by a coordinate polish. Afterwards each
  factor's **confidence** is measured (does ±5 % clearly worsen the match?);
  only well-determined factors are written to `vehicles/local/<car>_tuned.yaml`.

**Verification (self-test, `sysid_fit.py --synthetic`: a deliberately altered
911 recorded in the simulator, fit from the unaltered file).**

| Factor | Found | True | Determined? |
|---|---|---|---|
| Overall grip | 0.959 | 0.960 | yes |
| Grip balance | 1.100 | 1.087 | yes |
| Power | 0.985 | 1.000 | yes |
| Brakes | 0.913 | 0.900 | yes |
| Downforce | 1.373 | 1.000 | **no** → not applied |
| Drag | 0.886 | 1.000 | **no** → not applied |

Typical slice error: yaw rate 0.016 → 0.008 rad/s, lateral acceleration
0.34 → 0.15 m/s². Fast unit tests: `tests/test_sysid.py`.

**Problems & resolutions** (in the order they appeared)

| # | Problem | Root cause | Resolution |
|---|---|---|---|
| 19.1 | Even with the true parameters the replay missed by ±0.3 m/s² | A recorded spin dominated the error, and inputs were replayed half a tick late | Cauchy-type robust error per slice; inputs replayed at mid-tick (average of both samples) — error floor halved |
| 19.2 | Nelder-Mead stalled near the start | SciPy's default initial step for a zero start is 0.025 %, on a slightly jagged surface (gear shifts) | Replaced (see 19.3) |
| 19.3 | Robust least squares and coordinate search landed in wrong valleys | Factors partly stand in for each other: more front grip ≈ less rear grip (balance); less grip ≈ less downforce | Grip re-expressed as *level* and *balance*; global search (differential evolution) before polishing. 1-D scans had confirmed every factor has a clear minimum at the true value |
| 19.4 | Aero factors not recovered | The cautious recording driver rarely goes fast; downforce/drag barely affect the data | Accepted and made visible: confidence check, undetermined factors are not applied. To identify aero: record fast laps (high-speed corners, long straights) |
| 19.5 | **Dry-run mode in the game would have run at ~300 Hz instead of 25 Hz** (found while adding recording) | `session.observe()` returns at the next physics packet (~3 ms) | `drive()` paces dry-run ticks itself |

**Next.** First real use: in the first in-game session, record 3 laps of your
own driving at Imola and Monza (`ac_drive.py --dry-run --record`), then
`sysid_fit.py`. Extend the factor set if the residuals show a pattern (e.g.
roll-stiffness balance, tyre peak slip angle).

## WP20 — Track-preview aliasing: the real cause of the Imola/Monza failures ✅

**Finding.** A close trace of the Monza failure showed the driver braking hard
and continuously from 813 m (≈ 1.2 g), yet arriving at the 12 m-radius
chicane at 85 km/h where ~45–55 km/h is possible. It was not refusing to
brake — it **could not reliably see the corner**.

**Root cause.** The track preview sampled curvature at 23 *points*; far ahead
the points are 20–45 m apart. Monza's first chicane is ~15 m long. Depending
on where the points happened to fall, the radius the driver saw flickered
between 10.6 m and 23 m (and at 700 m the chicane was missed entirely: "70 m")
— exactly in the braking zone, where the decision is made. A corner at 23 m
allows ~40 % more speed than one at 12 m.

**Fix.** Each preview value now reports the **tightest curvature (and the
narrowest width) anywhere in its stretch** of road, scanned every ~1 m.
After the fix the preview shows the chicane steadily at its true 10.2 m from
700 m onward. Cost: moderate (129 000 decisions/s with 16 384 cars measured
while two other trainings shared the GPU).

**Verification — the v2 driver with the corrected preview, no retraining:**

| Track | Before | After (zero retraining) | Reference | Gap |
|---|---|---|---|---|
| Monza | off at 944 m, every time | **2:03.3** | 1:57.9 | +4.5 % |
| Imola | off at 3 364 m, every time | **1:59.3** | 1:52.0 | +6.5 % |
| Mugello | 2:07.9 | 2:09.4 | 2:02.7 | +5.5 % |
| Spa | 2:42.0 | 2:45.0 | 2:35.8 | +5.9 % |

All four real circuits completed for the first time since the cloned driver.
Slightly more cautious on Mugello/Spa (it now sees tight corners it used to
miss); fine-tuning should recover that.

**Lesson.** Before blaming learning, check what the driver can *see*. Two
experiments (v2, v3; ~3 GPU-hours) chased the wrong hypothesis; one trace of
the failure plus one line of arithmetic found the cause.

**Consequence.** The observation's *meaning* changed (not its size): drivers
trained before this change still load and still drive, but should be
fine-tuned. Running now: `gt3rs_v5_preview` (v2 + corrected preview only — the
clean test of this fix) and `gt3rs_v4_memory` (corrected preview + memory +
wider grip variation).

## WP21 — Limit-seeking training ✅ (built and tested; first run queued)

**Objective.** The owner's guiding principle (architecture §1b): *find the
limit, sometimes overstep it, adjust incrementally lap after lap; sliding is
fine; only lap time matters.* Two things in the training set-up worked against
it: (1) leaving the track **ended** the episode, so the driver never got its
"try again"; (2) episodes lasted 60 s — less than one lap of Imola — and the
driver had no memory across laps.

**Approach** (`EnvConfig`, switched on together by `train.py --limit-seeking`):
- **Recover and continue** (`recover_on_crash`): leaving the track, spinning or
  stalling costs the crash penalty, then the car is put back on the centre line
  at the same place, pointing the right way, at 40 % of the feasible speed
  there (battery charge kept). The time lost is the natural cost. After 6 offs
  the episode does end.
- **Lap memory** (`lap_memory`): the track is divided into 5 m sections; for
  every section the car remembers its slowest speed and its highest *grip use*
  last time through (1.0 = at the limit; 3.0 = went off there). Grip use is the
  larger of slip angle ÷ slip angle of peak grip and total acceleration ÷
  nominal grip — both measurable on a real car. For each of the 23 preview
  stretches the driver sees: last lap's minimum speed, maximum grip use, and
  whether anything is known (69 values, appended). This is exactly the
  information needed to push where there was margin and back off where it
  overstepped.
- **Multi-lap episodes**: 240 s.
- `--init-from` grows the network from a driver without these inputs (zero
  weights for the new ones), so experience is kept.
- Unchanged: no penalty on sliding, slip or tyre use — the reward is progress
  per time.

**Verification.** `test_recover_on_crash_puts_the_car_back_and_continues`
(two rejoins, back on the centre line pointing the right way, third off ends
the episode); `test_lap_memory_remembers_last_lap` (nothing known before the
first lap; afterwards every section ahead is known and the remembered speed
matches the speed driven). All 125 tests pass. CPU smoke run of
`train.py --limit-seeking` grown from the memory driver (218 → 287 inputs)
completes and validates.

**Queued:** `gt3rs_limit_v1` — from `gt3rs_v4_memory`, `--history 8
--limit-seeking --grip-range 0.82 1.08`, 3 h (lane 1, after v4). Evaluation
with ≥3 laps will show whether lap times fall from lap 2 to lap 3 — the direct
measure of "better each lap".

## WP22 — GitHub (private repository) ✅ published 2026-11-17: https://github.com/raduedu/Project-MV

The owner asked for the project to be kept on their GitHub as a private
repository "Project MV". GitHub does not allow spaces, so it will be
**`Project-MV`**. The GitHub CLI is not installed on the laptop and creating
the repository needs a one-time browser login, so it was prepared rather than
done: `scripts/publish_github.ps1` checks for the CLI (prints the install
command, installs nothing itself), logs in, refuses to publish if any Assetto
Corsa data, runs or checkpoints are tracked, then creates the **private**
repository and pushes (later runs: just pushes).

**Done (2026-11-17, with the owner).** GitHub CLI 2.104.0 installed with winget
(the owner approved the Windows administrator prompt); one-time device login
approved by the owner (account `raduedu`, scope: repositories); repository
created **private**. Verified on GitHub's side: visibility PRIVATE; remote
`main` = local HEAD; 11/11 commits and 84/84 files; 0 Assetto Corsa files, runs
or checkpoints.

| # | Problem | Root cause | Resolution |
|---|---|---|---|
| 22.1 | First publish attempt stopped at "repository not found" | With `$ErrorActionPreference = "Stop"`, Windows PowerShell 5.1 turns a native program's error message into a fatal error — and "not found" is the expected answer the first time | `try/catch` around the existence and login checks |

**Keeping it up to date:** commit, then
`powershell -ExecutionPolicy Bypass -File scripts\publish_github.ps1` (it
pushes; the safety check runs every time).

**Authorship (2026-11-17, owner's request).** All commit messages were rewritten
without the co-author lines (code unchanged, verified tree-identical) and the
history on GitHub replaced (force-push with lease). GitHub now lists `raduedu`
as sole author and contributor. Old history kept locally only, in branch
`backup/before-coauthor-removal`. From now on no attribution lines are added
(rule in `CLAUDE.md`).

## WP23 — Pause and resume ✅

**Objective (owner's request).** Be able to pause all training — e.g. to take
the laptop to university — and continue later without losing work.

**Approach.**
- `train.py` checks for `runs/PAUSE` after every update; if present it saves
  `latest.pt` and exits with code 3. On `--resume` the time budget counts the
  whole run (hours already trained are read from `log.csv`), `--reset-noise`
  is not applied again, and (new) the optimiser's momentum is restored from the
  checkpoint. A normally finished run writes `runs/<name>/FINISHED`.
- `scripts/train_plan.py` replaces `queue_runs.py`: the plan
  (`runs/plan.json`: lanes of jobs) plus the markers in `runs/` are the whole
  state, so `resume` always knows what is done, what to continue and what is
  next. `pause` creates the flag, waits for every trainer to save, stops any
  that do not react, and stops the supervisor. Double-click wrappers:
  `pause_training.cmd`, `resume_training.cmd`, `status_training.cmd`.

**Verification (live, on the two running trainings).**

| Test | Result |
|---|---|
| Hand-over: controller adopts the two already-running jobs | ✅ "already running; waiting for it" |
| Pause of runs started *before* the mechanism existed | ✅ stopped after the 4-min wait; v5 resumed at update 775 (lost 24 updates ≈ 1 min), v4 lost none |
| Resume | ✅ both continued from `latest.pt`; "already trained 1.49 h / 1.47 h of this run" |
| Pause of pause-aware runs | ✅ **10 s**, saved at the exact update (761, 788), nothing lost |
| Resume again | ✅ |

**Problems & resolutions**

| # | Problem | Resolution |
|---|---|---|
| 23.1 | First update after a resume changed the policy ~6× more than the target (KL 0.078 vs 0.012) | Optimiser state was not saved → now saved/restored (effective from the next save) |
| 23.2 | Resuming would have restarted the `--hours` budget and re-applied `--reset-noise` | Budget counts the whole run; noise reset only at the start |
| 23.3 | The old `queue_runs.py` processes were parents of a running training | Stopped individually (not as a process tree) so the training survived the hand-over |

## WP24 — Control panel (web UI) and terminal command ✅

**Objective (owner's request).** Control and follow training without asking the
assistant: a terminal command with progress bars, and a simple local web
interface that keeps running when the browser tab is closed.

**Approach.**
- `scripts/train_plan.py`: one `snapshot()` (plan, states, progress, latest
  validation) feeds the terminal views and the web UI; new `watch` command
  (refreshes every 5 s, bars, minutes left, decisions/s, pace, last lap times).
  The supervisor now **re-reads the plan** and starts jobs/lanes added while
  it runs; when everything is done it idles, waiting for new jobs.
- `mvctl.cmd` (project folder): `status | watch | pause | resume | ui | ui-stop | tests | help`.
- `scripts/webui.py` + `mv/ui/index.html`: Python standard library only, one
  HTML file, no external libraries; listens on 127.0.0.1:8765 only. Tabs:
  Dashboard (live lanes, progress, metrics, last validation, Pause/Resume),
  Plan (add job with car(s), tracks, hours, start-from, memory, limit-seeking,
  grip range; reorder; remove), Import (all Assetto Corsa cars and track
  layouts, searchable), Tools (evaluate any checkpoint with plots, vehicle
  check, reference lap — run as background tasks with live output), Runs
  (smoothed pace and crash curves, validation lap times of the real circuits
  over training, log tail, files, settings).

**Verification (in the built-in browser, against the live trainings).**

| Test | Result |
|---|---|
| All read endpoints | ✅ (state ~1.4 s because Windows is asked for the process list; cached 3 s) |
| Import a track (Red Bull Ring GP) and a car (911 GT3 Cup 2017) from the page | ✅ both imported |
| Vehicle check from the page | ✅ live output |
| Pause from the page | ✅ PAUSING → PAUSED, both jobs "resumable" |
| Resume from the page | ✅ RUNNING |
| **Stop the web server after resuming from it** | ✅ supervisor and both trainings kept running |
| New supervisor swapped in without interrupting training | ✅ adopted both running jobs |

**Problems & resolutions**

| # | Problem | Resolution |
|---|---|---|
| 24.1 | Progress bars crashed when output went to a pipe (Windows code page 1252) | Output forced to UTF-8 |
| 24.2 | Two tasks started in the same second shared one ID and one log | Millisecond suffix + collision check |
| 24.3 | **Stopping the web server would have stopped training** if Resume had been pressed in the page (the supervisor is then a child of the server, and stop killed the whole process tree) | Stop ends only the server process; verified by test |
| 24.4 | Charts noisy; crash axis went below zero | Running-average line over the raw data; zero floor |

## WP25 — Recorded laps, replay viewer, strict evaluation ✅

**Objective (owner).** Make results visual and verifiable: record fastest laps
and let anyone watch them.

**Approach.** `mv/laps.py`: full lap telemetry (25 Hz: position, heading, speed,
steering, throttle, brake, gear, distance) + provenance, saved in `runs/laps`;
physics reference laps converted to the same format ("ideal ghost").
Training validation saves each run's record laps automatically;
`scripts/evaluate.py` saves its laps. Web UI: **Laps** tab (track map with
speed-coloured racing line, chase camera, live HUD, ghost with live gap, speed
trace, CSV download, provenance) and a **Fastest laps** table on the dashboard.
`scripts/make_example_laps.py` publishes example laps (`docs/laps`) and the
README animation — generated circuits and public car files only.
`scripts/record_reference_laps.py` adds reference laps for any car.

**Findings (important).**

| # | Finding | Action |
|---|---|---|
| 25.1 | Evaluations of the limit-seeking driver let it rejoin after going off; strictly it completed only **2/5 real** and 7/12 generated circuits (v4/v5: 5/5 and 12/12) | `eval_config` is now strict (`recover_on_crash=False`); its 5 non-strict laps moved to `runs/laps_retracted_nonstrict` |
| 25.2 | Single-car drivers fail completely on another car file (v4/v5 with the public 992: **0/6**), most likely because the car-descriptor inputs never varied in training | Multi-car training (`multi_v1`, paused at 88 %); randomise descriptor-related parameters next |
| 25.3 | Clean best laps (v5, strict): Imola 1:58.1 (+5.4 %), RBR 1:42.3 (+3.9 %), Monza 2:02.9 (+4.3 %), Mugello 2:08.5 (+4.8 %), Spa 2:42.5 (+4.3 %) vs physics ideal; FS car 8/8 generated FS circuits | Recorded as replayable laps |

**Plain-language overview for presenting the project:** `docs/00-project-explained.md`.

## WP26 — Exact reproduction of every lap ✅

**Objective.** Any recorded lap can be driven again from its saved driver and
must give the same time — the strongest evidence that a lap time is real.

**Approach.** `scripts/reproduce_lap.py` (also the *Reproduce* button in the
viewer): loads the lap's checkpoint, car file and track (imported, or the
generator recipe now stored in the lap file), drives it again, compares.

**Problems & resolutions**

| # | Problem | Root cause | Resolution |
|---|---|---|---|
| 26.1 | Reproductions differed by 0.08–0.16 s (0.06–0.13 %) | The neural network was evaluated for all cars in one batch; the maths library rounds a matrix product slightly differently depending on the batch size, and the difference grows over a lap (the driving is chaotic). Proof: Mugello driven inside the original 5-track batch gave exactly 128.501 s, alone 128.581 s | Evaluation computes the network for each car separately (`_act_per_car`); verified identical lap times alone, in a batch of 2 and of 5 |
| 26.2 | Lap times were quantised to one decision step (0.04 s) | The lap counter fired at the first step past the line | The crossing time is interpolated within the step (`RacingEnv.advance_laps`) - lap times now accurate to milliseconds |

Older evaluation laps (batched, step-quantised) were moved to
`runs/laps_superseded` and re-recorded; laps saved by training before this
change still reproduce "within numerical noise" (the script says so).

## WP27 — Steering smoothness ✅ (mild learned smoothing kept)

**Observation (owner).** The driver's steering is jittery and probably costs
time. Measured "steering activity" (total wheel travel per second, in
fractions of full lock): **1.0–1.3** for the best driver vs ~0.01–0.02 for the
physics-ideal lap (whose steering is derived from the line's curvature, so it
is extra smooth).

**Quick check (driver NOT trained for it, so only a hint).** A low-pass filter
on its steering command: 0.03 s → activity -35 %, lap times unchanged
(±0.1 s); 0.06 s → activity -75 %, laps 0.1–0.4 s *slower* (the filter only
adds lag). → Heavy smoothing would make the driver timid, exactly like the
over-strict track-limit penalty earlier; the owner asked for it to stay on the
edge. The fix must be *learned*, and mild.

**Approach.** Two settings (`EnvConfig`, `train.py`): `--steer-filter S`
(time constant of a steering damper, also applied identically in Assetto
Corsa by `ActuatorModel`) and `--steer-penalty X` (penalty on the squared
change of the steering command). Lap files and the viewer now show steering
activity and a steering trace next to the ideal lap's.

**Experiment.** First started on the single-car 911 driver (`gt3rs_v6_*`),
then — at the owner's question "why not the multi-car one?" — stopped after
12 minutes and moved to the multi-car driver, because the goal is *any* car and
a single-car driver cannot drive other car files (25.2): running (2.5 h each,
from `multi_v1`, 911 + 918 + Taycan): `multi_v2_steer_light` (filter 0.03 s +
penalty 0.2) vs `multi_v2_steer_penalty` (penalty 0.5, no filter = no lag).
Judged by strict lap times first, steering activity second.

**Results (2026-11-23).** Steering activity halved or better, and the *light*
variant is the fastest multi-car driver so far:

| Driver (strict, 5 real circuits) | Completed | 911 gap | 918 gap | Taycan gap | Steering activity |
|---|---|---|---|---|---|
| `multi_v1` (no smoothing) | 10/15 | +4.3 % | +5.8 % | +6.1 % | 1.2 /s |
| `multi_v2_steer_light` (filter 0.03 s + penalty 0.2) | 12/15 | **+4.0 %** | +5.9 % | **+4.8 %** | 0.3-0.6 /s |
| `multi_v2_steer_penalty` (penalty 0.5) | 14/15 | +4.2 % | +7.0 % | +5.1 % | 0.3-0.6 /s |
| `multi_v3_cars` (car variation, penalty 0.3) | **15/15** | +4.4 % | +7.3 % | +5.8 % | 0.4-0.8 /s |

*Gap = mean over the circuits that car completed, against the physics
reference lap. Steering activity = wheel travel per second in fractions of
full lock (the reference lap: ~0.02 /s).*

**Interpretation.** Mild, learned smoothing does not cost lap time - it gained
0.3-1.3 %. What costs time is *reliability*: the drivers that finish more
circuits are 1-2 % slower where they finish (the 918 most). The owner's
impression "smooth but slower" was the 15/15 car-variation driver, which is
slower because it also learned to drive car files it has never seen (WP29),
not because it is smooth. → Keep the light setting (filter 0.03 s + penalty
0.2) for the next runs (WP30).

## WP28 — Shared replay viewer and published replay page ✅

* `mv/ui/replay.js`: one viewer for the control panel and the published page.
  New: line coloured by **time won/lost vs the ghost**, **sector times**,
  **steering trace** with steering activity, gap-to-ghost chart, space bar,
  links that open a lap directly (`#lap=...&ghost=...`), Reproduce button.
* `docs/index.html` (+ `docs/replay.js`, `docs/laps/index.json`): a stand-alone
  replay page for GitHub Pages - generated circuits and public-data cars only.
* `scripts/make_example_laps.py`: examples tried on listed circuits in order,
  first clean lap used, all attempts printed. Current examples: Taycan Turbo GT
  (multi-car driver) on a generated GP-style circuit, +10.3 % off ideal; the
  generic FS car on two generated FS circuits, +4.5 % and +1.7 %.
* README animation: track map + chase camera + dashboard (`--gif-only`
  re-renders it).
* `CLAUDE.md` (assistant working notes) is no longer published (owner's
  request): untracked, git-ignored and blocked by the publish safety check.

## WP29 — Car variation: learning to read the car ✅

**Problem (25.2).** A driver trained on one car file fails on another (0/6 for
the 911 drivers with the public 992; the multi-car driver: 1/6, and its one lap
at Mugello took 3:29 instead of ~2:08). Cause: the car-description inputs never
changed during training, so the driver never had to *use* them.

**Approach.** `EnvConfig.vary_car` / `train.py --vary-cars`: every episode each
car's mass (x0.85-1.20), power (x0.70-1.40), downforce (x0.5-1.6), drag
(x0.80-1.25) and tyre grip (x0.90-1.15) are drawn anew, the physics uses them
(new per-car downforce and drag factors in `VehicleSim`), and **the car
description in the observation is updated to match**. The hidden "conditions"
variation stays on top. Evaluation always uses the true car file.
`--holdout a,b`: validation also drives car files the driver never trained on
and prints how many tracks it completes - "works on an unseen car" becomes a
measured number.

**Verification.** `test_car_variation_changes_the_car_and_tells_the_driver`:
description = file x factors; acceleration at 45 m/s correlates with the
power-to-mass factor; evaluation uses the true car. (First version tested at
15 m/s and failed: there the car is grip-limited, so power cannot show - a
test-design error, not a code error.) 126 tests pass.

**Queued (after the steering A/B):**
* `multi_v3_cars` - 911 + 918 + Taycan, car variation, held-out test cars:
  public-data 992 GT3 RS and 918 Spyder; mild steering penalty 0.3; 3 h.
  Baseline to beat: 1/6 on the held-out 992.
* `fs_v2_vary` - FS car with car variation, 2.5 h: the generic FS car is an
  estimate of a real team car, so robustness across FS-like cars is what will
  transfer to the real one.

**Results.** `multi_v3_cars` (3 h): held-out public-data 992 GT3 RS **11/11**
validation tracks (baseline 1/6), held-out public 918 11/11 at its best check
(8/11 at the last); on the three training cars it is the first driver to
complete **all 15 strict real-circuit laps** (table in WP27), at +4.4 / +7.3
/ +5.8 % to the reference. `fs_v2_vary` (2.5 h): 100 % of validation tracks,
1-2 crashes per 100 km in training. Car variation works; its price is
about 1 % of pace on the 918, which the next work package attacks.

## WP30 — A bigger driver, distilled from the current one 🕓 (built and smoke-tested; run queued)

**Objective (owner, 2026-11-23).** "Train it as good and hard as you think it
would be beneficial - a larger model is fine as long as it runs on this
laptop." The drivers are still 4-7 % behind the reference lap.

**Approach.**
* `train.py --hidden 1024,1024,512`: the network grows from 0.2 M to 1.7 M
  parameters (actor). Driving it costs ~1 ms per decision on the CPU - far
  inside the 40 ms per decision in Assetto Corsa.
* `train.py --teacher runs/multi_v3_cars/best.pt`: a bigger network cannot
  load the smaller one's weights, and training it from zero would throw away
  ~10 GPU-hours of learning. Instead it **copies** the trained driver first
  ("distillation", the DAgger warm start of WP8 with the trained network as
  teacher), then PPO continues.
* Same car variation, held-out cars and light steering smoothing as before;
  new realistic training circuits (WP31).

**Verification (smoke tests, 4 096 cars).**

| Step | Result |
|---|---|
| 1st version: one gradient step per decision, own input scaling, 40 steps | imitation error 0.105, **9 %** of validation tracks |
| Buffered distillation: copy the teacher's input scaling, keep the last 48 steps of states, 4 mini-batches per step, learning rate falling to 5 %, 600 steps | imitation error **0.0012** (90x smaller); in training 157 km/h and 18 crashes/100 km - the teacher's level |
| ...but the first PPO update | KL **450** (target 0.012), 0 % of validation tracks: the copy destroyed in one update |
| Learning rate starts at 1e-5 after a teacher warm start + KL stop (`--kl-stop 4`) | KL **0.033** in update 1; 911 completes 4 of 5 real circuits (Imola **1:57.4**, best so far 1:58.1), 2 of 6 of the new, harder `complex` validation circuits |

**Problems & resolutions.**
1. *Rough copy.* One gradient step per decision cannot copy a 1.7 M-parameter
   network, and a student that builds its own input scaling sees different
   numbers than the teacher. → `warm_start(..., buffer_steps, grad_steps)`
   keeps a replay buffer of states labelled by the teacher, several
   mini-batches per step, cosine-falling learning rate; the teacher's input
   scaling is copied and frozen.
2. *First PPO update destroys the copy.* Root cause: a fresh Adam optimiser
   moves *every* weight by the full learning rate in its first steps (its
   step is normalised per weight). In a 1024-wide layer a thousand such
   moves add up: the steering output shifted by ~0.9 of full lock. The
   learning-rate rule only reacts after the damage. → (a) after a teacher
   warm start the learning rate starts at 1e-5 and the KL rule raises it
   x1.5 per mini-batch as far as the network tolerates; (b) new
   `PPOConfig.kl_stop` (`--kl-stop`): an update ends early once it has
   changed the driver by more than `kl_stop x target_kl` - the standard PPO
   safeguard, previously missing. Test
   `test_distillation_copies_a_trained_driver_into_a_bigger_network`.

**Queued (2026-11-23, plan lanes 1 and 2).**
* `multi_v4_big` - 1.7 M parameters, distilled from `multi_v3_cars/best.pt`
  (800 steps), car variation, held-out public 992 and 918, light steering
  smoothing (filter 0.03 s + penalty 0.2), KL stop 4, circuits
  `complex+complex+gp`, validation `complex`; **8 h**.
* `multi_v4_ctrl` - identical but the usual size, continuing from the same
  driver; 4 h. The control separates "bigger network" from "more training
  on better circuits".

Judged strictly on the five real circuits for the three training cars, plus
the held-out cars. *Results to be appended.*

## WP31 — Realistic circuits: sections, chicanes, esses ✅ (generator); used from WP30 on

**Objective (owner, 2026-11-23).** "The track generator doesn't seem all
that complex; I need it to have chicanes and more, similar to a normal track,
rather than a NASCAR track."

**Measurement first** (`track_stats` script, 30 circuits per preset, against
the five imported real circuits; *winding* = total turning, left and right
counted alike, in full circles per lap):

| | Lap | Corners/km | Slow (<40 m) /km | Quick left-right changes /km | Winding | On straights |
|---|---|---|---|---|---|---|
| Real circuits (mean of 5) | 5.4 km | 2.7 | 0.84 | 0.79 | **3.45** | 55 % |
| `mixed` (spline/polygon) | 3.7 km | 3.7 | 0.97 | 1.37 | 1.96 | 42 % |
| `gp` (segments, WP13) | 3.2 km | 3.3 | 2.09 | 1.14 | 2.15 | 71 % |
| **`complex` (new)** | 4.3 km | 3.2 | 1.45 | 1.32 | **2.94** | 60 % |

The owner was right: our circuits "go round once" (winding ~2: a convex
shape turns through one circle plus wiggles), while real circuits fold back
on themselves (Red Bull Ring's infield, Mugello, Spa: 2.6-4.7). In `mixed`
the left-right changes are random wiggles, not designed corner complexes.

**Approach.** `SegmentGenConfig(layout="sections")`, preset `complex`
(module docstring of `mv/track/segments.py`): a lap is 5-9 *sections* -
fast (long straight into hairpin / chicane / slow corner), technical (2-4 of:
slow corner, medium corner, hairpin, chicane, double-apex, tightening
corner) or flowing (esses of 3-5 alternating corners, medium corner, fast
sweeper) - joined by straights that may carry a flat-out kink. Whole
features are mirrored left/right at random, then more until the turning can
add up to one circle, so chicanes and esses keep their shape. The old
presets are byte-for-byte unchanged (recorded laps still reproduce: the
`gp` row above is identical before and after).

**Problems & resolutions.**
1. *Chicanes became kinks.* Corner softening up to 15 m smeared a 20 m
   chicane, and the gap between its corners was an ordinary straight that
   the loop closure stretches to at least 15 m. → fixed gaps ("G" items) that
   the closure does not touch; softening 2-6 m in `complex`.
2. *First version: big triangles.* Closing the loop sometimes added
   kilometres to two or three straights, so the drawn layout disappeared
   into a triangle. → `max_stretch=0.35`: a layout whose closure would add
   more than 35 % of the lap is redrawn.
3. *First version: too many slow corners* (2.75/km vs 0.84 real) and laps
   too short (3.3 km). → medium-speed corners added to technical and flowing
   sections, 5-9 sections instead of 4-7, fewer fast sections. The final mix
   is still a little more technical than an average real circuit - on
   purpose, for training.

**Verification.** Tests: `complex` added to the closed / valid / braking-zone
test; new `test_complex_tracks_fold_back_like_real_circuits` (winding well
above `gp`, quick left-right changes on nearly every lap). Generation ~0.5 s
per circuit (vs ~0.1 s), acceptable: 51 new circuits every 10 updates.

**Next steps.** Train on `complex+complex+gp` (WP30); validation on
`complex`. Later: width changes along the lap, run-off/kerbs, elevation (the
simulator is planar).

## WP32 — Laps in Assetto Corsa: "View in Assetto Corsa" with ghost 🕓 (built and tested offline; first in-game test by the owner)

**Objective (owner, 2026-11-23).** Film the fastest laps in Assetto Corsa
with the ghost in front, ideally a "View in Assetto Corsa" button, to present
the project to people who do not know the concepts and to observe the
driver's behaviour better.

**Approach.** Reuse the bridge's mechanism (chapter 07): Python writes the
lap into shared memory (`mv/ac/viewer_link.py`, `Local\mv_viewer_v1`, 672 kB:
up to 12 000 samples = 8 min at 25 Hz of car t/x/y/heading/speed/steering/
pedals/gear and ghost t/x/y/heading/speed); a second CSP Lua app, **MV
Viewer**, reads it and replays the lap on the player's car
(`physics.setCarPosition` + velocity + overridden controls + gear) and shows
the ghost as a second car (`physics.setAICarPosition`, collisions and AI
input off) or as a drawn footprint, plus the ghost's racing line coloured by
speed and the time gap. Entry points: web viewer button (only shown for laps
on imported circuits; the web server keeps the mapping open), and
`scripts/ac_view.py` (waits until the game confirms the lap). Every CSP call
was checked against the SDK shipped with the installed CSP 0.2.11.

**Verification.** `tests/test_ac_viewer.py` (5 tests): Lua layouts equal the
Python structs byte for byte, magic/version/commands match, map->AC frame and
steering sign in the Lua code, round trip of a lap through shared memory,
lap preparation (resampling, heading from the path for old laps, newest
matching ghost). End to end without the game: `ac_view.py --best
ks_porsche_911_gt3_rs imola` sends the 1:56.668 lap with the 1:52.035 ghost
and reports that no game picked it up; the web button does the same; a lap
on a generated circuit is refused with an explanation.

**Problems & resolutions.** Shared memory on Windows disappears when the last
handle closes, so a command-line tool that exits at once would lose the lap
if the game is not running yet. -> the web server keeps one link open for its
lifetime; `ac_view.py` waits until the game reports the lap id as loaded
(from then on the game holds its own handle).

**Next steps.** Owner's first in-game test (chapter 07, "Viewing laps in the
game"); then, if per-frame placement looks jerky, a velocity-driven mode with
position correction. Later: export generated circuits as AC tracks so every
lap can be shown.

## WP33 — Viewer: gears, engine sound, drift angle; one-click game start; desktop shortcut ✅ (offline tests; in-game check by the owner)

**Objective (owner, 2026-11-23, after the first in-game test of WP32).** "Seems
quite good, but there are no gear changes"; open the game directly in the
right settings ("it's just running on my hardware and my version of the
game"); look and feel as accurate as the lap really is. Plus a desktop
starter for the control panel.

**Root cause (gears).** The lap data has them (Imola, 911: gears 1-5) and the
app engaged them, but the game's automatic gearbox - and an H-shifter, which
CSP warns about - choose their own gear on the next physics step.

**Approach.**
* Viewer protocol v2: engine rpm per sample, computed in Python exactly like
  the simulator (wheel speed x gear ratio x final drive, launch rpm floor,
  limiter) from the car file the lap was driven with; QUIT command.
* MV Viewer: auto-shifter off while playing (restored after), gear requested
  through `requestedGearIndex` and forced with `engageGear` when refused,
  `setEngineRPM` every frame, velocity from the recorded path (drift angle),
  pause holds the car at that moment.
* `mv/ac/launch.py`: AC ids of the lap's car and circuit (car files imported
  from AC carry their AC folder name; cars built from public data such as
  the Taycan are refused with an explanation); `race.ini` rewritten from the
  owner's own file (only car, skin, circuit, cars, session, online/replay
  flags change; backup in `runs/ac_launch/`, last 10); `acs.exe` started
  detached; running session read from AC's shared memory; restart via the
  app's `ac.shutdownAssettoCorsa()` only after the owner confirms.
* Desktop shortcut "Project MV" (`scripts/make_desktop_shortcut.py`, icon
  drawn into `mv/ui/mv.ico`): `pythonw` + `webui.py start` - starts the
  control panel without a console window and opens the browser; never starts
  training.

**Verification.** 8 viewer tests (+3): rpm from gear ratios and limiter;
`race.ini` keeps weather/driver, replaces car/circuit/session, removes other
sessions and AI cars, never online, ghost car added on request; the three
launch cases (game closed -> file written + backup + start; same session ->
plays; other session -> asks, closes only on request) with the game process
replaced by a stand-in - **the game was not started by the tooling or the
tests**. Imola 911 lap: 5 256-8 914 rpm (limiter ~9 000). Shortcut read back
(target `pythonw.exe`, args `webui.py start`, icon); a windowless start
brings the panel up (HTTP 200) with training unaffected.

**Problems & resolutions.** (1) Patch scripts with backslashes in Python
strings corrupted text twice (an INI path in a docstring, a JavaScript
`\n`); fixed by direct edits - rule kept: no backslashes in generated
patch strings. (2) An AI car's own driver would fight the ghost placement ->
AI input off, collisions off.

**Next steps.** Owner: reinstall the apps (`install_plugin.ps1`, the app
changed) and try the button with the game closed. Open questions for the
in-game test: does `acs.exe` started directly start the session on this
install (Content Manager's "naive" start), is the AI ghost car accepted in
Practice, do the forced gears hold with the owner's H-shifter setting.

**Problem found in the owner's first one-click test (2026-11-23).** After
quitting, the game could not be started again: Steam said it was still
running and "Stop" looped. Root cause: the game started by the launcher wrote
"CLEAN EXIT" to its log at 22:20:04, but `acs.exe` stayed alive without a
window (75 threads, 860 MB) - so it was invisible in Task Manager's main list
and Steam waited for it forever. Resolution: the hung process (exit already
logged, nothing to lose) was ended; `start_game` now also starts a small
windowless watchdog (`watch_game`) that ends `acs.exe` only if it is still
alive 8 s after a clean exit was logged - Content Manager does the same - and
never touches a game that has not logged its exit
(`test_exit_watchdog_ends_only_a_game_that_hangs_after_a_clean_exit`, with a
stand-in process).

**Second problem, found while fixing the first.** A control panel started by
the desktop shortcut runs as `pythonw.exe`, but the process search
(`train_plan.python_processes`) listed only `python.exe`: "stop" did not find
the panel and a second start ran a duplicate. Fixed (both names); the
duplicate servers were ended one by one (never a process tree - training
was not affected) and one panel restarted.

**Third problem (owner's next test, Spa): the car was under the road and
facing backwards.** (a) Height: the car was placed at its *previous* height;
it starts in the pits and Spa's track climbs ~100 m from there, so it was
put below the road, where CSP's "align on the surface" cannot find the road
above it, and it kept falling. Fix: the road height is taken from the
track's own centre line at that point (`ac.worldCoordinateToTrack` ->
`ac.trackCoordinateToWorld`, works from any start height), refined by a
short ray down, a long ray from 300 m above as the last resort; the car is
placed at road + its ride height (measured while it stands in the pits).
(b) Direction: the sign of `setCarPosition`'s direction argument is not
documented; the one-time check ran in the same frame as the first
placement, before the game had applied it, so it compared with the pit
orientation and never flipped. Fix: checked every frame - three frames in a
row facing against the intended direction flip it - separately for the
ghost car. Checked offline: both Lua apps parse (luaparser, scratch only),
no unknown global names; in-game check by the owner.

## WP34 — AC viewer rebuilt: the car is steered by forces, the ghost is a see-through car ✅ (offline checks; in-game check by the owner)

**Owner's report (2026-11-23, Spa, 911, lap 2:41.534 of `multi_v2_steer_light`).** The picture
is jittery; the car's own dashboard shows nothing; it drops into neutral in some corners; the
ghost should be a semi-transparent car.

**Root cause (one cause, three symptoms).** WP32/33 teleported the car to the recorded pose on
every *rendered* frame (60-144 per second) while the game's physics run 333 steps per second:
between placements the physics moved the car, then it was snapped back - jitter - and every
placement reset the wheels and drivetrain, so the dashboard read nothing useful and the gearbox
was knocked out of gear. Requesting the gear on every frame restarted the shift as well.

**Approach.**
* *Follow by forces* (`mv_follow.lua`, a CSP physics worker, 333 Hz): the car stays a normal car
  - tyres, suspension, gearbox, engine, dashboard and sound are the game's own - and a
  critically damped spring-damper pulls it towards the recorded position/velocity (force at its
  reference point, horizontal) while a force couple at nose and tail turns it towards the
  recorded heading (with damping on the turning rate). Written with world directions only, so no
  undocumented sign convention matters. Teleport only at start/seek/restart/loop or if more than
  15 m off (crash). The worker owns the playback clock (physics time).
* Gears: requested once per recorded shift, forced with `engageGear` only if not taken after
  0.4 s; no engine-rpm override any more (the game's own rpm is now right).
* Ghost: the car's own 3D model (`content/cars/<id>/<model>.kn5`, chosen in Python: largest
  .kn5 that is not a LOD or the collider), loaded a second time, transparent, additive blue glow,
  no shadows/reflections, tilted with the road; moved only visually - no AI car, no physics. The
  "ghost as a real car" option is removed.
* Fallback: if the physics worker cannot start, per-frame placement as before (the app says so).

**Verification (offline - the game is not run by the tooling).** Protocol v3 (car model path,
mass, yaw inertia); tests: worker reads the same lap layout as Python, the shared control
structure is identical in both Lua files, frames/steering/gear rules (138 pass). Both Lua files
parse (luaparser) and use no unknown names. The controller was simulated in Python with the same
equations: planar rigid body, the 911's mass 1 517 kg and yaw inertia 1 893 kg m², 333 Hz, along
the owner's Spa lap:

| Case | Position error mean / 99 % / max | Heading error mean / 99 % |
|---|---|---|
| gains x1, no tyres | 0.81 / 1.55 / 1.61 m | 0.22 / 1.29 deg |
| gains x1, crude tyres resisting every slide (40 kN per m/s) | 1.45 / 5.27 / 5.58 m | 0.22 / 1.29 deg |
| **gains x1.5 (new default)**, crude tyres | 1.02 / 4.81 / 5.10 m | 0.11 / 0.64 deg |
| gains x1.5 + 50 % acceleration feed-forward, crude tyres | 1.09 / 5.03 / 5.33 m | 0.11 / 0.64 deg |
| gains x1.5 + feed-forward 50 %, no tyres | 0.21 / 0.47 / 0.52 m | 0.11 / 0.64 deg |

Feed-forward does not help once tyres act (in the game the tyres, given the recorded steering
and pedals, already produce most of the acceleration), so it is not used. The crude tyre model
is pessimistic (real tyres steer with the car instead of fighting every slide).

**Next steps.** Owner: reinstall the apps, try Spa again; the window shows "following: x m, y
deg" and a stiffness slider (0.3-3). If the worker does not start, the window says "fallback".

## WP35 — Owner's in-game test: gearbox damage, ghost fixes, steering zig-zag, shift points 🕓 (viewer fixed offline; zig-zag run queued)

**Owner's report (2026-11-23).** "It works well" (WP34 follower: smooth, dashboard and gears
show), but: the ghost is not see-through and floats; a pulsing red "driveshaft" indicator;
the steering looks chaotic; and a question whether gears are changed too late (redline vs
peak power).

**Red drivetrain indicator - root cause.** Kunos' damage display (gearbox). The owner's game
settings: damage 100 %, auto-clutch off, auto-blip off, H-shifter active. The viewer forced
gears in with the clutch released, which the game counts as mis-shifts -> gearbox damage, gears
popping out ("neutral in corners"). **It does not concern training**: our simulator has no
drivetrain damage and its lap times are unaffected; it only affected the in-game replay.
Fix: shifts like the car's PDK - one paddle pull per gear step, clutch dipped during the change,
throttle blip on downshifts; CSP's gear request after 0.25 s and `engageGear` after 0.6 s only
as fallbacks. (A gearbox already damaged stays damaged until the session restarts.)

**Ghost.** (1) Transparency was set on the model's top node; it has to be set on the meshes
(`findMeshes('?')`, `findSkinnedMeshes('?')`): transparent pass, additive blending, dimmed
diffuse/ambient, blue emissive glow. (2) It floated by ~0.45 m: it was placed at the physics
reference height (near the centre of gravity; the owner's Car Physics app shows CGH 0.455)
while a car model's origin is at road level. Now the real car's model origin relative to the
road (and to the physics point) is measured from `carRoot:0` while it stands still and the
ghost is placed the same way.

**Shift points - checked, already right.** The simulator does not shift at the redline by rule:
it shifts up where the next gear gives at least as much force at the wheels (computed from the
car's own torque curve), never later than 99 % of the limiter. 911 GT3 RS (peak power 325 kW at
8 250 rpm, limiter 9 000): 1-5 shift at 8 910 rpm (after a shift at 8 250 the engine would land at
~5 200 rpm and pull less - the gear steps are large), 5-6 at 8 883, 6-7 at 8 807 (closer
steps -> earlier). 918: 9 058 rpm. A car whose power falls after its peak with close ratios
(the owner's example: peak at 6 000, redline 6 600) gets an earlier shift automatically. The
driver network does not choose gears.

**Steering zig-zag - measured.** Direction reversals per second on the Spa laps: ideal lap
0.5; `multi_v1` 21.6; `multi_v2_steer_light` 13.7; `multi_v2_steer_penalty` 16.3;
`multi_v3_cars` 21.4 - i.e. the steering flips direction on almost every other decision, by
~1.6-2.8 % of lock. Neither existing setting targets that: a 0.03 s filter is shorter than one
decision (0.04 s), and the change penalty also taxes genuine fast steering. New
`EnvConfig.steer_zigzag_penalty` / `train.py --steer-zigzag`: penalty
`max(0, -change_now x change_before)` - zero for steering moving one way however fast, positive
only for direction flips (test `test_zigzag_penalty_hits_only_direction_flips`). Lap files now
record `steer_reversals` per second.

**Queued.** `multi_v5_zigzag` (lane 2, after `multi_v4_ctrl`): continues from it with
`--steer-zigzag 5`, everything else equal, 3 h. Judged by strict lap times (must not be slower)
and reversals per second.
