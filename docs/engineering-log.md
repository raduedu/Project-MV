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
| Reference lap (g-g-v + racing line) → `mv/planning/`, `docs/08-…` | Fable | 🕓 interrupted three times by the usage limit; `ggv.py`, `raceline.py`, `speed_profile.py`, `reference.py` exist but are unreviewed and the script, tests and document are missing. Resumed again |

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

## Next steps (in order)

1. **Long training run** of the 911 GT3 RS (roadmap phase 1) and evaluation
   against the baseline and the physics reference. Status of the run started at
   the end of this session: see "WP11" below if present, otherwise
   `runs/` and `docs/04-training-guide.md`.
2. **Finish and review the reference lap** (`mv/planning`): script, tests,
   document; then check the lead's own review of the helper's g-g-v against
   `VehicleSim`.
3. **First in-game session** (needs the owner): `docs/07-assetto-corsa-bridge.md`
   checklist → `scripts/ac_bridge_check.py` → `scripts/ac_drive.py --scripted
   --dry-run` → scripted driver with `--max-kmh 80` → trained driver.
4. **System identification** against AC telemetry and tuning of the importer
   (roadmap phase 2, step 4).
5. Hyper-parameter study, one change at a time (7.3).
6. Open items flagged ⚠️ above: 3.3 (generator shapes), 3.4 (flat tracks),
   5.3/5.4 (tier-2 physics, trail-braking balance), 6.6 (track-limit rule),
   8.2/8.3 (in-game verification), 10.2 (FS tracks).

