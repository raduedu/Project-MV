# Project MV — Decision Log

*MV-DEC-001. One entry per design decision: what was decided, what else was on
the table, and why. If a decision is reversed later, add a new entry that
supersedes it — do not edit history.*

| ID | Decision | Status |
|---|---|---|
| D-001 | Train in our own simulator; use Assetto Corsa to validate and fine-tune | active |
| D-002 | Simulator written in PyTorch, batched on the GPU | active |
| D-003 | Own PPO implementation as the first learning algorithm | active |
| D-004 | Physics tier 1: planar four-wheel model without wheel-rotation states | active |
| D-005 | Powertrain = list of power units + gearboxes (+ battery) | active |
| D-006 | Track = centre line + left/right width; random generator for training | active |
| D-007 | Observation: car-relative, nothing privileged, preview + lidar, blind mode | active |
| D-008 | Action: steering + one combined pedal; automatic gears | active |
| D-009 | Reward: progress along the centre line minus penalties | active |
| D-010 | Assetto Corsa data stays on the local machine, never in git | active |
| D-011 | Control the game through a CSP Lua app, not a virtual joystick driver | active, unverified in-game |
| D-012 | In-game observations come from a shadow copy of the training environment | active |
| D-013 | Combined-slip tyre model (replaces friction ellipse with force clamp) | active, supersedes part of D-004 |
| D-014 | Episodes start at feasible speeds derived from the corners ahead | active |
| D-015 | Warm start by imitating the rule-based driver | active (confirmed by experiment, engineering log WP7) |
| D-016 | Evaluation runs on the CPU | active |
| D-017 | Physics at 125 Hz (5 sub-steps per decision) | active |
| D-018 | Multi-car training = several single-car environments behind one interface | active |
| D-019 | Second track generator built from straights/corners/chicanes, mixed with the first | active |
| D-020 | `best.pt` = most validation tracks completed, then highest average speed | active |
| D-021 | Memory = stacked recent car states, appended; transfer by zero-initialised new inputs | active, under test |
| D-022 | Blind episodes cut the lidar range too | active |
| D-023 | Track preview = tightest curvature / narrowest width per stretch, not point samples | active |
| D-024 | Limit-seeking training: recover-and-continue, lap memory, multi-lap episodes | active, under test |
| D-025 | Evaluation is strict, per-car and timed to the crossing instant: every lap reproducible to the millisecond | active |
| D-026 | Steering smoothness only mild and learned (filter <= 0.03 s and/or jerk penalty), judged by lap time | under test |
| D-027 | Experiments run on the multi-car driver, not single-car ones | active |
| D-028 | Car variation told to the driver (descriptor follows the varied car); held-out car files in validation | active |
| D-029 | Bigger network (1.7 M) distilled from the best driver instead of trained from zero | active, under test |
| D-030 | Section-based realistic circuits (`complex`) as the main training preset | active |
| D-031 | No imitation of a named real driver; public telemetry only as a benchmark | active |
| D-032 | Laps are shown in AC by replaying our recorded lap on the game's car (CSP app), not by writing AC replay files | active, first in-game test pending |
| D-033 | The control panel may start Assetto Corsa for the owner (race.ini rewritten with backup), never on its own | active |
| D-034 | AC viewer: the real car is steered by forces at the physics rate (no per-frame teleports); ghost = see-through copy of the car model | active, in-game check pending |
| D-035 | Steering smoothness via a zig-zag penalty (direction flips only), not stronger filtering | under test |
| D-036 | Gearbox: always (nearly) in the gear with the most wheel force (downshift band 2 % instead of 7 %), adopted only after training with it | under test |

---

### D-001 — Own simulator for training, Assetto Corsa for validation

* **Context.** Requirement R6: results must hold in Assetto Corsa.
* **Options.** (a) Learn directly in AC. (b) Learn in an existing fast
  simulator. (c) Write our own simulator, parameterised from AC, validate in AC.
* **Decision.** (c).
* **Why.** AC runs one car at real time: 25 decisions/s. Racing needs
  10⁸–10⁹ decisions → months. Our simulator: ~170 000 decisions/s on the laptop
  GPU. Existing fast simulators either are not racing-grade or bring licence
  and dependency constraints that conflict with R7 (own the IP). Writing it
  ourselves also means every car parameter is under our control (R1, R2) and
  tracks can be randomised without limit (R3).
* **Cost.** A "reality gap" between our physics and AC's. Managed by: importing
  car data from AC, randomising physical parameters in training, system
  identification against AC telemetry (Phase 2), and fine-tuning in AC.

### D-002 — PyTorch, batched on the GPU

* **Options.** NumPy on CPU with many processes; JAX; Numba; C++.
* **Decision.** One PyTorch program in which every quantity is a tensor with
  one entry per car.
* **Why.** Simulation and learning live on the same device with no copying;
  one language; measured 168 000 decisions/s with 16 384 cars (800 MB of GPU
  memory). Cost per step is dominated by a fixed overhead per tensor operation,
  so more cars are almost free.
* **Consequence.** Small batches are slow on the GPU (54 ms per step
  regardless of size) → evaluation and in-game driving use the CPU (D-016).

### D-003 — Own PPO

* **Options.** Stable-Baselines3 PPO/SAC; RL-Games; own PPO; own SAC/QR-SAC
  (what *Gran Turismo Sophy* used).
* **Decision.** Own PPO, ~250 lines.
* **Why.** With a simulator this fast, samples are cheap and wall-clock time is
  what matters; PPO with thousands of parallel cars is the standard choice in
  that regime and is stable with little tuning. Library implementations expect
  NumPy environments and would force GPU↔CPU copies every step. Own code keeps
  the dependency list short (R7) and readable (R8).
* **Revisit when** fine-tuning in AC (few samples, real time): there an
  off-policy algorithm is the better tool (Phase 2, step 6).

### D-004 — Physics tier 1 without wheel-rotation states

* **Decision.** Three body degrees of freedom (forward, sideways, yaw); tyre
  forces from slip angle and *requested* longitudinal force; load transfer
  quasi-static with a short lag.
* **Why.** Wheel rotation dynamics are numerically stiff (time constant < 1 ms
  at low speed) and would need 4–10× more sub-steps. The first target car has
  ABS and traction control, which keep the wheels near peak slip anyway.
* **Known limits.** No true wheelspin/lock-up dynamics; needed for cars without
  ABS/TC (Formula Student) → Phase 3.

### D-005 — Powertrain as a list of power units

* **Decision.** `units: [engine | motor, …]`, each attached to a gearbox and an
  axle split; a battery whenever a motor exists.
* **Why.** ICE, hybrid and EV become data, not code paths (R2). The 918's
  rear motor through the PDK and front motor on a fixed gear, and the Taycan's
  two-speed rear axle, all fit without special cases.

### D-006 — Track = centre line + widths; random generator

* **Why this format.** It is the least information that defines a circuit, and
  every source can produce it (AC's AI line file, a generator, a cone map).
* **Why random tracks.** R3. A driver trained on few circuits memorises them.
  Each of the thousands of cars drives its own circuit and 10 % of circuits are
  replaced every ten updates.
* **Known limits.** Flat; star-shaped layouts only (Phase 3).

### D-007 — Observation design

* **Decision.** See architecture §4.3. Car state + position on track + track
  preview (curvature/width up to 340 m) + 19-ray lidar + car descriptor.
* **Why preview *and* lidar.** The preview is what a map gives you (after lap
  1); the lidar is what sensors give you now. Training with the preview
  randomly cut to 25–60 m forces the driver to cope with both — the blind
  first lap of FS autocross/trackdrive (R9).
* **Why 340 m.** Braking a road car from 300 to 80 km/h takes ≈ 230 m; the
  driver must see the corner before it has to brake.
* **Why slip angles instead of "grip used".** They are computed from speeds,
  yaw rate and steering angle only — available on any car — whereas tyre force
  utilisation would be privileged simulator knowledge.
* **Why a car descriptor.** So that later one network can drive different cars
  (Phase 4) without changing the interface.

### D-008 — Action space

* **Decision.** Steering target and a combined pedal, both in [-1, 1];
  automatic gear selection at the ideal shift speeds.
* **Why.** A combined pedal removes the useless "both pedals" region and halves
  what must be learned. Left-foot braking overlap is a refinement for later.
  Automatic gears keep the first problem small; manual shifting and hybrid
  deployment become extra actions in Phase 4.

### D-009 — Reward

* **Decision.** `0.25 × metres of centre-line progress − edge penalty −
  smoothness penalty`, `−5` on a crash (which also ends the episode).
* **Why not lap time.** One number per lap is far too sparse a signal.
  Progress per step is the same objective, delivered 25 times a second.
* **Why centre-line progress.** It rewards taking the short way round and
  carrying speed — exactly the racing line trade-off — without us prescribing a
  line.
* **Track limits.** Strict for now: any part of the body over the edge costs
  reward; more than 1 m beyond with the whole car ends the episode. AC and FS
  rules differ in what they tolerate (kerbs, 2 wheels, cones); the overhang
  allowance will become a parameter when those cases are implemented.

### D-010 — AC data stays local

* **Decision.** `vehicles/local/`, `tracks/local/`, `.cache/` are git-ignored.
* **Why.** R7. The importers read the owner's installed copy for the owner's
  private use; Kunos' data is not ours to redistribute.

### D-011 — Game control through a CSP Lua app

* **Options.** vJoy (virtual joystick driver), ViGEm (virtual gamepad), CSP Lua
  app with `ac.overrideCarControls`.
* **Decision.** CSP Lua app first; vJoy / ViGEm backends kept as fallbacks.
* **Why.** No driver installation, bypasses the game's input filtering, and the
  same app provides resets and exact orientation. **Not yet verified in the
  running game** — see `07-assetto-corsa-bridge.md`.

### D-012 — Shadow environment for in-game observations

* **Decision.** The in-game adapter owns a one-car `RacingEnv`, overwrites its
  state from telemetry and calls the same `observe()` used in training.
* **Why.** A second implementation of the observation would drift out of sync.
  `tests/test_ac_driver.py` checks equality to 2 × 10⁻³ for all library cars.

### D-013 — Combined-slip tyre model

* **Problem found.** With the original "friction ellipse + force clamp", a
  wheel working at the ABS/TC limit kept only √(1 − 0.97²) ≈ 24–31 % of its
  cornering grip. The rule-based driver spun at every heavy braking zone
  (Imola Tamburello, Monza first chicane, Spa La Source), and the first RL runs
  kept crashing at the same places.
* **Decision.** Similarity model: normalised longitudinal slip κ and slip angle
  α combine into σ = √(κ² + α²); force magnitude follows the magic formula of
  σ and points along the slip. ABS/TC clamp κ at the peak; without them κ runs
  away beyond the peak (wheelspin/lock-up).
* **Why.** It is the standard physically motivated way to share grip, it is
  what AC's own tyre model does in spirit, and at the ABS limit while cornering
  at the limit it leaves ≈ 70 % lateral force instead of ≈ 25 %.
* **Evidence.** Straight-line results unchanged (0–100 km/h 3.30 s, 100–0
  30.1 m); rule-based driver went from 3/14 to 14/14 circuits completed (with
  the driver fixes logged in WP6).

### D-014 — Feasible starting speeds

* **Problem found.** Episodes started at up to 90 % of top speed whenever no
  corner lay within 160 m, even if a hairpin followed at 170 m — an
  unavoidable crash that teaches nothing.
* **Decision.** Starting speed = random fraction (30–100 %) of the highest
  speed from which every corner in the next 450 m can still be made
  (corner speed + braking distance, same formula as the rule-based driver).
* **Why.** All starts are survivable, and roughly half of them are inside
  braking zones at racing speed — the situation that is hardest to discover by
  exploration.

### D-015 — Warm start by imitation (optional)

* **Idea.** Clone the rule-based driver into the network (DAgger), train the
  critic alone for a few updates, then run PPO as usual.
* **Why.** PPO from scratch spends tens of millions of decisions learning to
  brake for slow corners; the rule-based driver already does.
* **Risk.** The student could inherit the teacher's caution. PPO's reward is
  unchanged, so anything faster is still reinforced.
* **Evidence (engineering log WP7).** Same 150 updates: 3/6 unseen tracks
  completed with warm start vs 2/6 without, crash rate 31 % vs 59 %, and the
  student was already 10 % faster than its teacher at Imola. With 16 384 cars
  the cloned network completed 9/10 validation tracks before any PPO update.

### D-016 — Evaluation on the CPU

* **Measured.** 6 cars on the GPU: 54 ms per step; on one CPU thread: 11 ms.
* **Decision.** Timed validation laps and in-game driving run on the CPU,
  single-threaded.

### D-017 — Physics at 125 Hz

* **Measured.** Flying laps of the rule-based driver on six tracks at 500, 250,
  125 and 100 Hz agree within 0.05 s (engineering log WP5); straight-line and
  standstill behaviour identical.
* **Decision.** 5 physics sub-steps per 25 Hz decision (was 10).
* **Why.** Nearly doubles decisions per second (263 000/s with 16 384 cars).
* **Revisit when** wheel-rotation states are added (tier 2): they are stiffer
  and will need a higher rate or an implicit update.

### D-018 — Multi-car training by composition

* **Options.** (a) Make every physics parameter a per-car tensor inside one
  `VehicleSim`. (b) Run one `RacingEnv` per car model and concatenate.
* **Decision.** (b), `mv/envs/multi_car.py`.
* **Why.** Powertrains differ in *structure* (number of units and gearboxes),
  not only in numbers, which makes (a) invasive. (b) needed ~70 lines and no
  change to the physics. Cost: the GPU's fixed per-step overhead is paid once
  per car model, so three cars step about three times slower for the same fleet
  size — acceptable for a handful of cars.
* **Revisit when** training on dozens of car variants (then sample parameters
  per car inside one simulator).

### D-019 — Segment-based track generator, mixed with the original

* **Problem found (engineering log WP13).** After 1.6 billion decisions the
  driver was within ~4 % of the physics reference on Mugello and Spa but never
  completed Imola or Monza: it braked too late into the two heaviest braking
  zones. The original generator almost never builds a long straight followed by
  a slow corner or a real chicane.
* **Options.** (a) Train on the real circuits — rejected, the driver would
  memorise them and validation would be meaningless. (b) Tune the old
  generator's smoothing. (c) A second generator that composes circuits from the
  features real circuits are made of.
* **Decision.** (c), `mv/track/segments.py`, used *together* with the old one
  (`--preset mixed+gp`) so variety only grows.
* **Why.** It controls exactly the statistics that matter (straight length
  before a corner, corner radius, chicanes), and the same code at FS scale
  (`fs_segments`) is the starting point for Formula Student tracks.

### D-020 — How the "best" checkpoint is chosen

* **Problem found.** The first rule (fraction of distance completed, then mean
  lap time only if *all* tracks finished) selected the very first, cloned and
  slow driver of the 3-hour run.
* **Decision.** Score = number of validation tracks completed + mean speed over
  the completed ones / 1000 km/h. Reliability first; among equally reliable
  drivers, the faster one. Every validation also saves `eval_<update>.pt` so any
  point of a run can be re-examined later.

### D-021 — Memory by stacking recent states

* **Problem (engineering log WP16).** The driver completed 1/6 tracks at grip
  ×0.88 although training varied grip that much: from a single instant it
  cannot tell how much grip there is.
* **Options.** (a) Recurrent network (LSTM/GRU). (b) Stack the last N samples
  of the car state and actions into the observation. (c) Give the true grip as
  an input (privileged — rejected, a real car cannot measure it).
* **Decision.** (b) first: 8 samples, 0.12 s apart.
* **Why.** Works with the existing PPO and network unchanged; ~1 s is enough
  to see how the car responded to the last inputs (yaw rate vs steering, speed
  lost under braking). A recurrent network is the step after, if lap-to-lap
  learning needs longer memory.
* **Transfer.** New inputs are appended at the end and their first-layer
  weights start at zero, so training continues from an existing driver instead
  of from scratch (verified: identical outputs at the start).

### D-022 — Blind means blind

* **Problem (WP16).** "Blind" episodes cut the track preview but kept the
  150 m lidar, which still showed the road far ahead.
* **Decision.** In blind episodes the lidar range equals the preview distance
  (25–60 m), close to what an FS car's cone perception gives.

### D-023 — Preview by stretch, not by point

* **Problem (engineering log WP20).** Point samples 20–45 m apart let short,
  sharp corners fall between samples; the driver saw a chicane's radius flicker
  between 10.6 and 23 m in its braking zone and failed there every lap.
* **Decision.** Each of the 23 preview values reports the signed curvature of
  largest magnitude, and the smallest width, within its stretch of road
  (scanned every ~1 m).
* **Why.** "How tight is the worst corner in this stretch" is what braking
  depends on; it cannot alias. Same number of values, so the network and every
  tool keep working.
* **Evidence.** The previously trained driver completed Imola and Monza for the
  first time with this change alone, without retraining.

### D-024 — Limit-seeking training

* **Requirement (owner, 2026-11-17).** "The goal is for it to find the limit,
  sometimes surpass it and then try again with an incremental change, so it
  becomes as fast as it possibly can... no matter what parameters change the
  goal is to be at the limit."
* **Decision.** (1) Going off costs a penalty plus the time lost, and the car
  rejoins — the episode continues. (2) A per-section lap memory (last lap's
  minimum speed, maximum grip use, offs) is part of the observation.
  (3) Episodes span several laps (240 s).
* **Why.** Ending the episode on every mistake teaches "never make one", which
  is exactly a timid driver; rejoining teaches "a mistake costs time, so
  overstep only where it pays". The lap memory gives the driver the evidence to
  adjust incrementally, as a racing driver uses reference points and the
  feeling of last lap. All memory contents are measurable on a real car.
* **Alternatives considered.** A recurrent network with very long memory
  (implicit, slower to train, hard to inspect); a per-car grip estimate as
  input (privileged in simulation, unavailable on a real car).

### D-025 — Reproducible lap times

* **Decision.** Evaluation never lets the car rejoin (strict), computes the
  network for each car separately (batch-independent rounding) and times laps
  at the interpolated line-crossing instant.
* **Why.** A lap time anyone can reproduce exactly from the saved driver cannot
  be doubted; and a small numeric artefact (0.1 s from batching) must not be
  mistaken for driver improvement.

### D-026 — Steering smoothness: mild and learned

* **Decision.** Offer a short steering damper and a jerk penalty as training
  options; never smooth heavily after training.
* **Why.** Post-hoc smoothing of an untrained driver only adds lag and costs
  time (measured); the owner wants the driver to stay on the edge. The
  experiment decides by lap time.

### D-027 — Experiment on the multi-car driver

* **Context.** The steering A/B was first started on the single-car 911
  driver because it is the cleanest yardstick. The owner asked why.
* **Decision.** Run improvements on the multi-car driver.
* **Why.** The goal is any car; single-car drivers cannot drive other car
  files at all (25.2), so anything learned on them would have to be trained
  into the multi-car driver again. A slightly noisier comparison on the right
  model beats a clean comparison on a dead end.

### D-028 — Car variation, told to the driver

* **Options.** (a) More car files. (b) Hidden variation only (as before).
  (c) Vary the car and update the car description accordingly.
* **Decision.** (c), plus held-out car files in validation.
* **Why.** (b) teaches robustness but not adaptation - the driver cannot know
  which car it is in. (c) forces it to read the description; together with
  (a) over time it is what makes "add a car file and it drives" true. Held-out
  cars make that claim measurable instead of assumed.

### D-029 — A bigger driver, distilled

* **Options.** (a) Keep 0.2 M parameters, train longer. (b) Bigger network
  from zero. (c) Bigger network that first copies the best existing driver.
* **Decision.** (c), with a same-size control run continuing from the same
  driver on the same circuits, so the gain of size is measured, not assumed.
* **Why.** Three cars plus car variation plus held-out cars is a lot for
  0.2 M parameters (the multi-car driver is slower than the single-car one);
  (b) would waste ~10 GPU-hours of learning. A 1.7 M network still drives in
  ~1 ms on the laptop's CPU.

### D-030 — Realistic circuits as the main training preset

* **Context.** The owner: generated circuits look like ovals. Measured: they
  turn about 2 full circles per lap against 3.45 on real circuits.
* **Decision.** New `complex` preset (sections, chicanes, esses, folding
  layouts); training mixes `complex+complex+gp`, validation on `complex`.
  Old presets kept unchanged so earlier results and laps stay reproducible.
* **Why.** The driver can only learn corner sequences it meets; the real
  circuits it was weakest on (Imola, the 918 at Monza/Spa) are exactly the
  chicane and esses sections the old generators lacked.

### D-031 — No imitation of a named real driver

* **Context.** The owner suggested basing the style on Max Verstappen
  (versatile; drives F1 and GT3).
* **Decision.** Do not train the driver to imitate a real person. Use
  public timing/telemetry (e.g. F1 lap times, speed traces) at most as a
  benchmark, never as training targets; never brand the project with a
  driver's name.
* **Why.** (1) The data needed does not exist publicly: F1 public telemetry
  has speed, throttle, brake on/off, gear and position at ~4 Hz, but no
  steering, slip or tyre data, and only for F1 cars; GT3 telemetry is not
  public. (2) Style is car-specific: braking points and lines in an F1 car
  do not transfer to a 911 or an FS car - the project's goal is the
  opposite, any car. (3) Rights: data terms and personality rights. (4) The
  quality admired - versatility across cars - is exactly what car variation
  and held-out cars (D-028) measure and train. Reinforcement learning
  against the physics reference lap already aims at the limit, which is
  what any great driver does.

### D-032 — Showing laps in Assetto Corsa

* **Options.** (a) Write AC's own replay files (`.acreplay`). (b) A CSP Lua
  app that places the car on the recorded lap every frame. (c) Only drive
  live in AC with the trained driver.
* **Decision.** (b), with (c) kept for watching the driver on AC's physics.
* **Why.** (a) is an undocumented binary format and the files live in the
  user's documents - fragile and unverifiable without the game. (b) uses only
  documented CSP calls and the bridge's proven shared-memory link, shows
  exactly the lap we measured (lap times in the panel stay true), and adds
  what AC's replay cannot: the ideal-lap ghost and its racing line. (c) shows
  AC physics, which is a different (and valuable) question: sim-to-game gap.

### D-033 — Starting the game from the control panel

* **Context.** Until now the rule was: the tooling never starts the game. The
  owner asked for one click that opens the game in the right session.
* **Decision.** The "View in Assetto Corsa" button / `ac_view.py` may start
  the game and rewrite `Documents/Assetto Corsa/cfg/race.ini` - only when the
  owner clicks, never from training, tests or automation; the game folder is
  still never written; a running game is closed only after the owner
  confirms.
* **Why.** The owner's request, and it is what Content Manager itself does
  (it writes race.ini and starts acs.exe). Keeping the owner's file as the
  base and a backup of it keeps personal settings safe.

### D-034 — How the lap is shown in the game

* **Context.** Per-frame teleports (D-032) worked but the owner saw jitter, an empty dashboard
  and gears dropping to neutral; the ghost should look like a car.
* **Options.** (a) Teleport at the physics rate in a worker (still resets wheels/drivetrain).
  (b) Hide the real car and move a purely visual model (smooth, but no dashboard, sound or
  telemetry). (c) Keep the real car and steer it with forces at the physics rate.
* **Decision.** (c) for the car, (b) for the ghost.
* **Why.** Only (c) keeps everything the owner wants to observe genuine (suspension, gear
  changes, dashboard, sound) while following the recorded lap closely (offline simulation in
  WP34). The ghost needs no physics at all, so a visual copy is the smoothest and cannot collide.

### D-035 — Penalise steering zig-zag, not steering speed

* **Context.** The owner sees chaotic steering in the game; measured: 14-22 direction reversals
  per second (ideal lap 0.5). The owner wants it smooth but still at the limit.
* **Options.** (a) Longer steering filter (adds lag - measured 0.1-0.4 s slower at 0.06 s, WP27).
  (b) Stronger penalty on the change per step (also taxes the fast corrections a driver at the
  limit needs). (c) Penalise only direction flips between consecutive decisions.
* **Decision.** (c), tested against the same driver without it.
* **Why.** It removes exactly the sawing without making quick, decisive steering costly - the
  driver keeps its reflexes, which "at the limit" requires.

### D-036 — Gear choice: maximise force at the wheels

* **Context.** The owner: shift where the next gear makes more power. Upshifts already follow the
  wheel-force crossover; downshifts lagged by a 7 % band, costing 6-19 % pull for up to ~1 s per lap.
* **Decision.** Option `downshift_frac` (0.98 = 2 % band). Not switched on by default yet: with the
  same, untrained drivers it changed lap times only within the scatter (and caused one DNF), so it
  is first trained with (`multi_v5_gearbox`) and adopted if that driver is faster.
* **Why not RL-chosen gears.** Letting the network pick gears would add actions and change the
  observation contract (every trained model invalid) for a gain the force-optimal rule already
  captures; anticipation (e.g. holding a gear into a braking zone) is not needed - measured: no
  upshifts while lifting/braking.

### D-037 — Keep the zig-zag penalty; retry a bigger network only with smaller steps

* **Context.** WP40: the zig-zag penalty gave the fastest and most reliable driver so far
  (15/15, steering reversals halved); the 1.7 M-parameter network ended 2-9 % slower than the
  normal-size control.
* **Decision.** `--steer-zigzag 5` in every following run. The bigger-network idea is not
  dropped but retried at a medium size (768-512-256) with half the KL stop (`--kl-stop 2`),
  distilled from the new best driver, against a same-settings normal-size run.
* **Why.** The big network failed because its training steps were too large, not because of
  its size as such (WP30 interim); a medium size with tighter steps tests that directly at a
  third of the cost. Supersedes nothing; extends D-035 (zig-zag) and WP30.

### D-038 — Real F1 data: benchmark in equal machinery, private use only

* **Context.** The owner asks to race Max Verstappen in equal machinery, ideally with openly
  licensed data. Extends D-031 (no imitation of a named driver; public data at most a benchmark).
* **Options.** (a) No real-driver data. (b) Imitate his inputs (behaviour cloning). (c) His laps
  as the benchmark, and his telemetry to *define the car* (fit an F1 car until our physics
  reproduces his speeds), our driver learning by RL to beat that lap in that car.
* **Decision.** (c), with OpenF1 historical data (2023 on), cached and kept on the owner's computer
  only - never committed, published or shown on the public replay page.
* **Why.** (b) is impossible (no steering in the data) and against D-031. (c) uses only what the
  data does contain (speeds, gears, positions) and gives a real-world yardstick. The data belongs to
  Formula 1 and no licence is granted by OpenF1/FastF1; private, non-commercial research use with no
  redistribution is the most that can be justified - the same rule as for the game's data.

### D-039 — Real circuits in training, with a permanent held-out set

* **Context.** Owner: use all tracks. Until now imported circuits were validation only.
* **Decision.** `--real-tracks all` adds every imported circuit except those in
  `tracks/sets.yaml: never_train` (Mugello, Laguna Seca, Brands Hatch) and broken imports.
* **Why.** Real circuits contain combinations the generator does not make; the benchmark circuits
  are ones a real driver also practises on. Keeping three out forever keeps the "never seen this
  circuit" result honest. Supersedes the "never trains on these" line in 01-architecture §6.

### D-040 — Distillation copies the critic, not only the actor

* **Context.** Two distilled networks (WP30, WP44) sped up and crashed with normal update sizes.
* **Decision.** A teacher warm start copies actor, critic and return scaling.
* **Why.** PPO judges actions with the critic; an expert actor with a newborn critic is pushed
  towards whatever the critic over-values - here speed. Copying is cheap (same mini-batches).
  Revises D-037's reading of the WP30 failure (step size); the KL stop stays as a safeguard.

### D-041 — Cars the game does not have are shown with a stand-in body

* **Context.** The owner wants to see the FS electric car in Assetto Corsa; the game has no FS car and
  no electric car.
* **Options.** (a) Refuse (as before). (b) Ask the owner to install a mod car. (c) Show the lap with an
  installed car's body, chosen per car in a committed table.
* **Decision.** (c), table `vehicles/ac_stand_ins.yaml`; (b) stays possible (an installed mod is
  imported like any car and then needs no stand-in).
* **Why.** The lap is ours either way (our physics, our driver); the game only draws it. Gears and mass
  follow the stand-in so the game's own engine and the force follower behave sensibly.

### D-042 — The FS car is drawn by our own app, not shown as another car

* **Context.** The owner: the FS car must look like an FS car, generic, no names. The game has none.
* **Options.** (a) A branded FS or single-seater mod. (b) Write a complete game car (.kn5 model plus
  physics files) for the owner to install. (c) Draw a generic model from the viewer app on top of a
  hidden stand-in.
* **Decision.** (c). (b) stays possible later.
* **Why.** (a) is excluded by the owner. (b) means writing a binary model format and a full physics
  set that cannot be tested here, and a bad file can crash the game. (c) uses only drawing calls of
  the owner's CSP, is unbranded by construction, and falls back to the visible stand-in on failure.
  Supersedes D-041 for the FS car only.

### D-044 — The race bot learns with sensor range only, and races live as an AI car of the game

* **Context.** Owner: a bot to race against in the game, same car, any track, learns from nothing
  (Nordschleife, realistic lidar), recovers by itself, overtakes.
* **Decision.** (1) The bot sees only a sensor range (50-200 m in training, 120 m for the test) and
  never trains on the test circuit; it learns a circuit lap by lap through its lap memory (later: its
  own map of track already driven). (2) It races live as the second car of an offline session,
  controlled by our app through CSP's control override for AI cars; the owner drives car 0 in the
  same car model. (3) Recovery is learned (no teleports), overtaking by self-play in our simulator.
* **Why.** A preview of the whole track is a map; a real car racing an unknown circuit has sensors.
  Driving an AI slot keeps the game's own physics for both cars - equal machinery by construction.
