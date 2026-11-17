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
| D-015 | Optional warm start by imitating the rule-based driver | see engineering log WP9 for the experiment |
| D-016 | Evaluation runs on the CPU | active |

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
  unchanged, so anything faster is still reinforced; to be confirmed by
  experiment (engineering log WP9).

### D-016 — Evaluation on the CPU

* **Measured.** 6 cars on the GPU: 54 ms per step; on one CPU thread: 11 ms.
* **Decision.** Timed validation laps and in-game driving run on the CPU,
  single-threaded.
