# Project MV — Architecture

*MV-ARCH-001. Read this first if you are new to the code.*

## 1. What the system is, in one paragraph

Project MV is a racing driver made of software. It looks at a handful of numbers
describing the car and the road ahead (the **observation**) and answers with a
steering command and a pedal command (the **action**), 25 times per second. It
learns to do this by trial and error (**reinforcement learning**) in a
simulator we wrote ourselves, where thousands of cars practise at once on
randomly generated tracks. Nothing in the driver is specific to one car or one
track: the car is described by a parameter file, the track by its centre line
and width. The trained driver is then plugged into Assetto Corsa (and, later, a
real Formula Student car) through an adapter that produces the same observation
from that world's sensors.

## 1b. Guiding principle: always at the limit

*Stated by the project owner, 2026-11-17:* the driver's purpose is pure racing.
Whatever the car, the track or the conditions, it must drive **at the limit**:
find it, sometimes go past it, and adjust — incrementally, lap after lap — so it
becomes as fast as it possibly can. Sliding is normal and expected; it is how a
driver finds the limit. Nothing in the design should make it timid.

What this means in practice:

* **Reward = time only.** The reward is progress per unit time; there is no
  penalty for sliding, slip angle or tyre use. Mistakes cost what they cost on
  track — lost time — plus a penalty only for actually leaving the circuit.
* **Leaving the track is a mistake to learn from, not the end of the world.**
  In "limit-seeking" training the car is put back on track and continues (with
  the time lost), so it can try again on the next lap (engineering log WP21).
* **Lap-to-lap adaptation.** The driver remembers, per section of the track,
  how fast it went last lap and how close to the grip limit it was (or whether
  it went off), so it can push a little harder where it had margin and back off
  where it overstepped (WP21).
* **Robust, not conservative.** Parameter changes (grip, mass, power, aero,
  another car) are handled by sensing and adapting (memory, car descriptor),
  not by leaving a safety margin.

## 2. The big picture

```
                       ┌────────────────────────── TRAINING (our simulator, GPU) ──────────────────────────┐
 vehicles/*.yaml ──►   │  VehicleSim  ◄── action ──  Policy (neural network)  ◄── PPO (learning algorithm)  │
 (car parameters)      │   physics           ▲               ▲                         ▲                    │
                       │      │              │ observation   │ reward                  │                    │
 track generator ──►   │      ▼              │               │                         │                    │
 AC track import ──►   │  RacingEnv: locate car on track, lidar, preview, reward, track limits, lap timing  │
                       └────────────────────────────────────────────────────────────────────────────────────┘
                                                             │ trained driver (.pt file)
                                                             ▼
                       ┌────────────────────────── DEPLOYMENT ─────────────────────────────────────────────┐
                       │  Assetto Corsa ── telemetry ──► ObservationAdapter (shadow RacingEnv) ──► Policy  │
                       │        ▲                                                                    │      │
                       │        └──────── steering / throttle / brake ◄── ActuatorModel ◄────────────┘      │
                       │  (later: real FS car — state estimation + cone map replace AC telemetry)           │
                       └────────────────────────────────────────────────────────────────────────────────────┘
```

Why train in our own simulator and not directly in Assetto Corsa? AC runs one
car in real time. Learning to race takes on the order of 10⁸–10⁹ decisions. At
25 decisions per second that is 46–460 days in AC; our simulator does
~170 000 decisions per second on the laptop GPU, i.e. 10⁸ decisions in ten
minutes. AC stays the *judge*: it is where the result is validated and
fine-tuned (decision D-001).

## 3. Repository map

| Path | What it is | Key idea |
|---|---|---|
| `mv/vehicle/params.py` | The **vehicle contract**: every number that defines a car | A new car = a new YAML file; ICE, EV and hybrid share one structure |
| `mv/vehicle/summary.py` | Spec-sheet numbers from a parameter file | Sanity check before simulating |
| `vehicles/library/` | Cars built from public data (ours to share) | Taycan Turbo GT, 911 GT3 RS 992, 918 Spyder |
| `vehicles/local/` | Cars imported from the local AC install (**git-ignored**) | Kunos data never enters the repository |
| `mv/ac/acd.py`, `car_import.py` | Read a car out of Assetto Corsa | `scripts/import_ac_car.py` |
| `mv/ac/track_import.py` | Read a circuit out of Assetto Corsa | `scripts/import_ac_track.py` |
| `mv/track/track.py` | Track format + `TrackBatch` (many tracks on the GPU) | Centre line + left/right width, sampled every metre |
| `mv/track/generator.py` | Unlimited random circuits | The driver can only succeed by reading the road |
| `mv/sim/vehicle_sim.py` | **The physics**, batched over thousands of cars | 4-wheel planar model, combined-slip tyres, any powertrain |
| `mv/envs/racing_env.py` | **The game the driver plays**: sensors, reward, rules | Observation contract lives here |
| `mv/rl/ppo.py` | **The learning algorithm** (PPO), own implementation | Runs on the GPU next to the simulator |
| `mv/rl/evaluate.py` | Timed laps, no randomness | The number that matters: flying lap time |
| `mv/control/scripted.py` | Rule-based driver, no learning | Lower yardstick, self-test, first in-game test |
| `mv/planning/` | Physics reference: best possible lap (g-g-v + racing line) | Upper yardstick |
| `mv/ac/bridge/` | Live link to the running game | Telemetry in, controls out, resets |
| `mv/ac/driver.py` | Runs a driver in the game | Shadow environment → identical observations |
| `scripts/` | Command-line entry points | One script per task (see README) |
| `tests/` | Automated tests | Each checks one thing against a hand-computable number |
| `docs/` | This documentation | `engineering-log.md` is the diary |

## 4. The three contracts

Everything is modular because three interfaces are fixed. As long as a new
component honours them, nothing else has to change.

### 4.1 Vehicle contract — `VehicleParams`

Chassis (mass, inertia, wheelbase, weight distribution, CG height, track
widths, roll stiffness split), aero (drag area, front/rear downforce area),
tyres per axle (radius, peak friction, load sensitivity, slip angle of peak
grip, sliding grip), brakes (total torque, bias), steering (max wheel angle,
rate), electronics (ABS/TC), and the **powertrain as a list of power units**:

```
ICE     : [engine] ─ 7-speed gearbox ─ rear axle
Hybrid  : [engine, rear motor] ─ 7-speed gearbox ─ rear axle   +   [front motor] ─ fixed gear ─ front axle   + battery
EV      : [rear motor] ─ 2-speed gearbox ─ rear axle           +   [front motor] ─ fixed gear ─ front axle   + battery
```

The simulator loops over the list; it has no notion of "this is a hybrid".

### 4.2 Track contract — `Track`

A closed centre line sampled at uniform spacing, plus distance to the left and
right edge at each sample. Heading and curvature are derived. Source does not
matter: random generator, AC import, or (later) a cone map from a real car's
SLAM.

### 4.3 Observation contract — what the driver sees (98 numbers)

| Block | Size | Content | Where it comes from on a real car / in AC |
|---|---|---|---|
| car state | 13 | forward/sideways speed, yaw rate, long./lat. acceleration, steering angle, throttle, brake, gear, rpm, battery charge, front and rear slip angle ÷ slip angle of peak grip | IMU, wheel speeds, state estimator / game telemetry |
| previous action | 2 | what it commanded last tick | itself |
| position on track | 5 | lateral position between the edges, room to each edge, heading relative to track (sin, cos) | localisation against the track map |
| track preview | 23 + 23 | tightest curvature and narrowest width in each of 23 stretches ahead, up to 340 m | track map (after lap 1) — zeroed beyond the sensor horizon on a blind lap |
| preview horizon | 1 | how far the preview is valid | 1.0 with a map, small on a first lap |
| lidar | 19 | distance to the track edge along 19 rays over the front 180°, ≤ 150 m | lidar / cone detections / ray-cast on the map |
| car descriptor | 12 | mass, power-to-weight, grip, downforce, drag, wheelbase, weight distribution, steering lock, driven axle, electric?, width, top speed | the vehicle file |
| memory (optional) | 15 × N | the car-state block and previous action at N earlier moments (`history_steps`, `history_every`) | the driver's own recent past |

Design rules for this contract:

* **Nothing privileged.** Every entry can be measured or computed on a real
  car. Hidden quantities that vary in training (true grip, true mass) are *not*
  given to the driver; it must be robust to them.
* **Car-relative, not world-relative.** No absolute positions, so the driver
  cannot memorise a circuit.
* **The same code builds it everywhere.** In the game, `ObservationAdapter`
  owns a one-car copy of the training environment and overwrites its state from
  telemetry; `tests/test_ac_driver.py` proves the result is identical.

**Action:** `steer ∈ [-1, 1]` (fraction of full lock, + = left) and
`pedal ∈ [-1, 1]` (+ throttle, − brake). Gears are automatic (see roadmap for
agent-controlled shifting and hybrid deployment).

## 5. The physics in brief (`VehicleSim`)

Tier 1, "planar four-wheel model", 250 Hz:

1. **Actuators** — steering moves at a limited rate; pedals lag slightly.
2. **Tyre loads** — static weight + downforce + load transfer from the
   (smoothed) accelerations, split front/rear by roll stiffness.
3. **Tyre slip** — each wheel's slip angle from the car's speeds, yaw rate and
   steering angle.
4. **Powertrain** — every unit's torque at the current speed and gear,
   automatic shifting at the ideal speeds, shift time, rev limit, engine
   braking, battery power limit, regeneration, state of charge.
5. **Tyre forces** — combined-slip magic formula: cornering and
   traction/braking share one grip budget; ABS/TC hold the slip at the peak;
   without them a wheel asked for too much loses most of its cornering grip.
6. **Integration** — forces → accelerations → speeds → position.

Verified against road-test numbers (911 GT3 RS: 0–100 km/h 3.3 s, 400 m
11.1 s, 100–0 km/h 30 m, ≈1.2 g cornering). Not modelled yet: wheel rotation as
a state, suspension motion, tyre temperature/wear, differential locking,
elevation, banking, kerbs. These are tier 2 (roadmap) and are the main expected
sources of difference to Assetto Corsa.

## 6. The learning loop

```
repeat:
    collect   every car drives 32 decisions with the current policy       (GPU, ~1.5 s for 8 192 cars)
    learn     PPO update: make good decisions more likely, a little       (GPU, ~0.3 s)
    refresh   every 10 updates replace 10 % of the tracks with new ones
    validate  every N updates: timed laps on tracks never trained on      (CPU)
```

* **Reward** = metres of progress along the centre line − penalty for hanging
  over the track edge − penalty for jerky inputs; an episode that leaves the
  track by more than a metre, spins or stalls ends with a penalty. Lap time is
  never rewarded directly: maximising progress per unit time *is* minimising
  lap time, and it gives feedback at every step instead of once per lap.
* **Variation** ("domain randomisation") — every episode gets a random track,
  start point, speed, and ±grip, mass, power and downforce. 30 % of episodes
  are driven "blind" (preview limited to 25–60 m), which is what a first lap on
  an unknown Formula Student track looks like.
* **Validation** — fixed random circuits plus imported real circuits (Imola,
  Monza, Spa, Mugello). The driver never trains on these.

## 7. Where things run

| Task | Device | Why |
|---|---|---|
| Training (simulation + learning) | GPU | Thousands of cars; cost per step is almost independent of the number of cars |
| Evaluation, plots, scripted driver | CPU | A handful of cars; the GPU's fixed per-operation cost would dominate |
| Driving in Assetto Corsa | CPU | One car, 25 Hz, ~1 ms per decision |

## 8. Intellectual property boundaries

* All code in this repository was written for this project.
* Dependencies are permissively licensed (PyTorch, NumPy, SciPy, PyYAML,
  Matplotlib: BSD/MIT-style).
* **Assetto Corsa data is never committed.** `vehicles/local/`, `tracks/local/`
  and `.cache/` are git-ignored. The importers read *your* installed copy on
  *your* machine. Kunos' car and track data remain theirs; do not redistribute
  the generated files.
* Car files in `vehicles/library/` are built from public sources; each line is
  marked `SOURCE` or `ESTIMATE` (see `vehicle-data-sources.md`).
* The Formula Student rules digest quotes rule numbers and figures for
  engineering use; the rules document itself is not reproduced.
* This is engineering documentation, not legal advice. If the software is ever
  used by or for the team, agree ownership and licence in writing first.
