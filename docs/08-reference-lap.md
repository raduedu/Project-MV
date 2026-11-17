# 08 — Reference lap: what is theoretically achievable?

The RL driver is trained and graded in our own simulator. To say "the driver is
1.8 % off the pace" we need a pace that does **not** come from a driver. This
module computes one from physics alone:

> *Given this car (`VehicleParams`) and this track (`Track`): which line around
> the track is fastest, how fast can the car go at every point of it, and what
> lap time does that give?*

Code: `mv/planning/` (`ggv.py`, `raceline.py`, `speed_profile.py`,
`reference.py`), CLI `scripts/reference_lap.py`, tests `tests/test_planning.py`.

```
 VehicleParams ──► ggv.py ───────────► g-g-v envelope  ─┐   "what can the car do?"
                                                        ├─► speed_profile.py ─► speed, lap time
 Track ─────────► raceline.py ───────► line  n(s) ──────┘   "how fast along this line?"
                  (centre / minimum curvature / minimum time; the minimum-time
                   search calls the speed profile to judge candidate lines)

 reference.py :  reference_lap(track, vp) -> ReferenceLap  (.save / .load, one .npz)
```

Quick start:

```
python scripts/reference_lap.py --vehicle ks_porsche_911_gt3_rs --track imola
python scripts/reference_lap.py --vehicle porsche_taycan_turbo_gt_weissach --random 3 --preset mixed
```

writes `runs/reference/<vehicle name>__<track name>.npz` (+ `.png`);
`scripts/evaluate.py` reads the `lap_time` in that file and prints each
driver's gap to it. In Python:

```python
from mv.planning.reference import reference_lap, ReferenceLap
ref = reference_lap(track, vp)          # 5 s envelope + 2-30 s per track; pass ggv=build_ggv(vp) to reuse the envelope
ref.lap_time, ref.n, ref.speed          # arrays are indexed like the Track's stations
ref.save("runs/reference/x.npz"); ReferenceLap.load("runs/reference/x.npz")
```

---

## 1. The g-g-v diagram: what the car can do

Sit in a car with an accelerometer. Braking pushes the needle one way,
accelerating the other, cornering sideways. Drive as hard as you can for a
while and the needle never leaves a certain blob: the **g-g diagram**. Its edge
is the limit of the car.

* The top of the blob (forward acceleration) is set by engine power and
  traction; the bottom (braking) by tyre grip and brake balance; the sides
  (cornering) by tyre grip.
* The corners are rounded: a tyre has **one** grip budget. What it spends on
  braking it cannot spend on cornering.
* The blob changes with speed — downforce adds grip, drag helps braking,
  the engine runs out of force — so there is one blob per speed. Stacked, they
  are the **g-g-v diagram**.

The 911 GT3 RS in our simulator (in g; "steady" = cornering that can be held
at constant speed, see below):

| speed | cornering, steady | cornering, max | full throttle | full braking |
|---|---|---|---|---|
| 50 km/h  | 1.18 | 1.25 | +1.03 | −1.32 |
| 100 km/h | 1.22 | 1.27 | +0.69 | −1.36 |
| 150 km/h | 1.25 | 1.30 | +0.45 | −1.42 |
| 200 km/h | 1.28 | 1.35 | +0.29 | −1.48 |
| 250 km/h | 1.22 | 1.39 | +0.15 | −1.56 |

and one slice of it, at 100 km/h:

| cornering at | best acceleration | hardest braking |
|---|---|---|
| 0.00 g | +0.69 g | −1.36 g |
| 0.64 g (50 %) | +0.63 g | −1.13 g |
| 0.95 g (75 %) | +0.45 g | −0.75 g |
| 1.14 g (90 %) | +0.14 g | −0.34 g |
| 1.27 g (100 %) | −0.14 g | −0.14 g |

Note the last row: at the very limit of cornering the car *slows down* by
0.14 g, whatever the driver does. A tyre working at its limit drags (its side
force is tilted backwards by the slip angle), and the throttle needed to cancel
that drag would take grip from the rear tyres. So the largest lateral
acceleration that can be **held** at constant speed ("steady", what a skid-pad
measures) is about 4 % below the absolute maximum, which is only available
while the car is slowing — i.e. on the way into a corner.

### How it is computed (`mv/planning/ggv.py`)

The simulator integrates the car's equations in time. For the envelope we
solve the *same equations* for a steady state: pick a speed `v`, a lateral
acceleration `ay` and a pedal position `u` (throttle or brake) and ask whether
the car can do that, and with which `ax`.

1. **Tyre loads.** Static weight + downforce (front/rear split) + longitudinal
   load transfer (`m·ax·h/L` moves from the front to the rear axle) + lateral
   load transfer (`m·ay·h/track`, split between the axles by roll stiffness).
2. **Tyre grip.** Peak force of each tyre `D = mu(Fz)·Fz` with
   `mu(Fz) = mu_ref·(Fz/Fz_ref)^(exp−1)` — a tyre with twice the load has less
   than twice the grip, which is why load transfer costs grip.
3. **Tyre forces** from the simulator's combined-slip model: the pedal sets
   each wheel's longitudinal slip `kappa_n` (engine force through the gearbox
   in the gear the automatic box would be in, brake force with the front/rear
   bias, rolling resistance; traction control and ABS cap the slip), the slip
   angle sets `alpha_n`, and with `sigma = sqrt(kappa_n² + alpha_n²)`:
   `Fx = Dx·P(sigma)·kappa_n/sigma`, `Fy = Dy·P(sigma)·alpha_n/sigma`.
4. **Balance.** The front and the rear axle must together supply `m·ay` *and*
   produce no net yaw moment (otherwise the car would spin up or straighten).
   That fixes how much side force each axle needs. If an axle needs more than
   it can give at the slip angle of peak grip, the point is outside the
   envelope. (So the limit is set by the axle that saturates first —
   understeer or oversteer — not by the sum of all four tyres.)
5. **Result.** Sum of the forces along the direction of travel, minus drag,
   gives `ax`.

Steps 1–5 chase each other (loads depend on the accelerations, the
accelerations on the loads), so they are repeated until nothing changes — a
*fixed-point iteration*. Two bisections then find the edge: the largest `ay`
for each pedal position (→ `ay_max(v)`), and for each `ay` the pedal range that
is still feasible (→ `ax_max(v, ay)`, `ax_min(v, ay)`). The result is three
small tables; building them takes about 4–5 s per car.

Details that turned out to matter:

* **Body slip.** At the limit the car's nose points a few degrees into the
  corner. The simulator computes load transfer along the *body* axes, so part
  of the centripetal acceleration acts "forwards" and unloads the front tyres.
  Ignoring this over-estimated the cornering limit of the (understeering) 911
  by 4 %.
* **One pedal for four wheels.** The brake pedal feeds all wheels in the fixed
  bias ratio and the engine feeds the driven axle through an open differential,
  so the four tyres cannot all be at their individual optimum. This is why the
  envelope is found by searching over the *pedal*, not by adding up four
  friction ellipses.
* **TC and ABS** cap the longitudinal slip just below the peak, so a tyre under
  full ABS braking keeps about 70 % of its cornering force. The envelope's
  braking side is therefore much "squarer" than a friction circle.
* **Steering lock** (added in review). The simulator clamps the road-wheel
  angle at `max_wheel_angle_deg`, so an operating point that needs more is
  infeasible. Earlier versions also replaced the path curvature `ay/v²` by at
  most 0.15 1/m in the steering geometry; below ~9 m/s (FS hairpins) that
  made the envelope 7–98 % too optimistic against the simulator. It only
  changes the envelope below ~8 m/s, so the road-car reference times below
  are unchanged (only the Spa centre-line times moved, by 1–5 ms).

### Check against the simulator

| check (911 GT3 RS unless noted) | planner | simulator | difference |
|---|---|---|---|
| one simulator step from the predicted steady state, 90 operating points on 3 cars (50/100/200 km/h, five pedal positions, 50 % and 98 % of the cornering limit): rate of change of speed | | | max 0.066 m/s², median 0.003 m/s² |
| … yaw acceleration in that step (should be 0) | | | max 0.12 rad/s² at 50 km/h (median 0.06), 0.045 at 100 km/h, 0.016 at 200 km/h |
| steady cornering 60 / 100 / 150 / 200 km/h (skid-pad with speed controller) | 1.196 / 1.220 / 1.249 / 1.285 g | 1.200 / 1.222 / 1.251 / 1.288 g | −0.1 … −0.3 % |
| same, 918 Spyder | 1.293 / 1.319 / 1.349 / 1.384 g | 1.297 / 1.321 / 1.349 / 1.387 g | 0 … −0.3 % |
| same, Taycan Turbo GT | 1.231 / 1.247 / 1.259 / 1.272 g | 1.235 / 1.250 / 1.261 / 1.275 g | −0.2 … −0.3 % |
| 0–200 km/h | 10.01 s | 10.10 s | −0.9 % (simulator has throttle lag and shifts) |
| 200–0 km/h | 112.3 m | 113.7 m | −1.2 % (brake lag ≈ 0.05 s ≈ 2.8 m) |
| 60–200 km/h with the shift model of §3 (911 / 918 / Taycan) | 8.25 / 4.65 / 6.35 s | 8.21 / 4.67 / 6.35 s | +0.4 / −0.4 / 0.0 % |
| *independent review (Nov 2026), yaw-rate-controlled skid-pad:* 911 GT3 RS 60 / 100 / 150 / 200 km/h | 1.196 / 1.220 / 1.249 / 1.285 g | 1.201 / 1.224 / 1.254 / 1.291 g | −0.3 … −0.5 % |
| … FS car (`fs_generic_ev`, TC, no ABS) 18 / 22 / 25 / 29 / 36 / 50 / 72 km/h | 0.77 / 1.06 / 1.21 / 1.35 / 1.42 / 1.53 / 1.74 g | 0.69 / 0.99 / 1.29 / 1.36 / 1.44 / 1.54 / 1.69 g | +12 / +8 / −6 / −1 / −1 / −0.5 / +3 % (see §4: low-speed kinematics, and the 72 km/h value is the `ay_steady` caveat) |
| … FS car 0–100 km/h / 100–0 km/h | 2.70 s / 21.4 m | 2.72 s / 22.2 m | −1.0 / −3.6 % (brake lag 0.04 s ≈ 1.1 m) |
| … 60–200 km/h with shift model (911 / 918 / Taycan) | 8.28 / 4.68 / 6.38 s | 8.23 / 4.68 / 6.36 s | +0.6 / 0.0 / +0.4 % |

The first two rows are the strongest test: the simulator is put into exactly
the state the planner predicts (speed, body slip, yaw rate, steering angle,
pedal) and advanced one step; if the planner's equations were not the
simulator's, the car would immediately change speed at a different rate or
start to rotate. `tests/test_planning.py` repeats this, so a future change of
the simulator's tyre or powertrain model that is not mirrored in `ggv.py`
fails the tests.

---

## 2. The racing line (`mv/planning/raceline.py`)

A line is its sideways offset `n(s)` from the centre line (positive = left).
The car's centre keeps a `margin` = half the car's width + 0.2 m (parameter
`safety_m`) from both edges, so the whole car stays on the track:
`−(w_right − margin) ≤ n ≤ +(w_left − margin)`.

### (a) Minimum curvature

Corner speed obeys `v² · curvature ≤ ay_max`: halve the curvature and the car
can go 41 % faster. So a good first answer is "the line that bends least":
minimise `∫ curvature² ds`. That is the classic outside–inside–outside line.

Curvature is a complicated function of `n`, but if the *direction* of the line
is frozen at the current guess it becomes linear in `n`. "Minimise a sum of
squares of linear expressions subject to simple bounds" is a **quadratic
programme** with a single global optimum. We solve it (projected Newton
method with sparse matrices: milliseconds per round for 2 000 variables),
update the frozen direction, and repeat; 4–6 rounds converge.

To handle 7 km tracks the optimisation runs on a point every 3 m; the final
line is a periodic cubic spline through those points, evaluated at every
track station, with curvature taken analytically from the spline.

### Why minimum curvature is close to, but not, minimum time

Minimum curvature treats every metre of the lap alike. The stopwatch does not:

* **Time is spent where the car is slow.** One unit of curvature in a 60 km/h
  hairpin costs far more time than the same curvature in a 200 km/h sweep.
* **Not every part of a corner is grip-limited.** In a braking zone the car is
  limited by braking, not cornering, so the line may bend more there for free
  ("trail braking" into a later, tighter turn-in).
* **Cars accelerate worse than they brake.** A late apex that straightens the
  exit and lets the driver use full throttle earlier pays off all the way down
  the next straight; minimum curvature gives entry and exit equal weight.
* **Path length.** The minimum-curvature line uses all the road even where a
  shorter path would cost no speed.

### (b) Minimum-time refinement

We keep the well-behaved quadratic programme but tell it what the stopwatch
cares about. For the current line we *measure* with the speed-profile solver
how much lap time a little extra curvature costs at each point (make the path
slightly tighter at one point, re-solve the lap; all ~1 500 perturbed laps are
solved in one batch), and then minimise

```
   Σ  w_i · curvature_i²        +     Σ  ds_i / v_i
      (w_i = time cost of curvature        (time to cover the path
       at point i / curvature_i)            at the current speeds)
```

At the current line the first sum has exactly the same slope as the lap time,
so each round moves the line downhill in lap time, while track limits and
smoothness are handled exactly by the QP. Speeds and weights are re-measured
on the new line and the round is repeated (7–15 rounds in practice) until three
rounds in a row gain less than 0.02 %. The three fastest lines found are then
re-evaluated at full resolution and the best one is kept; if none beats
minimum curvature, minimum curvature is returned (`meta['fell_back_to']`).

**Measured gain** of (b) over (a): 0.66 % to 2.29 % of lap time (mean 1.3 %),
positive on all 21 car/track combinations of §5. (b) is therefore kept as the
default (`method="auto"`).

*What did not work:* direct descent on lap time with a spline parameterisation
of `n(s)` and finite-difference gradients (L-BFGS-B). Wherever the line runs
along the track edge a perturbation is clipped, which puts a tiny kink in the
line; on a fast straight a kink of a few centimetres already looks like a
corner the car must slow for, so the lap time jumps by tenths of a second and
the "gradient" is noise (forward and backward differences were uncorrelated).
The optimiser stalled at the start on two of four tracks. The re-weighted QP
never creates kinks because the bounds are part of the problem it solves.

---

## 3. Speed along a given line: the forward–backward pass (`mv/planning/speed_profile.py`)

Now the line is fixed and so is its curvature at every station. Three sweeps:

1. **Corner limit.** At every station, the speed at which cornering alone
   would use all the grip: solve `v²·|curvature| = ay_max(v)`.
2. **Backward (braking) sweep.** Walk the lap backwards. "I must not be faster
   than `v[i+1]` at the next station; braking as hard as the envelope allows
   *while cornering as much as this station needs*, how fast may I be here?"
   `v[i]² = v[i+1]² + 2·|a_brake|·ds`, capped by the corner limit.
3. **Forward (acceleration) sweep.** Walk forwards. "From `v[i]`, with the best
   acceleration the envelope gives while cornering, how fast am I at the next
   station?" `v[i+1]² = v[i]² + 2·a_accel·ds`, capped by the result of sweep 2.

The lap is closed, so each sweep simply keeps going past the start line until
nothing changes (a fraction of a second lap). Lap time = Σ `ds / average speed`.
Each step averages the acceleration at both ends of the segment (Heun's
method): on a 1 m grid the lap time is within 0.02 % of the converged value.

### Worked example

A made-up car: cornering limit 12 m/s², braking 10 m/s², acceleration 5 m/s²,
and (to keep the arithmetic simple) the "square" rule: full braking or
acceleration is available until the cornering limit is reached. Stations are
20 m apart; stations C and D lie in a corner of radius 50 m (curvature 0.02).

| station | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| curvature (1/m) | 0 | 0 | 0.02 | 0.02 | 0 | 0 |
| 1. corner limit `sqrt(12/0.02)` | — | — | 24.5 | 24.5 | — | — |
| 2. backward: `v² = v_next² + 2·10·20` | 37.4 | 31.6 | 24.5 | 24.5 | — | — |
| 3. forward: `v² = v_prev² + 2·5·20` | 37.4 | 31.6 | 24.5 | 24.5 | 28.3 | 31.6 |

Backward from C: `v_B² = 24.5² + 400 = 1000` → 31.6 m/s; `v_A² = 1000 + 400` →
37.4 m/s. So the driver must be down to 37.4 m/s at A and brakes 40 m before
the corner. Forward from D: `v_E² = 600 + 200 = 800` → 28.3 m/s, `v_F² = 1000`
→ 31.6 m/s. The exit takes longer to regain speed than the entry took to lose
it (5 against 10 m/s²) — which is exactly why exits matter more than entries.
Time from A to F: `20/34.5 + 20/28.1 + 20/24.5 + 20/26.4 + 20/30.0 = 3.53 s`.

The real solver does the same with the full envelope: between C and D the car
is *at* the cornering limit, where our real car can only slow down slightly
(see §1), so the real minimum speed sits a little below the corner limit.

### Gear shifts

During an up-shift the simulator cuts the drive torque of that gearbox for the
shift time (911: 23 ms, 918: 60 ms, Taycan rear box: 100 ms). The forward
sweep does the same: when the speed crosses a shift speed, the interrupted
share of the drive force is removed for the distance covered in that time.
With it, 60–200 km/h matches the simulator within 0.4 % for all three cars
(without it the 918 is 3.7 % too quick). Cost on a lap of Imola: 0.11 s (911),
0.22 s (918), 0.03 s (Taycan).

---

## 4. Simplifying assumptions and what they do to the lap time

"Optimistic" = the reference is faster than a perfect driver could be in our
simulator; "pessimistic" = slower.

| assumption | effect on reference lap time | size (estimate) |
|---|---|---|
| Quasi-steady state: the car jumps from one steady state to the next. No yaw inertia, no steering-rate limit, no throttle/brake lag (50 ms), no load-transfer lag (60 ms) | **optimistic** | 0.5–1.5 %; largest in chicanes and quick direction changes |
| Tyres never used beyond the slip angle of peak grip; both tyres of an axle at the same slip angle | slightly **pessimistic** | < 0.3 % |
| Small-angle axle kinematics, same slip angle on both wheels of an axle: wrong at very low speed in very tight corners, where yaw rate × half track is 10–20 % of the speed | **either way** there | FS car below ~25 km/h: skid-pad +12 % (18 km/h) … −6 % (25 km/h); < 1 % above 30 km/h (FS) and at road-car circuit speeds (one-step check at 45 km/h: ay within 1 %). None of the stored laps goes below 48 km/h |
| `ay_steady` (skid-pad number printed by the CLI) assumes any acceleration between `ax_min` and `ax_max` can be held with one pedal; with TC that is not always so | reporting only, not lap time | FS car at 72 km/h: 1.74 g printed, 1.70 g holdable in the simulator |
| One pedal at a time (no left-foot braking / throttle against the brake to cancel engine braking) | slightly **pessimistic** | < 0.2 % |
| Feasible pedal range treated as one piece (see `ggv.py`); the envelope's "cliff" in combined braking/cornering is located to ~2 % in `ay` | either way | < 0.3 % |
| Gear = the one the automatic box is in while *accelerating* through that speed. The simulator's box changes down 7 % below the up-shift speed, so after braking to a speed just below a shift point the real car is briefly in too high a gear | **optimistic** | 0–0.3 % |
| Up-shift losses modelled, down-shift time (torque cut while braking) ignored | negligible | — |
| Battery never below its minimum charge; hybrid/EV always has full electric power. `meta['battery_lasts_the_lap']` reports whether one lap from the initial charge is possible | **optimistic** when the flag is `False` | can be seconds; none of the 21 laps in §5 runs out of charge (the 918 uses up to 65 % of its usable charge, at Spa) |
| Envelope tabulated every 2 m/s and interpolated; gear-change saw-tooth of the drive force is smoothed | either way | < 0.1 % |
| Line: car's centre stays `width/2 + 0.2 m` inside the track edge, no kerbs, no track-limit abuse | **pessimistic** against a driver who uses kerbs; equal to our simulator's rule "whole car on track" plus 0.2 m | measured at Imola (911, min. curvature): 0.2 m of buffer costs 0.39 %, 0.5 m costs 1.0 % |
| On the inside of a corner the line uses at most 70 % of the centre line's radius (`INSIDE_FRAC`). Only binds where the track is wider than its centre-line radius: La Source and the Bus Stop at Spa | **pessimistic** there | a few tenths of a second at Spa, zero elsewhere (not measured) |
| Line optimised on a 3 m grid with smooth splines; minimum-time search stops at 0.02 % gain per round and is a local search from the minimum-curvature line | slightly **pessimistic** | grid: ±0.15 % (2 m and 4 m grids gave −0.15 % and −0.06 % at Imola); search: unknown, probably a few tenths of a per cent. On FS-scale tracks (5 m corner radii) 3 m is coarse: ±0.3 % between 1–3 m grids and up to 7 cm of hard clipping; use `step_m=1.5` there |
| Speed profile integration (1 m steps, second order) | negligible | 0.02 % |
| Track is flat and 2-D: no elevation, banking, bumps; grip is the same everywhere | same as our simulator; **differs from Assetto Corsa** (either way) | track dependent |
| No tyre temperature or wear, no fuel burn, no wind, no slipstream | same as our simulator | — |

Net: against our own simulator the reference should be **optimistic by
roughly 1–2 %** (transients dominate). An RL lap within ~2 % of the reference
is therefore essentially at the limit of the simulated car; a gap of 5 % or
more is driver, not model. Beating the reference by more than a few tenths of
a per cent would mean the driver found something the reference excludes
(drifting, kerb/track-limit use beyond the margin) — worth a look at the
replay.

---

## 5. Results

All laps: car's centre at least half its width + 0.2 m inside the track edge,
up-shift losses included, 3 m optimisation grid, results at the track's 1 m
stations. "random N" = `random_track(N, "mixed")`. Times were measured on one
CPU core of the development PC while other jobs were running, so read them as
±30 %. The envelope is built once per car (4.7–5.0 s) and is
not included in the per-track times.

| car | track | length | centre line | (a) min. curvature | (b) min. time = reference | gain of (b) over (a) | top / slowest speed | compute: centre / (a) / (b) |
|---|---|---|---|---|---|---|---|---|
| 911 GT3 RS | Imola | 4900 m | 2:12.376 | 1:53.630 | **1:52.035** | 1.60 s (1.40 %) | 268 / 66 km/h | 1.2 / 1.4 / 19.0 s |
| 911 GT3 RS | Monza | 5787 m | 2:11.035 | 1:58.994 | **1:57.899** | 1.10 s (0.92 %) | 277 / 54 km/h | 0.8 / 1.4 / 21.2 s |
| 911 GT3 RS | Spa | 6991 m | 2:56.991 | 2:38.583 | **2:35.794** | 2.79 s (1.76 %) | 273 / 48 km/h | 1.6 / 1.9 / 27.1 s |
| 911 GT3 RS | Mugello | 5235 m | 2:16.383 | 2:05.532 | **2:02.655** | 2.88 s (2.29 %) | 272 / 85 km/h | 0.9 / 1.5 / 20.4 s |
| 911 GT3 RS | random 1 | 752 m | 0:25.510 | 0:23.025 | **0:22.671** | 0.35 s (1.54 %) | 168 / 82 km/h | 0.2 / 0.2 / 1.8 s |
| 911 GT3 RS | random 2 | 2055 m | 0:49.368 | 0:42.196 | **0:41.324** | 0.87 s (2.07 %) | 218 / 134 km/h | 0.5 / 0.6 / 6.8 s |
| 911 GT3 RS | random 3 | 4188 m | 1:25.605 | 1:19.199 | **1:18.601** | 0.60 s (0.76 %) | 280 / 65 km/h | 0.9 / 1.1 / 11.2 s |
| 918 Spyder | Imola | 4900 m | 2:01.447 | 1:43.506 | **1:42.186** | 1.32 s (1.27 %) | 302 / 70 km/h | 1.0 / 1.7 / 16.2 s |
| 918 Spyder | Monza | 5787 m | 1:58.889 | 1:48.139 | **1:47.187** | 0.95 s (0.88 %) | 310 / 51 km/h | 1.4 / 2.5 / 24.4 s |
| 918 Spyder | Spa | 6991 m | 2:42.880 | 2:25.643 | **2:23.840** | 1.80 s (1.24 %) | 305 / 52 km/h | 1.6 / 2.2 / 26.5 s |
| 918 Spyder | Mugello | 5235 m | 2:06.382 | 1:56.538 | **1:54.374** | 2.16 s (1.86 %) | 306 / 88 km/h | 1.1 / 1.4 / 23.8 s |
| 918 Spyder | random 1 | 752 m | 0:23.863 | 0:21.559 | **0:21.299** | 0.26 s (1.20 %) | 188 / 83 km/h | 0.3 / 0.2 / 1.9 s |
| 918 Spyder | random 2 | 2055 m | 0:45.819 | 0:39.208 | **0:38.633** | 0.58 s (1.47 %) | 247 / 140 km/h | 0.4 / 0.6 / 6.7 s |
| 918 Spyder | random 3 | 4188 m | 1:17.013 | 1:10.667 | **1:10.198** | 0.47 s (0.66 %) | 314 / 68 km/h | 0.8 / 1.2 / 9.5 s |
| Taycan Turbo GT | Imola | 4900 m | 2:08.099 | 1:49.419 | **1:48.269** | 1.15 s (1.05 %) | 292 / 67 km/h | 1.0 / 1.4 / 16.4 s |
| Taycan Turbo GT | Monza | 5787 m | 2:05.640 | 1:54.163 | **1:53.181** | 0.98 s (0.86 %) | 304 / 52 km/h | 1.1 / 1.2 / 22.1 s |
| Taycan Turbo GT | Spa | 6991 m | 2:51.446 | 2:33.407 | **2:31.332** | 2.08 s (1.35 %) | 298 / 50 km/h | 1.2 / 1.9 / 20.8 s |
| Taycan Turbo GT | Mugello | 5235 m | 2:12.389 | 2:01.900 | **1:59.639** | 2.26 s (1.85 %) | 298 / 84 km/h | 0.9 / 1.6 / 22.0 s |
| Taycan Turbo GT | random 1 | 752 m | 0:24.884 | 0:22.374 | **0:22.158** | 0.22 s (0.97 %) | 174 / 81 km/h | 0.1 / 0.1 / 2.1 s |
| Taycan Turbo GT | random 2 | 2055 m | 0:48.114 | 0:41.280 | **0:40.651** | 0.63 s (1.52 %) | 228 / 133 km/h | 0.3 / 0.6 / 6.0 s |
| Taycan Turbo GT | random 3 | 4188 m | 1:22.065 | 1:15.109 | **1:14.614** | 0.49 s (0.66 %) | 306 / 66 km/h | 0.9 / 1.2 / 10.8 s |

**Gain of the minimum-time refinement over minimum curvature:** 0.66 % to
2.29 %, mean 1.31 %, positive on all 21 combinations
(the code falls back to minimum curvature otherwise; it never had to). It
needed 7–15 rounds. It costs about 12 times the computing time of
minimum curvature, almost all of it in measuring the curvature sensitivities.
The minimum-curvature line alone is 7–15 % faster than the centre line.

The files behind this table are in `runs/reference/` (`<vehicle>__<track>.npz`
and `.png`); `scripts/evaluate.py` picks up the four circuit files for each car.

### Battery check (hybrid and electric car)

The envelope assumes full electric power everywhere. The table shows whether
one lap from the initial state of charge supports that (energy in kWh;
"usable" = initial charge minus the minimum the simulator allows).

| car | track | discharged | regenerated | net | usable | lasts the lap? |
|---|---|---|---|---|---|---|
| 918 Spyder | Imola | 3.60 | 1.03 | 2.57 | 5.44 | yes |
| 918 Spyder | Monza | 3.65 | 0.87 | 2.78 | 5.44 | yes |
| 918 Spyder | Spa | 4.86 | 1.32 | 3.54 | 5.44 | yes |
| 918 Spyder | Mugello | 3.10 | 0.85 | 2.25 | 5.44 | yes |
| 918 Spyder | random 1 | 0.51 | 0.23 | 0.29 | 5.44 | yes |
| 918 Spyder | random 2 | 1.30 | 0.29 | 1.01 | 5.44 | yes |
| 918 Spyder | random 3 | 2.63 | 0.59 | 2.04 | 5.44 | yes |
| Taycan Turbo GT | Imola | 10.17 | 2.46 | 7.71 | 82.45 | yes |
| Taycan Turbo GT | Monza | 11.78 | 2.19 | 9.60 | 82.45 | yes |
| Taycan Turbo GT | Spa | 13.66 | 3.15 | 10.51 | 82.45 | yes |
| Taycan Turbo GT | Mugello | 9.07 | 2.09 | 6.98 | 82.45 | yes |
| Taycan Turbo GT | random 1 | 1.44 | 0.54 | 0.89 | 82.45 | yes |
| Taycan Turbo GT | random 2 | 3.21 | 0.64 | 2.57 | 82.45 | yes |
| Taycan Turbo GT | random 3 | 9.04 | 1.54 | 7.50 | 82.45 | yes |

No lap runs out of charge, so the full-power assumption holds for a single lap from the initial state of charge. It would not hold for consecutive laps of the 918 (the net use per lap is a large share of its small battery).

---

## 6. Plausibility

### Against a driver in our own simulator

`mv/control/scripted.py` is a deliberately cautious rule-based driver that
follows the centre line. Its flying laps with the 911 GT3 RS (figures supplied
by the simulator team) against the reference:

| track | scripted centre-line driver | reference on the centre line | reference (min. time) | reference vs. scripted |
|---|---|---|---|---|
| Imola   | 2:26.6 | 2:12.4 | 1:52.0 | −23.6 % |
| Monza   | 2:24.2 | 2:11.0 | 1:57.9 | −18.2 % |
| Mugello | 2:33.0 | 2:16.4 | 2:02.7 | −19.8 % |
| Spa     | 3:17.0 | 2:57.0 | 2:35.8 | −20.9 % |

Both comparisons make sense. On the *same* (centre) line the reference is
9–11 % quicker than a driver that uses 72 % of the grip and half the braking
(lap time goes roughly with the square root of usable grip in corners:
√0.72 = 0.85, diluted by the straights). The racing line is worth another
10–15 % on these circuits, more where the track is wide relative to the
corner radii.

### Against Assetto Corsa (rough expectations only)

**Caveat first:** no measured Assetto Corsa laps for this car were available on this
machine. The numbers in the "expected" column are my recollection of what a
quick sim-racer does with the Kunos 911 GT3 RS on its standard tyres; treat
them as ±3–4 s. They are here to catch a gross error (a reference of 1:35 or
2:15 at Imola would be wrong), not to validate tenths.

| track | reference (911 GT3 RS) | rough expectation in AC | comment |
|---|---|---|---|
| Imola   | 1:52.0 | about 1:50–1:55 | flat-out on the main straight at 268 km/h, Tosa ~66 km/h: both in the expected range |
| Monza   | 1:57.9 | about 1:55–2:00 | 277 km/h before the first chicane (expected ~275–285 with slipstream-free air) |
| Mugello | 2:02.7 | about 1:58–2:04 | 272 km/h at the end of the straight; no slow corner below 85 km/h |
| Spa     | 2:35.8 | about 2:32–2:38 | 273 km/h on Kemmel; La Source 48 km/h |

So the reference lands where one would expect for this car — which is mildly
surprising, because the model differs from Assetto Corsa in ways that do not
obviously cancel:

* **Flat, 2-D track.** Spa, Mugello and Imola have large elevation changes.
  Compressions (Eau Rouge, Arrabbiata, Acque Minerali) add grip, crests and
  downhill braking zones (Imola's Rivazza, Mugello's Casanova–Savelli) remove
  it; uphill straights cost top speed. Our reference ignores all of it.
* **No kerbs, and a 0.2 m buffer.** A sim-racer uses every kerb and often a
  bit more; our line keeps the whole car inside the imported edge plus 0.2 m
  (worth 0.4 % at Imola, see §4). This makes the reference *slower* than AC.
* **Imported track width.** Widths come from AC's AI line file
  (`side_left/right`); whether they stop at the white line or include the
  kerb varies from track to track.
* **Tyres.** Our tyre is a simplified fit of the AC tyre (one friction value
  with load sensitivity, no temperature, no pressure, no wear, no camber).
* **A perfect, quasi-steady driver.** No reaction time, no correction, no
  transients: this makes the reference *faster* than any real driver
  (1–2 %, §4).

The realistic statement is therefore: the reference is the right order of
magnitude for the real car in Assetto Corsa (within a few seconds), and it is
a much sharper yardstick for drivers in *our* simulator, whose physics it
shares to within the tolerances of §1.

---

## 7. Limitations and next steps

* **The reference has not been driven in the simulator.** I tried: a simple
  tracker (steering and pedal fed forward from the steady-state model, plus
  Stanley-type feedback on position/heading and proportional speed control)
  lost the car in the first corner even on a reference computed with 80 % of
  the grip. The reason is instructive: on corner entry the reference
  trail-brakes with the *rear* axle at its limit, a state that in steady
  conditions needs a touch of opposite lock and is unstable without yaw
  feedback. A competent limit controller (or the RL driver) is needed to find
  out how much of that entry phase is really usable. Until then the
  "transients cost 0.5–1.5 %" in §4 is an estimate, not a measurement, and
  corner entry is the part of the reference most likely to be optimistic.
* Battery state of charge is checked after the fact, not planned. A proper
  energy-management reference (where to deploy and where to save) is a
  different optimisation.
* The minimum-time line is a local refinement of the minimum-curvature line
  and uses a quasi-steady car. A transient optimal-control solution would be
  the next level; expect it to move the line little and the lap time by the
  transient penalty above.
* Cars without TC/ABS are handled by the equations (wheelspin/lock-up beyond
  the peak). The FS car (TC, no ABS) was checked in review: one-step test,
  skid-pad and straight-line braking agree with the simulator (§1), and on
  `segment_track(seed, "fs_segments")` seeds 0–3 the lap times fall centre →
  minimum curvature → minimum time with the line inside the edges. Its
  envelope is still unreliable below ~25 km/h (§4). A car with neither TC nor
  ABS has not been tested.
* `margin` is the single most influential user parameter: check it against the
  track-limit rule the driver is actually judged by.
