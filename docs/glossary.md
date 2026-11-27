# Project MV — Glossary

Plain-language meanings of the terms used in the code and documents, grouped by
topic. Where a term has a symbol in the code it is given in `code font`.

## Learning

| Term | Meaning |
|---|---|
| Reinforcement learning (RL) | Learning by trial and error: try actions, get a score ("reward"), do more of what scored well. No human shows the right answer. |
| Policy / driver / actor | The neural network that turns an observation into an action. |
| Critic / value function | A second network that predicts how much reward the driver will collect from the current situation onward. Used only during training. |
| Distillation (teacher warm start) | Copying a trained driver (the teacher) into a new, e.g. bigger, network before training continues. Since WP44 both halves are copied: the actor (what to do) and the critic (how good the situation is). |
| Held-out (car or circuit) | Kept out of training on purpose, so how well the driver handles something it has never seen is measured, not assumed. |
| Observation | The numbers the driver is given each tick (98 of them; see architecture §4.3). |
| Action | What the driver outputs: steering and pedal, each between −1 and +1. |
| Reward | The score for one tick: metres of progress minus penalties. |
| Episode | One attempt, from a start state until a crash or the time limit. |
| Terminated / truncated | Episode ended because of the car's own doing (crash) / merely because time ran out. They are treated differently when learning. |
| Rollout | A short recording (32 decisions per car) of the current driver driving, used for one learning update. |
| PPO | Proximal Policy Optimisation — the learning algorithm used here. "Proximal" = each update may only change the driver a little. |
| Advantage | "Was this decision better or worse than expected?" Positive advantages are reinforced. |
| GAE | Generalised Advantage Estimation — a way of computing advantages that balances noise against bias. |
| Discount `gamma` | How much future reward counts. 0.995 at 25 Hz means the driver cares about roughly the next 8 seconds. |
| KL (divergence) | A measure of how much an update changed the driver's behaviour; the learning rate is steered to keep it small. |
| Exploration noise `std` | Random variation added to actions during training so new things get tried. Shrinks as the driver becomes sure. |
| Behaviour cloning / imitation | Supervised learning: copy what a teacher does. |
| DAgger | Imitation where the student drives part of the time and the teacher labels what it should have done, so the student learns to recover. |
| Warm start | Initialising the network by imitation before RL begins. |
| Domain randomisation | Varying physics (grip, mass, power, downforce) between episodes so the driver does not depend on exact values. |
| Sim-to-real / sim-to-sim gap | The difference between the training simulator and the target world (a real car, or Assetto Corsa). |
| Fine-tuning | Continuing to learn in the target world, starting from the simulator-trained driver. |
| Zero-shot | Using the trained driver in a new world with no further learning. |
| Checkpoint (`.pt`) | A saved driver. |
| Decisions per second | How many (car × tick) control decisions the simulator produces per wall-clock second. |

## Vehicle dynamics

| Term | Meaning |
|---|---|
| Body frame | Coordinate system fixed to the car: x forward, y left, z up. |
| Yaw, yaw rate `r` | Rotation about the vertical axis and how fast it changes. Positive = turning left. |
| `vx`, `vy` | Speed forwards and sideways in the body frame. |
| Slip angle | The angle between where a tyre points and where it is actually moving. Tyres only make cornering force when this is non-zero. |
| Peak slip angle | The slip angle at which a tyre grips most (~7° for these tyres). Beyond it the tyre slides and grip drops. |
| Slip ratio | How much faster (wheelspin) or slower (locking) a wheel turns than rolling freely would need. Tyres only make traction/braking force when this is non-zero. |
| Combined slip | A tyre has one grip budget; cornering and accelerating/braking share it. |
| Friction coefficient `mu` | Maximum tyre force divided by the load on the tyre. 1.3 means 1.3 g of grip. |
| Load sensitivity | Doubling the load on a tyre gives less than double the grip; this is why load transfer costs grip. |
| Magic formula (Pacejka) | The standard curve shape `D·sin(C·atan(B·slip))` for tyre force versus slip. |
| Load transfer | Under braking weight moves to the front tyres, under acceleration to the rear, in corners to the outside. |
| Roll stiffness distribution | How the cornering load transfer is shared between front and rear axle; sets understeer/oversteer balance. |
| Understeer / oversteer | Front tyres run out of grip first (car runs wide) / rear tyres do (car rotates, may spin). |
| CdA, ClA | Drag area and downforce area: force = ½ · air density · speed² · area. |
| Downforce | Aerodynamic force pressing the car down; adds grip at speed. |
| ABS / TC | Anti-lock brakes / traction control: electronics that stop the wheels from locking or spinning. |
| Trail braking | Still braking while turning into a corner. Fast, but easy to overdo. |
| g-g-v diagram | For each speed, the envelope of combined longitudinal and lateral acceleration the car can reach. The car's whole performance in one picture. |
| Racing line | The path through corners that minimises lap time; generally uses all the width to straighten the corner. |
| Apex | The point of a corner where the line comes closest to the inside edge. |
| Power unit | In this project: one engine or one electric motor. |
| SoC | State of charge of the battery, 0–1. |
| Regeneration | Electric motors braking the car and charging the battery. |

## Tracks and sensing

| Term | Meaning |
|---|---|
| Centre line | The line midway between the track edges; a track is stored as this line plus the distance to each edge. |
| Curvature `kappa` | 1 / corner radius, in 1/m. Positive = left-hander. 0.02 = 50 m radius. |
| Arc length `s` | Distance along the centre line from the start line. |
| Lateral offset `lat` | How far the car is to the left (+) or right (−) of the centre line. |
| Heading error | Angle between the car's direction and the track's direction. |
| Track preview / lookahead | Curvature and width of the track at set distances ahead. |
| Blind lap | A lap with the preview cut to sensor range, as on a first lap of an unknown track. |
| Lidar | A sensor that measures distance along rays. Here: distance to the track edge along 19 directions. |
| Track limits | The rule for how far off the track a car may go. Here (strict): no part of the body over the edge. |
| Standing / flying lap | A lap from a standstill on the start line / a lap entered at full speed. |
| Run-off | The area past the track edge (grass, gravel) before the barrier. In self-recovery training (WP61) it is 2-20 m wide, drawn per episode; reaching its end ends the episode. |
| Ploughing drag | The force that slows a car rolling through grass or gravel: the wheels push the surface aside. Modelled as a share of each wheel's load (0.02-0.30), acting against the direction of motion and not through the tyres. |
| Excursion | One trip off the road: three or more wheels past the edge, then (if all goes well) back on. AC invalidates a lap at three wheels out. |
| Self-recovery | Training mode where leaving the road does not end the episode or put the car back: the driver must rejoin by itself (race bot stage 2, WP61). |
| Own map | The stretch of circuit the car's sensors have already seen, remembered and shown as track preview on later laps (WP62). |
| Self-play | Training against copies of oneself: every car in a race is the same driver, so opponents get better as the driver does (WP64). |
| Restitution | How bouncy a collision is: 0 = the bodies stick together, 1 = they bounce apart as fast as they met. Car contacts use 0.3. |
| CUDA graph | A recording of a sequence of GPU operations that can be replayed with one command, saving the cost of launching each one from Python (WP63). |

## Assetto Corsa

| Term | Meaning |
|---|---|
| Equal machinery | Comparing two drivers in the same car. For the F1 benchmark (WP43) "the same car" is a car file fitted until our physics reproduces the real car's speeds at the corners and on the straights of that lap. |
| OpenF1 | A free, unofficial service that serves Formula 1 live-timing data (speed, gear, throttle, brake on/off, position) from 2023 on. The data belongs to Formula 1: used privately only (D-038). |
| Alignment (of a real lap) | Finding the rotation, scale and shift that lay a real car's recorded positions onto our copy of the circuit (iterative closest point: match each point to the nearest point of our circuit, solve for the best transform, repeat). |
| Timing line | Where a lap starts and ends. F1's timing line and the game's start line can be up to ~180 m apart; real laps are re-cut at our line. |
| AC | Assetto Corsa (the original, 2014) by Kunos Simulazioni. |
| CSP | Custom Shaders Patch — community extension to AC that, among much else, lets Lua scripts control the car. |
| Content Manager | The community launcher for AC. |
| `data.acd` | The packed file in each car folder holding its physics definition. |
| `fast_lane.ai` | The file in each track folder holding the AI racing line and the distance to both track edges. |
| Shared memory | A block of memory AC publishes with live telemetry that other programs can read. |
| Bridge | Our code that reads that telemetry and sends controls back (`mv/ac/bridge`). |
| Shadow environment | A one-car copy of the training environment kept in step with the game so observations are computed by the training code itself. |
| `ui_car.json` | The description file in each car folder (`ui/`): display name, class "race"/"street", tags such as `#GT3`, and advertised figures ("specs": bhp, weight, top speed, 0-100). The catalogue reads it to sort cars and to check our numbers. |
| DLC | Downloadable content - paid add-on car packs. Steam creates empty folders for DLC cars that are not owned; the importer reports them as "not installed". |
| bhp / whp | Brake horsepower = engine power (1 hp = 0.7457 kW). "whp" = horsepower measured at the wheels, after drivetrain losses, so ~10-20 % less. |
| Car catalogue | `mv/vehicle/catalog.py`: every car we can drive, its category, favourite star and check status. |
| Favourites | Car ids listed in `vehicles/favourites.yaml`; shown first (starred) wherever the control panel asks for a car. |
| Plausibility check | After importing a car: our power, mass and top speed against the game's own figures, plus a few seconds of full throttle and full braking in our simulator. "Suspicious" = usable but a number is far off; "failed" = unusable. |

## Racing car categories

| Term | Meaning |
|---|---|
| Formula / single-seater | Open-wheel, one-seat racing car (F1, Formula Abarth, Formula Student, IndyCar). |
| Prototype (LMP, Group C) | Purpose-built closed sports-racing car for endurance races: Le Mans Prototypes (LMP1/2/3) and their 1980s ancestors, Group C. |
| GT3 | Race cars derived from road sports cars, power and weight equalised by the organiser ("balance of performance"); ~370-440 kW, ~1250-1300 kg. |
| GT2 / GTE / GT1 | The faster professional GT classes (GTE = Le Mans GT Endurance; GT1 = the 1990s top GT class). |
| GT4 | Entry-level GT race cars, closer to the road car than GT3; ~300 kW. |
| Touring / cup car | Race cars from saloons (touring, DTM, Group A) and one-make "cup" series where everyone drives the same car. |

## Formula Student

| Term | Meaning |
|---|---|
| DV / DC | Driverless Vehicle / Driverless Cup. |
| AS | Autonomous System; its state machine (Off, Ready, Driving, Finished, Emergency) is defined in the rules. |
| EBS | Emergency Brake System. |
| RES | Remote Emergency System (the remote stop/go). |
| ASSI | The status lights showing the AS state. |
| Trackdrive | 10-lap driverless event on an unknown closed track. |
| Autocross | Single-lap driverless event on an unknown track. |
| Skidpad | Figure-of-eight event measuring cornering. |
| Acceleration | 75 m straight-line event. |
| DOO / OC | Down-or-out (cone knocked over) / off-course. |
