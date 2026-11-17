# Formula Student Rules 2027 – Driverless (DV) Digest

**Source:** "Formula Student Rules 2027", Version 1.0 (printed on every page footer). The PDF does not print a release date. Its file metadata says it was created on 2026-09-29.

Rule numbers in brackets can be checked against the PDF. "Not specified in the rules" means the 2027 rules PDF does not contain it. Many DV details (cone types, colours, exact track drawings, data logger signals, RES model) are delegated to the **Event Handbook**, which is a separate document [D 2.4.1, T 14.2.1, T 14.3.1]. The Event Handbook was not reviewed for this digest.

Abbreviations: AS = Autonomous System, ASB = Autonomous System Brake, EBS = Emergency Brake System, RES = Remote Emergency System, ASMS = AS Master Switch, ASSI = AS Status Indicator, AMI = Autonomous Mission Indicator, ASR = AS Responsible, SDC = Shutdown Circuit, TS = Tractive System, R2D = Ready-to-drive, DC = Driverless Cup, DOO = Down or Out (cone), OC = Off-Course, USS = Unsafe Stop, DQ = disqualified, DNF = Did Not Finish.

---

## 0. Competition structure (DV-relevant)

- Classes are CV (combustion, including hybrid) and EV. Cars from both classes may also enter the separate **Driverless Cup (DC)** [A 1.2.1, A 1.2.6].
- Maximum points [A 1.2.5, A 1.2.7, Table 1]:

| Discipline | CV & EV class | DC |
|---|---|---|
| Business Plan | 75 | – |
| Cost & Manufacturing | 100 | – |
| Engineering Design | 150 | 150 |
| Skidpad (manual) | 50 | – |
| DV Skidpad | 75 | 75 |
| Acceleration (manual) | 50 | – |
| DV Acceleration | 75 | 75 |
| Autocross (manual) | 100 | – |
| DV Autocross | – | 100 |
| Endurance | 250 | – |
| Efficiency | 75 | – |
| Trackdrive | – | 200 |
| **Overall** | **1000** | **600** |

- In the CV/EV classes, DV Skidpad and DV Acceleration are compulsory scored disciplines. They are worth 150 points even for a team that never enters the DC [Table 1].
- Points per discipline and overall are bounded to 0 ≤ P_team ≤ P_max [A 1.2.8].
- DC Engineering Design weights [Table 6]: Software/Compute 35, Vehicle Performance 30, Autonomous Functionality 85. There are separate DC Engineering Design finals [S 4.3.7, S 4.3.8]. Engineering Design also assesses the AS hardware and software [S 4.1.3].
- **Runs per discipline** [D 1.3.1, Table 7]:

| Discipline | Manual runs | Autonomous runs |
|---|---|---|
| Acceleration | 2 | – |
| DV Acceleration | – | 2 |
| Autocross | 4 | – |
| [DC] DV Autocross | – | 2 |
| Endurance | 1 | – |
| Skidpad | 2 | – |
| DV Skidpad | – | 2 |
| [DC] Trackdrive | – | 1 |

- **Vehicle Status Video, DC** [A 5.6.2–A 5.6.5]: the video must be driven in autonomous mode. It must show straight driving of at least 30 m [A 5.6.3; the 10 km/h minimum applies to manual mode only], a 180° turn, and straight driving back. It must end with an emergency brake manoeuvre (T 15). The ASSI must be visible. A time-synchronised split screen must show the onboard view plus a visualisation of perception and path planning. Maximum length is 60 s. Failing only the DC part de-registers the team from the DC only.

---

## 1. Dynamic disciplines (as they apply to driverless)

### 1.1 General definitions and procedures

- **Start/finish:** a lap starts when the car crosses the light barrier at the start line and ends at the finish line, which can be the same line [D 1.1.1, D 1.1.2]. A run starts with the Green Flag or the officials' go-signal [D 1.1.4]. A DNF is scored as a DQ [D 1.1.5]. At least one valid run is needed to score [D 1.1.9], except in Trackdrive (see 1.6).
- **Turning diameter** = diameter of the smallest circle within which the vehicle can turn round completely [D 1.1.10].
- **Track walks** take place before Autocross, Endurance and Trackdrive. Only analog measuring devices may be used [D 1.5.1, D 1.5.2].
- **No flags in autonomous mode** [D 2.1.2].
- **Wireless / teleoperation:** after the car crosses the start line, nobody may change parameters, send commands or change software over a wireless link until the car crosses the finish line. One-way telemetry from the car is allowed [D 2.2.1]. The RES is the only device that may send wireless commands at all times [D 2.2.2].
- **No reverse driving** [D 2.3.2]. For EVs, wheels must not be spun in reverse [EV 2.2.4]. Burnouts are prohibited [D 2.3.5].
- **Race control:** an ASR must be at race control with the RES. One monitoring device (laptop, tablet, …) may be brought, with "no complicated antenna construction" [D 2.3.6].
- **Start-up at the line** [D 2.5]:
  - No additional equipment (laptop, jack, pressure tank, …) may be used to start the car at the staging/start line [D 2.5.1].
  - The car must reach "AS Ready" within **1 min** of staging, or it may be sent back and lose priority [D 2.5.2].
  - The car must be staged with the steering straight [D 2.5.3].
  - The ASR switches on the ASMS only after an official approves it at the start line [D 2.5.4].
  - If the car cannot start the run within **30 s** of the Green Flag/go-signal, it is DQ for that run [D 10.1.11].
- **Standstill during a run:** the car has up to **30 s** to continue driving. Otherwise it is DQ [D 2.6.1].
- **RES signal loss:** a re-run may be granted if the loss is traceable and provably not self-inflicted. Data logger data must be uploaded before notifying officials [D 2.6.3].
- **After the run:** the ASR deactivates the car with the RES after officials approve. The ASR plus one more team member then collect the car [D 2.7.1, D 2.7.2].
- **Re-runs** may be granted for: a Red Flag, RES use to keep the track safe, traceable RES signal loss, or technical/organisational issues decided by officials [D 2.8.1]. An accepted re-run replaces the original run [D 2.8.3].
- **Cones:** cones that are DOO are **not reset** during autonomous running. There is no re-run for cones in the driving path or for "disorientation due to missing cones" [D 10.1.3].
- **Cone details, colours, sizes and detailed track figures:** these are in the Event Handbook [D 2.4.1]. **Cone colours, which side each colour is on, cone sizes and cone spacing (except Acceleration and slaloms) are not specified in the rules.**
- **Weather and tyres:** track conditions are Dry, Damp or Wet, set by the officials [D 3.1]. Allowed tyres: Dry → dry tyres; Damp → dry or wet; Wet → wet [D 3.2.1]. The one-set-per-type limit does **not** apply to DV Autocross, Trackdrive, the Brake Test or the practice area [D 3.2.4].

### 1.2 Penalties [D 10.1]

| Penalty | Acceleration | Skidpad | Autocross | Endurance | Trackdrive |
|---|---|---|---|---|---|
| DOO (per cone) | 2 s | 0.2 s | 2 s | 2 s | 2 s |
| OC | DQ | DQ | 10 s | 10 s | 10 s |
| USS | DQ | DQ | DQ | n/a | −50 points |

[D 10.1.7]

- **DOO:** the cone is knocked over, or its entire base lies outside the box marked around its undisturbed position [D 10.1.2]. The penalty counts every DOO, including entry/exit gate cones before the start and after the finish [D 10.1.4].
- **OC:** all four wheels are outside the track boundary marked by the edge markings. Missing one or more slalom gates (Autocross/Endurance) counts as one OC per occurrence. The car must re-enter at the next possible point [D 10.1.5].
- **USS:** in autonomous running, the car does not stop within the specified area, and/or does not keep the expected orientation while stopping, and/or does not enter the finish state (AS Finished) within **30 s** after stopping [D 10.1.6].
- Penalties are not assessed for accident avoidance [D 10.1.1].
- **Post-inspection rule violations** (per violation, applied to all runs since the car last entered the dynamic area) [IN 12.1.4]:

| | Acceleration | Skidpad | Autocross | Endurance | Trackdrive |
|---|---|---|---|---|---|
| Group A (no advantage) | 0.3 s | 0.2 s | 2 s | 30 s | 30 s |
| Group B (advantage) | 1 s | 0.6 s | 5 s | 2 min | 2 min |

- Losing a part of the car on track → DQ for that run [IN 12.1.6]. Safety-related violations → DQ since entering the dynamic area [IN 12.1.5].
- **EV power/voltage violation:** power above 80 kW, current above 500 A or voltage above 600 V, judged on a 500 ms moving average of the data-logger signal. Each violation DQs the fastest valid run of that discipline [D 10.4.1, D 10.4.2]. Missing data-logger data counts as a violation [D 10.4.4].

### 1.3 DV Acceleration [D 5]

| Item | Value | Rule |
|---|---|---|
| Layout | Straight, **75 m** from start line to finish line | D 5.1.1 |
| Track width | **≥ 3 m** | D 5.1.1 |
| Cone spacing | Along the track at intervals of **about 5 m** | D 5.1.1 |
| Laps per run | 1 | D 5.2.1 |
| Runs (autonomous) | 2 | Table 7 |
| Staging | Foremost part of car **0.30 m behind** the start line | D 5.2.3 |
| Stop | Full stop within **75 m after** the finish line, all four wheels inside the track boundaries, then enter AS Finished | D 5.2.4 |
| Max run time | Run time without penalties **> 25 s → DQ** | D 9.2.1 |
| Penalties | DOO 2 s; OC DQ; USS DQ | D 10.1.7 |

### 1.4 DV Skidpad [D 4]

| Item | Value | Rule |
|---|---|---|
| Layout | Two pairs of concentric circles in a figure eight | D 4.1.1 |
| Circle centre distance | **18.25 m** | D 4.1.2 |
| Inner circle diameter | **15.25 m** | D 4.1.2 |
| Outer circle diameter | **21.25 m** | D 4.1.2 |
| Driving path width | **3 m** between inner and outer circles; entry/exit path also 3 m wide, tangent to the circles where they meet | D 4.1.5 |
| Cones | **17** around the inside of each inner circle; **13** around the outside of each outer circle (pattern per Figure 22) | D 4.1.3 |
| Line marking | Each circle marked with a line outside the inner circle and inside the outer circle | D 4.1.4 |
| Start/finish line | The line between the two circle centres | D 4.1.6 |
| Lap | One trip around one circle, from start/finish line to start/finish line | D 4.1.6 |
| Staging | Foremost part of the car **15 m in front of** the start line | D 4.2.4 |
| Stop | Full stop within **25 m after** the finish line, all four wheels in bounds, then AS Finished | D 4.2.5 |
| Max run time | Run time without penalties **> 25 s → DQ** | D 9.2.1 |
| Penalties | DOO **0.2 s**; OC DQ; USS DQ | D 10.1.7 |

**Run sequence** [D 4.2.1], 4 laps:
1. Enter perpendicular to the figure eight.
2. Lap 1: right circle, untimed, to establish the turn.
3. Lap 2: right circle, **timed**.
4. Lap 3: left circle, untimed.
5. Lap 4: left circle, **timed**.
6. Exit perpendicular, in the same direction as the entry.

**Run time** = mean of the timed right lap and the timed left lap. Penalties are added after averaging [D 4.2.2].

Figure 22 (rendered from the PDF): the car enters from one side of the straight tangent corridor. The right circle is driven clockwise and the left circle counter-clockwise. The figure's legend shows two cone symbols (open and filled squares), but the rules text does not say what colours they are.

### 1.5 DV Autocross (DC only) [D 6]

- **Layout for DC:** built to the **Trackdrive guidelines of D 8.1** instead of the manual Autocross guidelines [D 6.1.3]. That means straights **≤ 80 m**, minimum track width **3 m**, minimum required turning diameter **9 m**, with chicanes, multiple turns, decreasing-radius turns and hairpins. D 8.1 has no slalom rule.
- Lap length: D 8.1.2 gives **≈ 200–500 m**. Whether the manual Autocross limit of < 1.5 km [D 6.1.2] still applies alongside it is not stated. See Ambiguities.
- For reference, the manual Autocross layout is [D 6.1.1, D 6.1.2]: straights ≤ 80 m; slaloms are cones in a straight line with **7.5–12 m** spacing; minimum width 3 m; minimum turning diameter 9 m; length < 1.5 km.
- 1 lap per run [D 6.2.1]. 2 autonomous runs [Table 7].
- **Staging:** front wheels **6 m in front of** the start line, on the track [D 6.2.3].
- **Stop:** full stop within **30 m after** the finish line, four wheels in bounds, then AS Finished [D 6.2.4].
- Penalties: DOO 2 s, OC 10 s, USS DQ [D 10.1.7].
- Track walk is allowed, with analog devices only [D 1.5].

### 1.6 Trackdrive (DC only) [D 8]

| Item | Value | Rule |
|---|---|---|
| Layout | Closed-loop circuit | D 8.1.1 |
| Straights | **≤ 80 m** | D 8.1.1 |
| Features | Chicanes, multiple turns, decreasing-radius turns, hairpins, etc. | D 8.1.1 |
| Min track width | **3 m** | D 8.1.1 |
| Min required turning diameter | **9 m** | D 8.1.1 |
| Lap length | **≈ 200 m to 500 m** | D 8.1.2 |
| Laps per run | **10** | D 8.3.1 |
| Runs | 1 | Table 7 |
| Staging | Front wheels **6 m in front of** the start line | D 8.3.2 |
| Stop | After 10 laps: full stop within **30 m after** the finish line, four wheels in bounds, then AS Finished | D 8.3.3 |
| Lap counting | **No last-lap signal**; the car must count laps itself | D 8.3.4 |
| Timing | Each lap timed individually; corrected time = total time + penalties | D 8.3.5 |
| Min speed | DQ if the average speed of the first 3 laps (checked after lap 3) is **< 2.5 m/s**, or if the average of any later lap is **< 3.5 m/s** | D 2.6.2 |
| Penalties | DOO 2 s; OC 10 s; **USS −50 points** | D 10.1.7 |
| Running order | Set by officials from earlier dynamic results | D 8.2.1 |
| Afterwards | Car goes to parc fermé | IN 12.1.2 |

### 1.7 Endurance and Efficiency

Endurance and Efficiency are **manual-only** disciplines. Table 7 lists no autonomous runs for them. They are listed here only as layout references:
- Closed lap; straights ≤ 80 m; slaloms **9–15 m** spacing; min width 3 m; min turning diameter 9 m [D 7.1.1].
- Lap ≈ 1 km; total ≈ 22 km [D 7.1.2, D 7.1.3].
- Too slow: two consecutive laps averaging < 8 m/s (dry) or < 5 m/s (wet/damp) [D 10.2.4].

### 1.8 Scoring formulas

**Default formula** [D 9.1.1]. The PDF typesets the bracketed term with a superscript 2, so it is squared:

`SCORE = (P_max − P_min) · ((T_max − T_team) / (T_max − T_min))² + P_min`

- T_team = the team's best time including penalties, capped at T_max.
- T_min = the fastest car's time including penalties.
- Constants [Table 9]:

| Discipline | T_max | P_min |
|---|---|---|
| Skidpad (manual) | 1.35 T_min | 0.05 P_max |
| Acceleration (manual) | 1.7 T_min | 0.05 P_max |
| Autocross (manual) | 1.4 T_min | 0.1 P_max |
| Endurance | 1.5 T_min | 0.1 P_max |
| **[DC] Skidpad** | **1.7 T_min** | **0.05 P_max** |
| **[DC] Acceleration** | **2.25 T_min** | **0.05 P_max** |

**DV Skidpad / DV Acceleration in the CV/EV classes** [D 9.2]:
- Runs with a run time without penalties of **> 25 s** are DQ [D 9.2.1].
- `DV_SCORE = P_max · (N_all + 1 − R_DV,team) / N_all` [D 9.2.2]
  - R_DV,team = rank of the team's best autonomous time, including penalties, among all teams' best autonomous times.
  - N_all = number of teams with at least one valid manual **or** autonomous run.
  - This is rank-based, so only finishing order matters, not the size of the time gap.

**Driverless Cup** [D 9.3]:
- DV Skidpad and DV Acceleration use D 9.1.1 with the [DC] rows of Table 9 [D 9.3.1].
- **DV Autocross** [D 9.3.2]. The formula is linear (no exponent in the PDF):
  `AUTOCROSS_SCORE = 0.9 P_max · (T_max − T_team,total) / (T_max − T_min) + 0.1 P_max`
  - `T_team,total = min(T_team,1, avg(T_team,1, T_team,2))`. Run 1 always counts. Run 2 can only help by improving the average.
  - **T_max = time to drive the lap at 6 m/s.**
  - T_min = the fastest T_team,total of all teams.
  - A DNF/DQ run, or a run slower than T_max, is set to T_max.
- **Trackdrive** [D 9.3.3]:
  `TRACKDRIVE_SCORE = 0.75 P_max · (T_max / T_team − 1)`
  - T_team = the corrected elapsed time, capped at T_max.
  - T_max = 2 × the fastest car's corrected elapsed time.
- **Per-lap bonus:** in deviation from D 1.1.9, an extra **2.5 % of P_max per completed lap** is awarded [D 9.3.4]. For Trackdrive (P_max 200): 10 laps × 5 = 50 points, plus up to 150 for time.

**Efficiency** (manual Endurance only) [D 9.4]:
- `EFFICIENCY_SCORE = P_max · ((EF_max − EF_team) / (EF_max − EF_min))²`
- EF_max = 2 · EF_min.
- `EF = T² · E`, where T = uncorrected driving time and E = energy (EV) or corrected fuel mass (CV).

---

## 2. Autonomous System technical rules

### 2.1 Modes and general

- **Manual mode:** a human drives and the ASMS is off [T 1.4.1].
- **Autonomous mode:** the AS is activated, and **no person may be inside the vehicle** [T 1.4.2].
- A full AS according to T 14 is required to run in autonomous mode [T 14.1.1].
- The car must pass AS Inspection and the EBS Test before running autonomously [IN 1.2.2].
- First-year rule: an existing chassis may be reused for one more year when an AS is implemented for the first time [A 2.2.3].
- **ASR** [A 4.4]: at least one is required. For EVs the ASR also fulfils the ESO role. ASRs plus ESOs may number at most 4. The ASR is the only person who may declare the AS safe. If there is only one ASR, that person must not be a driver. ASR qualification is a bachelor's degree in CS, EE, mechatronics, automation, robotics or similar.

### 2.2 AS state machine [T 14.8, Figure 16]

Figure 16 (rendered from the PDF) is evaluated top-down as a decision tree:

```
EBS activated?
├─ no  → (Mission selected AND ASMS On AND ASB checks OK AND TS active)?
│        ├─ no  → AS Off
│        └─ yes → R2D?
│                 ├─ yes → AS Driving
│                 └─ no  → Brakes engaged?
│                          ├─ yes → AS Ready
│                          └─ no  → AS Off
└─ yes → (Mission finished AND vehicle at standstill)?
         ├─ no  → AS Emergency
         └─ yes → SDC open at RES?
                  ├─ yes → AS Emergency
                  └─ no  → AS Finished
```

- **EBS "activated"** means the EBS supply path of T 15.2.2 is cut after the initial checkup has passed. Brakes can then only be released at the manual deactivation points [T 14.8.1]. The EBS stops counting as activated once all brakes have been released there [T 14.8.2].
- **Ready → Driving:** R2D may only be entered by the RES **"Go"** signal, and only after the system has been in **AS Ready for ≥ 5 s** [T 14.8.4].
- **Driving → motion:** the car must not move until it has been in **AS Driving for ≥ 3 s** [T 14.8.5].
- **Steering actuation** is allowed only in R2D or during AS Emergency [T 14.6.1].
- AS Ready can only be reached after the ASB initial check has passed [T 15.3.1].
- **Finish:** at the end of every DV discipline the car must enter **AS Finished** within the stop zone. Not doing so within 30 s of stopping is a USS [D 4.2.5, D 5.2.4, D 6.2.4, D 8.3.3, D 10.1.6].
- **SDC closing by the AS** [T 14.4.1]:
  - Manual mission: only once the AS has checked that the ASB is deactivated.
  - Autonomous mission: only with the ASMS on and enough brake pressure built up (brakes closed).
- **EV:** the AS must not be able to (re)activate the TS [EV 4.11.5]. R2D is left immediately when the SDC opens [EV 4.11.8].
- **CV:** the AS must not be able to (re)start the engine [CV 1.2.4]. For CVs, "R2D" means engine running with a gear engaged [T 14.1.2].

### 2.3 ASSI lights and sound [T 14.9]

| AS state | ASSI |
|---|---|
| AS Off | off |
| AS Ready | yellow, continuous |
| AS Driving | yellow, flashing |
| AS Emergency | blue, flashing |
| AS Finished | blue, continuous |

- Flashing is continuous at **2–5 Hz**, 50 % duty cycle. ASSIs must have no other function [T 14.9.1].
- There are **three** ASSIs [T 14.9.2]:
  - one on each side behind the driver's compartment, in a region **160 mm below the top of the main hoop and 600 mm above the ground**;
  - one at the rear on the centreline, in a region **160 mm below the main-hoop top and 100 mm above the brake light**.
- At least one ASSI must be fully visible from any angle at 1.60 m height within a 3 m horizontal radius of the main-hoop top, and in direct sunlight [T 14.9.3].
- Each ASSI must meet the system status light rules [T 14.9.4, T 11.10.1]: black background; ≥ 15 cm² illuminated area; LEDs ≤ 20 mm apart if there is no diffuser; a single LED line must be ≥ 150 mm long.
- **AS Emergency sound** [T 14.9.5]: intermittent at 1–5 Hz, 50 % duty cycle, **80–90 dB(A)** (fast weighting, 2 m radius), lasting **8–10 s** after entering AS Emergency.
- SCS safe state for the ASSI is indicating "AS Off" [T 11.9.5].

### 2.4 RES – Remote Emergency System [T 14.3]

- A standard RES specified in the Event Handbook is required. It has a remote control and a vehicle module, and the team buys it [T 14.3.1, T 14.3.2].
- **Functions** [T 14.3.3]:
  - the emergency stop button opens the SDC;
  - the **"Go"** button starts the preselected autonomous mission.
- The RES relay is hard-wired in series with the shutdown buttons, connected directly (T 1.3.1) [T 14.3.4]. In manual driving it is bypassed by a normally-closed relay supplied by the ASMS [T 14.3.5].
- The antenna must be unobstructed [T 14.3.6].
- The RES is part of the EBS supply chain [T 15.2.2]. It is the only wireless command device allowed [D 2.2.2].

### 2.5 EBS / ASB [T 15]

- An ASB with an integrated EBS is required for autonomous mode [T 15.1.1]. All ASB parts must be inside the rollover protection envelope, except the T 6 brake system and the deactivation points [T 15.1.2]. The TS does not count as a brake [T 15.1.3].
- Manual braking must always be possible. If manual and autonomous braking act together, the higher pressure applies [T 15.1.4]. Master cylinders must not be in series [T 15.1.5].
- **Deactivation (brake release):** two points [T 15.1.7]. Each must [T 15.1.8]:
  - work without electrical power;
  - be next to the other point and within the surface envelope;
  - be near the ASMS, or on top of the car between the front bulkhead and front hoop near the centreline;
  - be protected against accidental actuation (e.g. a cone strike);
  - need at most 2 simple push/pull/turn actions;
  - be marked "Brake release" and have a red handle.
- **EBS energy:** only passive systems designed to store energy mechanically. Loss of electrical power must directly trigger an emergency brake manoeuvre [T 15.2.1].
- **EBS supply chain** [T 15.2.2]: LVMS → ASMS → forcibly-guided/mirrored NO contact of the RES-bypass relay → RES → an undelayed relay supplied by the SDC.
- **Functional safety** [T 15.3]:
  - An initial check (ASB works and builds pressure as expected) must pass before AS Ready [T 15.3.1].
  - The ASB and its SCS are then monitored continuously [T 15.3.2].
  - The check, monitoring and brake actuation must be on **one PCB** [T 15.3.3].
  - The car must go automatically to the safe state if EBS function cannot be ensured, or if one more single failure would cause total loss of braking [T 15.3.4].
  - **Safe state** = car at standstill, brakes engaged, SDC open [T 15.3.5]. It is reached by an autonomous brake manoeuvre [T 15.3.6].
- **Performance** [T 15.4]:
  - **System reaction time** (SDC opens → deceleration starts) **≤ 200 ms** [T 15.4.1].
  - **Mean deceleration > 10 m/s²** on a dry track [T 15.4.2].
  - With a single failure, the ASB "should" achieve **≥ half** of that, i.e. 5 m/s² [T 15.4.3].
  - The car must stay stable while braking [T 15.4.4].
- The data logger gets a basic signal set plus a vehicle-specific set that the ASB monitors [T 14.2.2].
- "Brake-by-wire" is prohibited in **manual** mode only [T 6.1.6]. If a brake circuit is disconnected, the ASB must fail its initial check [T 6.1.4].

### 2.6 ASMS – AS Master Switch [T 14.5, T 11.2]

- Rotary mechanical switch with a red removable handle at least 50 mm wide. It must be direct-acting [T 11.2.1]. It sits on the right side of the car near the main hoop, next to the other master switches [T 11.2.2, T 11.2.5]. "ON" is horizontal [T 11.2.3].
- Mounted in the middle of a fully **blue** circle of at least 50 mm diameter, marked "AS" [T 14.5.2, T 14.5.3].
- Power to all steering and brake actuators is switched by the LVMS **and** the ASMS. A relay coil supplied by the ASMS may be used in parallel [T 14.5.4].
- **ASMS Off** [T 14.5.5]:
  - no steering, braking or propulsion actuation can come from the AS;
  - sensors and computers may stay powered;
  - the car must be pushable and drivable manually.
- **ASMS On** [T 14.5.6]: nobody may be in the car. No manual steps except TS activation are allowed.
- Lockout/tagout is required. The ASMS is locked off outside the dynamic area and in manual driving [T 14.5.7, A 6.6.4].
- External start: EVs have an external TS activation button near the TSMS, which works only when the ASMS is On [EV 4.11.3]. CVs have an external engine start button that works only with the ASMS On and the gearbox in neutral, plus a green "N" light [CV 1.2.2, CV 1.2.3].

### 2.7 Steering and actuators

- Steering may move only in R2D or AS Emergency [T 14.6.1]. With the ASMS Off, manual steering must work without manual release steps [T 14.6.2].
- The ban on cable/belt steering does not apply to autonomous steering actuators [T 2.8.1]. Total steering free play is ≤ 7° at the wheel [T 2.8.4]. Positive steering stops on the rack are required [T 2.8.3].
- Actuators may be decoupled, and sensors moved between multiple fixed positions, if all positions comply with the rules. Parts must stay attached, with no tethers, and be locked [T 14.7]. This is allowed after inspection [IN 1.5.1].

### 2.8 Missions [T 14.10]

- Required missions: **Acceleration, Skidpad, [DC] Autocross, [DC] Trackdrive, EBS Test, Inspection, Manual Driving** [T 14.10.1].
- **Inspection mission:** with the car jacked up and all wheels removed, slowly spin the drivetrain and move the steering in a sine wave. After **25–30 s**, transition to AS Finished [T 14.10.2].
- The mission must be selectable without an external device [T 14.10.3]. It is shown on the **AMI**, which must be readable by someone standing next to the ASMS [T 14.10.4, T 14.10.5].

### 2.9 Sensors, computing, data logger, comms

- **Sensor types:** no rule restricts sensor type (lidar, camera, radar, GNSS, etc.), count or performance. Not specified in the rules.
- **Placement:** all sensors, cameras and components, including housings and mounts, must be within the **surface envelope** (T 1.1.18) or within the aerodynamic box of T 8.2 [T 11.12.3].
  - The T 8.2 box: ahead of the front tyres < 350 mm high and ≤ 700 mm forward of the front tyres; rearward of the head-restraint plane < 1100 mm high and ≤ 250 mm behind the rear tyres; plus width limits [T 8.2.1–T 8.2.3].
  - Passive antennas with longest side < 100 mm may stick out of the surface envelope, but their mounts may not [T 11.12.4].
  - Sensors must be securely mounted and must never touch the driver's helmet [T 11.12.1, T 11.12.2].
- **Legal:** sensors must meet local law, e.g. laser eye-safety class and radar power [T 11.13.1]. Datasheets and proof of compliance for all perception sensors are required at AS Inspection [IN 6.1.1].
- **Data logger:** a standard logger is supplied by the officials and must be installed. Signal sets and interfaces are defined in the Event Handbook [T 14.2.1, T 14.2.2]. The EV logger measures TS voltage and current [EV 4.6.1].
- **Wireless:** see D 2.2 above. SCS sent over wireless must detect data corruption and lost or delayed messages. Maximum message delay is 500 ms [T 11.9.2(d), T 11.9.4].
- **Pre-loaded maps / track knowledge:** **not specified in the rules.** The only related rules are: track walks with analog devices only [D 1.5.2]; no wireless commands or parameter changes after the start line [D 2.2.1]; no laptop at the start line [D 2.5.1]; the mission is selected on the car [T 14.10.3]. "Software calibration changes" between runs are allowed [IN 1.5.1].

### 2.10 Allowed human interaction (autonomous mode)

1. At the line: an official approves, then the ASR switches on the ASMS [D 2.5.4]. TS activation, or engine start for CVs, is the only other manual step allowed [T 14.5.6]. The mission is selected on the car [T 14.10.3].
2. The ASR at race control presses RES "Go" [T 14.3.3, D 2.3.6].
3. During the run: only the RES emergency stop. Telemetry is one-way only [D 2.2].
4. After AS Finished: the ASR deactivates via the RES after officials approve, then two team members collect the car [D 2.7].

---

## 3. Vehicle-level limits that constrain the policy

| Item | Value | Rule |
|---|---|---|
| EV max TS power (TSAC outlet) | **80 kW** | EV 2.2.1 |
| EV max TS current | **500 A** | EV 2.2.2 |
| EV max TS voltage | **600 V DC** | EV 4.1.1 |
| Power-check filter | 500 ms moving average | D 10.4.1 |
| Regen | Allowed, power unrestricted | EV 2.2.3 |
| CV engine | Four-stroke piston, ≤ 710 cm³ | CV 1.1.1 |
| LV system max voltage | 60 V DC / 50 V AC RMS | T 11.1.2 |
| Wheelbase | **≥ 1525 mm** | T 2.9.1 |
| Track ratio | Smaller track ≥ **75 %** of the larger | T 2.9.2 |
| Min track width (absolute) | not specified in the rules | – |
| Rollover stability | Tilt table at **60°**, all wheels in contact | IN 7.1.3 |
| Ground clearance | ≥ 30 mm static | T 2.2.1 |
| Suspension travel | ≥ 50 mm usable, ≥ 25 mm jounce | T 2.5.1 |
| Rear-wheel steering | Allowed, mechanically limited to ±6° (max 6°) | T 2.8.11 |
| Wheel layout | 4 wheels, open-wheel, single seat | T 2.1.2 |
| Reverse driving | Prohibited | D 2.3.2, EV 2.2.4 |
| **Max driverless speed** | **not specified in the rules** | – |
| Min speed, Trackdrive | 2.5 m/s (laps 1–3 average), 3.5 m/s (any later lap) | D 2.6.2 |
| Min speed, DV Autocross scoring | Runs slower than a 6 m/s lap score as T_max (0.1 P_max) | D 9.3.2 |
| Max run time, DV Accel/Skidpad | 25 s | D 9.2.1 |
| Tyres | Dry set = tyres presented at inspection; wet tyres need ≥ 2.4 mm moulded/approved tread; same make, size and compound per axle; no tyre warmers; no traction agents | T 2.7.1–T 2.7.4 |
| Tyre/rim combination | Fixed for the whole event | IN 2.1.1 |
| Brake Test tyre pressure | ≤ cold pressure + 1.0 bar | IN 11.1.7 |
| Tyre size / compound limits | not specified in the rules | – |
| Vehicle mass limit | not specified (weighed only; ±5 kg change tolerance) | IN 8, IN 12.1.7 |

---

## 4. DV-specific inspection and tests

| Test | Procedure / pass criteria | Rule |
|---|---|---|
| Inspection order | All inspections except AS Inspection and EBS Test are needed to drive at all. **All, including those two,** are needed for autonomous mode. | IN 1.2.1, IN 1.2.2 |
| AS Inspection | Run by an ASR. Bring perception-sensor datasheets, proof of legal compliance, the RES remote, the ASF and tools. | IN 1.4.1, IN 6.1.1 |
| ASF | Structured documentation of the whole AS including the ASB, submitted before the event | T 14.11.1 |
| EBS Test | Straight line marked with cones like Acceleration. In autonomous mode, accelerate to **≥ 40 km/h within 20 m**. When the RES is triggered, the car must come to a **safe stop** within the distance implied by T 15.4 (≤ 200 ms reaction, > 10 m/s² mean deceleration, stable). In the wet, officials scale the distance. Done after all other parts of IN 11. | IN 11.2.1–IN 11.2.5 |
| Brake Test (manual) | Lock all four wheels and stop straight. EV: TS off via the cockpit button, mechanical brakes only. | IN 11.1 |
| Post Inspection | If any discipline was attempted autonomously: after a manual discipline the car must reach **AS Ready**; after an autonomous discipline it must **fully run the Inspection Mission**. Otherwise DQ. | IN 12.1.8 |
| Parc fermé | After Trackdrive and Endurance | IN 12.1.2 |
| Data logger removal | After the last dynamic discipline | IN 12.1.11 |
| Inspection Mission | Jacked up, all wheels removed, drivetrain spins slowly, steering sine wave, AS Finished after 25–30 s | T 14.10.2 |
| Safe testing guideline | Working ASSI, EBS and RES; "no running at speeds above typical competition speeds" | A 6.3.4 |

Derived figure, **not in the rules**: from 40 km/h (11.1 m/s), 0.2 s reaction plus 10 m/s² gives ≈ 2.2 m + 6.2 m ≈ **8.4 m** stopping distance. The rules give no distance number.

---

## 5. Implications for a simulator and RL driver

### 5.1 Track-generator parameters

| Parameter | Value from rules | Rule | Notes |
|---|---|---|---|
| Min track width | 3 m | D 5.1.1, D 6.1.1, D 8.1.1, D 4.1.5 | No max width given; also train on wider tracks |
| Max track width | not specified | – | |
| Max straight length (Autocross/Trackdrive) | 80 m | D 8.1.1, D 6.1.1 | |
| Min turning requirement | Vehicle must manage a 9 m minimum turning diameter (the tightest required turn) | D 8.1.1, D 1.1.10 | Treat ≈ 4.5 m as the tightest outer turn radius |
| Max corner radius | not specified | – | |
| Hairpin size | not specified (bounded only by the 9 m turning diameter) | – | |
| Decreasing-radius turns, chicanes, multiple turns | Required feature types | D 8.1.1 | |
| Slalom spacing | DC Autocross/Trackdrive: none specified. Manual Autocross 7.5–12 m; Endurance 9–15 m | D 6.1.1, D 7.1.1 | Optional augmentation |
| Trackdrive lap length | ≈ 200–500 m | D 8.1.2 | |
| DV Autocross lap length | ≈ 200–500 m (via D 8.1); manual Autocross < 1.5 km | D 6.1.3, D 6.1.2 | |
| Trackdrive laps | 10, self-counted | D 8.3.1, D 8.3.4 | |
| Cone spacing (boundary) | not specified (Event Handbook); Acceleration ≈ 5 m | D 2.4.1, D 5.1.1 | |
| Cone colours / sides / sizes | not specified (Event Handbook) | D 2.4.1 | |
| Acceleration length | 75 m, + stop within 75 m | D 5.1.1, D 5.2.4 | Staging 0.30 m behind the start line |
| Skidpad | Centres 18.25 m apart; inner Ø 15.25 m; outer Ø 21.25 m; 17 inner and 13 outer cones per circle; lane 3 m; stop within 25 m; staging 15 m before start | D 4.1.2–D 4.2.5 | Driving-line radius ≈ 7.6–10.6 m |
| Autocross/Trackdrive staging | Front wheels 6 m before the start line | D 6.2.3, D 8.3.2 | |
| Stop zone | Autocross/Trackdrive 30 m; Skidpad 25 m; Acceleration 75 m, all four wheels in bounds | D 6.2.4, D 8.3.3, D 4.2.5, D 5.2.4 | Must end in AS Finished within 30 s |
| Standstill timeout | 30 s → DQ | D 2.6.1 | |
| Start timeout | 30 s after go → DQ | D 10.1.11 | |
| Start sequencing | ≥ 5 s in AS Ready before Go; ≥ 3 s in AS Driving before moving | T 14.8.4, T 14.8.5 | |
| Reward-relevant penalties | DOO 2 s (Skidpad 0.2 s); OC 10 s (DQ in Accel/Skidpad); USS DQ or −50 points | D 10.1.7 | DOO cones stay knocked down for later laps |
| Min speeds | Trackdrive 2.5 / 3.5 m/s; Autocross 6 m/s lap = zero time points | D 2.6.2, D 9.3.2 | |
| Emergency brake model | ≤ 200 ms latency, > 10 m/s² | T 15.4 | |
| Power cap (EV) | 80 kW, 500 A, 600 V | EV 2.2, EV 4.1.1 | |
| Wheelbase | ≥ 1525 mm | T 2.9.1 | |
| No reverse | – | D 2.3.2 | Action space must exclude reverse |

### 5.2 Missions the car must support [T 14.10.1]

1. **Acceleration:** 75 m straight, stop within 75 m.
2. **Skidpad:** 2 right laps then 2 left laps (laps 2 and 4 timed), exit, stop within 25 m.
3. **Autocross** (DC): 1 unknown lap, stop within 30 m.
4. **Trackdrive** (DC): 10 laps with self lap counting, stop within 30 m.
5. **EBS Test:** reach ≥ 40 km/h within 20 m, then emergency stop on RES.
6. **Inspection:** jacked, wheels off; spin drivetrain slowly and steer in a sine wave; AS Finished after 25–30 s.
7. **Manual Driving:** AS hands over control; ASB confirmed deactivated.

Each mission also needs: the full AS state machine (T 14.8), ASSI and sound outputs (T 14.9), finish-state entry within the stop zone, and RES handling.

---

## 6. Ambiguities and gaps

- **Cones:** colours, sizes, spacing, side assignment, and orange start/finish or timing cones are not in the rules. They are only in the Event Handbook [D 2.4.1].
- **D 6.1.2 vs D 6.1.3:** it is unclear whether the manual Autocross limit of "< 1.5 km" still applies to DC Autocross, given that D 8.1.2 sets ≈ 200–500 m.
- **"Minimum required turning diameter 9 m":** this reads as the tightest turn the course requires (the car must manage a 9 m diameter). The rules do not map it to cone geometry such as an outer hairpin radius.
- **D 9.2.1 (> 25 s DQ):** it sits under "[Autonomous Mode ONLY] Scoring for Skidpad and Acceleration". It is not stated whether it also applies to DC Skidpad/Acceleration scored under D 9.3.1. Most likely it applies to all autonomous runs.
- **D 9.3.4 per-lap bonus:** "for every completed lap" is written generically under DC scoring. It is clearly meaningful for Trackdrive (10 laps). For one-lap disciplines it would add 2.5 % per run or lap. Its scope is not stated.
- **Scoring exponents:** pypdf flattens the formulas. Superscript "2" was confirmed from font size and position for D 9.1.1, D 9.4.1 and EF = T²·E. No exponent was found in D 9.3.2 or D 9.3.3.
- **T 15.4.3** uses "should" (single-failure half performance). It is a recommendation, not a requirement.
- **EBS Test stopping distance** is not given as a number [IN 11.2.3].
- **Not specified in the rules:** maximum DV speed, sensor type limits, pre-loaded maps or SLAM, maximum corner radius, hairpin dimensions, maximum track width, and absolute track-width minimum for the car.
