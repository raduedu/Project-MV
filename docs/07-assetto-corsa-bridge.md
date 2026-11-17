# 07 — Assetto Corsa live bridge

The bridge connects a Python process to a **running Assetto Corsa** (the original
game, v1.16.x, *not* ACC/EVO) with **Custom Shaders Patch** (installed here:
CSP 0.2.11, build 3465). It reads telemetry, sends controls, resets episodes and
runs a fixed-rate loop with a fail-safe. It deliberately builds **no observation
vectors and no rewards**; those sit on top.

Code: `mv/ac/bridge/` (Python), `mv/ac/plugin/` (files for the game folder),
`tests/test_ac_bridge.py`, `scripts/ac_bridge_check.py`.

---

## 1. Architecture

```
 ┌───────────────────────── Assetto Corsa (acs.exe) ─────────────────────────┐
 │  physics 333 Hz ──► Local\acpmf_physics   (Kunos, read-only for us)       │
 │  render frame   ──► Local\acpmf_graphics                                  │
 │  session load   ──► Local\acpmf_static                                    │
 │                                                                           │
 │  CSP Lua app "MV Bridge" (apps/lua/mv_bridge, runs every frame)           │
 │     reads  Local\mv_bridge_cmd_v1   ──► ac.overrideCarControls(0)         │
 │                                     ──► ac.resetCar / tryToTeleportToPits │
 │                                         / loadCarState / setCarPosition   │
 │     writes Local\mv_bridge_state_v1 ◄── car.look/up, track lateral,       │
 │                                         collisions, acks, fail-safe flag  │
 └──────────▲───────────────────────▲─────────────────────────▲──────────────┘
            │ OpenFileMapping (RO)  │ mmap (RW, ours)          │ (optional)
 ┌──────────┴───────────────────────┴──────────┐   vJoy / ViGEm virtual device
 │ Python (mv.ac.bridge)                       │   (VJoyBackend / VGamepadBackend)
 │  shm.py        tear-checked page snapshots  │
 │  shm_layout.py exact ctypes structs, pack 4 │
 │  plugin_link.py seqlocked cmd/state structs │
 │  reader.py     ACReader ─► telemetry.py     │
 │                 Telemetry (SI, project axes)│
 │  backends.py   ControlBackend.send(...)     │
 │  session.py    ACSession: tick, wait packet,│
 │                stats, watchdog, fail-safe   │
 │  episode.py    off-track/wrong-way/collision│
 │  env.py        ACRawEnv.reset()/step()      │
 └──────────────────────▲──────────────────────┘
                        │ Telemetry
            lead's observation builder / policy
```

Why read the Kunos pages with `OpenFileMappingW` (not `mmap(tagname=…)`): `mmap`
*creates* the mapping if the game is not running yet, and AC would then attach to
our (possibly smaller) object and break its own shared memory. We open
read-only and never create AC's names.

---

## 2. Frames, units and exact formulas

AC frames, verified from this install (not assumed): every left-hand-drive car has
`DRIVEREYES` x > 0 (ferrari_458 0.34, bmw_m3_e30 0.31, ks_mazda_mx5_nd 0.34),
every right-hand-drive car x < 0 (ks_nissan_skyline_r34 −0.38,
ks_toyota_supra_mkiv −0.34, ks_toyota_ae86 −0.32). So **car-local X = left,
Y = up, Z = forward**, right-handed; the world uses the same handedness with Y up.

| quantity | frame | formula (AC vector `(X, Y, Z)`) |
|---|---|---|
| `velocity_local`, `angular_velocity_local`, `acceleration_local` | body: x fwd, y left, z up | `(Z, X, Y)` (`frames.ac_local_to_body`) |
| `position`, `velocity_world`, plugin `forward`/`up` | **map**: x right, y up on the map, z up | `(X, −Z, Y)` (`frames.ac_world_to_map3`; identical to `mv.ac.track_import.ac_world_to_map` plus height) |
| `yaw` | map | CCW positive, `yaw = 0` along +map_x |

Both maps are proper rotations (det +1), so angular velocities keep their sense:
`yaw_rate = localAngularVel[1]` (> 0 = turning left), `pitch` > 0 nose down,
`roll` > 0 right side down (right-hand rule, ISO 8855).

Yaw, by source (`Telemetry.yaw_source`):

* `"plugin"` (MV Bridge app running): `look` is AC's unit forward vector;
  `f = (look.X, −look.Z, look.Y)`, `yaw = atan2(f_y, f_x) = atan2(−look.Z, look.X)`,
  `pitch = −asin(f_z)`, `roll = atan2(left_z, up_z)` with `left = up × f`. Exact.
* `"velocity"` (speed ≥ 3 m/s): `yaw = atan2(v_world_y, v_world_x) − atan2(v_local_y, v_local_x)`.
  Exact on flat ground and independent of any undocumented convention, because
  it compares the same velocity in world and body frames (slip included).
* `"heading_calibrated"`: Kunos `heading` has no documented zero/sign, so the
  decoder learns `yaw = s·heading + offset` (s = ±1) from 50 moving samples and
  uses it at low speed.
* `"heading_raw"`: only before calibration and while stationary without the app.

Units: m, s, m/s, m/s² (`accG × 9.80665`), rad, rad/s, N (wheel load), °C, L.
`speed = speedKmh / 3.6`. Lap times are ms → s; `0`/`INT_MAX` → `nan`.
`gear`: AC `0=R, 1=N, 2=1st` → ours `−1, 0, 1…`. Wheel arrays: FL, FR, RL, RR.
`steer`: normalised, **+1 = full left** (project convention); AC/CSP use +1 =
right (verified in CSP's bundled `extension/lua/mouse-steering/base/assist.lua`:
rightward mouse motion increases `state.steer`). `sim_time` is CSP's physics
timestamp when the app runs, otherwise `packetId / 333` (estimate).

---

## 3. Shared-memory layouts

Sizes (pack 4, `wchar_t` = UTF-16): **physics 712 B, graphics 288 B, static 684 B**
(asserted by the tests together with 24 field offsets).

Cross-checked sources:
1. `mdjarv/assettocorsasharedmemory` (C#, `Pack = 4`) — physics up to
   `localVelocity`, full graphics (to `mandatoryPitDone`) and static.
2. `Spacefreak18/simapi` `acdata.h` (`#pragma pack(4)`) and Kunos' AC-EVO
   shared-memory reference, which carries the AC1 physics header verbatim —
   fields after `localVelocity`: `P2PActivations, P2PStatus, currentMaxRpm (int),
   mz[4], fx[4], fy[4], slipRatio[4], slipAngle[4], tcinAction, absInAction,
   suspensionDamage[4], tyreTemp[4]` (AC 1.14+, added with Push-To-Pass).
3. The game's own `changelog.txt` (version history of each field). It declares
   `wchar_t trackConfiguration[33]` (source 1 says `[15]`); we follow Kunos.
   Only static fields after it depend on this, and none is used for control.

The bridge maps the *whole* section and reads `min(sizeof, section size)`, so an
older AC with a shorter struct reads zeros instead of failing.

---

## 4. Control backends — recommendation

| backend | driver needed | 1:1 mapping | resets | notes |
|---|---|---|---|---|
| **`csp`** `CSPLuaBackend` (**recommended**) | none | yes, AC input filters bypassed (to verify) | yes | needs the MV Bridge app; pedals combine as max(user, ours) |
| `vjoy` `VJoyBackend` | vJoy | yes if AC settings neutral (§5.4) | no | AC sees a normal wheel |
| `vgamepad` `VGamepadBackend` | ViGEmBus | only with gamepad filters zeroed | no | last resort |

Why `csp`: CSP 0.2.11's SDK (`extension/internal/lua-sdk/ac_apps/lib.lua`)
documents for **Lua apps** `ac.overrideCarControls(carIndex)` ("Easier than
simulating inputs with fake gamepads and keypresses"), with fields `steer`
(−1..1, replaces the user's value; `math.huge` = leave it), `gas`/`brake`
("final value will be the maximum of original and this"), `clutch` (minimum),
`gearUp/gearDown`, `requestedGearIndex`, and `ac.writeMemoryMappedFile(name,
layout)` which CSP itself uses to talk to external processes (CEF browser,
Mumble). No driver install, no input binding, sub-frame latency, and it also
gives resets and exact orientation. Python steering is negated exactly once, in
the backend.

All backends: `send(steer, throttle, brake, gear_up=False, gear_down=False)`,
`failsafe()` (wheel straight, throttle 0, brake 1), `release()`, `close()`.
`pyvjoy`/`vgamepad` are imported lazily only when those backends are built.

---

## 5. Setup (do this before the first in-game test)

### 5.1 Install the Lua app (required for `csp`, recommended always)

1. Close Assetto Corsa.
2. In PowerShell run (copies two files into `assettocorsa\apps\lua\mv_bridge\`):
   ```
   powershell -ExecutionPolicy Bypass -File "D:\Visual Studio\Project MV\mv\ac\plugin\install_plugin.ps1"
   ```
   (`-AcRoot <path>` for another install, `-Uninstall` to remove.)
3. The app has `LAZY = NONE`, so CSP loads it with every session. Optionally open
   its window in-game (right-side app bar → "MV Bridge"): it shows
   `waiting for Python` / `driving` / `FAIL-SAFE`, `physics.allowed` and
   `reset allowed`.
4. Lua apps are enabled by default in CSP; if the app does not show up in the
   in-game app list, check CM → Settings → Custom Shaders Patch for a setting
   that disables Lua apps (exact location unverified).

### 5.2 Session and game settings

1. **Offline Practice, single car (no AI)**, from Content Manager → Drive →
   Practice. This is the mode in which `ac.resetCar`, `ac.loadCarState` and
   restart are documented to work ("available in few basic modes in offline only
   (such as practice with a single car)"). Online servers disable most of this.
2. Assists (CM → Drive → Assists, or `cfg\assists.ini`):
   **Automatic gearbox ON** (`AUTO_SHIFTER=1`) — the policy outputs no gears;
   **Auto clutch ON**, **Auto blip ON**; **Stability control 0**; **Ideal line
   OFF**; **Damage 0 %** (else crashes end runs via engine/suspension damage);
   **Fuel rate 0** and **Tyre wear 0** for reproducible episodes; ABS / TC: set
   them to whatever the training simulator models (factory or off) — they change
   how `brake`/`throttle` are applied.
3. Your current `controls.ini` has an H-shifter active (`[SHIFTER] ACTIVE=1`);
   CSP warns gear forcing "might not work well … with H-shifter". With automatic
   gearbox this does not matter; if you ever use `gear_up/gear_down`, disable the
   shifter.
4. While the bridge drives: hands off the wheel and **feet off the pedals**
   (overrides are `max(user, ours)` for pedals). With the G923 bound, consider
   FFB gain 0 so the rim does not spin violently. Keyboard input mode is fine.
5. Avoid pausing / alt-tabbing if your setup pauses AC on focus loss (look for a
   "pause when unfocused"-type option in CM/CSP settings; exact name
   unverified): a pause stops physics packets and triggers the fail-safe (by
   design). Run Python from a second monitor or with AC windowed.

### 5.3 First test

1. Start the session, drive out of the pits onto the track, stop.
2. ```
   "D:\Visual Studio\Project MV\.venv\Scripts\python.exe" "D:\Visual Studio\Project MV\scripts\ac_bridge_check.py"
   ```
   During the 10 s, drive yourself: accelerate, brake, one left and one right turn.
3. Then, standing still on a straight with nothing ahead:
   ```
   "D:\Visual Studio\Project MV\.venv\Scripts\python.exe" "D:\Visual Studio\Project MV\scripts\ac_bridge_check.py" --drive
   ```
   It applies throttle 0.3 for 2 s, brakes 1.5 s, prints loop stats and releases.

### 5.4 Optional: vJoy backend

1. Download vJoy from its official repository/release page (the maintained
   Windows 10/11 build is `jshafer817/vJoy` on GitHub, release v2.1.9.1; the
   original project is vjoystick.sourceforge.net) and install it yourself; run
   "Configure vJoy": device 1, axes X, Y, Z enabled, ≥ 2 buttons.
2. `"…\.venv\Scripts\python.exe" -m pip install pyvjoy` (yourself; the bridge never
   installs packages).
3. Content Manager → Settings → Assetto Corsa → Controls → Input method
   "Wheel". Bind with the sweep helper (click the field, then run):
   * Steering: `ac_bridge_check.py --vjoy-sweep X`
   * Throttle: `--vjoy-sweep Y`; Brake: `--vjoy-sweep Z`
   * Gear up / down: vJoy buttons 1 / 2 (only needed with manual gearbox).
4. Neutralise AC processing (CM controls page, or `cfg\controls.ini [STEER]`):
   `STEER_GAMMA=1`, `STEER_FILTER=0`, `SPEED_SENSITIVITY=0`, `DEBOUNCING_MS=0`
   (yours is 50!), `SCALE=1`, **`LOCK` = 2 × the car's `[CONTROLS] STEER_LOCK`**
   (controls `LOCK` is total wheel rotation, car `STEER_LOCK` is per side, so
   full axis = full car lock; verify with the check script: full virtual axis
   should give `steer ≈ ±1`), deadzone 0;
   pedals `GAMMA=1` (your `[BRAKES] GAMMA=2.4` must become 1), full range
   calibrated by the sweep.
5. Raw mapping: steer X `0x0001` = full left … `0x8000` = full right; throttle Y
   and brake Z `0x0001` = released … `0x8000` = floored.

### 5.5 Optional: vgamepad backend

Install ViGEmBus (official: `nefarius/ViGEmBus` GitHub releases; project is
archived but the driver works) and `pip install vgamepad`. AC input method
"Gamepad": left stick = steer, RT = throttle, LT = brake, A/X = gear up/down.
Set `[X360] STEER_GAMMA=1, STEER_FILTER=0, SPEED_SENSITIVITY=0,
STEER_DEADZONE=0, STEER_SPEED=1` (yours are 3.01 / 0.6 / 0.2 / 0.1 / 0.2).
Even then AC's gamepad helper may remain; prefer `csp`.

---

## 6. Episode control

`ACRawEnv(session, reset_mode=…)`, executed by the Lua app (`csp` backend only):

| mode | CSP call | result |
|---|---|---|
| `reset_car` (default) | `ac.resetCar()` | car back on track near its position, stationary; needs `ac.isCarResetAllowed()` (offline single-car practice) |
| `step_back` | `ac.takeAStepBack()` | as above, moved back a bit |
| `pits` | `ac.tryToTeleportToPits()` | pit box |
| `restart` | `ac.tryToRestartSession()` | session restart (offline) — slow, resets lap/time |
| `load_state` | `ac.loadCarState(slot)` | restores a snapshot taken with `env.save_state(slot)` (position, velocities, engine, gear) — **"start at spline s with speed v"** on stock tracks: drive there once, `save_state`, then reuse |
| `spline` | `physics.setCarPosition` + `setCarVelocity` | exact spline/lateral/speed teleport, **only if `physics.allowed()`** |
| `none` | — | brake to a stop (works with any backend) |

`physics.allowed()` for Lua apps requires the track's `surfaces.ini` to grant it
(`[_SCRIPTING_PHYSICS] ALLOW_APPS=1` plus CSP extended physics) — stock Kunos
tracks do not, so `spline` returns `PHYSICS_NOT_ALLOWED` there. Because CSP does
not document the direction convention of `setCarPosition(pos, dir)` (its own
debug app passes `-cameraLook`), the app checks the car's heading 5 frames after
teleporting and flips once if it faces backwards; the learned sign is published
as `teleport_dir_sign`.

Detection (`EpisodeMonitor`, result in `env.flags`):

* **off-track**: Kunos `numberOfTyresOut ≥ 3` (AC's own track-limit logic), or
  app `|track_lateral| > 1.15` (`ac.worldCoordinateToTrack(pos).x`, ±1 = AI-spline
  sides, not necessarily painted limits).
* **wrong-way**: wrap-aware spline progress rate < −2 m/s for 1 s.
* **collision**: app collision counter (rising edges of `car.collisionDepth > 0`,
  `collided_with` 0 = track object, > 0 = car); fallbacks: any `carDamage` zone
  increased (needs damage > 0 %), or |Δv|/Δt > 5 g between packets.
* **stuck**: speed < 0.5 m/s for 5 s.

---

## 7. Real-time loop (`ACSession`)

`step(steer, throttle, brake)`: send now → sleep to the next tick of a
drift-free `t0 + k/f` schedule (re-sync on overrun) → ensure ≥ 1 new physics
packet since sending → read `Telemetry`. Controls are zero-order-held for one
period, like the training sim. Default 50 Hz (25–50 Hz supported; 1–200 allowed).

Stats (`session.stats.format()`): rate, period mean / std (jitter) / p99,
observation latency (send → first new physics packet), **ack latency** (send →
Lua app read the command; `csp` only), physics packets per step, timeouts,
overruns, watchdog trips.

Fail-safe (wheel straight, throttle 0, **full brake**):
* no new physics packet within 250 ms, or game status not LIVE → fail-safe +
  `ACTimeoutError`;
* any exception inside `step` → fail-safe + re-raise;
* watchdog thread: no `step` for 300 ms → fail-safe (policy hang / debugger);
* Lua app: command sequence stale for `stale_timeout_s` (0.5 s) → brake;
  stale > 10 s → release to the user, so a killed Python never leaves the car
  unusable;
* `close()`: brake, then release control to the user.

Usage:

```python
from mv.ac.bridge import ACRawEnv
env = ACRawEnv.make(backend="csp", rate_hz=50, reset_mode="reset_car")
tel = env.reset()
while True:
    steer, thr, brk = policy(tel)            # project convention: steer + = left
    tel = env.step(steer, thr, brk)
    if env.flags.off_track or env.flags.wrong_way or env.flags.collision:
        tel = env.reset()
```

---

## 8. Verification checklist (first session)

- [ ] `ac_bridge_check.py` prints car/track names correctly (wchar decoding) and
      `track length` > 0.
- [ ] "MV Bridge Lua app: RUNNING" with a CSP version code.
- [ ] Physics packet rate ≈ 333/s (AC physics step) while driving, 0 when paused.
- [ ] `pos=(x, y)` moves consistently with the imported track map (same frame as
      `track_import`); `spl` increases while driving forward and wraps at the line.
- [ ] Convention checks all PASS: ax vs d(vx)/dt, ay vs yaw_rate·vx, yaw_rate vs
      d(yaw)/dt, steer vs yaw_rate. Turning **left** shows `steer > 0`, `r > 0`.
- [ ] `heading calibration: sign=…` gets a value after ~2 s of driving.
- [ ] `--drive`: speed increases, `reported throttle ≈ 0.30`,
      `override_active=True`, then the car brakes and control returns to you.
- [ ] Ack latency p95 < one frame (≈ 17 ms at 60 fps); obs latency p95 < 5 ms.
- [ ] Python killed (Ctrl+C) during `--drive` → car brakes (Lua fail-safe), then
      after 10 s you can drive again.
- [ ] In a Python shell: `env.reset()` with `reset_car` and `load_state` work in
      single-car practice.

---

## 9. Verified vs. not yet verified in-game

Verified from documentation / the install / tests:
* struct layouts (three sources + Kunos changelog), sizes and offsets (tests);
* AC car-local axes (DRIVEREYES of LHD vs RHD cars in this install);
* CSP control steer sign (+ = right) from CSP's own mouse-steering script;
* existence and documented semantics of every CSP function the app calls, in the
  CSP 0.2.11 SDK shipped with this install;
* Python ↔ Lua struct texts are identical (test); reader, decoding, fail-safe,
  watchdog, timing and episode logic (54 tests, synthetic pages in real named
  mappings).

**Not yet verified in-game** (each has a check in §8):
1. That `ac.overrideCarControls` is honoured for a Lua *app* in offline practice
   on this CSP build, and that its steering bypasses AC's controller filters.
2. That `script.update` of a `LAZY = NONE` app runs every frame with its window closed.
3. That `ac.writeMemoryMappedFile` uses natural C alignment for the layouts
   (expected: LuaJIT FFI) — a mismatch would show as garbage in the app fields.
4. Kunos `steerAngle` normalisation/sign (assumed −1..1, + = right) and `accG`
   axis order/signs (assumed [0] lateral, [1] vertical, [2] longitudinal).
5. Sign of Kunos `pitch`/`roll` (only used without the app).
6. Physics packet rate (assumed ≈ 333 Hz; `sim_time` fallback uses it).
7. `iBestTime` value before a best lap exists (handled: ≤ 0 or INT_MAX → nan).
8. `setCarPosition` direction convention (auto-corrected) and whether
   `physics.allowed()` is false on stock tracks (expected).
9. Whether `ac.resetCar` / `loadCarState` succeed in CM "Practice" (expected yes,
   per SDK docs) and how long they take.
10. Static-page fields after `trackConfiguration` (`[33]` vs `[15]`): unused.

---

## 10. Known limitations

* Pedal overrides are `max(user, ours)`: a touched pedal adds to the policy's.
* Lua runs at render rate: control updates reach physics at the frame rate
  (≥ 60 fps recommended); physics then holds the value for its 333 Hz steps.
* Exact spline teleport needs track permission; on stock tracks use
  `load_state` (saved snapshots, kept only for the running session) or
  `reset_car` (position chosen by AC).
* No reset with `vjoy`/`vgamepad` backends (AC has no vanilla reset binding);
  use the app for resets even when driving through a virtual device.
* Kunos off-track is per-tyre and lags AC's own logic; `track_lateral` uses
  AI-spline sides, which differ from painted limits on some tracks.
* Collision detection without the app misses light contacts (damage 0 %, < 5 g).
* Online servers: overrides/resets are restricted or disabled; offline only.
* Windows only (named mappings, `c_wchar` = 2 bytes).
