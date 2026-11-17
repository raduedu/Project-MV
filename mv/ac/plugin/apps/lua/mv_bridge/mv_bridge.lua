--[[
  MV Bridge — CSP Lua app connecting Assetto Corsa to the Project MV Python process.

  Installed to: assettocorsa/apps/lua/mv_bridge/   (use mv/ac/plugin/install_plugin.ps1)

  Python -> Lua : Local\mv_bridge_cmd_v1   (controls + episode commands)
  Lua -> Python : Local\mv_bridge_state_v1 (extra telemetry, acknowledgements)

  The two layouts below MUST match mv/ac/bridge/plugin_link.py byte for byte;
  tests/test_ac_bridge.py checks the text of both blocks against Python.

  Safety:
    * enable == 0 (or Python never connected)  -> controls released to the user.
    * enable == 1 but no new command for stale_timeout_s -> wheel straight,
      zero gas, full brake ("fail-safe").
    * stale for RELEASE_AFTER_S -> controls released again, so a crashed Python
      process can never leave the car unusable.

  APIs used (all documented in extension/internal/lua-sdk/ac_apps of CSP 0.2.11):
    ac.writeMemoryMappedFile, ac.overrideCarControls, ac.resetCar, ac.isCarResetAllowed,
    ac.tryToTeleportToPits, ac.tryToRestartSession, ac.takeAStepBack, ac.saveCarStateAsync,
    ac.loadCarState, ac.worldCoordinateToTrack, ac.trackCoordinateToWorld,
    physics.allowed, physics.setCarPosition, physics.setCarVelocity.
]]

local CMD_MAGIC = 0x31444D43    -- "CMD1"
local STATE_MAGIC = 0x31545353  -- "SST1"
local LAYOUT_VERSION = 1
local RELEASE_AFTER_S = 10.0

local CMD_LAYOUT = [[
uint32_t magic;
uint32_t version;
uint32_t seq;
uint32_t enable;
float steer;
float gas;
float brake;
float clutch;
float handbrake;
int32_t gear_up_count;
int32_t gear_down_count;
int32_t requested_gear;
float stale_timeout_s;
uint32_t reset_seq;
int32_t reset_cmd;
int32_t reset_slot;
float reset_spline;
float reset_speed;
float reset_lateral;
uint32_t host_time_ms;
]]

local STATE_LAYOUT = [[
uint32_t magic;
uint32_t version;
uint32_t seq;
uint32_t frame;
double game_time_s;
double car_timestamp_ms;
double lua_clock_s;
float position[3];
float velocity[3];
float local_velocity[3];
float local_angular_velocity[3];
float look[3];
float up[3];
float acceleration_g[3];
float spline;
float track_lateral;
float track_length_m;
float steer_deg;
float collision_depth;
float fps;
int32_t collided_with;
int32_t collision_count;
int32_t wheels_outside;
int32_t reset_counter;
int32_t lap_valid;
int32_t lap_cuts;
int32_t in_pitlane;
int32_t in_pit;
int32_t physics_allowed;
int32_t reset_allowed;
int32_t override_active;
int32_t failsafe_active;
int32_t is_paused;
int32_t ai_controlled;
uint32_t cmd_seq_seen;
uint32_t reset_seq_done;
int32_t reset_result;
int32_t saved_slots_mask;
int32_t csp_version;
int32_t teleport_dir_sign;
]]

-- ResetCommand / ResetResult (mirror of plugin_link.py)
local CMD = { NONE = 0, RESET_CAR = 1, TELEPORT_PITS = 2, RESTART_SESSION = 3,
              TELEPORT_SPLINE = 4, SAVE_STATE = 5, LOAD_STATE = 6, STEP_BACK = 7 }
local RES = { PENDING = 0, OK = 1, FAILED = -1, NOT_ALLOWED_IN_SESSION = -2,
              PHYSICS_NOT_ALLOWED = -3, EMPTY_SLOT = -4, UNKNOWN_COMMAND = -5, LUA_ERROR = -6 }

local cmdMap = ac.writeMemoryMappedFile('mv_bridge_cmd_v1', CMD_LAYOUT)
local st = ac.writeMemoryMappedFile('mv_bridge_state_v1', STATE_LAYOUT)

local sim = ac.getSim()
local car = ac.getCar(0)

local U32 = 4294967296
local controls = nil          -- ac.CarControlsInput, created on first enable
local cmdState = nil          -- last consistent copy of the command struct
local lastCmdSeq = nil
local lastCmdChange = os.preciseClock()
local lastGearUp, lastGearDown = nil, nil
local lastResetSeq = nil      -- initialised from the map so stale commands never run
local busy = nil              -- reset command in progress
local savedStates = {}
local savedMask = 0
local collisionCount = 0
local wasColliding = false
local teleportDirSign = 1
local failsafe = false
local overrideOn = false
local frame = 0
local statusText = 'waiting for Python'

local function clamp(x, a, b) if x ~= x then return 0 end return math.max(a, math.min(b, x)) end

-- Seqlock read of the command block; returns a table or nil if inconsistent / not written yet.
local function readCommand()
  if cmdMap.magic ~= CMD_MAGIC or cmdMap.version ~= LAYOUT_VERSION then return nil end
  local s1 = cmdMap.seq
  if s1 % 2 == 1 then return nil end
  local c = {
    seq = s1, enable = cmdMap.enable, steer = cmdMap.steer, gas = cmdMap.gas,
    brake = cmdMap.brake, clutch = cmdMap.clutch, handbrake = cmdMap.handbrake,
    gearUp = cmdMap.gear_up_count, gearDown = cmdMap.gear_down_count,
    gear = cmdMap.requested_gear, staleTimeout = cmdMap.stale_timeout_s,
    resetSeq = cmdMap.reset_seq, resetCmd = cmdMap.reset_cmd, resetSlot = cmdMap.reset_slot,
    resetSpline = cmdMap.reset_spline, resetSpeed = cmdMap.reset_speed,
    resetLateral = cmdMap.reset_lateral,
  }
  if cmdMap.seq ~= s1 then return nil end
  return c
end

local function releaseControls()
  if controls then
    controls.steer = math.huge   -- documented: leave the original steering value
    controls.gas = 0             -- final = max(user, this)
    controls.brake = 0
    controls.handbrake = 0
    controls.clutch = 1          -- final = min(user, this); 1 = pedal released
    controls.gearUp = false
    controls.gearDown = false
    controls.requestedGearIndex = 0
  end
  overrideOn = false
end

local function applyControls(c, safe)
  controls = controls or ac.overrideCarControls(0)
  if not controls then overrideOn = false return end
  if safe then
    controls.steer, controls.gas, controls.brake = 0, 0, 1
    controls.handbrake, controls.clutch = 0, 1
  else
    controls.steer = clamp(c.steer, -1, 1)
    controls.gas = clamp(c.gas, 0, 1)
    controls.brake = clamp(c.brake, 0, 1)
    controls.handbrake = clamp(c.handbrake, 0, 1)
    controls.clutch = clamp(c.clutch, 0, 1)
  end
  -- gear requests are edge counters; fire once per change, clear next frame
  controls.gearUp = lastGearUp ~= nil and c.gearUp ~= lastGearUp and not safe
  controls.gearDown = lastGearDown ~= nil and c.gearDown ~= lastGearDown and not safe
  lastGearUp, lastGearDown = c.gearUp, c.gearDown
  controls.requestedGearIndex = safe and 0 or c.gear
  overrideOn = true
end

-- ------------------------------------------------------------------ episode commands
local function finish(seq, result)
  st.reset_result = result
  st.reset_seq_done = seq
  busy = nil
end

local function trackPoint(spline, lateral)
  return ac.trackCoordinateToWorld(vec3(lateral, 0, spline % 1))
end

local function teleport(b)
  local len = math.max(sim.trackLengthM or 0, 100)
  local p0 = trackPoint(b.spline, b.lateral)
  local p1 = trackPoint(b.spline + 2 / len, b.lateral)
  local dir = (p1 - p0):normalize()
  physics.setCarPosition(0, p0, dir * teleportDirSign)
  physics.setCarVelocity(0, dir * b.speed)
  b.dir, b.pos, b.wait = dir, p0, 5
end

local function startCommand(c)
  local b = { seq = c.resetSeq, kind = c.resetCmd, slot = c.resetSlot,
              spline = c.resetSpline, speed = c.resetSpeed, lateral = c.resetLateral }
  st.reset_result = RES.PENDING
  local k = b.kind
  if k == CMD.RESET_CAR then
    if not ac.isCarResetAllowed() then return finish(b.seq, RES.NOT_ALLOWED_IN_SESSION) end
    ac.resetCar()
    return finish(b.seq, RES.OK)
  elseif k == CMD.STEP_BACK then
    if not ac.isCarResetAllowed() then return finish(b.seq, RES.NOT_ALLOWED_IN_SESSION) end
    ac.takeAStepBack()
    return finish(b.seq, RES.OK)
  elseif k == CMD.TELEPORT_PITS then
    return finish(b.seq, ac.tryToTeleportToPits() and RES.OK or RES.FAILED)
  elseif k == CMD.RESTART_SESSION then
    return finish(b.seq, ac.tryToRestartSession() and RES.OK or RES.FAILED)
  elseif k == CMD.TELEPORT_SPLINE then
    if not physics.allowed() then return finish(b.seq, RES.PHYSICS_NOT_ALLOWED) end
    busy = b
    teleport(b)
  elseif k == CMD.SAVE_STATE then
    busy = b
    ac.saveCarStateAsync(function(err, data)
      if err or not data then
        ac.log('mv_bridge: saveCarStateAsync failed: ' .. tostring(err))
        return finish(b.seq, RES.FAILED)
      end
      savedStates[b.slot] = data
      if b.slot >= 0 and b.slot < 31 then savedMask = bit.bor(savedMask, bit.lshift(1, b.slot)) end
      finish(b.seq, RES.OK)
    end)
  elseif k == CMD.LOAD_STATE then
    local data = savedStates[b.slot]
    if not data then return finish(b.seq, RES.EMPTY_SLOT) end
    return finish(b.seq, ac.loadCarState(data) and RES.OK or RES.FAILED)
  else
    return finish(b.seq, RES.UNKNOWN_COMMAND)
  end
end

-- Called every frame while a teleport waits for its orientation check: the
-- meaning of setCarPosition's `dir` argument is not documented precisely, so
-- verify the car ended up facing along the track and flip once if not.
local function updateBusy()
  local b = busy
  if not b or b.kind ~= CMD.TELEPORT_SPLINE then return end
  b.wait = b.wait - 1
  if b.wait > 0 then return end
  if car.look:dot(b.dir) < 0 and not b.flipped then
    teleportDirSign = -teleportDirSign
    b.flipped = true
    teleport(b)
    return
  end
  finish(b.seq, car.look:dot(b.dir) > 0 and RES.OK or RES.FAILED)
end

-- ------------------------------------------------------------------ main loop
local function publish(now)
  local s = st.seq
  if s % 2 == 1 then s = s + 1 end
  st.seq = (s + 1) % U32              -- odd: writing
  st.magic = STATE_MAGIC
  st.version = LAYOUT_VERSION
  st.frame = frame % U32
  st.game_time_s = sim.gameTime or 0
  st.car_timestamp_ms = car.timestamp or 0
  st.lua_clock_s = now
  local p, v, lv, la, lk, up, ag = car.position, car.velocity, car.localVelocity,
    car.localAngularVelocity, car.look, car.up, car.acceleration
  st.position[0], st.position[1], st.position[2] = p.x, p.y, p.z
  st.velocity[0], st.velocity[1], st.velocity[2] = v.x, v.y, v.z
  st.local_velocity[0], st.local_velocity[1], st.local_velocity[2] = lv.x, lv.y, lv.z
  st.local_angular_velocity[0], st.local_angular_velocity[1], st.local_angular_velocity[2] = la.x, la.y, la.z
  st.look[0], st.look[1], st.look[2] = lk.x, lk.y, lk.z
  st.up[0], st.up[1], st.up[2] = up.x, up.y, up.z
  st.acceleration_g[0], st.acceleration_g[1], st.acceleration_g[2] = ag.x, ag.y, ag.z
  st.spline = car.splinePosition
  local tc = ac.worldCoordinateToTrack(p)
  st.track_lateral = tc.x
  st.track_length_m = sim.trackLengthM or 0
  st.steer_deg = car.steer
  st.collision_depth = car.collisionDepth
  st.fps = sim.fps or 0
  st.collided_with = car.collidedWith
  st.collision_count = collisionCount
  st.wheels_outside = car.wheelsOutside
  st.reset_counter = car.resetCounter or 0
  st.lap_valid = car.isLapValid and 1 or 0
  st.lap_cuts = car.lapCutsCount or 0
  st.in_pitlane = car.isInPitlane and 1 or 0
  st.in_pit = car.isInPit and 1 or 0
  st.physics_allowed = physics.allowed() and 1 or 0
  st.reset_allowed = ac.isCarResetAllowed() and 1 or 0
  st.override_active = (controls and overrideOn and controls:active()) and 1 or 0
  st.failsafe_active = failsafe and 1 or 0
  st.is_paused = sim.isPaused and 1 or 0
  st.ai_controlled = car.isAIControlled and 1 or 0
  st.cmd_seq_seen = (lastCmdSeq or 0) % U32
  st.saved_slots_mask = savedMask
  st.csp_version = ac.getPatchVersionCode()
  st.teleport_dir_sign = teleportDirSign
  st.seq = (s + 2) % U32              -- even: consistent
end

local function tick()
  frame = frame + 1
  local now = os.preciseClock()

  local colliding = (car.collisionDepth or 0) > 0
  if colliding and not wasColliding then collisionCount = collisionCount + 1 end
  wasColliding = colliding

  local c = readCommand()
  if c then
    if c.seq ~= lastCmdSeq then lastCmdSeq, lastCmdChange = c.seq, now end
    if lastResetSeq == nil then lastResetSeq = c.resetSeq end
    if c.resetSeq ~= lastResetSeq and not busy then
      lastResetSeq = c.resetSeq
      local ok, err = pcall(startCommand, c)
      if not ok then
        ac.log('mv_bridge: command error: ' .. tostring(err))
        finish(c.resetSeq, RES.LUA_ERROR)
      end
    end
    cmdState = c
  end
  if busy then
    local ok, err = pcall(updateBusy)
    if not ok then
      ac.log('mv_bridge: teleport error: ' .. tostring(err))
      finish(busy.seq, RES.LUA_ERROR)
    end
  end

  local cs = cmdState
  local stale = now - lastCmdChange
  if cs and cs.enable == 1 and stale <= math.max(cs.staleTimeout, 0.05) then
    failsafe = false
    applyControls(cs, false)
    statusText = 'driving (Python connected)'
  elseif cs and cs.enable == 1 and stale <= RELEASE_AFTER_S then
    failsafe = true
    applyControls(cs, true)
    statusText = string.format('FAIL-SAFE: no command for %.1f s', stale)
  else
    failsafe = false
    if overrideOn then releaseControls() end
    lastGearUp, lastGearDown = cs and cs.gearUp or nil, cs and cs.gearDown or nil
    statusText = cs and 'released (Python idle)' or 'waiting for Python'
  end

  publish(now)
end

function script.update(dt)
  local ok, err = pcall(tick)
  if not ok then
    ac.log('mv_bridge: tick error: ' .. tostring(err))
    statusText = 'ERROR: ' .. tostring(err)
  end
end

function script.windowMain(dt)
  ui.text('MV Bridge')
  ui.text(statusText)
  ui.text(string.format('cmd seq %s  frame %d', tostring(lastCmdSeq), frame))
  ui.text(string.format('physics.allowed: %s  reset allowed: %s',
    tostring(physics.allowed()), tostring(ac.isCarResetAllowed())))
  ui.text(string.format('override active: %s', tostring(controls ~= nil and controls:active())))
end
