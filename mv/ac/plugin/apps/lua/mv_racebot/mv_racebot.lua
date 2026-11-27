--[[
  MV Race Bot — CSP Lua app: the Project MV race bot drives an AI car (normally car 1) against the owner.

  Installed to: assettocorsa/apps/lua/mv_racebot/   (mv/ac/plugin/install_plugin.ps1 - the owner runs it)

  Python -> Lua : Local\mv_racebot_cmd_v1   (enable, which car, steering, pedals)
  Lua -> Python : Local\mv_racebot_state_v1 (session, the bot car, the other cars, control check)
  Both layouts MUST match mv/ac/racebot_link.py byte for byte (tests/test_racebot.py checks the text).

  Taking the car over.  For an AI car the game combines our pedals with the AI's: "final value will be
  the maximum of original and this" (CSP SDK, ac.CarControlsInput.gas/brake).  So the AI is silenced
  first: takeover 1 = physics.setAINoInput(car) (documented: "disables input for an AI car"), takeover 2
  = physics.setAIThrottleLimit(car, 0) (fallback: AI gas off, it may still brake).  Steering is replaced
  outright.  The game's automatic gearbox shifts (physics.setCarAutoShifter), as in our simulator.
  control_ok reports whether the car's gas and brake really equal ours - Python stops if they do not.

  Safety: no fresh command for stale_timeout_s -> wheel straight, full brake; for RELEASE_AFTER_S -> the
  car is handed back to the game's AI.  enable == 0 hands it back at once.

  APIs (CSP 0.2.11 SDK, extension/internal/lua-sdk/ac_apps/lib.lua): ac.writeMemoryMappedFile,
  ac.overrideCarControls(carIndex), ac.getCar, ac.getSim, physics.allowed, physics.setAINoInput,
  physics.setAIThrottleLimit, physics.setCarAutoShifter.
]]

local CMD_MAGIC = 0x31424D43    -- "CMB1"
local STATE_MAGIC = 0x31425353  -- "SSB1"
local LAYOUT_VERSION = 1
local RELEASE_AFTER_S = 10.0
local MAX_OTHERS = 8
local TAKEOVER_NO_INPUT, TAKEOVER_THROTTLE_LIMIT = 1, 2

local CMD_LAYOUT = [[
uint32_t magic;
uint32_t version;
uint32_t seq;
uint32_t enable;
int32_t bot_index;
int32_t takeover;
float steer;
float gas;
float brake;
float stale_timeout_s;
uint32_t host_time_ms;
]]

local STATE_LAYOUT = [[
uint32_t magic;
uint32_t version;
uint32_t seq;
uint32_t frame;
double game_time_s;
int32_t bot_index;
int32_t cars_count;
int32_t session_started;
int32_t in_menu;
int32_t is_paused;
int32_t physics_allowed;
int32_t takeover_active;
int32_t override_active;
int32_t failsafe_active;
int32_t control_ok;
float time_to_start_s;
uint32_t cmd_seq_seen;
float position[3];
float velocity[3];
float local_velocity[3];
float local_angular_velocity[3];
float look[3];
float gas;
float brake;
float steer_deg;
float rpm;
float kers_charge;
float spline;
float collision_depth;
int32_t gear;
int32_t wheels_outside;
int32_t lap_count;
int32_t lap_time_ms;
int32_t last_lap_ms;
int32_t best_lap_ms;
int32_t race_position;
int32_t collided_with;
int32_t other_index[8];
int32_t other_active[8];
float other_position[24];
float other_velocity[24];
float other_look[24];
]]

local cmdMap = ac.writeMemoryMappedFile('mv_racebot_cmd_v1', CMD_LAYOUT)
local st = ac.writeMemoryMappedFile('mv_racebot_state_v1', STATE_LAYOUT)
local sim = ac.getSim()
local U32 = 4294967296

local cmdState = nil
local lastCmdSeq = nil
local lastCmdChange = os.preciseClock()
local controls = nil          -- ac.CarControlsInput of the bot car
local controlsIndex = nil
local taken = nil             -- {index=, mode=} while the AI is silenced
local takeoverOk = false
local failsafe = false
local overrideOn = false
local okFrames = 0
local frame = 0
local statusText = 'waiting for Python'

local function clamp(x, a, b) if x ~= x then return 0 end return math.max(a, math.min(b, x)) end

local function readCommand()
  if cmdMap.magic ~= CMD_MAGIC or cmdMap.version ~= LAYOUT_VERSION then return nil end
  local s1 = cmdMap.seq
  if s1 % 2 == 1 then return nil end
  local c = { seq = s1, enable = cmdMap.enable, index = cmdMap.bot_index, takeover = cmdMap.takeover,
              steer = cmdMap.steer, gas = cmdMap.gas, brake = cmdMap.brake, staleTimeout = cmdMap.stale_timeout_s }
  if cmdMap.seq ~= s1 then return nil end
  return c
end

-- ------------------------------------------------------------------ taking the car over / giving it back
local function call(f, ...)
  if not f then return false end
  local ok, res = pcall(f, ...)
  return ok and res ~= false
end

local function giveBack()
  if controls then
    controls.steer = math.huge   -- documented: leave the original (AI) steering
    controls.gas, controls.brake, controls.handbrake = 0, 0, 0
  end
  if taken and physics.allowed() then
    if taken.mode == TAKEOVER_NO_INPUT then call(physics.setAINoInput, taken.index, false, false)
    else call(physics.setAIThrottleLimit, taken.index, 1) end
  end
  taken, takeoverOk, overrideOn, okFrames = nil, false, false, 0
end

local function takeOver(index, mode)
  if taken and (taken.index ~= index or taken.mode ~= mode) then giveBack() end
  if not taken then
    takeoverOk = false
    if physics.allowed() then
      if mode == TAKEOVER_NO_INPUT then takeoverOk = call(physics.setAINoInput, index, true, false)
      else takeoverOk = call(physics.setAIThrottleLimit, index, 0) end
      call(physics.setCarAutoShifter, index, true)
    end
    taken = { index = index, mode = mode }
  end
  if controlsIndex ~= index then controls, controlsIndex = ac.overrideCarControls(index), index end
end

local function apply(c, safe)
  takeOver(c.index, c.takeover)
  if not controls then overrideOn = false return end
  controls.handbrake, controls.clutch, controls.requestedGearIndex = 0, 1, 0
  if safe then
    controls.steer, controls.gas, controls.brake = 0, 0, 1
  else
    controls.steer = clamp(c.steer, -1, 1)
    controls.gas = clamp(c.gas, 0, 1)
    controls.brake = clamp(c.brake, 0, 1)
  end
  overrideOn = true
end

-- ------------------------------------------------------------------ publishing
local function put3(arr, k, v)
  arr[3 * k], arr[3 * k + 1], arr[3 * k + 2] = v.x, v.y, v.z
end

local function publish(now, c)
  local index = c and c.index or 1
  local car = ac.getCar(index)
  local s = st.seq
  if s % 2 == 1 then s = s + 1 end
  st.seq = (s + 1) % U32
  st.magic, st.version, st.frame = STATE_MAGIC, LAYOUT_VERSION, frame % U32
  st.game_time_s = sim.gameTime or 0
  st.bot_index, st.cars_count = index, sim.carsCount or 1
  st.session_started = sim.isSessionStarted and 1 or 0
  st.in_menu = sim.isInMainMenu and 1 or 0
  st.is_paused = sim.isPaused and 1 or 0
  st.physics_allowed = physics.allowed() and 1 or 0
  st.takeover_active = takeoverOk and 1 or 0
  st.override_active = overrideOn and 1 or 0
  st.failsafe_active = failsafe and 1 or 0
  st.time_to_start_s = (sim.timeToSessionStart or -1000) / 1000
  st.cmd_seq_seen = (lastCmdSeq or 0) % U32
  if car then
    -- does the car really get our pedals? (the AI's would show up as a larger value)
    if overrideOn and c and not failsafe and math.abs(car.gas - clamp(c.gas, 0, 1)) < 0.05
        and math.abs(car.brake - clamp(c.brake, 0, 1)) < 0.05 then
      okFrames = math.min(okFrames + 1, 1000)
    else
      okFrames = 0
    end
    st.control_ok = okFrames >= 15 and 1 or 0
    local p, v, lv, la, lk = car.position, car.velocity, car.localVelocity, car.localAngularVelocity, car.look
    st.position[0], st.position[1], st.position[2] = p.x, p.y, p.z
    st.velocity[0], st.velocity[1], st.velocity[2] = v.x, v.y, v.z
    st.local_velocity[0], st.local_velocity[1], st.local_velocity[2] = lv.x, lv.y, lv.z
    st.local_angular_velocity[0], st.local_angular_velocity[1], st.local_angular_velocity[2] = la.x, la.y, la.z
    st.look[0], st.look[1], st.look[2] = lk.x, lk.y, lk.z
    st.gas, st.brake, st.steer_deg, st.rpm = car.gas, car.brake, car.steer, car.rpm
    st.kers_charge = car.kersCharge or 0
    st.spline, st.collision_depth = car.splinePosition, car.collisionDepth or 0
    st.gear, st.wheels_outside = car.gear, car.wheelsOutside
    st.lap_count, st.lap_time_ms = car.lapCount, car.lapTimeMs
    st.last_lap_ms, st.best_lap_ms = car.previousLapTimeMs, car.bestLapTimeMs
    st.race_position, st.collided_with = car.racePosition or 0, car.collidedWith or -1
  end
  local k = 0
  for i = 0, (sim.carsCount or 1) - 1 do
    if i ~= index and k < MAX_OTHERS then
      local o = ac.getCar(i)
      if o and o.isConnected and o.isActive ~= false then
        st.other_index[k], st.other_active[k] = i, 1
        put3(st.other_position, k, o.position)
        put3(st.other_velocity, k, o.velocity)
        put3(st.other_look, k, o.look)
        k = k + 1
      end
    end
  end
  for j = k, MAX_OTHERS - 1 do st.other_active[j] = 0 end
  st.seq = (s + 2) % U32
end

-- ------------------------------------------------------------------ main loop
local function tick()
  frame = frame + 1
  local now = os.preciseClock()
  local c = readCommand()
  if c then
    if c.seq ~= lastCmdSeq then lastCmdSeq, lastCmdChange = c.seq, now end
    cmdState = c
  end
  local cs = cmdState
  local stale = now - lastCmdChange
  if cs and cs.enable == 1 and not sim.isInMainMenu and stale <= math.max(cs.staleTimeout, 0.05) then
    failsafe = false
    apply(cs, false)
    statusText = string.format('driving car %d%s', cs.index, takeoverOk and '' or ' (AI NOT silenced!)')
  elseif cs and cs.enable == 1 and not sim.isInMainMenu and stale <= RELEASE_AFTER_S then
    failsafe = true
    apply(cs, true)
    statusText = string.format('FAIL-SAFE: no command for %.1f s', stale)
  else
    failsafe = false
    if taken or overrideOn then giveBack() end
    statusText = cs and 'released (car back to the game AI)' or 'waiting for Python'
  end
  publish(now, cs)
end

function script.update(dt)
  local ok, err = pcall(tick)
  if not ok then
    ac.log('mv_racebot: tick error: ' .. tostring(err))
    statusText = 'ERROR: ' .. tostring(err)
  end
end

function script.windowMain(dt)
  ui.text('MV Race Bot')
  ui.text(statusText)
  ui.text(string.format('physics.allowed: %s   AI silenced: %s', tostring(physics.allowed()), tostring(takeoverOk)))
  ui.text(string.format('pedals follow the bot: %s', tostring(okFrames >= 15)))
end
