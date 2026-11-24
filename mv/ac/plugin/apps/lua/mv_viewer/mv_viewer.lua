--[[
  MV Viewer — CSP Lua app that plays a Project MV lap in Assetto Corsa, with a see-through ghost.

  Installed to: assettocorsa/apps/lua/mv_viewer/   (use mv/ac/plugin/install_plugin.ps1)
  Python side : mv/ac/viewer_link.py  (control panel button "View in Assetto Corsa", scripts/ac_view.py)
  Companion   : mv_follow.lua (physics worker, 333 Hz) - steers the car along the lap with forces.

  Python -> Lua : Local\mv_viewer_v1        (the lap: car + ghost samples, car model, play/pause/stop)
  Lua -> Python : Local\mv_viewer_state_v1  (loaded lap id, play time, gap)
  The two layouts below MUST match mv/ac/viewer_link.py byte for byte (tests/test_ac_viewer.py).

  How the lap is shown (engineering log WP34)
    * "follow" mode (normal): the car stays an ordinary game car - suspension, tyres, gearbox,
      engine sound, dashboard - and mv_follow.lua pulls it along the recorded path and heading with
      forces at the physics rate. The car is teleported only at the start, on seek/restart/loop,
      or if it ends up far from the lap (crash). Steering, pedals and gears follow the recording.
    * "teleport" mode (fallback if the physics worker cannot start): the car is placed on the
      recorded pose every frame (WP32/33); works, but the picture is less smooth.
    * The ghost (ideal lap) is drawn by this app: CSP's simplified 3D shape of the car with a
      see-through blue "hologram" shader (WP39), moved only visually - no physics, cannot touch anything.

  Frames: AC x = map_x, AC z = -map_y; heading yaw -> AC direction (cos yaw, 0, -sin yaw).
  Steering: +1 = left in our files, AC's steer is +1 = right: negated once, here.
]]

local VIEWER_MAGIC = 0x3157564D        -- "MVW1"
local VIEWER_STATE_MAGIC = 0x3153564D  -- "MVS1"
local VIEWER_VERSION = 3
local CMD = { NONE = 0, PLAY = 1, PAUSE = 2, STOP = 3, RESTART = 4, QUIT = 5 }

local DATA_LAYOUT = [[
uint32_t magic;
uint32_t version;
uint32_t seq;
uint32_t lap_id;
uint32_t command_seq;
int32_t command;
float speed;
int32_t loop;
int32_t n_car;
int32_t n_ghost;
float car_lap_time;
float ghost_lap_time;
uint8_t label[96];
uint8_t model[160];
float mass_kg;
float yaw_inertia_kgm2;
float car_t[12000];
float car_x[12000];
float car_y[12000];
float car_yaw[12000];
float car_speed[12000];
float car_steer[12000];
float car_thr[12000];
float car_brk[12000];
float car_gear[12000];
float car_rpm[12000];
float ghost_t[12000];
float ghost_x[12000];
float ghost_y[12000];
float ghost_yaw[12000];
float ghost_speed[12000];
]]

local STATE_LAYOUT = [[
uint32_t magic;
uint32_t version;
uint32_t seq;
uint32_t frame;
uint32_t lap_id_loaded;
uint32_t command_seq_seen;
int32_t playing;
int32_t physics_allowed;
int32_t cars_count;
int32_t ghost_is_car;
float play_time;
float gap_s;
]]

-- shared with mv_follow.lua (identical table there)
local FOLLOW_LAYOUT = {
  key = ac.StructItem.key('ProjectMV.viewer.follow.v1'),
  instanceKey = ac.StructItem.int32(),
  lapId = ac.StructItem.int32(),
  active = ac.StructItem.int32(),
  playing = ac.StructItem.int32(),
  speed = ac.StructItem.float(),
  seekSeq = ac.StructItem.int32(),
  seekT = ac.StructItem.float(),
  stiffness = ac.StructItem.float(),
  t = ac.StructItem.float(),
  posErr = ac.StructItem.float(),
  yawErrDeg = ac.StructItem.float(),
  atEnd = ac.StructItem.int32(),
  steps = ac.StructItem.int32(),
}

local data = ac.writeMemoryMappedFile('mv_viewer_v1', DATA_LAYOUT)
local st = ac.writeMemoryMappedFile('mv_viewer_state_v1', STATE_LAYOUT)
local follow = ac.connect(FOLLOW_LAYOUT)
local sim = ac.getSim()
local car = ac.getCar(0)
local U32 = 4294967296

-- ------------------------------------------------------------------ state
local lap = nil              -- copied lap: {car = {...}, ghost = {...}, label, model, carTime}
local lapId = nil
local cmdSeq = nil
local playing = false        -- clock running
local active = false         -- a lap is being shown (playing or held); false = stopped, car is the user's
local playT = 0
local speed = 1.0
local loop = true
local stiffness = 1.0
local status = 'waiting for a lap from the MV control panel'
local frame = 0
local gap = 0
local ghostIdx = 1
local heights = {}           -- ghost sample index -> road height (racing line drawing)
local options = { ghost = true, ghostOpacity = 0.6, racingLine = true, lineAheadM = 450 }
local controls = nil
local savedAutoShift = nil
local rideHeight = 0.5       -- car reference point above the road (m), measured at rest
local lastGround = nil
local dirSign = 1            -- meaning of setCarPosition's `dir`: verified after each placement
local verify = nil           -- {dir, frames, wrong} while checking a placement
-- follow worker
local workerKey = math.floor(os.preciseClock() * 1000) % 2000000000 + 1
local workerStarted, workerFailed = false, nil
local lastSteps, lastStepsChange = -1, 0
-- gears
local shift = nil            -- {target, t0, lastPulse} while a gear change is in progress (applyInputs)
-- ghost model
local ghostTransform = nil   -- mat4x4: where the ghost car is drawn this frame (nil = not shown)
local ghostMesh = nil        -- ac.SimpleMesh.carShape(0): CSP's simplified 3D shape of this car

-- ------------------------------------------------------------------ helpers
local function readString(field, len)
  local chars = {}
  for i = 0, len - 1 do
    local c = data[field][i]
    if c == 0 then break end
    chars[#chars + 1] = string.char(c)
  end
  return table.concat(chars)
end

local function copyArrays(prefix, fields, n)
  local out = { n = n }
  for _, f in ipairs(fields) do
    local src = data[prefix .. f]
    local t = {}
    for i = 0, n - 1 do t[i + 1] = src[i] end
    out[f] = t
  end
  return out
end

-- index i with t[i] <= time < t[i+1] (binary search), and the fraction between them
local function locate(arr, time)
  local t, n = arr.t, arr.n
  if time <= t[1] then return 1, 0 end
  if time >= t[n] then return n - 1, 1 end
  local lo, hi = 1, n
  while hi - lo > 1 do
    local mid = math.floor((lo + hi) / 2)
    if t[mid] <= time then lo = mid else hi = mid end
  end
  local span = t[hi] - t[lo]
  return lo, span > 0 and (time - t[lo]) / span or 0
end

local function lerp(a, b, f) return a + (b - a) * f end

local function lerpAngle(a, b, f)
  local d = (b - a + math.pi) % (2 * math.pi) - math.pi
  return a + d * f
end

local function sample(arr, time, fields)
  local i, f = locate(arr, time)
  local out = {}
  for _, k in ipairs(fields) do
    out[k] = k == 'yaw' and lerpAngle(arr.yaw[i], arr.yaw[i + 1], f) or lerp(arr[k][i], arr[k][i + 1], f)
  end
  return out, i
end

local function worldDir(yaw) return vec3(math.cos(yaw), 0, -math.sin(yaw)) end

-- Road surface under (x, z): height from the track's own centre line (works from any start height:
-- Spa climbs ~100 m from the pits), then a short ray down; `normalOut` receives the road's normal.
local function groundY(x, z, guessY, normalOut)
  local y = guessY
  local ok, tc = pcall(ac.worldCoordinateToTrack, vec3(x, guessY, z))
  if ok and tc then
    local ok2, w = pcall(ac.trackCoordinateToWorld, vec3(tc.x, 0, tc.z))
    if ok2 and w then y = w.y end
  end
  local from = vec3(x, y + 4, z)
  local d = physics.raycastTrack(from, vec3(0, -1, 0), 12, nil, normalOut)
  if d and d >= 0 then return from.y - d end
  from = vec3(x, y + 300, z)
  d = physics.raycastTrack(from, vec3(0, -1, 0), 700, nil, normalOut)
  if d and d >= 0 then return from.y - d end
  return y
end

-- Where the car's 3D model sits: its origin relative to the road and to the physics reference point
-- (car.position is near the centre of gravity, 0.455 m up on the 911; the model's origin is usually
-- at road level). WP34's ghost floated because it was placed at the physics height (WP35).
local modelHeight, modelForward = 0.0, 0.0

local function measureRideHeight()
  if (car.speedKmh or 99) < 2 then
    local ground = groundY(car.position.x, car.position.z, car.position.y)
    local h = car.position.y - ground
    if h > 0.05 and h < 1.5 then rideHeight = h end
    local m = car.bodyTransform                            -- the transform applied to the car's 3D model
    if m then
      local mh = m.position.y - ground
      if mh > -0.5 and mh < 1.5 then modelHeight = mh end
      local mf = (m.position - car.position):dot(car.look)
      if math.abs(mf) < 2 then modelForward = mf end
    end
  end
end

-- ------------------------------------------------------------------ the car: gearbox and controls
local function takeGearbox()
  if savedAutoShift == nil then
    savedAutoShift = car.autoShift and true or false
    physics.setCarAutoShifter(0, false)
  end
end

local function releaseCar()
  if controls then
    controls.steer = math.huge   -- leave the user's steering
    controls.gas, controls.brake, controls.handbrake = 0, 0, 0
    controls.clutch = 1
    controls.requestedGearIndex = 0
  end
  if savedAutoShift ~= nil then
    physics.setCarAutoShifter(0, savedAutoShift)
    savedAutoShift = nil
  end
  shift = nil
  follow.active = 0
end

-- Recorded steering and pedals every frame. Gears are changed the way the car's PDK does it
-- (WP35): one paddle pull per gear step, the clutch dipped while the gear changes and a throttle
-- blip on downshifts. Before, gears were forced in with the clutch released; with the owner's
-- settings (damage 100 %, no auto-clutch, H-shifter) the game counted every shift as a mis-shift
-- and damaged the gearbox (the flashing red drivetrain, gears popping out = "neutral in corners").
-- CSP's gear request and, last of all, engageGear are fallbacks if the paddles are ignored.
local function applyInputs(s, gear)
  takeGearbox()
  controls = controls or ac.overrideCarControls(0)
  if not controls then return end
  controls.steer = math.max(-1, math.min(1, -s.steer))
  controls.gas = math.max(0, math.min(1, s.thr))
  controls.brake = math.max(0, math.min(1, s.brk))
  controls.clutch = 1
  controls.requestedGearIndex = 0
  controls.gearUp, controls.gearDown = false, false
  local g = math.floor(gear + 0.5)
  if g < 1 or car.gear == g then shift = nil return end
  local now = os.preciseClock()
  if not shift or shift.target ~= g then shift = { target = g, t0 = now, lastPulse = -1 } end
  local since = now - shift.t0
  controls.clutch = 0                                      -- clutch in while the gear changes
  if car.gear > g then controls.gas = math.max(controls.gas, 0.45) end   -- blip: match revs going down
  if now - shift.lastPulse > 0.12 then                     -- one paddle pull per gear step
    if car.gear < g then controls.gearUp = true else controls.gearDown = true end
    shift.lastPulse = now
  end
  if since > 0.25 then controls.requestedGearIndex = g end -- paddles ignored (H-shifter set up)?
  if since > 0.6 then physics.engageGear(0, g); shift.t0 = now end   -- last resort, clutch is in
end

-- Put the car on the recorded pose at time T (road height, verified direction) - start/seek/loop only
local function placeAt(T)
  local s = sample(lap.car, T, { 'x', 'y', 'yaw', 'speed' })
  local dir = worldDir(s.yaw)
  lastGround = groundY(s.x, -s.y, lastGround or car.position.y)
  physics.setCarPosition(0, vec3(s.x, lastGround + rideHeight + 0.05, -s.y), dir * dirSign)
  local i = locate(lap.car, T)
  local a = lap.car
  local span = math.max(a.t[i + 1] - a.t[i], 1e-4)
  physics.setCarVelocity(0, vec3((a.x[i + 1] - a.x[i]) / span, 0, -(a.y[i + 1] - a.y[i]) / span))
  physics.awakeCar(0)
  verify = { dir = dir, frames = 0, wrong = 0, T = T }
end

-- After a placement: if the car faces against the intended direction for 3 frames, flip and redo
local function checkPlacement()
  if not verify then return end
  verify.frames = verify.frames + 1
  if verify.frames < 2 then return end
  if car.look:dot(verify.dir) < -0.3 then verify.wrong = verify.wrong + 1 else verify.wrong = 0 end
  if verify.wrong >= 3 then
    dirSign = -dirSign
    placeAt(verify.T)
    return
  end
  if verify.frames > 12 then verify = nil end
end

-- ------------------------------------------------------------------ follow worker
local function startWorker()
  if workerStarted or not physics.allowed() then return end
  workerStarted = true
  follow.instanceKey = workerKey
  follow.active = 0
  local ok, err = pcall(physics.startPhysicsWorker, 'mv_follow', workerKey, function(e)
    workerFailed = 'follower stopped: ' .. tostring(e)
  end)
  if not ok then workerFailed = 'follower could not start: ' .. tostring(err) end
  lastStepsChange = os.preciseClock()
end

local function followMode() return workerStarted and not workerFailed end

-- start/seek in follow mode: place the car, then tell the worker where the clock is
local function seek(T)
  playT = math.max(0, math.min(T, lap.carTime))
  placeAt(playT)
  follow.lapId = lapId
  follow.seekT = playT
  follow.seekSeq = follow.seekSeq + 1
  follow.atEnd = 0
  ghostIdx = 1
end

-- ------------------------------------------------------------------ the ghost: see-through car
-- WP39: loading the car model a second time and switching its materials to blending did not make it
-- see-through in the game (the car's own shaders ignore that). Now the app draws the ghost itself,
-- the way CSP's own debug app draws see-through cars: CSP's simplified 3D shape of the car
-- (ac.SimpleMesh.carShape) with a small shader in the transparent pass - faint where you look straight
-- at a surface, brighter towards its edges (a blue "hologram"), and hidden behind walls like a real car.
local ghostNormal = vec3(0, 1, 0)
local ghostGround = nil

local function updateGhost()
  ghostTransform = nil
  if not (options.ghost and active and lap and lap.ghost) then return end
  local gs = sample(lap.ghost, playT, { 'x', 'y', 'yaw' })
  ghostGround = groundY(gs.x, -gs.y, ghostGround or lastGround or car.position.y, ghostNormal)
  local look = worldDir(gs.yaw)
  local up = ghostNormal:length() > 0.5 and ghostNormal or vec3(0, 1, 0)
  look = (look - up * look:dot(up)):normalize()            -- follow the road's slope (Eau Rouge)
  -- placed where the real car's 3D model would be: on the road, not at the centre of gravity
  ghostTransform = mat4x4.look(vec3(gs.x, ghostGround, -gs.y) + up * modelHeight + look * modelForward, look, up)
end

-- one table reused every frame (CSP: never add new values later - that would recompile the shader)
local ghostDraw = {
  mesh = nil, transform = nil, async = true, textures = {},
  values = { gColor = rgbm(0.25, 0.6, 1.0, 1.0), gOpacity = 0.6 },
  shader = [[float4 main(PS_IN pin) {
    float g = dot(normalize(pin.NormalW), normalize(pin.PosC));
    float edge = pow(1 - abs(g), 2);
    return float4(gColor.rgb * gWhiteRefPoint * (0.5 + edge), saturate(0.15 + 0.6 * edge) * gOpacity);
  }]],
}

local function drawGhostCar()
  if not ghostTransform then return false end
  ghostMesh = ghostMesh or ac.SimpleMesh.carShape(0, false)
  ghostDraw.mesh, ghostDraw.transform = ghostMesh, ghostTransform
  ghostDraw.values.gOpacity = options.ghostOpacity
  render.setBlendMode(render.BlendMode.AlphaBlend)
  render.setDepthMode(render.DepthMode.ReadOnlyLessEqual)  -- behind walls it is hidden, like a real car
  render.setCullMode(render.CullMode.None)
  render.mesh(ghostDraw)
  return true
end

-- Time gap: when did the ghost pass the point where our car is now? (search near the last match)
local function updateGap(cs)
  local gh = lap.ghost
  if not gh then gap = 0 return end
  local best, bestD = ghostIdx, math.huge
  local n = gh.n
  for k = -40, 200 do
    local i = ((ghostIdx - 1 + k) % n) + 1
    local dx, dy = gh.x[i] - cs.x, gh.y[i] - cs.y
    local d = dx * dx + dy * dy
    if d < bestD then best, bestD = i, d end
  end
  if bestD > 900 then
    for i = 1, n do
      local dx, dy = gh.x[i] - cs.x, gh.y[i] - cs.y
      local d = dx * dx + dy * dy
      if d < bestD then best, bestD = i, d end
    end
  end
  ghostIdx = best
  gap = playT - gh.t[best]
end

-- ------------------------------------------------------------------ loading and commands
local function tryLoad()
  if data.magic ~= VIEWER_MAGIC or data.version ~= VIEWER_VERSION then return end
  local s1 = data.seq
  if s1 % 2 == 1 or data.lap_id == lapId then return end
  local nCar, nGhost = data.n_car, data.n_ghost
  local l = {
    car = copyArrays('car_', { 't', 'x', 'y', 'yaw', 'speed', 'steer', 'thr', 'brk', 'gear', 'rpm' }, nCar),
    ghost = nGhost > 1 and copyArrays('ghost_', { 't', 'x', 'y', 'yaw', 'speed' }, nGhost) or nil,
    label = readString('label', 96), model = readString('model', 160), carTime = data.car_lap_time,
  }
  local id = data.lap_id
  if data.seq ~= s1 then return end
  lap, lapId = l, id
  playT, ghostIdx, heights, lastGround, ghostGround = 0, 1, {}, nil, nil
  if not active then pcall(measureRideHeight) end
  status = 'lap loaded'
end

local function start(T)
  active, playing = true, true
  if followMode() then
    seek(T)
    follow.active = 1
  else
    playT = T
  end
end

local function handleCommand()
  if data.magic ~= VIEWER_MAGIC then return end
  local s1 = data.seq
  if s1 % 2 == 1 then return end
  local cs, c, sp, lp = data.command_seq, data.command, data.speed, data.loop
  if data.seq ~= s1 or cs == cmdSeq then return end
  cmdSeq = cs
  speed = (sp > 0.05 and sp < 8) and sp or 1
  loop = lp == 1
  if not lap then return end
  if c == CMD.PLAY then
    if active then playing = true else start(0) end
  elseif c == CMD.PAUSE then playing = false
  elseif c == CMD.STOP then active, playing, playT = false, false, 0; releaseCar()
  elseif c == CMD.RESTART then start(0)
  elseif c == CMD.QUIT then                              -- the control panel restarts the game elsewhere
    active, playing = false, false; releaseCar()
    local ok, err = pcall(ac.shutdownAssettoCorsa)
    if not ok then status = 'could not close the game: ' .. tostring(err) end
  end
end

-- ------------------------------------------------------------------ main loop
local function tick(dt)
  frame = frame + 1
  tryLoad()
  local allowed = physics.allowed()
  if lap and allowed then startWorker() end
  handleCommand()

  if followMode() and follow.active == 1 then              -- worker alive? (heartbeat)
    if follow.steps ~= lastSteps then lastSteps, lastStepsChange = follow.steps, os.preciseClock()
    elseif os.preciseClock() - lastStepsChange > 2 then workerFailed = 'follower stopped responding' end
  end

  if active and lap and not allowed then
    status = 'no physics access in this session: use offline Practice (see docs chapter 07)'
  elseif active and lap and not sim.isPaused then
    if followMode() then
      follow.playing = playing and 1 or 0
      follow.speed = speed
      follow.stiffness = stiffness
      follow.active = 1
      playT = follow.t
      if follow.atEnd == 1 then
        if loop then seek(0) else playing = false end
      elseif follow.posErr > 15 then                       -- crashed / stuck: put it back on the lap
        seek(playT)
      end
      checkPlacement()
    else                                                   -- fallback: place it every frame
      if playing then
        playT = playT + dt * speed
        if playT > lap.carTime then
          if loop then playT = 0; ghostIdx = 1 else playT = lap.carTime; playing = false end
        end
      end
      local s = sample(lap.car, playT, { 'x', 'y', 'yaw' })
      local dir = worldDir(s.yaw)
      lastGround = groundY(s.x, -s.y, lastGround or car.position.y)
      physics.setCarPosition(0, vec3(s.x, lastGround + rideHeight + 0.05, -s.y), dir * dirSign)
      local i = locate(lap.car, playT)
      local a = lap.car
      local span = math.max(a.t[i + 1] - a.t[i], 1e-4)
      physics.setCarVelocity(0, playing and vec3((a.x[i + 1] - a.x[i]) / span, 0, -(a.y[i + 1] - a.y[i]) / span)
        or vec3(0, 0, 0))
      if car.look:dot(dir) < -0.3 then dirSign = -dirSign end
    end
    local cs, i = sample(lap.car, playT, { 'x', 'y', 'steer', 'thr', 'brk' })
    applyInputs(cs, lap.car.gear[i])
    updateGap(cs)
    status = (playing and string.format('playing %.2fx', speed) or 'paused (car held at this moment)')
      .. (followMode() and '' or '  [fallback: per-frame placement - ' .. tostring(workerFailed or 'no worker') .. ']')
  elseif not active then
    if frame % 30 == 0 then pcall(measureRideHeight) end   -- standing on its wheels: learn its height
    if lap then status = sim.isPaused and 'game paused' or 'stopped - press Play' end
  end
  pcall(updateGhost)

  local s = st.seq
  if s % 2 == 1 then s = s + 1 end
  st.seq = (s + 1) % U32
  st.magic, st.version, st.frame = VIEWER_STATE_MAGIC, VIEWER_VERSION, frame % U32
  st.lap_id_loaded = (lapId or 0) % U32
  st.command_seq_seen = (cmdSeq or 0) % U32
  st.playing = playing and 1 or 0
  st.physics_allowed = allowed and 1 or 0
  st.cars_count = sim.carsCount or 1
  st.ghost_is_car = (ghostTransform ~= nil) and 1 or 0   -- (1 = ghost drawn as the see-through car, WP39)
  st.play_time = playT
  st.gap_s = gap
  st.seq = (s + 2) % U32
end

function script.update(dt)
  local ok, err = pcall(tick, dt)
  if not ok then
    ac.log('mv_viewer: ' .. tostring(err))
    status = 'ERROR: ' .. tostring(err)
  end
end

-- ------------------------------------------------------------------ 3D drawing: racing line (+ marker fallback)
local function speedColor(v, vmax)
  local f = math.max(0, math.min(1, v / math.max(vmax, 1)))
  return rgbm(1 - f, f, 0.15, 1)                          -- red = slow, green = fast
end

local function lineHeight(i, guess)
  local h = heights[i]
  if not h then
    h = groundY(lap.ghost.x[i], -lap.ghost.y[i], guess)
    heights[i] = h
  end
  return h
end

local function draw()
  if not lap or not lap.ghost or not active then return end
  local gh = lap.ghost
  local y0 = lastGround or car.position.y
  if options.racingLine then
    local vmax = 0
    for i = 1, gh.n do if gh.speed[i] > vmax then vmax = gh.speed[i] end end
    local dist, i, prev = 0, ghostIdx, nil
    while dist < options.lineAheadM do
      local p = vec3(gh.x[i], lineHeight(i, y0) + 0.08, -gh.y[i])
      if prev then
        render.debugLine(prev, p, speedColor(gh.speed[i], vmax))
        dist = dist + (p - prev):length()
      end
      prev, i = p, (i % gh.n) + 1
      if i == ghostIdx then break end
    end
  end
  local carDrawn = false
  if options.ghost then
    local ok, res = pcall(drawGhostCar)
    carDrawn = ok and res
    if not ok then ac.log('mv_viewer ghost car: ' .. tostring(res)) end
  end
  if options.ghost and not carDrawn then                   -- ghost car not available: a footprint marker
    local gs = sample(gh, playT, { 'x', 'y', 'yaw' })
    local f = worldDir(gs.yaw)
    local l = vec3(-f.z, 0, f.x)
    local c = vec3(gs.x, groundY(gs.x, -gs.y, y0) + 0.05, -gs.y)
    local a, b = f * 2.3, l * 0.95
    render.setBlendMode(render.BlendMode.AlphaBlend)
    render.quad(c + a + b, c + a - b, c - a - b, c - a + b, rgbm(0.3, 0.7, 1, 0.45))
    render.debugArrow(c, c + f * 3.5, 0.6, rgbm(0.3, 0.7, 1, 1))
  end
  if options.ghost then
    local gs = sample(gh, playT, { 'x', 'y' })
    render.debugText(vec3(gs.x, (ghostGround or y0) + 2.0, -gs.y), string.format('GHOST  %+.2f s', -gap),
      rgbm(0.6, 0.85, 1, 1), 1.1)
  end
end

render.on('main.root.transparent', function()
  local ok, err = pcall(draw)
  if not ok then ac.log('mv_viewer draw: ' .. tostring(err)) end
end)

-- ------------------------------------------------------------------ window
function script.windowMain(dt)
  ui.text('MV Viewer')
  ui.textWrapped(status)
  if lap then
    ui.textWrapped(lap.label)
    ui.text(string.format('time %.2f / %.2f s   gap to ghost %+.2f s', playT, lap.carTime, gap))
    if followMode() and active then
      ui.text(string.format('following: %.2f m, %.1f deg off the recorded lap', follow.posErr, follow.yawErrDeg))
    end
    if ui.button(active and playing and 'Pause' or 'Play') then
      if active then playing = not playing else start(0) end
    end
    ui.sameLine()
    if ui.button('Restart') then start(0) end
    ui.sameLine()
    if ui.button('Stop') then active, playing, playT = false, false, 0; releaseCar() end
    local v, changed = ui.slider('##speed', speed, 0.1, 2.0, 'speed %.2fx')
    if changed then speed = v end
    local t2, moved = ui.slider('##seek', playT, 0, lap.carTime, 'seek %.1f s')
    if moved and active then
      if followMode() then seek(t2) else playT = t2 end
    end
    local k, kChanged = ui.slider('##stiff', stiffness, 0.3, 3.0, 'follow stiffness %.1f')
    if kChanged then stiffness = k end
    if ui.checkbox('Ghost (see-through car)', options.ghost) then options.ghost = not options.ghost end
    local o, oChanged = ui.slider('##ghostop', options.ghostOpacity, 0.1, 1.0, 'ghost opacity %.2f')
    if oChanged then options.ghostOpacity = o end
    if ui.checkbox('Ghost racing line', options.racingLine) then options.racingLine = not options.racingLine end
    if ui.checkbox('Loop', loop) then loop = not loop end
  end
  ui.text(string.format('physics.allowed: %s   mode: %s', tostring(physics.allowed()),
    followMode() and 'follow (forces, 333 Hz)' or 'per-frame placement'))
end
