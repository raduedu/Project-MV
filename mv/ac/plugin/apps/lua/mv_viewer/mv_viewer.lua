--[[
  MV Viewer — CSP Lua app that plays a Project MV lap in Assetto Corsa, with the ghost.

  Installed to: assettocorsa/apps/lua/mv_viewer/   (use mv/ac/plugin/install_plugin.ps1)
  Python side : mv/ac/viewer_link.py  (web UI button "View in Assetto Corsa", scripts/ac_view.py)

  Python -> Lua : Local\mv_viewer_v1        (the lap: car + ghost samples, play/pause/stop)
  Lua -> Python : Local\mv_viewer_state_v1  (loaded lap id, play time, gap)

  The two layouts below MUST match mv/ac/viewer_link.py byte for byte;
  tests/test_ac_viewer.py checks the text of both blocks against Python.

  What it does every frame while playing:
    * the player's car is placed where the recorded car was at the play time
      (physics.setCarPosition: CSP aligns it on the track surface), with the
      recorded velocity, steering, pedals and gear, so the car model, wheels,
      steering wheel and brake lights move as in the lap;
    * the ghost (ideal lap) is a second car if the session has one (collisions
      off), otherwise a translucent footprint + label; its racing line ahead is
      drawn, coloured by speed; the time gap is shown.

  Session: Practice on the same circuit and car as the lap. Physics access
  (physics.allowed) is needed - the window says if it is missing.
  Map frame -> AC world: AC x = map_x, AC z = -map_y; heading yaw (CCW from
  +map_x) -> AC direction (cos yaw, 0, -sin yaw). Steering: +1 = left in our
  files, AC's steer is +1 = right, so it is negated once, here.
]]

local VIEWER_MAGIC = 0x3157564D        -- "MVW1"
local VIEWER_STATE_MAGIC = 0x3153564D  -- "MVS1"
local VIEWER_VERSION = 1
local CMD = { NONE = 0, PLAY = 1, PAUSE = 2, STOP = 3, RESTART = 4 }

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
float car_t[12000];
float car_x[12000];
float car_y[12000];
float car_yaw[12000];
float car_speed[12000];
float car_steer[12000];
float car_thr[12000];
float car_brk[12000];
float car_gear[12000];
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

local data = ac.writeMemoryMappedFile('mv_viewer_v1', DATA_LAYOUT)
local st = ac.writeMemoryMappedFile('mv_viewer_state_v1', STATE_LAYOUT)
local sim = ac.getSim()
local car = ac.getCar(0)
local U32 = 4294967296

-- ------------------------------------------------------------------ state
local lap = nil              -- copied lap: {car = {...}, ghost = {...}, label, carTime, ghostTime}
local lapId = nil
local cmdSeq = nil
local playing = false
local playT = 0
local speed = 1.0
local loop = true
local dirSign = 1            -- meaning of setCarPosition's `dir` is checked once, like mv_bridge
local dirChecked = false
local controls = nil
local lastGear = nil
local gap = 0
local ghostIdx = 1
local options = { ghostAsCar = true, ghostMarker = true, racingLine = true, lineAheadM = 450 }
local status = 'waiting for a lap from the MV control panel'
local frame = 0
local ghostCarReady = false
local heights = {}           -- ghost sample index -> ground height (raycast once)

-- ------------------------------------------------------------------ helpers
local function readLabel()
  local chars = {}
  for i = 0, 95 do
    local c = data.label[i]
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

-- Load a new lap when Python published one (seqlock: copy only a consistent version).
local function tryLoad()
  if data.magic ~= VIEWER_MAGIC or data.version ~= VIEWER_VERSION then return end
  local s1 = data.seq
  if s1 % 2 == 1 or data.lap_id == lapId then return end
  local nCar, nGhost = data.n_car, data.n_ghost
  local l = {
    car = copyArrays('car_', { 't', 'x', 'y', 'yaw', 'speed', 'steer', 'thr', 'brk', 'gear' }, nCar),
    ghost = nGhost > 1 and copyArrays('ghost_', { 't', 'x', 'y', 'yaw', 'speed' }, nGhost) or nil,
    label = readLabel(), carTime = data.car_lap_time, ghostTime = data.ghost_lap_time,
  }
  local id = data.lap_id
  if data.seq ~= s1 then return end               -- Python wrote meanwhile: try again next frame
  lap, lapId = l, id
  playT, ghostIdx, heights, lastGear = 0, 1, {}, nil
  status = 'lap loaded'
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
  return out
end

local function groundY(x, z, guessY)
  local from = vec3(x, guessY + 40, z)
  local d = physics.raycastTrack(from, vec3(0, -1, 0), 120)
  if d and d >= 0 then return from.y - d end
  return guessY
end

local function worldDir(yaw) return vec3(math.cos(yaw), 0, -math.sin(yaw)) end

local function releaseControls()
  if controls then
    controls.steer = math.huge   -- leave the user's steering
    controls.gas, controls.brake, controls.handbrake = 0, 0, 0
    controls.clutch = 1
  end
end

local function placeCar(s)
  local dir = worldDir(s.yaw)
  local pos = vec3(s.x, car.position.y, -s.y)
  physics.setCarPosition(0, pos, dir * dirSign)
  physics.setCarVelocity(0, dir * s.speed)
  controls = controls or ac.overrideCarControls(0)
  if controls then
    controls.steer = math.max(-1, math.min(1, -s.steer))
    controls.gas = math.max(0, math.min(1, s.thr))
    controls.brake = math.max(0, math.min(1, s.brk))
    controls.clutch = 1
  end
  local g = math.floor(s.gear + 0.5)
  if g >= 1 and g ~= lastGear then
    physics.engageGear(0, g)
    lastGear = g
  end
  return dir
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
  if bestD > 900 then                                   -- lost (lap restarted, seek): global search
    for i = 1, n do
      local dx, dy = gh.x[i] - cs.x, gh.y[i] - cs.y
      local d = dx * dx + dy * dy
      if d < bestD then best, bestD = i, d end
    end
  end
  ghostIdx = best
  gap = playT - gh.t[best]
end

-- ------------------------------------------------------------------ main loop
local function handleCommand()
  if data.magic ~= VIEWER_MAGIC then return end
  local s1 = data.seq
  if s1 % 2 == 1 then return end
  local cs, c, sp, lp = data.command_seq, data.command, data.speed, data.loop
  if data.seq ~= s1 or cs == cmdSeq then return end
  cmdSeq = cs
  speed = (sp > 0.05 and sp < 8) and sp or 1
  loop = lp == 1
  if c == CMD.PLAY then playing = lap ~= nil
  elseif c == CMD.PAUSE then playing = false
  elseif c == CMD.STOP then playing = false; playT = 0; releaseControls()
  elseif c == CMD.RESTART then playT = 0; ghostIdx = 1; playing = lap ~= nil end
end

local function tick(dt)
  frame = frame + 1
  tryLoad()
  handleCommand()
  local allowed = physics.allowed()
  if playing and lap and not allowed then
    status = 'no physics access in this session: use offline Practice (see docs chapter 07)'
  elseif playing and lap and not sim.isPaused then
    playT = playT + dt * speed
    if playT > lap.carTime then
      if loop then playT = 0; ghostIdx = 1 else playT = lap.carTime; playing = false end
    end
    local cs = sample(lap.car, playT, { 'x', 'y', 'yaw', 'speed', 'steer', 'thr', 'brk', 'gear' })
    local dir = placeCar(cs)
    if not dirChecked and frame % 3 == 0 then           -- verify the car faces the way it drives, once
      if car.look:dot(dir) < 0 then dirSign = -dirSign end
      dirChecked = true
    end
    updateGap(cs)
    if lap.ghost and options.ghostAsCar and sim.carsCount > 1 then
      local gs = sample(lap.ghost, playT, { 'x', 'y', 'yaw', 'speed' })
      local gdir = worldDir(gs.yaw)
      if not ghostCarReady then                          -- the AI must not drive or collide: it is a ghost
        physics.disableCarCollisions(1, true)
        physics.setAINoInput(1, true, false)
        ghostCarReady = true
      end
      physics.setAICarPosition(1, vec3(gs.x, car.position.y, -gs.y), gdir * dirSign)
      physics.setCarVelocity(1, gdir * gs.speed)
    end
    status = string.format('playing %.2fx', speed)
  elseif lap then
    status = playing and 'paused (game paused)' or 'stopped / paused'
  end

  local s = st.seq
  if s % 2 == 1 then s = s + 1 end
  st.seq = (s + 1) % U32
  st.magic, st.version, st.frame = VIEWER_STATE_MAGIC, VIEWER_VERSION, frame % U32
  st.lap_id_loaded = (lapId or 0) % U32
  st.command_seq_seen = (cmdSeq or 0) % U32
  st.playing = playing and 1 or 0
  st.physics_allowed = allowed and 1 or 0
  st.cars_count = sim.carsCount or 1
  st.ghost_is_car = (ghostCarReady and options.ghostAsCar) and 1 or 0
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

-- ------------------------------------------------------------------ 3D drawing
local function speedColor(v, vmax)
  local f = math.max(0, math.min(1, v / math.max(vmax, 1)))
  return rgbm(1 - f, f, 0.15, 1)                          -- red = slow, green = fast
end

local function ghostHeight(i, guess)
  local h = heights[i]
  if not h then
    h = groundY(lap.ghost.x[i], -lap.ghost.y[i], guess)
    heights[i] = h
  end
  return h
end

local function draw()
  if not lap or not lap.ghost or not playing then return end
  local gh = lap.ghost
  local y0 = car.position.y
  if options.racingLine then                             -- ghost's line ahead, coloured by speed
    local vmax = 0
    for i = 1, gh.n do if gh.speed[i] > vmax then vmax = gh.speed[i] end end
    local dist, i = 0, ghostIdx
    local prev = nil
    while dist < options.lineAheadM do
      local j = (i % gh.n) + 1
      local p = vec3(gh.x[i], ghostHeight(i, y0) + 0.08, -gh.y[i])
      if prev then
        render.debugLine(prev, p, speedColor(gh.speed[i], vmax))
        dist = dist + (p - prev):length()
      end
      prev, i = p, j
      if i == ghostIdx then break end
    end
  end
  if options.ghostMarker and not (options.ghostAsCar and sim.carsCount > 1) then
    local gs = sample(gh, playT, { 'x', 'y', 'yaw' })
    local f = worldDir(gs.yaw)
    local l = vec3(-f.z, 0, f.x)                           -- left of the direction (y up)
    local c = vec3(gs.x, groundY(gs.x, -gs.y, y0) + 0.05, -gs.y)
    local a, b = f * 2.3, l * 0.95                         -- half length / half width of a car (m)
    render.setBlendMode(render.BlendMode.AlphaBlend)
    render.quad(c + a + b, c + a - b, c - a - b, c - a + b, rgbm(0.3, 0.7, 1, 0.45))
    render.debugArrow(c, c + f * 3.5, 0.6, rgbm(0.3, 0.7, 1, 1))
    render.debugText(c + vec3(0, 1.6, 0), string.format('GHOST  %+.2f s', -gap), rgbm(0.6, 0.85, 1, 1), 1.2)
  end
end

render.on('main.root.transparent', function()
  local ok, err = pcall(draw)
  if not ok then ac.log('mv_viewer draw: ' .. tostring(err)) end
end)

-- ------------------------------------------------------------------ window
function script.windowMain(dt)
  ui.text('MV Viewer')
  ui.text(status)
  if lap then
    ui.textWrapped(lap.label)
    ui.text(string.format('time %.2f / %.2f s   gap to ghost %+.2f s', playT, lap.carTime, gap))
    if ui.button(playing and 'Pause' or 'Play') then playing = not playing end
    ui.sameLine()
    if ui.button('Restart') then playT, ghostIdx = 0, 1; playing = true end
    ui.sameLine()
    if ui.button('Stop') then playing = false; playT = 0; releaseControls() end
    local v, changed = ui.slider('##speed', speed, 0.1, 2.0, 'speed %.2fx')
    if changed then speed = v end
    local t2, moved = ui.slider('##seek', playT, 0, lap.carTime, 'seek %.1f s')
    if moved then playT = t2 end
    if ui.checkbox('Ghost as 2nd car (needs a session with 2 cars)', options.ghostAsCar) then
      options.ghostAsCar = not options.ghostAsCar
    end
    if ui.checkbox('Ghost marker', options.ghostMarker) then options.ghostMarker = not options.ghostMarker end
    if ui.checkbox('Ghost racing line', options.racingLine) then options.racingLine = not options.racingLine end
    if ui.checkbox('Loop', loop) then loop = not loop end
  end
  ui.text(string.format('physics.allowed: %s   cars: %d', tostring(physics.allowed()), sim.carsCount or 1))
end
