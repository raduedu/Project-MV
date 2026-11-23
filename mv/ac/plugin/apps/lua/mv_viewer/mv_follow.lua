--[[
  MV Viewer — force follower (CSP physics worker, runs at the physics rate: 333 steps per second).

  Started by mv_viewer.lua with physics.startPhysicsWorker('mv_follow', key). Why it exists
  (engineering log WP34): placing the car by teleport every rendered frame fought the game's
  physics - the picture shook, the wheels and drivetrain were reset each time (empty dashboard,
  gear dropping to neutral). Here the car stays an ordinary car: suspension, tyres, gearbox,
  engine, dashboard and sound are the game's own. This worker only adds forces that pull it
  along the recorded lap:

    * translation: a spring-damper towards where the car was at this moment of the lap,
          a = kp * (position error) + kd * (velocity error),     force = mass * a
      applied at the car's reference point, horizontal only (height is the road's business);
    * rotation: a pair of opposite sideways forces at the nose and the tail (a couple) that turn
      the car towards the recorded heading, with damping on the turning rate.

  Everything is written with directions in the world (push the nose towards the target heading),
  so no sign convention of the game has to be guessed. The playback clock lives here (physics
  time), the main script reads it for the picture, the ghost and the controls.

  Shared with mv_viewer.lua through ac.connect (layout FOLLOW_LAYOUT, identical in both files).
  The lap itself is read straight from the viewer's shared memory (DATA_LAYOUT, identical to
  mv_viewer.lua and mv/ac/viewer_link.py; tested).
]]

local startKey = worker.input

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

local FOLLOW_LAYOUT = {
  key = ac.StructItem.key('ProjectMV.viewer.follow.v1'),
  instanceKey = ac.StructItem.int32(),
  lapId = ac.StructItem.int32(),       -- lap the follower may use (main script checks it is loaded)
  active = ac.StructItem.int32(),      -- 1: apply forces
  playing = ac.StructItem.int32(),     -- 1: clock runs; 0: hold the car at the current moment
  speed = ac.StructItem.float(),
  seekSeq = ac.StructItem.int32(),     -- changed by the main script after it placed the car at seekT
  seekT = ac.StructItem.float(),
  stiffness = ac.StructItem.float(),   -- 1 = default gains
  t = ac.StructItem.float(),           -- worker -> main: playback time (s)
  posErr = ac.StructItem.float(),      -- worker -> main: distance to the recorded position (m)
  yawErrDeg = ac.StructItem.float(),   -- worker -> main: heading error (deg)
  atEnd = ac.StructItem.int32(),       -- worker -> main: the lap is over (main loops or stops)
  steps = ac.StructItem.int32(),       -- worker -> main: heartbeat
}

local d = ac.readMemoryMappedFile('mv_viewer_v1', DATA_LAYOUT)
local s = ac.connect(FOLLOW_LAYOUT)

local ARM = 1.2                 -- m: nose and tail points of the turning couple (car-local z, forward)
local A_MAX = 30.0              -- m/s^2: limit of the pulling acceleration (a crash must not explode)
local ALPHA_MAX = 10.0          -- rad/s^2: limit of the turning acceleration
local vZero, vNose, vTail = vec3(0, 0, 0), vec3(0, 0, ARM), vec3(0, 0, -ARM)
local force, f = vec3(), vec3()
local idx = 0
local lastSeek = nil
local t = 0

local function wrap(a) return (a + math.pi) % (2 * math.pi) - math.pi end

-- recorded state at time tt: position (AC x, z), velocity, heading direction and its rate of change
local function reference(tt)
  local n = d.n_car
  local ct = d.car_t
  if idx > n - 2 or ct[idx] > tt then idx = 0 end
  while idx < n - 2 and ct[idx + 1] <= tt do idx = idx + 1 end
  local i = idx
  local t0, t1 = ct[i], ct[i + 1]
  local span = math.max(t1 - t0, 1e-4)
  local fr = math.max(0, math.min(1, (tt - t0) / span))
  local x0, x1, y0, y1 = d.car_x[i], d.car_x[i + 1], d.car_y[i], d.car_y[i + 1]
  local yaw0 = d.car_yaw[i]
  local dyaw = wrap(d.car_yaw[i + 1] - yaw0)
  local yaw = yaw0 + dyaw * fr
  local r = dyaw / span
  -- map frame -> AC world: x = map_x, z = -map_y; heading look = (cos yaw, 0, -sin yaw)
  return x0 + (x1 - x0) * fr, -(y0 + (y1 - y0) * fr),
         (x1 - x0) / span, -(y1 - y0) / span,
         math.cos(yaw), -math.sin(yaw),
         -math.sin(yaw) * r, -math.cos(yaw) * r          -- d(look)/dt
end

function script.update(dt)
  if s.instanceKey ~= startKey then worker.terminate() return end
  s.steps = s.steps + 1
  if s.active ~= 1 or d.magic ~= 0x3157564D or d.lap_id ~= s.lapId or d.n_car < 3 then return end

  if s.seekSeq ~= lastSeek then lastSeek = s.seekSeq; t = s.seekT; idx = 0; s.atEnd = 0 end
  local holding = s.playing ~= 1
  if not holding then
    t = t + dt * math.max(0.05, s.speed)
    if t >= d.car_lap_time then t = d.car_lap_time; s.atEnd = 1 end
  end
  s.t = t

  local px, pz, vx, vz, lx, lz, dlx, dlz = reference(t)
  if holding then vx, vz, dlx, dlz = 0, 0, 0, 0 end

  local T = physics.getBodyTransform(0)
  local p, L = T.position, T.look
  local v = physics.getPointVelocity(0, vZero, true)
  local m = math.max(d.mass_kg, 300)
  local iz = math.max(d.yaw_inertia_kgm2, 200)
  local k = math.max(0.2, s.stiffness)
  -- rad/s: natural frequencies (critically damped). Chosen offline (WP34): a rigid body with the
  -- 911's mass and yaw inertia driven by these equations along the recorded Spa lap follows it to
  -- 0.2 m / 0.1 deg on average; with crude tyres fighting every slide, 1.0 m / 0.1 deg (peaks ~5 m).
  local wp, wy = 4.5 * k, 9.0 * k
  local kp, kd, kpy, kdy = wp * wp, 2 * wp, wy * wy, 2 * wy

  -- translation (horizontal)
  local ex, ez = px - p.x, pz - p.z
  local ax = kp * ex + kd * (vx - v.x)
  local az = kp * ez + kd * (vz - v.z)
  local amag = math.sqrt(ax * ax + az * az)
  if amag > A_MAX then ax, az = ax * A_MAX / amag, az * A_MAX / amag end
  force:set(m * ax, 0, m * az)
  physics.addForce(0, vZero, true, force, false)
  s.posErr = math.sqrt(ex * ex + ez * ez)

  -- rotation: couple at nose/tail, towards the recorded heading
  local hx, hz = L.x, L.z
  local hl = math.sqrt(hx * hx + hz * hz)
  if hl < 1e-3 then return end
  hx, hz = hx / hl, hz / hl
  local dot = hx * lx + hz * lz
  local sx, sz = lx - dot * hx, lz - dot * hz               -- part of the target heading sideways to the car
  local sl = math.sqrt(sx * sx + sz * sz)
  local angle = math.atan2(sl, dot)                          -- 0..pi, the heading error
  s.yawErrDeg = math.deg(angle)
  if sl > 1e-6 then sx, sz = sx / sl, sz / sl else sx, sz = -hz, hx end
  -- damping: sideways speed of the nose relative to the tail, actual vs recorded (2*ARM*dlook/dt)
  local vn = physics.getPointVelocity(0, vNose, true)
  local vt = physics.getPointVelocity(0, vTail, true)
  local rx, rz = vn.x - vt.x, vn.z - vt.z
  local ra = rx * hx + rz * hz
  rx, rz = rx - ra * hx, rz - ra * hz
  local qx, qz = 2 * ARM * dlx, 2 * ARM * dlz
  local qa = qx * hx + qz * hz
  qx, qz = qx - qa * hx, qz - qa * hz
  -- sideways acceleration wanted at the nose (relative to the centre)
  local nx = ARM * kpy * angle * sx + kdy * (qx - rx) * 0.5
  local nz = ARM * kpy * angle * sz + kdy * (qz - rz) * 0.5
  local nmag = math.sqrt(nx * nx + nz * nz)
  local nmax = ALPHA_MAX * ARM
  if nmag > nmax then nx, nz = nx * nmax / nmag, nz * nmax / nmag end
  local c = iz / (2 * ARM * ARM)                             -- force per (m/s^2) of nose acceleration
  f:set(c * nx, 0, c * nz)
  physics.addForce(0, vNose, true, f, false)
  f:set(-c * nx, 0, -c * nz)
  physics.addForce(0, vTail, true, f, false)
end
