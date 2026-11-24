--[[
  MV generic Formula Student car - a 3-D model drawn by the MV Viewer app itself (engineering log WP47).

  Why: the owner's Assetto Corsa has no Formula Student car (and no electric car at all), and the
  project shows only generic, unbranded cars of its own. So the viewer draws this simple model on
  the position of the car the game is actually moving (a stand-in, hidden while this one is shown).

  How: a list of triangles in the car's own frame (metres; x = right, y = up, z = forward; origin on
  the ground midway between the axles), built once from a few shapes (tapered boxes, beams, a
  sphere, cylinders). Each frame every triangle is moved to the car's pose, shaded by how much it
  faces a fixed light from above, and drawn with CSP's immediate-mode calls (render.glBegin ...).
  Triangles are grouped by colour and shade step, so there are a few dozen draw calls, not hundreds.

  Proportions follow vehicles/library/fs_generic_ev.yaml (wheelbase 1.55 m, track 1.22/1.18 m,
  tyre radius 0.203 m, width 1.42 m, length 2.95 m). Colours: white body, carbon-black parts, one
  red accent. No names, numbers or logos.
]]

local M = {}

local DIM = {
  wheelbase = 1.55, trackF = 1.22, trackR = 1.18, tyreR = 0.203, tyreW = 0.19, rimR = 0.13,
}

local COL = {
  body = { 0.86, 0.87, 0.89 },
  carbon = { 0.07, 0.07, 0.08 },
  accent = { 0.84, 0.0, 0.11 },
  tyre = { 0.035, 0.035, 0.04 },
  rim = { 0.55, 0.56, 0.58 },
  spoke = { 0.20, 0.20, 0.22 },
  helmet = { 0.92, 0.92, 0.94 },
  visor = { 0.05, 0.06, 0.08 },
}

-- ------------------------------------------------------------------ building the shape (car frame)
local function tri(list, a, b, c, col)
  -- face normal from the winding; drawing is double-sided, the normal is only for shading
  local ux, uy, uz = b[1] - a[1], b[2] - a[2], b[3] - a[3]
  local vx, vy, vz = c[1] - a[1], c[2] - a[2], c[3] - a[3]
  local nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
  local l = math.sqrt(nx * nx + ny * ny + nz * nz)
  if l < 1e-9 then return end
  list[#list + 1] = { a, b, c, { nx / l, ny / l, nz / l }, col }
end

local function quad(list, a, b, c, d, col)
  tri(list, a, b, c, col); tri(list, a, c, d, col)
end

-- a box tapered along z: rear face at z0 (width w0, from y=b0 to t0), front face at z1 (w1, b1..t1)
local function frustum(list, cx, z0, z1, w0, w1, b0, t0, b1, t1, col)
  local r = { { cx - w0 / 2, b0, z0 }, { cx + w0 / 2, b0, z0 }, { cx + w0 / 2, t0, z0 }, { cx - w0 / 2, t0, z0 } }
  local f = { { cx - w1 / 2, b1, z1 }, { cx + w1 / 2, b1, z1 }, { cx + w1 / 2, t1, z1 }, { cx - w1 / 2, t1, z1 } }
  quad(list, r[1], r[2], r[3], r[4], col)          -- rear
  quad(list, f[2], f[1], f[4], f[3], col)          -- front
  quad(list, r[1], f[1], f[2], r[2], col)          -- bottom
  quad(list, r[4], r[3], f[3], f[4], col)          -- top
  quad(list, r[1], r[4], f[4], f[1], col)          -- left
  quad(list, r[2], f[2], f[3], r[3], col)          -- right
end

local function box(list, cx, cy, cz, sx, sy, sz, col)
  frustum(list, cx, cz - sz / 2, cz + sz / 2, sx, sx, cy - sy / 2, cy + sy / 2, cy - sy / 2, cy + sy / 2, col)
end

-- a thin square beam from p to q (tubes, wishbones, pylons)
local function beam(list, p, q, t, col)
  local dx, dy, dz = q[1] - p[1], q[2] - p[2], q[3] - p[3]
  local l = math.sqrt(dx * dx + dy * dy + dz * dz)
  if l < 1e-6 then return end
  dx, dy, dz = dx / l, dy / l, dz / l
  local ax, ay, az = 0, 1, 0                         -- any vector not along the beam
  if math.abs(dy) > 0.9 then ax, ay, az = 1, 0, 0 end
  local ux, uy, uz = dy * az - dz * ay, dz * ax - dx * az, dx * ay - dy * ax
  local ul = math.sqrt(ux * ux + uy * uy + uz * uz)
  ux, uy, uz = ux / ul * t / 2, uy / ul * t / 2, uz / ul * t / 2
  local vx, vy, vz = dy * uz - dz * uy, dz * ux - dx * uz, dx * uy - dy * ux
  local function c(o, s1, s2)
    return { o[1] + s1 * ux + s2 * vx, o[2] + s1 * uy + s2 * vy, o[3] + s1 * uz + s2 * vz }
  end
  local P = { c(p, -1, -1), c(p, 1, -1), c(p, 1, 1), c(p, -1, 1) }
  local Q = { c(q, -1, -1), c(q, 1, -1), c(q, 1, 1), c(q, -1, 1) }
  for i = 1, 4 do
    local j = i % 4 + 1
    quad(list, P[i], P[j], Q[j], Q[i], col)
  end
end

local function sphere(list, cx, cy, cz, r, col, nLon, nLat)
  for j = 0, nLat - 1 do
    local a0, a1 = math.pi * (j / nLat - 0.5), math.pi * ((j + 1) / nLat - 0.5)
    for i = 0, nLon - 1 do
      local b0, b1 = 2 * math.pi * i / nLon, 2 * math.pi * (i + 1) / nLon
      local function p(a, b) return { cx + r * math.cos(a) * math.sin(b), cy + r * math.sin(a), cz + r * math.cos(a) * math.cos(b) } end
      quad(list, p(a0, b0), p(a0, b1), p(a1, b1), p(a1, b0), col)
    end
  end
end

-- one wheel in its own frame (x = axle, outward positive for a right wheel), centred at the origin;
-- returns triangles whose points are later rotated by spin and steer
local function wheelShape(side)
  local list, n = {}, 20
  local r, w, rr = DIM.tyreR, DIM.tyreW, DIM.rimR
  local xo, xi = side * w / 2, -side * w / 2      -- outer and inner face
  for i = 0, n - 1 do
    local t0, t1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
    local function p(x, rad, t) return { x, rad * math.sin(t), rad * math.cos(t) } end
    quad(list, p(xi, r, t0), p(xi, r, t1), p(xo, r, t1), p(xo, r, t0), COL.tyre)               -- tread
    quad(list, p(xo, r, t0), p(xo, r, t1), p(xo, rr, t1), p(xo, rr, t0), COL.tyre)             -- sidewalls
    quad(list, p(xi, r, t0), p(xi, rr, t0), p(xi, rr, t1), p(xi, r, t1), COL.tyre)
    -- rim face, two-tone so that the wheel is seen turning
    local rim = (i % 4 < 2) and COL.rim or COL.spoke
    tri(list, { xo * 0.8, 0, 0 }, p(xo * 0.8, rr, t0), p(xo * 0.8, rr, t1), rim)
    tri(list, { xi * 0.8, 0, 0 }, p(xi * 0.8, rr, t1), p(xi * 0.8, rr, t0), COL.spoke)
  end
  return list
end

local function bodyShape()
  local L = {}
  local zf, zr = DIM.wheelbase / 2, -DIM.wheelbase / 2
  -- monocoque and nose
  frustum(L, 0, -0.40, 0.95, 0.62, 0.40, 0.05, 0.62, 0.09, 0.45, COL.body)
  frustum(L, 0, 0.95, 1.40, 0.40, 0.16, 0.09, 0.45, 0.11, 0.24, COL.body)
  box(L, 0, 0.627, 0.22, 0.40, 0.012, 0.62, COL.carbon)                 -- cockpit opening
  box(L, 0, 0.47, 1.08, 0.36, 0.012, 0.20, COL.accent)                  -- accent stripe on the nose
  -- driver
  sphere(L, 0, 0.80, -0.02, 0.13, COL.helmet, 10, 6)
  box(L, 0, 0.81, 0.10, 0.17, 0.05, 0.03, COL.visor)
  -- roll hoops
  beam(L, { -0.22, 0.58, -0.22 }, { -0.17, 1.00, -0.25 }, 0.035, COL.carbon)
  beam(L, { 0.22, 0.58, -0.22 }, { 0.17, 1.00, -0.25 }, 0.035, COL.carbon)
  beam(L, { -0.17, 1.00, -0.25 }, { 0.17, 1.00, -0.25 }, 0.035, COL.carbon)
  beam(L, { -0.17, 0.62, 0.55 }, { 0.17, 0.62, 0.55 }, 0.03, COL.carbon)
  -- sidepods (cooling) with accent top edge
  for _, s in ipairs({ -1, 1 }) do
    frustum(L, s * 0.43, -0.55, 0.40, 0.24, 0.22, 0.07, 0.40, 0.09, 0.32, COL.body)
    box(L, s * 0.53, 0.40, -0.08, 0.03, 0.012, 0.90, COL.accent)
  end
  -- rear: accumulator / motor housing and diffuser
  frustum(L, 0, -1.10, -0.38, 0.40, 0.58, 0.08, 0.40, 0.06, 0.60, COL.body)
  box(L, 0, 0.07, -1.18, 0.90, 0.05, 0.40, COL.carbon)
  -- front wing: two elements and end plates
  box(L, 0, 0.075, 1.34, 1.30, 0.02, 0.30, COL.carbon)
  box(L, 0, 0.14, 1.24, 1.30, 0.02, 0.16, COL.carbon)
  for _, s in ipairs({ -1, 1 }) do box(L, s * 0.66, 0.12, 1.30, 0.02, 0.17, 0.40, COL.accent) end
  -- rear wing: three elements, end plates, pylons
  box(L, 0, 1.00, -1.32, 1.00, 0.02, 0.36, COL.carbon)
  box(L, 0, 1.10, -1.42, 1.00, 0.02, 0.22, COL.carbon)
  box(L, 0, 1.19, -1.49, 1.00, 0.02, 0.14, COL.carbon)
  for _, s in ipairs({ -1, 1 }) do
    box(L, s * 0.51, 1.06, -1.40, 0.02, 0.38, 0.50, COL.accent)
    beam(L, { s * 0.14, 0.55, -1.00 }, { s * 0.14, 0.99, -1.30 }, 0.03, COL.carbon)
  end
  -- wishbones (upper and lower, each a "V" from the chassis to the upright)
  for _, s in ipairs({ -1, 1 }) do
    for _, ax in ipairs({ { zf, DIM.trackF / 2, 0.95, 0.55, 0.22 }, { zr, DIM.trackR / 2, -0.50, -1.00, 0.26 } }) do
      local z, half, za, zb, cx = ax[1], ax[2], ax[3], ax[4], ax[5]
      local ux = s * (half - 0.09)
      for _, h in ipairs({ { 0.15, 0.11 }, { 0.40, 0.30 } }) do
        beam(L, { s * cx, h[1], za }, { ux, h[2], z }, 0.022, COL.carbon)
        beam(L, { s * cx, h[1], zb }, { ux, h[2], z }, 0.022, COL.carbon)
      end
    end
  end
  return L
end

local BODY = bodyShape()
local WHEELS = {
  { x = -DIM.trackF / 2, z = DIM.wheelbase / 2, steer = true, tris = wheelShape(-1) },
  { x = DIM.trackF / 2, z = DIM.wheelbase / 2, steer = true, tris = wheelShape(1) },
  { x = -DIM.trackR / 2, z = -DIM.wheelbase / 2, steer = false, tris = wheelShape(-1) },
  { x = DIM.trackR / 2, z = -DIM.wheelbase / 2, steer = false, tris = wheelShape(1) },
}
M.triangles = #BODY + 4 * #WHEELS[1].tris
M.dim = DIM

-- ------------------------------------------------------------------ drawing
local LIGHT = { 0.33, 0.86, 0.39 }               -- fixed light from above (normalised)
local STEPS = 6                                  -- shade steps: triangles of equal colour+step share a draw call
local buckets = {}
local v = vec3()

local function addTri(bk, col, n, p1, p2, p3, ambient)
  local d = n[1] * LIGHT[1] + n[2] * LIGHT[2] + n[3] * LIGHT[3]
  if d < 0 then d = -d end                       -- windings are not consistent and both sides are drawn
  local s = math.floor((ambient + (1 - ambient) * d) * STEPS + 0.5)
  local key = col
  local byCol = bk[key]
  if not byCol then byCol = {}; bk[key] = byCol end
  local list = byCol[s]
  if not list then list = {}; byCol[s] = list end
  list[#list + 1] = p1; list[#list + 1] = p2; list[#list + 1] = p3
end

--- Draw the car.
--- pose: {o = vec3 ground point under the car centre, r = right, u = up, f = forward (unit vec3)}
--- steer: front-wheel angle (rad, + = left); spin: wheel rotation (rad)
--- style: {brightness = 1, alpha = 1 (below 1: see-through, e.g. the ghost), tint = rgb or nil}
function M.draw(pose, steer, spin, style)
  local o, R, U, F = pose.o, pose.r, pose.u, pose.f
  local bright, alpha, tint = style.brightness or 1, style.alpha or 1, style.tint
  for k in pairs(buckets) do buckets[k] = nil end
  local function W(x, y, z)                    -- car frame -> world
    return { o.x + R.x * x + U.x * y + F.x * z, o.y + R.y * x + U.y * y + F.y * z, o.z + R.z * x + U.z * y + F.z * z }
  end
  local function N(n)                          -- direction car frame -> world (light is in world space)
    return { R.x * n[1] + U.x * n[2] + F.x * n[3], R.y * n[1] + U.y * n[2] + F.y * n[3], R.z * n[1] + U.z * n[2] + F.z * n[3] }
  end
  for _, t in ipairs(BODY) do
    addTri(buckets, t[5], N(t[4]), W(t[1][1], t[1][2], t[1][3]), W(t[2][1], t[2][2], t[2][3]), W(t[3][1], t[3][2], t[3][3]), 0.45)
  end
  local cs, sn = math.cos(spin), math.sin(spin)
  for _, wh in ipairs(WHEELS) do
    local d = wh.steer and steer or 0
    local cd, sd = math.cos(d), math.sin(d)
    local function P(p)                        -- spin about the axle, steer about the vertical, then place
      local y, z = p[2] * cs - p[3] * sn, p[2] * sn + p[3] * cs
      local x = p[1]
      return x * cd - z * sd + wh.x, y + DIM.tyreR, x * sd + z * cd + wh.z
    end
    for _, t in ipairs(wh.tris) do
      local ax, ay, az = P(t[1]); local bx, by, bz = P(t[2]); local cx, cy, cz = P(t[3])
      local nx, ny, nz = t[4][1], t[4][2] * cs - t[4][3] * sn, t[4][2] * sn + t[4][3] * cs
      addTri(buckets, t[5], N({ nx * cd - nz * sd, ny, nx * sd + nz * cd }), W(ax, ay, az), W(bx, by, bz), W(cx, cy, cz), 0.35)
    end
  end
  for col, byStep in pairs(buckets) do
    for s, list in pairs(byStep) do
      local k = s / STEPS * bright
      local c = tint or col
      render.glSetColor(rgbm(c[1] * k, c[2] * k, c[3] * k, alpha))
      render.glBegin(render.GLPrimitiveType.Triangles)
      for i = 1, #list do
        local p = list[i]
        v:set(p[1], p[2], p[3])
        render.glVertex(v)
      end
      render.glEnd()
    end
  end
end

return M
