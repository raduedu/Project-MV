"""A rule-based driver that needs no training.

Three reasons it exists:

1. **Yardstick.**  The learned driver has to beat this, otherwise learning adds
   nothing.  (The physics reference in ``mv/planning`` is the upper yardstick.)
2. **Self-test.**  It drives using *only the observation vector*, so if it laps
   a track, the observation, track-following and lap-timing code are right.
3. **First in-game test.**  It can drive the real Assetto Corsa car through the
   bridge before any neural network is involved.

How it drives
-------------
*Steering* (a "Stanley" style path follower on the centre line):
feed-forward from the curvature just ahead (the steering angle a car needs to
follow a circle of that curvature is about ``wheelbase * curvature``), plus
corrections proportional to the heading error and to the distance from the
centre line.

*Speed*: for every preview point work out how fast the corner there can be
taken (``v^2 * curvature = usable lateral acceleration``, including
downforce) and how fast we may be going *now* to still brake down to that
speed in the distance available (``v_now^2 = v_corner^2 + 2 * braking *
distance``), working backwards from the furthest preview point to the car.
The result is the target speed; throttle or brake proportionally to the speed
error.

*Traction*: the throttle is capped by how much grip the car is already using
sideways (measured lateral acceleration), and cut when the rear slip angle
passes the tyre's peak.

It follows the centre line and brakes cautiously, so it is deliberately slower
than a racing driver.
"""

from __future__ import annotations

import math

import torch

from mv.vehicle.params import G

__all__ = ["ScriptedDriver"]


class ScriptedDriver:
    def __init__(self, env, grip_use: float = 0.72, brake_use: float = 0.5, k_heading: float = 1.2,
                 k_lateral: float = 0.35, k_speed: float = 0.25):
        """``env`` is a :class:`mv.envs.racing_env.RacingEnv` (used for its observation layout and car)."""
        vp, cfg = env.vp, env.cfg
        self.layout = env.obs_layout
        self.look_m = torch.tensor(cfg.lookahead_m, device=env.device)
        self.wheelbase = vp.chassis.wheelbase_m
        self.steer_max = math.radians(vp.steering.max_wheel_angle_deg)
        self.mu = env.mu_ref
        # downforce per unit mass per (m/s)^2: extra "gravity" that grows with speed
        cla = vp.aero.cla_front_m2 + vp.aero.cla_rear_m2
        self.k_aero = 0.5 * vp.aero.air_density * cla / vp.chassis.mass_kg
        self.v_cap = env.v_cap
        self.grip_use, self.brake_use = grip_use, brake_use
        self.k_heading, self.k_lateral, self.k_speed = k_heading, k_lateral, k_speed

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = True):
        lay = self.layout
        car, pos = obs[:, lay["car"]], obs[:, lay["position"]]
        kappa = obs[:, lay["kappa"]] / 25.0                       # undo the observation scaling -> 1/m
        seen = obs[:, lay["width"]] > 0                           # preview points beyond the horizon are zeroed
        vx = car[:, 0] * 50.0
        lat = (pos[:, 2] - pos[:, 1]) * 2.5                       # metres left of the centre line
        head_err = torch.atan2(pos[:, 3], pos[:, 4])

        # ---- steering ---------------------------------------------------------
        k_near = kappa[:, 1:4].mean(1)                            # curvature 5-12 m ahead
        steer = (torch.atan(self.wheelbase * k_near)
                 - self.k_heading * head_err
                 - torch.atan(self.k_lateral * lat / (0.5 * vx.clamp(min=0.0) + 2.0)))
        steer_cmd = (steer / self.steer_max).clamp(-1, 1)

        # ---- speed --------------------------------------------------------------
        mu = self.mu * self.grip_use
        # corner speed with downforce:  v^2 * |k| = mu * (g + k_aero * v^2)
        denom = (kappa.abs() - mu * self.k_aero).clamp(min=1e-5)
        v_corner = torch.sqrt(mu * G / denom).clamp(max=self.v_cap)
        v_corner = torch.where(seen, v_corner, torch.full_like(v_corner, 12.0))   # unknown road ahead: assume slow
        # Backward pass over the preview points, far to near: the speed allowed at a
        # point is the lower of its own corner speed and the speed from which we can
        # still brake down to the next point's allowed speed.  Braking capacity shrinks
        # with the share of grip that cornering already uses there (traction circle),
        # which is what makes the plan safe for corners that tighten gradually.
        brake = self.mu * self.brake_use * G
        k_abs = kappa.abs()
        v = v_corner[:, -1]
        for j in range(len(self.look_m) - 2, -1, -1):
            gap = float(self.look_m[j + 1] - self.look_m[j])
            lat_use = (v * v * k_abs[:, j] / (mu * (G + self.k_aero * v * v))).clamp(0.0, 1.0)
            decel = brake * torch.sqrt((1.0 - lat_use ** 2).clamp(min=0.04))
            v = torch.minimum(v_corner[:, j], torch.sqrt(v * v + 2.0 * decel * gap))
        v_target = v.clamp(max=self.v_cap)
        pedal = (self.k_speed * (v_target - vx)).clamp(-1, 1)
        # Traction circle: a tyre that is busy cornering has little grip left for
        # accelerating.  Cap the throttle by the share of grip not used sideways, and
        # lift completely once the rear tyres slide past their peak slip angle
        # (a powerful rear-driven car spins otherwise).
        ay = car[:, 4] * 15.0
        usage = (ay.abs() / (self.mu * (G + self.k_aero * vx * vx))).clamp(0.0, 1.0)
        throttle_cap = 0.1 + 0.9 * (1.0 - usage ** 2)
        rear_slip = car[:, 12].abs()                              # 1.0 = at the peak of the tyre curve
        throttle_cap = throttle_cap * (1.0 - ((rear_slip - 0.6) / 0.4).clamp(0.0, 1.0))
        pedal = torch.where(pedal > 0, torch.minimum(pedal, throttle_cap), pedal)
        # Same for the brakes: braking hard while cornering near the limit overloads the
        # rear tyres and spins the car (trail-braking is for experts).  The speed plan
        # below is cautious enough that most braking is done in a straight line.
        slide = torch.maximum(rear_slip, car[:, 11].abs())
        brake_cap = (0.15 + 0.85 * (1.0 - usage ** 2)) * (1.0 - 0.8 * ((slide - 0.7) / 0.5).clamp(0.0, 1.0))
        pedal = torch.where(pedal < 0, torch.maximum(pedal, -brake_cap), pedal)
        action = torch.stack([steer_cmd, pedal], 1)
        zeros = torch.zeros(obs.shape[0], device=obs.device)
        return action, zeros, zeros
