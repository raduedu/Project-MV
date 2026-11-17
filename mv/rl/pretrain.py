"""Warm start: teach the network to imitate the rule-based driver before PPO.

Why
---
A freshly initialised network outputs noise.  PPO then spends its first tens of
millions of decisions discovering the basics: steer along the road, and - the
hard one - brake *long before* a slow corner.  Braking is hard to discover by
trial and error because the punishment (leaving the road) comes seconds after
the mistake (not braking).

The rule-based driver in ``mv/control/scripted.py`` already knows those basics.
Copying it first ("behaviour cloning") gives PPO a driver that completes laps
from update one; PPO's job is then only to make it *fast*, which is what
reinforcement learning is good at.  The teacher is slow and follows the centre
line, but nothing ties the student to that style: once PPO runs, every change
that gains progress is reinforced.

How (DAgger, Ross et al. 2011)
------------------------------
Plain imitation only sees the states the teacher visits, so the student is
lost as soon as it drifts off the teacher's path.  DAgger fixes that: let the
*student* drive part of the time (with noise), but always record what the
*teacher would have done* in each state, and train on that.  The student thus
learns how to recover from its own mistakes.

After cloning, the critic (the network that predicts future reward) still
knows nothing, so a few critic-only PPO updates follow before normal training.
"""

from __future__ import annotations

import math

import torch

__all__ = ["warm_start"]


def warm_start(ppo, teacher, steps: int = 600, noise: float = 0.25, lr: float = 1e-3, critic_updates: int = 15,
               start_std: float = 0.25, printer=print) -> dict:
    """Clone ``teacher`` into ``ppo.net`` using the PPO's own environment.

    ``steps`` environment steps are taken (each yields ``env.n`` labelled
    samples).  Returns a small dict of diagnostics.
    """
    env, net = ppo.env, ppo.net
    opt = torch.optim.Adam(net.actor.parameters(), lr=lr)
    obs = env.observe()
    loss_avg, crashes, episodes = None, 0, 0
    for i in range(steps):
        net.obs_norm.update(obs)
        with torch.no_grad():
            target, _, _ = teacher.act(obs)
            # who drives this step?  teacher early on, student later (per car, at random)
            share = min(1.0, i / (0.5 * steps))
            student_drives = torch.rand(env.n, device=env.device) < share
            drive = torch.where(student_drives[:, None], net.actor(net.obs_norm(obs)), target)
            drive = (drive + noise * torch.randn_like(drive)).clamp(-1, 1)
        pred = net.actor(net.obs_norm(obs))
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_avg = float(loss.detach()) if loss_avg is None else 0.98 * loss_avg + 0.02 * float(loss.detach())
        obs, _, term, trunc, _ = env.step(drive)
        crashes += int(term.sum()); episodes += int((term | trunc).sum())
        if (i + 1) % 100 == 0:
            printer(f"  warm start {i + 1}/{steps}: imitation error {loss_avg:.4f}, "
                    f"student driving {share * 100:.0f} % of cars")
    with torch.no_grad():
        net.log_std.fill_(math.log(start_std))
    ppo.obs = env.observe()
    for _ in range(critic_updates):                 # teach the critic what this driver's future looks like
        ppo.collect()
        ppo.update(train_policy=False)
    ppo.total_steps += steps * env.n
    return {"imitation_error": loss_avg, "crashes": crashes, "episodes": episodes}
