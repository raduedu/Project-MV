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
               start_std: float = 0.25, printer=print, buffer_steps: int = 0, grad_steps: int = 1,
               batch: int = 16384) -> dict:
    """Clone ``teacher`` into ``ppo.net`` using the PPO's own environment.

    ``steps`` environment steps are taken (each yields ``env.n`` labelled
    samples).  Returns a small dict of diagnostics.

    Two modes:

    * ``buffer_steps=0`` (default, for the rule-based teacher): one gradient
      step per environment step on the newest states.
    * ``buffer_steps>0`` (*distillation*, for copying a trained network into a
      bigger one, log WP30): the newest ``buffer_steps`` steps of states are
      kept in memory and ``grad_steps`` random mini-batches of ``batch``
      samples are learned per environment step, with the learning rate
      lowered smoothly to 5 % by the end.  A big network needs many more
      passes than one per step to copy a driver closely; a rough copy crashes
      and PPO then has to relearn what the teacher already knew.

    If the teacher carries an input scaler (``obs_norm``) of the same size it
    is copied and frozen, so student and teacher see identical numbers.

    If the teacher also has a ``value(obs)`` method (a trained driver's critic),
    the student's critic copies it as well, in the same mini-batches (log WP44).
    Why: without it the student's critic starts from nothing and gets only a few
    million decisions of PPO practice, while the actor is already an expert. A
    critic that has not learnt that "this speed into this corner ends in a crash"
    rewards going faster - both distilled networks (WP30 big, WP44 medium) then
    sped up and crashed on almost every episode (crashes 10 -> 230 per 100 km)
    although their update sizes were normal. (The caller must also copy the
    teacher's return scaling, ``ret_norm``, so the copied values mean the same.)
    """
    env, net = ppo.env, ppo.net
    tnorm = getattr(teacher, "obs_norm", None)
    copy_norm = tnorm is not None and tuple(tnorm.mean.shape) == tuple(net.obs_norm.mean.shape)
    if copy_norm:
        net.obs_norm.load_state_dict(tnorm.state_dict())
    copy_value = hasattr(teacher, "value")
    params = list(net.actor.parameters()) + (list(net.critic.parameters()) if copy_value else [])
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, k / max(1, steps * grad_steps)))))
    cap = buffer_steps * env.n
    if cap:
        buf_o = torch.empty((cap, env.num_obs), device=env.device)
        buf_t = torch.empty((cap, env.num_actions), device=env.device)
        buf_v = torch.empty((cap,), device=env.device) if copy_value else None
        filled, head = 0, 0
    obs = env.observe()
    loss_avg, vloss_avg, crashes, episodes = None, None, 0, 0
    for i in range(steps):
        if not copy_norm:
            net.obs_norm.update(obs)
        with torch.no_grad():
            target, _, _ = teacher.act(obs)
            v_target = teacher.value(obs) if copy_value else None
            # who drives this step?  teacher early on, student later (per car, at random)
            share = min(1.0, i / (0.5 * steps))
            student_drives = torch.rand(env.n, device=env.device) < share
            drive = torch.where(student_drives[:, None], net.actor(net.obs_norm(obs)), target)
            drive = (drive + noise * torch.randn_like(drive)).clamp(-1, 1)
        if cap:
            n = obs.shape[0]
            idx = (torch.arange(n, device=env.device) + head) % cap
            buf_o[idx], buf_t[idx] = obs, target
            if copy_value:
                buf_v[idx] = v_target
            head, filled = (head + n) % cap, min(cap, filled + n)
            for _ in range(grad_steps):
                j = torch.randint(0, filled, (min(batch, filled),), device=env.device)
                o = net.obs_norm(buf_o[j])
                loss = ((net.actor(o) - buf_t[j]) ** 2).mean()
                total = loss
                if copy_value:
                    vloss = ((net.critic(o).squeeze(-1) - buf_v[j]) ** 2).mean()
                    total = loss + vloss
                    vloss_avg = float(vloss.detach()) if vloss_avg is None else 0.98 * vloss_avg + 0.02 * float(vloss.detach())
                opt.zero_grad(set_to_none=True)
                total.backward()
                opt.step()
                sched.step()
        else:
            pred = net.actor(net.obs_norm(obs))
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        loss_avg = float(loss.detach()) if loss_avg is None else 0.98 * loss_avg + 0.02 * float(loss.detach())
        obs, _, term, trunc, _ = env.step(drive)
        crashes += int(term.sum()); episodes += int((term | trunc).sum())
        if (i + 1) % 100 == 0:
            printer(f"  warm start {i + 1}/{steps}: imitation error {loss_avg:.5f}"
                    + (f", critic copy error {vloss_avg:.5f}" if vloss_avg is not None else "")
                    + f", student driving {share * 100:.0f} % of cars, crashes so far {crashes}")
    with torch.no_grad():
        net.log_std.fill_(math.log(start_std))
    ppo.obs = env.observe()
    for _ in range(critic_updates):                 # teach the critic what this driver's future looks like
        ppo.collect()
        ppo.update(train_policy=False)
    ppo.total_steps += steps * env.n
    return {"imitation_error": loss_avg, "value_error": vloss_avg, "crashes": crashes, "episodes": episodes}
