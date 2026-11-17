"""Proximal Policy Optimisation (PPO), written from scratch for Project MV.

Why our own implementation: it runs entirely on the GPU next to the simulator
(no copying between devices, which is what makes >50 000 decisions per second
possible), it has no dependency we do not control, and it is short enough to
read in one sitting.

The idea of PPO in four sentences
---------------------------------
1. Let the current driver ("policy") drive all cars for a short while and
   record what it saw, what it did and what reward it got (a "rollout").
2. For every recorded decision estimate the *advantage*: did things go better
   or worse than the driver's own expectation (the "value" network)?
3. Nudge the policy so that decisions with positive advantage become more
   likely and those with negative advantage less likely...
4. ...but never let one update change the policy by much ("proximal" - the
   clip range), because a driver that changes too abruptly forgets how to drive.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

__all__ = ["PPOConfig", "ActorCritic", "RunningNorm", "PPO", "load_policy"]


@dataclass
class PPOConfig:
    rollout_steps: int = 32            # decisions per car between updates
    epochs: int = 5                    # passes over each rollout
    minibatches: int = 4
    gamma: float = 0.995               # discount: how far ahead rewards count (0.995 at 25 Hz ~ 8 s)
    lam: float = 0.95                  # GAE smoothing of the advantage estimate
    clip: float = 0.2                  # the "proximal" limit on policy change
    lr: float = 3e-4
    target_kl: float = 0.012           # learning rate is steered to keep each update about this big
    entropy_coef: float = 0.002        # small bonus for keeping some randomness (exploration)
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    hidden: tuple[int, ...] = (512, 256, 128)
    init_std: float = 0.6              # initial exploration noise on steering and pedal
    min_std: float = 0.03


class RunningNorm(nn.Module):
    """Keeps a running mean/variance and rescales inputs to roughly unit scale."""

    def __init__(self, size: int, clip: float = 6.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        self.var.copy_((self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot) / tot)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


def _mlp(sizes: list[int], out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        last = i == len(sizes) - 2
        nn.init.orthogonal_(lin.weight, gain=out_gain if last else 2 ** 0.5)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if not last:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Two networks: the *actor* picks actions, the *critic* predicts future reward."""

    def __init__(self, num_obs: int, num_actions: int, cfg: PPOConfig):
        super().__init__()
        self.obs_norm = RunningNorm(num_obs)
        self.actor = _mlp([num_obs, *cfg.hidden, num_actions], out_gain=0.01)
        self.critic = _mlp([num_obs, *cfg.hidden, 1], out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((num_actions,), float(torch.tensor(cfg.init_std).log())))
        self.min_log_std = float(torch.tensor(cfg.min_std).log())

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(self.obs_norm(obs))
        std = self.log_std.clamp(min=self.min_log_std).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.obs_norm(obs)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.dist(obs)
        action = d.mean if deterministic else d.sample()
        return action, d.log_prob(action).sum(-1), self.value(obs)


class PPO:
    def __init__(self, env, cfg: PPOConfig | None = None):
        self.env, self.cfg = env, cfg or PPOConfig()
        cfg, dev = self.cfg, env.device
        self.net = ActorCritic(env.num_obs, env.num_actions, cfg).to(dev)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.lr = cfg.lr
        t, n = cfg.rollout_steps, env.n
        self.obs_buf = torch.zeros(t, n, env.num_obs, device=dev)
        self.act_buf = torch.zeros(t, n, env.num_actions, device=dev)
        self.logp_buf = torch.zeros(t, n, device=dev)
        self.rew_buf = torch.zeros(t, n, device=dev)
        self.done_buf = torch.zeros(t, n, device=dev)
        self.val_buf = torch.zeros(t, n, device=dev)
        self.ret_norm = RunningNorm(1).to(dev)       # scale of returns, so the critic sees ~unit targets
        self.obs = env.observe()
        self.total_steps = 0

    # ------------------------------------------------------------------------
    def collect(self) -> dict:
        """Drive ``rollout_steps`` decisions with the current policy and record everything."""
        cfg, env, net = self.cfg, self.env, self.net
        stats = {"ep_return": [], "ep_len": [], "ep_dist": [], "crashes": 0, "episodes": 0, "speed": 0.0,
                 "progress_m": 0.0, "reward": 0.0}
        net.obs_norm.update(self.obs)
        for t in range(cfg.rollout_steps):
            action, logp, value = net.act(self.obs)
            obs, rew, term, trunc, info = env.step(action)
            if "final_obs" in info:
                ids = info["done_ids"]
                # An episode cut off by the clock is not a failure: add the value of
                # the state it was cut at, as if it had carried on.
                cut = trunc[ids]
                if cut.any():
                    rew[ids[cut]] += cfg.gamma * net.value(info["final_obs"][cut]).detach() * self._ret_scale()
                stats["ep_return"].append(info["ep_return"]); stats["ep_len"].append(info["ep_len"])
                stats["ep_dist"].append(info["ep_dist"])
                stats["crashes"] += int(term.sum()); stats["episodes"] += len(ids)
            self.obs_buf[t], self.act_buf[t], self.logp_buf[t] = self.obs, action, logp
            self.rew_buf[t], self.done_buf[t], self.val_buf[t] = rew, (term | trunc).float(), value
            self.obs = obs
            stats["speed"] += float(info["speed"].mean()) / cfg.rollout_steps
            stats["progress_m"] += float(torch.nan_to_num(info["progress"]).sum())
            stats["reward"] += float(rew.mean()) / cfg.rollout_steps
            net.obs_norm.update(obs)
        self.total_steps += cfg.rollout_steps * env.n
        return stats

    def _ret_scale(self) -> torch.Tensor:
        return torch.sqrt(self.ret_norm.var[0] + 1e-8)

    def update(self, train_policy: bool = True) -> dict:
        """One PPO update from the recorded rollout.

        ``train_policy=False`` trains only the critic (used right after a warm
        start, when the policy is already sensible but the critic knows nothing:
        letting an ignorant critic steer the policy would wreck it).
        """
        cfg, net = self.cfg, self.net
        scale = self._ret_scale()
        with torch.no_grad():
            # Generalised Advantage Estimation, computed backwards through time.
            values = self.val_buf * scale                      # critic works in normalised units
            last = net.value(self.obs) * scale
            adv = torch.zeros_like(self.rew_buf)
            gae = torch.zeros(self.env.n, device=self.env.device)
            for t in reversed(range(cfg.rollout_steps)):
                nxt = last if t == cfg.rollout_steps - 1 else values[t + 1]
                alive = 1.0 - self.done_buf[t]
                delta = self.rew_buf[t] + cfg.gamma * nxt * alive - values[t]
                gae = delta + cfg.gamma * cfg.lam * alive * gae
                adv[t] = gae
            returns = adv + values
            self.ret_norm.update(returns.reshape(-1, 1))
            scale = self._ret_scale()
            ret_n = (returns / scale).reshape(-1)
            adv = adv.reshape(-1)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs = self.obs_buf.reshape(-1, self.obs_buf.shape[-1])
        act = self.act_buf.reshape(-1, self.act_buf.shape[-1])
        logp_old = self.logp_buf.reshape(-1)
        total = obs.shape[0]
        mb = total // cfg.minibatches
        out = {"kl": 0.0, "value_loss": 0.0, "policy_loss": 0.0, "clip_frac": 0.0}
        updates = 0
        for _epoch in range(cfg.epochs):
            perm = torch.randperm(total, device=obs.device)
            for i in range(cfg.minibatches):
                j = perm[i * mb:(i + 1) * mb]
                d = net.dist(obs[j])
                logp = d.log_prob(act[j]).sum(-1)
                ratio = (logp - logp_old[j]).exp()
                a = adv[j]
                policy_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                value_loss = 0.5 * (net.value(obs[j]) - ret_n[j]).pow(2).mean()
                entropy = d.entropy().sum(-1).mean()
                if train_policy:
                    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                else:
                    loss = cfg.value_coef * value_loss
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kl = (logp_old[j] - logp).mean().clamp(min=0.0)
                    out["kl"] += float(kl); out["value_loss"] += float(value_loss)
                    out["policy_loss"] += float(policy_loss)
                    out["clip_frac"] += float(((ratio - 1).abs() > cfg.clip).float().mean())
                updates += 1
                # steer the learning rate so each update changes the policy by about target_kl
                if not train_policy:
                    continue
                if kl > 2.0 * cfg.target_kl:
                    self.lr = max(self.lr / 1.5, 1e-5)
                elif kl < 0.5 * cfg.target_kl:
                    self.lr = min(self.lr * 1.5, 1e-3)
                for g in self.opt.param_groups:
                    g["lr"] = self.lr
        for k in out:
            out[k] /= max(updates, 1)
        out["lr"] = self.lr
        out["std"] = float(net.log_std.detach().exp().mean())
        return out

    # ------------------------------------------------------------------------
    def save(self, path, extra: dict | None = None) -> None:
        torch.save({"model": self.net.state_dict(), "ret_norm": self.ret_norm.state_dict(),
                    "cfg": asdict(self.cfg), "num_obs": self.env.num_obs, "num_actions": self.env.num_actions,
                    "total_steps": self.total_steps, "lr": self.lr,
                    # optimiser momentum: without it a resumed run takes one oversized first step
                    "opt": self.opt.state_dict(), **(extra or {})}, path)

    def load(self, path) -> dict:
        ck = torch.load(path, map_location=self.env.device, weights_only=False)
        self.net.load_state_dict(ck["model"])
        self.ret_norm.load_state_dict(ck["ret_norm"])
        self.total_steps, self.lr = ck.get("total_steps", 0), ck.get("lr", self.cfg.lr)
        if "opt" in ck:
            self.opt.load_state_dict(ck["opt"])
        for g in self.opt.param_groups:
            g["lr"] = self.lr
        return ck


def load_policy(path, device="cpu") -> ActorCritic:
    """Load a trained driver from a checkpoint for driving only (no training state)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = PPOConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()})
    net = ActorCritic(ck["num_obs"], ck["num_actions"], cfg).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net
