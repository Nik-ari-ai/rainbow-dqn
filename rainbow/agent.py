"""Rainbow DQN agent (Hessel et al., 2018).

Six components, each can be switched off for ablations / a vanilla DQN baseline:
  double       - Double Q-learning target (online net picks a', target net evaluates it)
  per          - prioritized replay (alpha > 0) with importance-sampling weights
  dueling      - value / advantage streams
  n_step       - multi-step returns
  distributional - C51 return distribution with KL (cross-entropy) loss
  noisy        - NoisyNet exploration instead of epsilon-greedy
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import RainbowNet


@dataclass
class AgentConfig:
    arch: str = "canonical"
    hidden: int = 512
    atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    noisy_std: float = 0.5
    gamma: float = 0.99
    n_step: int = 3
    lr: float = 6.25e-5
    adam_eps: float = 1.5e-4
    batch_size: int = 32
    grad_clip: float = 10.0
    # components
    double: bool = True
    dueling: bool = True
    distributional: bool = True
    noisy: bool = True
    per: bool = True
    # epsilon-greedy (only used when noisy=False)
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 250_000
    eval_eps: float = 0.001

    def to_dict(self):
        return asdict(self)


def project_distribution(next_probs, returns, nonterminal, gamma_n, support, v_min, v_max):
    """C51 projection of r + gamma^n * z onto the fixed support (Bellemare et al., 2017).

    next_probs: (B, atoms) target distribution for the chosen next action.
    returns, nonterminal: (B,). Returns m: (B, atoms) projected target distribution.
    """
    B, atoms = next_probs.shape
    dz = (v_max - v_min) / (atoms - 1)
    tz = returns.unsqueeze(1) + nonterminal.unsqueeze(1) * gamma_n * support.unsqueeze(0)
    tz = tz.clamp(v_min, v_max)
    b = (tz - v_min) / dz                       # fractional atom index in [0, atoms-1]
    l, u = b.floor().long(), b.ceil().long()
    # when b is an exact integer, l == u and the mass would vanish -> shift one side
    l[(u > 0) & (l == u)] -= 1
    u[(l < atoms - 1) & (l == u)] += 1
    m = torch.zeros(B, atoms, device=next_probs.device, dtype=next_probs.dtype)
    offset = (torch.arange(B, device=next_probs.device) * atoms).unsqueeze(1)
    m.view(-1).index_add_(0, (l + offset).view(-1), (next_probs * (u.float() - b)).view(-1))
    m.view(-1).index_add_(0, (u + offset).view(-1), (next_probs * (b - l.float())).view(-1))
    return m


class RainbowAgent:
    def __init__(self, obs_shape, n_actions: int, cfg: AgentConfig | None = None,
                 device: str | torch.device = "cpu", pixel_input: bool = True):
        self.cfg = cfg = cfg or AgentConfig()
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.obs_shape = tuple(obs_shape)
        self.pixel_input = pixel_input
        kw = dict(atoms=cfg.atoms, hidden=cfg.hidden, arch=cfg.arch, noisy=cfg.noisy,
                  noisy_std=cfg.noisy_std, dueling=cfg.dueling, distributional=cfg.distributional,
                  pixel_input=pixel_input)
        self.online = RainbowNet(self.obs_shape, n_actions, **kw).to(self.device)
        self.target = RainbowNet(self.obs_shape, n_actions, **kw).to(self.device)
        self.update_target()
        self.target.train()  # keep noise on in the target net (as in the paper)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optim = torch.optim.Adam(self.online.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
        self.support = torch.linspace(cfg.v_min, cfg.v_max, cfg.atoms, device=self.device)
        self.steps = 0  # environment steps, used for epsilon decay
        self.rng = np.random.default_rng()

    # -------------------------------------------------------------- acting
    def _tensor(self, obs):
        if isinstance(obs, torch.Tensor):
            t = obs.to(self.device)
        else:
            t = torch.as_tensor(np.asarray(obs), device=self.device)
        return t if self.pixel_input else t.float()

    @torch.no_grad()
    def q_values(self, obs_batch, net=None):
        net = self.online if net is None else net
        out = net(self._tensor(obs_batch))
        return (out * self.support).sum(-1) if self.cfg.distributional else out

    def epsilon(self):
        if self.cfg.noisy:
            return 0.0
        c = self.cfg
        frac = min(1.0, self.steps / max(1, c.eps_decay_steps))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def act(self, obs, greedy: bool = False) -> int:
        """Training action (noise / epsilon exploration) or greedy evaluation action."""
        eps = self.cfg.eval_eps if greedy else self.epsilon()
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return int(self.q_values(np.asarray(obs)[None]).argmax(1).item())

    def act_batch(self, obs_batch, greedy: bool = False):
        """Vectorised version of ``act`` for several environments at once."""
        q = self.q_values(np.asarray(obs_batch)).argmax(1).cpu().numpy()
        eps = self.cfg.eval_eps if greedy else self.epsilon()
        rand = self.rng.random(len(q)) < eps
        q[rand] = self.rng.integers(self.n_actions, size=rand.sum())
        return q

    def reset_noise(self):
        if self.cfg.noisy:
            self.online.reset_noise()

    # ------------------------------------------------------------ learning
    def learn(self, memory, beta: float):
        c = self.cfg
        batch = memory.sample(c.batch_size, beta)
        obs = self._tensor(batch["obs"])
        next_obs = self._tensor(batch["next_obs"])
        actions = torch.as_tensor(batch["actions"], device=self.device)
        returns = torch.as_tensor(batch["returns"], device=self.device)
        nonterminal = torch.as_tensor(batch["nonterminal"], device=self.device)
        weights = torch.as_tensor(batch["weights"], device=self.device)
        rows = torch.arange(c.batch_size, device=self.device)
        gamma_n = c.gamma ** c.n_step

        with torch.no_grad():
            self.target.reset_noise() if c.noisy else None
            if c.double:   # online network chooses a'
                next_a = self.q_values(next_obs, self.online).argmax(1)
            else:          # target network chooses a'
                next_a = self.q_values(next_obs, self.target).argmax(1)
            target_out = self.target(next_obs)

        if c.distributional:
            log_p = self.online(obs, log=True)[rows, actions]                       # (B, atoms)
            with torch.no_grad():
                m = project_distribution(target_out[rows, next_a], returns, nonterminal,
                                         gamma_n, self.support, c.v_min, c.v_max)
            loss = -(m * log_p).sum(1)                                               # cross-entropy
            with torch.no_grad():
                q_taken = (log_p.exp() * self.support).sum(1)
        else:
            q = self.online(obs)[rows, actions]
            with torch.no_grad():
                y = returns + nonterminal * gamma_n * target_out[rows, next_a]
            loss = F.smooth_l1_loss(q, y, reduction="none")
            q_taken = q.detach()

        w = weights if c.per else torch.ones_like(weights)
        self.optim.zero_grad(set_to_none=True)
        (w * loss).mean().backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), c.grad_clip)
        self.optim.step()
        if c.per:
            memory.update_priorities(batch["idx"], loss.detach().abs().cpu().numpy())
        return {"loss": float(loss.mean().item()), "q": float(q_taken.mean().item())}

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def train(self):
        self.online.train()

    def eval(self):
        self.online.eval()

    # ------------------------------------------------------------ saving
    def state_dict(self):
        return {"online": self.online.state_dict(), "target": self.target.state_dict(),
                "optim": self.optim.state_dict(), "steps": self.steps,
                "agent_cfg": self.cfg.to_dict(), "obs_shape": self.obs_shape,
                "n_actions": self.n_actions, "pixel_input": self.pixel_input}

    def load_state_dict(self, d, load_optim: bool = True):
        self.online.load_state_dict(d["online"])
        self.target.load_state_dict(d.get("target", d["online"]))
        if load_optim and "optim" in d:
            self.optim.load_state_dict(d["optim"])
        self.steps = d.get("steps", 0)

    @classmethod
    def from_file(cls, path, device="cpu"):
        d = torch.load(path, map_location=device, weights_only=False)
        agent = cls(d["obs_shape"], d["n_actions"], AgentConfig(**d["agent_cfg"]), device, d["pixel_input"])
        agent.load_state_dict(d, load_optim=False)
        agent.eval()
        return agent, d
