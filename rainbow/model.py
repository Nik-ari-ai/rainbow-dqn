"""Rainbow network: (noisy) dueling head on top of a CNN or MLP torso,
outputting either a return distribution over atoms (C51) or plain Q-values."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet layer (Fortunato et al., 2018).

    In ``eval()`` mode the noise is switched off and the mean weights are used.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features, self.out_features, self.std_init = in_features, out_features, std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int, device):
        x = torch.randn(size, device=device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        dev = self.weight_mu.device
        eps_in = self._scale_noise(self.in_features, dev)
        eps_out = self._scale_noise(self.out_features, dev)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return F.linear(x, w, b)
        return F.linear(x, self.weight_mu, self.bias_mu)


def _torso(arch: str, in_shape):
    if arch == "canonical":   # Nature DQN CNN (Mnih et al., 2015)
        net = nn.Sequential(nn.Conv2d(in_shape[0], 32, 8, 4), nn.ReLU(),
                            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                            nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())
    elif arch == "data-efficient":  # smaller CNN from van Hasselt et al., 2019
        net = nn.Sequential(nn.Conv2d(in_shape[0], 32, 5, 5), nn.ReLU(),
                            nn.Conv2d(32, 64, 5, 5), nn.ReLU(), nn.Flatten())
    elif arch == "mlp":       # for the compact state vector
        net = nn.Sequential(nn.Flatten(), nn.Linear(int(math.prod(in_shape)), 256), nn.ReLU(),
                            nn.Linear(256, 256), nn.ReLU())
    else:
        raise ValueError(arch)
    with torch.no_grad():
        out = net(torch.zeros(1, *in_shape)).shape[1]
    return net, out


class RainbowNet(nn.Module):
    def __init__(self, in_shape, n_actions: int, atoms: int = 51, hidden: int = 512,
                 arch: str = "canonical", noisy: bool = True, noisy_std: float = 0.5,
                 dueling: bool = True, distributional: bool = True, pixel_input: bool = True):
        super().__init__()
        self.n_actions = n_actions
        self.atoms = atoms if distributional else 1
        self.distributional = distributional
        self.dueling = dueling
        self.pixel_input = pixel_input
        self.torso, feat = _torso(arch, tuple(in_shape))
        Lin = (lambda i, o: NoisyLinear(i, o, noisy_std)) if noisy else nn.Linear
        self.adv = nn.Sequential(Lin(feat, hidden), nn.ReLU(), Lin(hidden, n_actions * self.atoms))
        if dueling:
            self.val = nn.Sequential(Lin(feat, hidden), nn.ReLU(), Lin(hidden, self.atoms))

    def forward(self, x, log: bool = False):
        """Returns (B, A, atoms) probabilities (or log-probs) if distributional, else (B, A) Q-values."""
        if self.pixel_input:
            x = x.float() / 255.0
        h = self.torso(x)
        a = self.adv(h).view(-1, self.n_actions, self.atoms)
        if self.dueling:
            v = self.val(h).view(-1, 1, self.atoms)
            q = v + a - a.mean(1, keepdim=True)
        else:
            q = a
        if not self.distributional:
            return q.squeeze(-1)
        return F.log_softmax(q, dim=2) if log else F.softmax(q, dim=2)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
