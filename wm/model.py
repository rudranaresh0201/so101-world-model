"""Ensemble dynamics model:  next_obs = obs + f(obs, action), with 5 independently-trained MLPs.

Why each piece:
  - predict the DELTA, not next_obs: most of the 41 numbers barely change in 0.05 s, so the delta is a
    small, well-scaled target. Predicting next_obs directly makes the net relearn "copy the input".
  - normalise inputs and delta targets (mean/std from training data): positions are ~0.1 m, velocities
    ~1 rad/s, flags 0/1. Without this the loss is dominated by whichever numbers happen to be big.
  - ENSEMBLE: 5 nets, different init + different bootstrap sample of the data. On states like the
    training data they agree; off it they guess, and guess DIFFERENTLY. Their spread (disagreement)
    is our "the model doesn't know" signal: the lie detector for Milestone 4.
  - all 5 members live in one tensor of shape (E, in, out), so one matmul runs the whole ensemble.
"""
import torch
import torch.nn as nn

OBS, ACT = 41, 6


class EnsembleLinear(nn.Module):
    def __init__(self, E, d_in, d_out):
        super().__init__()
        self.w = nn.Parameter(torch.empty(E, d_in, d_out))
        self.b = nn.Parameter(torch.zeros(E, 1, d_out))
        for e in range(E):
            nn.init.kaiming_uniform_(self.w[e].T, a=5 ** 0.5)   # same init as nn.Linear, per member

    def forward(self, x):            # x: (E, B, d_in) -> (E, B, d_out)
        return torch.baddbmm(self.b, x, self.w)


class EnsembleDynamics(nn.Module):
    def __init__(self, E=5, hidden=256, layers=4):
        super().__init__()
        dims = [OBS + ACT] + [hidden] * layers
        self.net = nn.ModuleList(EnsembleLinear(E, a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = EnsembleLinear(E, hidden, OBS)
        self.E = E
        # normalisation stats, filled by set_stats() and saved with the weights
        for k, n in [("in_mu", OBS + ACT), ("in_sd", OBS + ACT), ("d_mu", OBS), ("d_sd", OBS)]:
            self.register_buffer(k, torch.zeros(n) if "mu" in k else torch.ones(n))

    def set_stats(self, obs, act, next_obs):
        x, d = torch.cat([obs, act], -1), next_obs - obs
        self.in_mu.copy_(x.mean(0)); self.in_sd.copy_(x.std(0).clamp_min(1e-6))
        self.d_mu.copy_(d.mean(0));  self.d_sd.copy_(d.std(0).clamp_min(1e-6))

    def forward_norm(self, obs, act):
        """(E,B,41),(E,B,6) -> normalised delta prediction (E,B,41). Used by the loss."""
        h = (torch.cat([obs, act], -1) - self.in_mu) / self.in_sd
        for layer in self.net:
            h = nn.functional.silu(layer(h))
        return self.head(h)

    @torch.no_grad()
    def predict(self, obs, act):
        """obs (B,41), act (B,6) -> next_obs per member (E,B,41), in real units."""
        o, a = obs.expand(self.E, *obs.shape), act.expand(self.E, *act.shape)
        return obs + self.forward_norm(o, a) * self.d_sd + self.d_mu
