"""Stochastic gates for sparse reaction-coordinate input selection."""

import math

import torch
import torch.nn as nn


class StochasticGates(nn.Module):
    """Learn one stochastic hard-sigmoid gate for each input descriptor."""

    def __init__(self, n_in, sigma=0.5, init_mu=1.0, mu_min=None):
        super().__init__()
        if n_in <= 0:
            raise ValueError("n_in must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.n_in = int(n_in)
        self.n_out = int(n_in)
        self.sigma = float(sigma)
        # mu is hard-clamped to [mu_min, mu_max] after every optimizer step.
        # mu_max = 1.0 keeps a fully-open gate off the erf-saturation plateau
        # (d prob/d mu stays ~0.2 at mu=1, sigma=0.5), so the sparsity penalty
        # can always pull a gate back down. mu_min = -3 sigma is "effectively
        # off" (prob ~1e-3) while still leaving a small gradient.
        self.mu_max = 1.0
        self.mu_min = float(mu_min) if mu_min is not None else -3.0 * self.sigma
        if self.mu_min >= self.mu_max:
            raise ValueError("mu_min must be below mu_max (1.0)")
        self.init_mu = float(min(max(init_mu, self.mu_min), self.mu_max))
        self.mu = nn.Parameter(torch.full((self.n_in,), self.init_mu))
        self.bypass = False
        self.call_kwargs = {
            "n_in": self.n_in,
            "sigma": self.sigma,
            "init_mu": self.init_mu,
            "mu_min": self.mu_min,
        }

    def forward(self, descriptors):
        if descriptors.shape[-1] != self.n_in:
            raise ValueError(
                f"Expected {self.n_in} descriptors, got {descriptors.shape[-1]}"
            )
        if self.bypass:
            return descriptors
        if self.training:
            gates = self.mu + torch.randn_like(self.mu) * self.sigma
        else:
            gates = self.mu
        return descriptors * gates.clamp(0.0, 1.0)

    def expected_gate_probabilities(self):
        """Return each feature's probability of having a nonzero gate."""
        return 0.5 * (1.0 + torch.erf(self.mu / (math.sqrt(2.0) * self.sigma)))

    def expected_active_features(self):
        """Return the expected number of active descriptor gates."""
        return self.expected_gate_probabilities().sum()

    def gate_probability_gradient_magnitude(self):
        """Return |d(gate_prob)/d(mu)| per feature, used to detect erf saturation."""
        x = self.mu / (math.sqrt(2.0) * self.sigma)
        return (2.0 / math.sqrt(math.pi)) * torch.exp(-x.square()) / (math.sqrt(2.0) * self.sigma)

    def deterministic_gate_values(self):
        """Return inference-time gate values."""
        return self.mu.clamp(0.0, 1.0)

    def clamp_mu(self):
        """Project mu back into [mu_min, mu_max]. Call after every optimizer step."""
        with torch.no_grad():
            self.mu.clamp_(self.mu_min, self.mu_max)

    def reset_parameters(self):
        with torch.no_grad():
            self.mu.fill_(self.init_mu)