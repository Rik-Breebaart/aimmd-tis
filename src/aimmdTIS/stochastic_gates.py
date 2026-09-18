"""Stochastic gates for sparse reaction-coordinate input selection."""

import math

import torch
import torch.nn as nn


class StochasticGates(nn.Module):
    """Learn one stochastic hard-sigmoid gate for each input descriptor."""

    def __init__(self, n_in, sigma=0.5, init_mu=1.0, mu_min=None, clamp_mu=True):
        super().__init__()
        if n_in <= 0:
            raise ValueError("n_in must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.n_in = int(n_in)
        self.n_out = int(n_in)
        self.sigma = float(sigma)
        self.clamp_mu_enabled = bool(clamp_mu)
        # mu is hard-clamped to [mu_min, mu_max] after every optimizer step,
        # when clamp_mu_enabled (the default). Both bounds scale with sigma
        # and sit symmetrically 3 sigma outside the "core" [0,1] gate range:
        # mu_min = -3*sigma is "effectively off" (prob ~1e-3), mu_max =
        # 1+3*sigma is "effectively fully open" (prob ~1-1e-3) -- keeping a
        # fully-open gate off the erf-saturation plateau regardless of sigma
        # (a fixed mu_max=1.0 put mu=1 deep in saturation at small sigma,
        # e.g. exp(-50) at sigma=0.1 -- see stg_pilot_v10), so the sparsity
        # penalty can always pull a gate back down no matter how open it
        # currently sits. The runtime gate value used in forward() is always
        # separately clamped to [0,1] regardless of mu_max, so mu sitting
        # above 1 still means "fully open," just with headroom before
        # hitting the gradient-vanishing boundary. Set clamp_mu=False to
        # test whether the (now-normalized) sparsity penalty keeps mu
        # bounded on its own, without the hard box.
        self.mu_min = float(mu_min) if mu_min is not None else -3.0 * self.sigma
        self.mu_max = 1.0 + 3.0 * self.sigma
        if self.clamp_mu_enabled and self.mu_min >= self.mu_max:
            raise ValueError(f"mu_min ({self.mu_min}) must be below mu_max ({self.mu_max})")
        if self.clamp_mu_enabled:
            self.init_mu = float(min(max(init_mu, self.mu_min), self.mu_max))
        else:
            self.init_mu = float(init_mu)
        self.mu = nn.Parameter(torch.full((self.n_in,), self.init_mu))
        self.bypass = False
        # When True, forward uses the deterministic gate (clamp(mu)) even in
        # training mode -- for the clean fine-tune phase after selection.
        self.frozen = False
        self.call_kwargs = {
            "n_in": self.n_in,
            "sigma": self.sigma,
            "init_mu": self.init_mu,
            "mu_min": self.mu_min,
            "clamp_mu": self.clamp_mu_enabled,
        }

    def forward(self, descriptors):
        if descriptors.shape[-1] != self.n_in:
            raise ValueError(
                f"Expected {self.n_in} descriptors, got {descriptors.shape[-1]}"
            )
        if getattr(self, "bypass", False):
            return descriptors
        if self.training and not getattr(self, "frozen", False):
            # One independent noise draw per (frame, feature), matching
            # descriptors' shape -- NOT torch.randn_like(self.mu), which has
            # shape (n_in,) and broadcasts the *same* noise realization onto
            # every frame in the batch. Literal STG (Yamada et al. 2020)
            # samples z_d = mu_d + eps_d per data point; sharing eps_d across
            # a whole batch turns the intended per-sample stochastic
            # regularizer into one shared random offset per batch, which
            # also inflates gradient variance between batches (no
            # within-batch Monte-Carlo averaging over independent noise
            # draws) compared to proper per-frame sampling.
            noise = torch.randn(
                descriptors.shape[0], self.n_in,
                device=self.mu.device, dtype=self.mu.dtype,
            ) * self.sigma
            gates = self.mu + noise
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
        """Project mu back into [mu_min, mu_max]. Call after every optimizer
        step. No-op when clamp_mu_enabled is False."""
        if not self.clamp_mu_enabled:
            return
        with torch.no_grad():
            self.mu.clamp_(self.mu_min, self.mu_max)

    def reset_parameters(self):
        with torch.no_grad():
            self.mu.fill_(self.init_mu)