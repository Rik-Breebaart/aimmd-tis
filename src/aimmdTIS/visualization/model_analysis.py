"""Model-dependent along-q analysis methods.

All methods in :class:`ModelAnalysisMixin` require

* ``self._model_output_rpe(model)`` — inherited from
  :class:`~aimmdTIS.visualization.base.BaseVisualizer`; returns the cached
  ``(p_B, q)`` pair so the forward pass runs **at most once per model per
  session**.
* ``self._w``, ``self._shot`` — set by
  :meth:`~aimmdTIS.visualization.base.BaseVisualizer.load_trainset`.

Intended usage: mix into :class:`~aimmdTIS.visualization.general.SystemVisualizer`
*after* :class:`~aimmdTIS.visualization.base.BaseVisualizer`::

    class SystemVisualizer(BaseVisualizer, ModelAnalysisMixin):
        ...
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from ..training import snapshot_loss_original


class ModelAnalysisMixin:
    """Along-q model diagnostic plots.

    Mix-in: does **not** define ``__init__``; relies on attributes provided by
    :class:`~aimmdTIS.visualization.base.BaseVisualizer`.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _q_and_weights(self, model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(q, weights, shot_results)`` ready for histogram/loss calls."""
        if self._desc is None:
            raise ValueError("load_trainset() must be called before model analysis.")
        _, q = self._model_output_rpe(model)
        return q, self._w, self._shot

    def _loss_per_frame(self, model) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(loss, q)`` where loss is the per-frame negative log-likelihood.

        Loss is weighted by the normalised MBAR weights and computed via
        ``snapshot_loss_original`` from :mod:`aimmdTIS.training`.
        """
        q, w, shot = self._q_and_weights(model)
        w_norm = w / w.sum()
        q_t    = torch.as_tensor(q.reshape(-1, 1), dtype=torch.float64)
        w_t    = torch.as_tensor(w_norm,            dtype=torch.float64)
        shot_t = torch.as_tensor(shot,              dtype=torch.float64)
        loss = snapshot_loss_original(q_t, w_t, shot_t).detach().numpy()
        return loss, q

    # ------------------------------------------------------------------
    # Individual along-q plots
    # ------------------------------------------------------------------

    def plot_loss_along_q(
        self,
        model,
        ax: Optional[plt.Axes] = None,
        n_bins: int = 100,
        color: str = "black",
        density: bool = False,
    ) -> plt.Axes:
        r"""Plot the per-bin negative log-likelihood loss along :math:`q`.

        Parameters
        ----------
        model
            Trained committor model.
        ax
            Target axes; created if *None*.
        n_bins
            Number of q-bins.
        color
            Line colour.
        density
            If *True*, normalise by q-bin width.

        Returns
        -------
        ax
        """
        loss, q = self._loss_per_frame(model)
        label = r"$\mathcal{L}(q')$" + ("*" if density else "")
        hist, edges = np.histogram(q, bins=n_bins, weights=loss, density=density)
        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.plot(edges[1:], hist, label=label, color=color, linewidth=self.plot_settings.linewidth)
        return ax

    def plot_loss_normalized_along_q(
        self,
        model,
        ax: Optional[plt.Axes] = None,
        n_bins: int = 100,
        color: str = "orange",
    ) -> plt.Axes:
        r"""Plot the loss divided by the q-bin density :math:`\mathcal{L}(q')/\rho^{PE}(q')`.

        This reveals where the model is poorly calibrated independent of how
        densely sampled each q-region is.
        """
        q, w, shot = self._q_and_weights(model)

        # Density-normalised weights for loss: scale each frame by 1/ρ(q)
        H_q, q_edges = np.histogram(q, bins=n_bins, density=True)
        bin_idx      = np.clip(np.digitize(q, q_edges[:-1]) - 1, 0, len(H_q) - 1)
        rho_inv      = np.nan_to_num(1.0 / H_q[bin_idx])

        w_scaled = w / w.sum() * rho_inv
        q_t    = torch.as_tensor(q.reshape(-1, 1), dtype=torch.float64)
        w_t    = torch.as_tensor(w_scaled,          dtype=torch.float64)
        shot_t = torch.as_tensor(shot,              dtype=torch.float64)
        loss   = snapshot_loss_original(q_t, w_t, shot_t).detach().numpy()

        hist, edges = np.histogram(q, bins=n_bins, weights=loss)
        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.plot(
            edges[1:], hist,
            label=r"$\mathcal{L}(q')/\rho^{PE}(q')$",
            color=color,
            linewidth=self.plot_settings.linewidth,
        )
        return ax

    def plot_plnp_along_q(
        self,
        model,
        ax: Optional[plt.Axes] = None,
        n_bins: int = 100,
        color: str = "steelblue",
    ) -> plt.Axes:
        r"""Plot :math:`p(q')\ln p_\text{model}(q')` — loss per unit weight along q."""
        q, w, shot = self._q_and_weights(model)
        w_norm = w / w.sum()
        q_t    = torch.as_tensor(q.reshape(-1, 1), dtype=torch.float64)
        w_t    = torch.as_tensor(w_norm,            dtype=torch.float64)
        shot_t = torch.as_tensor(shot,              dtype=torch.float64)
        loss   = snapshot_loss_original(q_t, w_t, shot_t).detach().numpy()

        loss_hist, edges = np.histogram(q, bins=n_bins, weights=loss)
        w_hist, _        = np.histogram(q, bins=n_bins, weights=w_norm)
        with np.errstate(invalid="ignore"):
            plnp = np.where(w_hist > 0, loss_hist / w_hist, np.nan)

        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.plot(
            edges[1:], plnp,
            label=r"$p^{RPE}(q')\ln p^{\rm model}(q')$",
            color=color,
            linewidth=self.plot_settings.linewidth,
        )
        return ax

    def plot_weight_along_q(
        self,
        model,
        ax: Optional[plt.Axes] = None,
        n_bins: int = 100,
        color: str = "seagreen",
        density: bool = False,
    ) -> plt.Axes:
        r"""Plot the normalised MBAR-weight distribution :math:`\rho^{RPE}(q')` along q."""
        q, w, _ = self._q_and_weights(model)
        w_norm = w / w.sum()
        label  = r"$\rho^{RPE}(q')$" + ("*" if density else "")
        hist, edges = np.histogram(q, bins=n_bins, weights=w_norm, density=density)
        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.plot(edges[1:], hist, label=label, color=color, linewidth=self.plot_settings.linewidth)
        return ax

    def plot_distribution_along_q(
        self,
        model,
        ax: Optional[plt.Axes] = None,
        n_bins: int = 100,
        color: str = "tomato",
        density: bool = False,
    ) -> plt.Axes:
        r"""Plot the raw frame-count distribution :math:`\rho^{PE}(q')` along q."""
        q, _, _ = self._q_and_weights(model)
        label   = r"$\rho^{PE}(q')$" + ("*" if density else "")
        hist, edges = np.histogram(q, bins=n_bins, density=density)
        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.plot(edges[1:], hist, label=label, color=color, linewidth=self.plot_settings.linewidth)
        return ax

    def plot_pA_pB_along_q(
        self,
        model,
        ax: Optional[plt.Axes] = None,
        n_bins: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Plot :math:`p_A` and :math:`p_B` from shot results binned along model q.

        Returns
        -------
        edges, p_A, p_B
            Bin edges, A-committor per bin, B-committor per bin.
        """
        q, w, shot = self._q_and_weights(model)
        rho_A, edges = np.histogram(q, bins=n_bins, weights=w * shot[:, 0])
        rho_B, _     = np.histogram(q, bins=edges,  weights=w * shot[:, 1])
        total = rho_A + rho_B
        with np.errstate(invalid="ignore"):
            p_A = np.where(total > 0, rho_A / total, np.nan)
            p_B = np.where(total > 0, rho_B / total, np.nan)

        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.plot(edges[1:], p_A, label=r"$p_A^{RPE}(q')$", linewidth=self.plot_settings.linewidth)
        ax.plot(edges[1:], p_B, label=r"$p_B^{RPE}(q')$", linewidth=self.plot_settings.linewidth)
        return edges, p_A, p_B

    # ------------------------------------------------------------------
    # Combined overview figure
    # ------------------------------------------------------------------

    def overview_along_q(
        self,
        model,
        q_min: float = -20.0,
        q_max: float = 20.0,
        n_bins: int = 200,
    ) -> plt.Figure:
        r"""Two-panel overview of model quality along :math:`q`.

        **Left panel** — diagnostic losses on a log scale:

        * :math:`\mathcal{L}(q')` — raw per-bin loss
        * :math:`p^{RPE}(q')\ln p^{\rm model}(q')` — plnp (loss per unit weight)
        * :math:`\mathcal{L}(q')/\rho^{PE}(q')` — density-normalised loss
        * :math:`\rho^{RPE}(q')` — MBAR weight distribution

        **Right panel** — calibration check:

        * :math:`p_A^{RPE}(q')`, :math:`p_B^{RPE}(q')` from shot results
        * Theoretical :math:`p_A(q') = \sigma(-q')`, :math:`p_B(q') = \sigma(q')`

        Parameters
        ----------
        model
            Trained committor model.
        q_min, q_max
            x-axis limits.
        n_bins
            Histogram bin count.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

        self.plot_loss_along_q(model,              ax=ax_left,  n_bins=n_bins, color="black")
        self.plot_plnp_along_q(model,              ax=ax_left,  n_bins=n_bins, color="steelblue")
        self.plot_loss_normalized_along_q(model,   ax=ax_left,  n_bins=n_bins, color="orange")
        self.plot_weight_along_q(model,            ax=ax_left,  n_bins=n_bins, color="seagreen")

        ax_left.set_yscale("log")
        ax_left.set_xlim([q_min, q_max])
        ax_left.set_xlabel(r"$q'$", fontsize=self.plot_settings.fontsize)
        ax_left.set_ylabel("Value", fontsize=self.plot_settings.fontsize)
        ax_left.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=10)
        ax_left.grid(True, which="both", linestyle="--", alpha=0.5)

        edges, p_A, p_B = self.plot_pA_pB_along_q(model, ax=ax_right, n_bins=n_bins)
        # Theoretical curves
        q_space = edges[1:]
        ax_right.plot(q_space, 1 / (1 + np.exp( q_space)), label=r"$p_A$ theory", linestyle="--", color="C0")
        ax_right.plot(q_space, 1 / (1 + np.exp(-q_space)), label=r"$p_B$ theory", linestyle="--", color="C1")

        ax_right.set_yscale("log")
        ax_right.set_xlim([q_min, q_max])
        ax_right.set_xlabel(r"$q'$", fontsize=self.plot_settings.fontsize)
        ax_right.set_ylabel(r"$p_A$, $p_B$", fontsize=self.plot_settings.fontsize)
        ax_right.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=10)
        ax_right.grid(True, which="both", linestyle="--", alpha=0.5)

        fig.tight_layout()
        return fig
