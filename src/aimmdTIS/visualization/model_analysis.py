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

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..training import snapshot_loss_softplus, snapshot_loss_original


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
        ``snapshot_loss_softplus`` from :mod:`aimmdTIS.training`.
        """
        q, w, shot = self._q_and_weights(model)
        w_norm = w/len(w)  # normalise by total frame count to get per-frame loss
        q_t    = torch.as_tensor(q.reshape(-1, 1), dtype=torch.float64)
        w_t    = torch.as_tensor(w_norm,            dtype=torch.float64)
        shot_t = torch.as_tensor(shot,              dtype=torch.float64)
        loss = snapshot_loss_softplus(q_t, w_t, shot_t).detach().numpy()

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
        loss   = snapshot_loss_softplus(q_t, w_t, shot_t).detach().numpy()

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
        w_norm = w / len(w)  # normalise by total frame count to get per-frame loss
        loss = self._loss_per_frame(model)[0]

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
        ax.plot(edges[1:], p_A,".", label=r"$p_A^{RPE}(q')$", linewidth=self.plot_settings.linewidth)
        ax.plot(edges[1:], p_B,".", label=r"$p_B^{RPE}(q')$", linewidth=self.plot_settings.linewidth)
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
        plot_ideal: bool = True,
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

        if plot_ideal:
            q_theory = np.linspace(q_min, q_max, 100)
            p_B_theory = 1.0 / (1.0 + np.exp(-q_theory))
            Information = -(p_B_theory * np.log(p_B_theory) + (1 - p_B_theory) * np.log(1 - p_B_theory))
            ax_left.plot(q_theory, Information, color="darkblue", linestyle="--", label="Ideal I(q')")


        ax_left.set_yscale("log")
        ax_left.set_xlim([q_min, q_max])
        ax_left.set_xlabel(r"$q'$", fontsize=self.plot_settings.fontsize)
        ax_left.set_ylabel("Value", fontsize=self.plot_settings.fontsize)
        ax_left.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=10)
        ax_left.grid(True, which="both", linestyle="--", alpha=0.5)
        ax_left.set_ylim(bottom=1/self._w.sum())

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
        ax_right.set_ylim(bottom=1/self._w.sum())


        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Gradient / descriptor-importance methods
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_mean_std(
        values: np.ndarray,
        weights: np.ndarray,
        axis: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted mean and standard deviation along *axis*."""
        mean = np.average(values, weights=weights, axis=axis)
        variance = np.average((values - mean) ** 2, weights=weights, axis=axis)
        return mean, np.sqrt(variance)

    def _compute_gradients_raw(
        self,
        model,
        descriptors: np.ndarray,
    ) -> np.ndarray:
        """Backprop through *model.nnet* for *descriptors* and return gradients.

        Parameters
        ----------
        model
            AIMMD RCModel with a ``.nnet`` attribute.
        descriptors
            Array of shape ``(N, D)`` in **scaled** (model-input) space.

        Returns
        -------
        gradients : ndarray, shape (N, D)
            ``d(output) / d(input)`` for each sample.
        """
        device = getattr(model, "_device", "cpu")
        input_tensor = torch.tensor(
            descriptors, requires_grad=True, device=device, dtype=torch.float32
        )
        model.nnet.eval()
        output = model.nnet(input_tensor)
        output.backward(torch.ones_like(output))
        gradients = input_tensor.grad.detach().cpu().numpy()
        model.nnet.zero_grad(set_to_none=True)
        model.nnet.train()
        return gradients

    def compute_gradient_stats_per_q(
        self,
        model,
        norm_factors: Optional[np.ndarray] = None,
        q_bins: Optional[np.ndarray] = None,
        q_min: float = -50.0,
        q_max: float = 18.0,
        dq: float = 0.05,
        weights_ones: bool = False,
        use_cache: bool = True,
    ) -> Dict[str, object]:
        """Compute per-q-bin weighted gradient statistics for the loaded trainset.

        The result is cached in ``self._gradient_output_cache[id(model)]`` so
        that repeated calls for the same model object are free.  Use
        :meth:`save_gradient_cache` / :meth:`load_gradient_cache` to persist
        results across notebook restarts.

        Parameters
        ----------
        model
            Trained AIMMD RCModel.
        norm_factors
            Per-descriptor normalisation factors applied to raw gradients before
            computing unit vectors and statistics.  Defaults to
            ``1 / max(descriptors, axis=0)`` (same as the legacy notebook).
        q_bins
            Explicit bin edges.  If *None*, constructed from ``q_min``/``q_max``/``dq``.
        q_min, q_max, dq
            Used only when ``q_bins`` is *None*.
        weights_ones
            If *True*, use uniform weights (ignore MBAR weights) — useful for
            debugging.

        Returns
        -------
        result : dict with keys

        * ``q_bins``                              – bin edges
        * ``bin_centers``                         – bin centres
        * ``counts``                              – raw frame count per bin
        * ``gradients_per_q_bin``                 – weighted mean of ``grad / norm_factors`` per bin, shape (B, D)
        * ``gradients_per_q_bin_std``             – weighted std, shape (B, D)
        * ``abs_gradients_per_q_bin``             – weighted mean of ``|grad / norm_factors|`` per bin
        * ``abs_gradients_per_q_bin_std``         – weighted std
        * ``gradient_unit_vector_per_q_bin``      – weighted mean of per-frame unit gradient, shape (B, D)
        * ``gradient_unit_vector_per_q_bin_std``  – weighted std
        * ``gradient_magnitude_per_q_bin``        – mean ``||grad_normalised||`` per bin, shape (B,)
        """
        if self._desc is None:
            raise ValueError("load_trainset() must be called before gradient computation.")

        cache_key = id(model)
        if use_cache and cache_key in self._gradient_output_cache:
            print("Using cached gradient statistics for this model.")
            return self._gradient_output_cache[cache_key]

        # Retrieve already-cached model forward pass (free if model was evaluated before)
        _, q_all = self._model_output_rpe(model)

        if q_bins is None:
            q_bins = np.arange(q_min, q_max, dq)
        bin_centers = (q_bins[:-1] + q_bins[1:]) / 2
        num_bins = len(q_bins) - 1
        num_desc = self._desc.shape[1]

        # Bin assignment (digitize returns 1-based; shift to 0-based)
        ind_bin = np.digitize(q_all, q_bins, right=False)
        valid = (ind_bin > 0) & (ind_bin <= num_bins)
        ind_bin_valid = ind_bin[valid] - 1

        desc_valid    = self._desc[valid]
        w_valid       = self._w[valid]
        q_valid       = q_all[valid]  # noqa: F841 — kept for potential future use

        if norm_factors is None:
            norm_factors = 1.0 / np.maximum(np.max(desc_valid, axis=0), 1e-12)
        norm_factors = np.asarray(norm_factors, dtype=np.float64)

        epsilon = 1e-12

        result: Dict[str, object] = {
            "q_bins":                             q_bins,
            "bin_centers":                        bin_centers,
            "counts":                             np.zeros(num_bins, dtype=np.int64),
            "gradients_per_q_bin":                np.zeros((num_bins, num_desc)),
            "gradients_per_q_bin_std":            np.zeros((num_bins, num_desc)),
            "abs_gradients_per_q_bin":            np.zeros((num_bins, num_desc)),
            "abs_gradients_per_q_bin_std":        np.zeros((num_bins, num_desc)),
            "gradient_unit_vector_per_q_bin":     np.zeros((num_bins, num_desc)),
            "gradient_unit_vector_per_q_bin_std": np.zeros((num_bins, num_desc)),
            "gradient_magnitude_per_q_bin":       np.zeros(num_bins),
            "norm_factors":                       norm_factors,
        }

        for bin_idx in range(num_bins):
            mask = ind_bin_valid == bin_idx
            count = int(mask.sum())
            result["counts"][bin_idx] = count
            if count == 0:
                continue

            desc_bin = desc_valid[mask]
            w_bin    = np.ones(count) if weights_ones else w_valid[mask]

            grads = self._compute_gradients_raw(model, desc_bin)          # (N, D)
            grads_norm = grads * norm_factors                              # (N, D)
            grad_mag   = np.linalg.norm(grads_norm, axis=1)               # (N,)

            # Guard zero-magnitude frames
            safe_mag = np.where(grad_mag > epsilon, grad_mag, epsilon)
            unit_vecs = grads_norm / safe_mag[:, np.newaxis]              # (N, D)

            result["gradient_magnitude_per_q_bin"][bin_idx] = float(np.mean(grad_mag))

            result["gradients_per_q_bin"][bin_idx],     result["gradients_per_q_bin_std"][bin_idx]     = self._weighted_mean_std(grads_norm,        w_bin, axis=0)
            result["abs_gradients_per_q_bin"][bin_idx], result["abs_gradients_per_q_bin_std"][bin_idx] = self._weighted_mean_std(np.abs(grads_norm), w_bin, axis=0)
            result["gradient_unit_vector_per_q_bin"][bin_idx], result["gradient_unit_vector_per_q_bin_std"][bin_idx] = self._weighted_mean_std(unit_vecs, w_bin, axis=0)

        self._gradient_output_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Gradient cache persistence
    # ------------------------------------------------------------------

    def save_gradient_cache(self, model, path: Union[str, Path]) -> None:
        """Save gradient statistics for *model* to a ``.npz`` file on disk.

        Parameters
        ----------
        model
            The model whose gradient cache should be saved.
        path
            File path (the ``.npz`` extension is added automatically if absent).
        """
        cache_key = id(model)
        if cache_key not in self._gradient_output_cache:
            raise ValueError(
                "No gradient cache found for this model. "
                "Call compute_gradient_stats_per_q() first."
            )
        result = self._gradient_output_cache[cache_key]
        np.savez(str(path), **{k: np.asarray(v) for k, v in result.items()})

    def load_gradient_cache(
        self,
        model,
        path: Union[str, Path],
    ) -> Dict[str, object]:
        """Load gradient statistics from a ``.npz`` file and inject into the cache.

        Parameters
        ----------
        model
            The model to associate the loaded cache with (keyed by ``id(model)``).
        path
            Path to the ``.npz`` file written by :meth:`save_gradient_cache`.

        Returns
        -------
        result : dict  (same structure as :meth:`compute_gradient_stats_per_q`)
        """
        path = Path(path)
        if not path.exists():
            # Try with .npz appended
            path = path.with_suffix(".npz")
        data = np.load(str(path), allow_pickle=False)
        result: Dict[str, object] = {k: data[k] for k in data.files}
        self._gradient_output_cache[id(model)] = result
        return result


    def compute_gradient_histograms_vs_q(
        self,
        model,
        q_edges=None,
        grad_edges=None,
        norm_factors=None,
        batch_size=10000,
        use_cache=True,
    ):
        """
        Histogram of normalised raw gradients versus model q.

        Returns
        -------
        result dict:
            H: shape (n_q_bins, n_grad_bins, n_desc)
            q_edges
            grad_edges
            norm_factors
        """
        if self._desc is None:
            raise ValueError("load_trainset() must be called first.")

        _, q = self._model_output_rpe(model)
        n_desc = self._desc.shape[1]

        if q_edges is None:
            q_edges = np.linspace(-50, 18, 150)
        if grad_edges is None:
            grad_edges = np.linspace(-70, 70, 1000)

        q_edges = np.asarray(q_edges, dtype=float)
        grad_edges = np.asarray(grad_edges, dtype=float)

        if norm_factors is None:
            norm_factors = np.ones(n_desc, dtype=float)
        norm_factors = np.asarray(norm_factors, dtype=float)

        cache_key = (
            "gradient_hist_vs_q",
            id(model),
            tuple(q_edges[[0, -1]]),
            len(q_edges),
            tuple(grad_edges[[0, -1]]),
            len(grad_edges),
            tuple(norm_factors),
        )

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        H = np.zeros((len(q_edges) - 1, len(grad_edges) - 1, n_desc), dtype=float)

        for start in range(0, self._desc.shape[0], batch_size):
            end = min(start + batch_size, self._desc.shape[0])
            desc_batch = self._desc[start:end]
            q_batch = q[start:end]

            gradients = self._compute_gradients_raw(model, desc_batch)
            gradients = gradients / norm_factors

            for dim in range(n_desc):
                H_dim, _, _ = np.histogram2d(
                    q_batch,
                    gradients[:, dim],
                    bins=[q_edges, grad_edges],
                    density=False,
                )
                H[:, :, dim] += H_dim

        result = {
            "H": H,
            "q_edges": q_edges,
            "grad_edges": grad_edges,
            "norm_factors": norm_factors,
        }

        self._cache[cache_key] = result
        return result
    # ------------------------------------------------------------------
    # Gradient plots
    # ------------------------------------------------------------------

    @staticmethod
    def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
        """1-D moving average with ``mode='same'`` (uniform bin-index window)."""
        if window <= 1:
            return arr
        return np.convolve(arr, np.ones(window) / window, mode="same")

    @staticmethod
    def _q_smooth(arr: np.ndarray, bin_centers: np.ndarray, q_width: float) -> np.ndarray:
        """Box-smooth *arr* with a fixed width *q_width* in q-space.

        For each bin ``i`` the smoothed value is the mean of all bins whose
        centre falls within ``[c_i - q_width/2,  c_i + q_width/2]``.  This
        gives consistent physical smoothing regardless of local bin density,
        making it the right choice when ``q_bins`` has variable bin widths.

        Falls back to the raw array if *q_width* ≤ 0.
        """
        if q_width <= 0:
            return arr.copy()
        half = q_width / 2.0
        out = np.empty_like(arr, dtype=float)
        for i, c in enumerate(bin_centers):
            mask = (bin_centers >= c - half) & (bin_centers <= c + half)
            out[i] = arr[mask].mean() if mask.any() else arr[i]
        return out

    def plot_gradient_along_q(
        self,
        model,
        ax: Optional[plt.Axes] = None,
        moving_avg: int = 1,
        q_smooth: float = 0.0,
        q_min: float = -50.0,
        q_max: float = 18.0,
        descriptor_labels: Optional[Sequence[str]] = None,
        plot_error: bool = True,
        norm_factors: Optional[np.ndarray] = None,
        q_bins: Optional[np.ndarray] = None,
        absolute: bool = False,
    ) -> plt.Axes:
        r"""Plot weighted-mean normalised gradient :math:`\langle \partial q / \partial d_i \rangle` per q-bin.

        One line per descriptor, shaded ±std region when ``plot_error=True``.
        Calls :meth:`compute_gradient_stats_per_q` (cached).

        Parameters
        ----------
        model
            Trained committor model.
        ax
            Target axes; created if *None*.
        moving_avg
            Bin-index smoothing window (used when ``q_smooth`` is 0).  Not
            recommended for variable-width bins — use ``q_smooth`` instead.
        q_smooth
            Smoothing half-width in **q-space** (physical units).  When > 0,
            overrides ``moving_avg`` and applies a box filter of this width,
            giving consistent smoothing regardless of local bin density.
            A value of ~0.5 is a good starting point for the default binning.
        q_min, q_max
            x-axis limits.
        descriptor_labels
            Legend labels; falls back to ``self.descriptor_labels`` then generic indices.
        plot_error
            Whether to shade ±std.
        norm_factors
            Forwarded to :meth:`compute_gradient_stats_per_q`.
        q_bins
            Forwarded to :meth:`compute_gradient_stats_per_q`.
        absolute
            If *True*, plot ``|gradient|`` instead of signed gradient.
        """
        result = self.compute_gradient_stats_per_q(
            model, norm_factors=norm_factors, q_bins=q_bins
        )
        labels = (
            descriptor_labels
            or (self.descriptor_labels if hasattr(self, "descriptor_labels") else None)
            or [f"d{i}" for i in range(result["gradients_per_q_bin"].shape[1])]
        )
        key     = "abs_gradients_per_q_bin"     if absolute else "gradients_per_q_bin"
        key_std = "abs_gradients_per_q_bin_std" if absolute else "gradients_per_q_bin_std"
        data     = result[key]
        data_std = result[key_std]
        bin_centers = result["bin_centers"]

        def _smooth(arr1d):
            if q_smooth > 0:
                return self._q_smooth(arr1d, bin_centers, q_smooth)
            return self._moving_average(arr1d, moving_avg)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(12, 4))

        for dim, label in enumerate(labels):
            y     = _smooth(data[:, dim])
            y_err = _smooth(data_std[:, dim])
            ax.plot(bin_centers, y, label=label, alpha=0.8, lw=2)
            if plot_error:
                ax.fill_between(bin_centers, y - y_err, y + y_err, alpha=0.2)

        ax.axhline(0, color="black", linewidth=1.5, alpha=0.7)
        ax.set_xlim([q_min, q_max])
        ax.set_xlabel(r"$q(x|\theta)$", fontsize=self.plot_settings.fontsize)
        ylabel = (
            r"$\langle |\partial q / \partial d_i| \rangle$"
            if absolute
            else r"$\langle \partial q / \partial d_i \rangle$"
        )
        ax.set_ylabel(ylabel, fontsize=self.plot_settings.fontsize)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small", frameon=False)
        return ax

    def plot_gradient_unit_along_q(
        self,
        model,
        q_ranges: Optional[List[float]] = None,
        q_spacing: Optional[List[float]] = None,
        columns: Optional[List[Tuple[int, int]]] = None,
        moving_avg: int = 1,
        q_smooth: float = 0.0,
        descriptor_labels: Optional[Sequence[str]] = None,
        plot_error: bool = True,
        norm_factors: Optional[np.ndarray] = None,
        q_bins: Optional[np.ndarray] = None,
        figsize: Tuple[float, float] = (14, 8),
    ) -> plt.Figure:
        r"""Segmented importance plot: unit gradient component :math:`\hat{g}_i(q)` vs. q.

        Reproduces the multi-panel "axis-break" figure from the archive notebook
        (``HG_Importance_final_it_2``).  The unit gradient is computed as

        .. math::
            \hat{g}_i(q') = \frac{\langle \partial q / \partial d_i \rangle_{q'}}
                            {||\langle \partial q / \partial d \rangle_{q'}||}

        which equals the component of the mean normalised gradient unit vector
        in dimension *i*.

        Parameters
        ----------
        model
            Trained committor model.
        q_ranges
            Breakpoints defining q-axis segments, e.g. ``[-45, -5, 0, 12]``.
            Defaults to ``[-45, -5, 0, 12]``.
        q_spacing
            Major grid / tick spacing per segment.  Defaults to ``[5, 1, 5]``.
        columns
            ``(start_col, end_col)`` pairs for ``gridspec`` column slices — one
            per segment.  Defaults to ``[(0, 2), (2, 5), (5, 8)]``.
        moving_avg
            Bin-index smoothing window (used when ``q_smooth`` is 0).  Not
            recommended for variable-width bins — use ``q_smooth`` instead.
        q_smooth
            Smoothing half-width in **q-space** (physical units).  When > 0,
            overrides ``moving_avg`` and applies a box filter of this width,
            giving consistent smoothing regardless of local bin density.
        descriptor_labels
            Legend labels; falls back to ``self.descriptor_labels``.
        plot_error
            Whether to shade ±std of the unit-vector estimate.
        norm_factors
            Forwarded to :meth:`compute_gradient_stats_per_q`.
        q_bins
            Forwarded to :meth:`compute_gradient_stats_per_q`.
        figsize
            Overall figure size.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        result = self.compute_gradient_stats_per_q(
            model, norm_factors=norm_factors, q_bins=q_bins
        )
        labels = (
            descriptor_labels
            or (self.descriptor_labels if hasattr(self, "descriptor_labels") else None)
            or [f"d{i}" for i in range(result["gradient_unit_vector_per_q_bin"].shape[1])]
        )

        bin_centers = result["bin_centers"]
        grad_mean   = result["gradients_per_q_bin"]                # (B, D)
        unit_mean   = result["gradient_unit_vector_per_q_bin"]     # (B, D)
        unit_std    = result["gradient_unit_vector_per_q_bin_std"] # (B, D)

        # Unit vector computed from the mean gradient (alternative definition,
        # matches the ``plot_relative_mean_unit_grad=True`` branch in the archive)
        mag = np.linalg.norm(grad_mean, axis=1, keepdims=True)     # (B, 1)
        unit_from_mean = np.where(mag > 1e-12, grad_mean / mag, 0.0)  # (B, D)

        # Defaults for segmented layout
        if q_ranges is None:
            q_ranges = [-45, -5, 0, 12]
        if q_spacing is None:
            q_spacing = [5, 1, 5]
        n_segs = len(q_ranges) - 1
        if columns is None:
            # Distribute columns automatically: narrow middle region gets fewer columns
            base = 3
            columns = [(i * base, (i + 1) * base) for i in range(n_segs)]

        # Build GridSpec width ratios from column extents
        total_cols = max(end for _, end in columns)
        width_ratios = [1] * total_cols

        fig = plt.figure(figsize=figsize)
        gs  = gridspec.GridSpec(1, total_cols, width_ratios=width_ratios, wspace=0.1)

        ax_list: List[plt.Axes] = []
        for i, (col_start, col_end) in enumerate(columns):
            ax = fig.add_subplot(
                gs[0, col_start:col_end],
                sharey=ax_list[0] if ax_list else None,
            )
            ax_list.append(ax)
            q_start, q_end = q_ranges[i], q_ranges[i + 1]
            ax.set_xlim(q_start, q_end)
            ax.set_ylim(-1, 1)
            if i != 0:
                ax.spines["left"].set_visible(False)
                ax.tick_params(labelleft=False, left=False)

        def _smooth(arr1d):
            if q_smooth > 0:
                return self._q_smooth(arr1d, bin_centers, q_smooth)
            return self._moving_average(arr1d, moving_avg)

        num_desc = unit_from_mean.shape[1]
        for dim in range(num_desc):
            label = labels[dim] if dim < len(labels) else f"d{dim}"
            y     = _smooth(unit_from_mean[:, dim])
            y_err = _smooth(unit_std[:, dim])

            for i, ax in enumerate(ax_list):
                q_start, q_end = q_ranges[i], q_ranges[i + 1]
                ax.plot(bin_centers, y, label=label, linewidth=2)
                ax.axhline(0, color="black", linewidth=1, alpha=0.8)
                if plot_error:
                    ax.fill_between(bin_centers, y - y_err, y + y_err, alpha=0.2)
                ax.set_xlim(q_start, q_end)
                spacing = q_spacing[i] if i < len(q_spacing) else 5
                ax.set_xticks(np.arange(q_start, q_end + 1e-9, spacing))
                ax.tick_params(axis="x", labelrotation=0, labelsize=9)
                ax.grid(True, which="major", linestyle="--", alpha=0.5)

        # Axis-break indicators
        d = 0.012
        kwargs_break = dict(color="k", clip_on=False, linewidth=1)
        for i in range(1, len(ax_list)):
            ax_l = ax_list[i - 1]
            ax_r = ax_list[i]
            ax_l.plot((1 - d, 1 + d), (-d, +d), transform=ax_l.transAxes, **kwargs_break)
            ax_l.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_l.transAxes, **kwargs_break)
            ax_r.plot((-d, +d), (-d, +d), transform=ax_r.transAxes, **kwargs_break)
            ax_r.plot((-d, +d), (1 - d, 1 + d), transform=ax_r.transAxes, **kwargs_break)

        ax_list[0].set_ylabel(
            r"Importance $\hat{g}_i(q')$", fontsize=self.plot_settings.fontsize
        )
        ax_list[-1].legend(
            loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="medium", frameon=False
        )
        fig.subplots_adjust(left=0.08, right=0.85, bottom=0.2, top=0.88)
        return fig

