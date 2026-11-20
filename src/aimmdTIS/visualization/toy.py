"""Toy-system visualization utilities.

Extends BaseVisualizer with PES-dependent plotting and theory overlays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import openpathsampling as paths

from .base import BaseVisualizer
from .model_analysis import ModelAnalysisMixin
from ..Tools import create_discrete_cmap
from ..Tools import CallableVolume


class ToyVisualizer(BaseVisualizer, ModelAnalysisMixin):
    """Toy-system visualization class.

    Parameters
    ----------
    pes
        Toy potential object providing extent, states, and plot utilities.
    """

    def __init__(
        self,
        pes,
        temperature: float = 1.0,
        resolution: int = 501,
        descriptor_dims: Iterable[int] = (0, 1),
        standard_value: Optional[Iterable[float]] = None,
    ) -> None:
        self.pes = pes
        dims_extent = pes.extent if pes is not None else [-10, 10, -10, 10]
        total_num_descriptors = None
        if pes is not None:
            total_num_descriptors = pes.n_harmonics + pes.n_dims_pot
        standard_value = np.zeros(total_num_descriptors) if standard_value is None else np.asarray(standard_value)

        super().__init__(
            temperature=temperature,
            resolution=resolution,
            descriptor_dims=descriptor_dims,
            total_num_descriptors=total_num_descriptors,
            dims_extent=dims_extent,
            standard_value=standard_value,
        )

        self._init_pes_grid()
        self._init_plot_defaults()

    def _init_pes_grid(self) -> None:
        if self.pes is None:
            return
        self.x = np.linspace(self.dims_extent[0], self.dims_extent[1], self.resolution)
        self.y = np.linspace(self.dims_extent[2], self.dims_extent[3], self.resolution)
        self.x_2d, self.y_2d, self.U = self.pes.plot_2d_pes(self.x, self.y)

    def _init_plot_defaults(self) -> None:
        self.cmap_committor = create_discrete_cmap(24)
        self.levels_committor = np.linspace(0, 1, 11)
        self.levels_q = np.arange(-16, 17, 2)
        self.levels_theory = self.levels_q

    # -------------------------
    # Toy-specific overlays
    # -------------------------
    def plot_states(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot state volumes for the toy PES."""
        opA = paths.CoordinateFunctionCV(
            name="opA", f=self.pes.stable_interface_function, center=self.pes.state_A
        )
        opB = paths.CoordinateFunctionCV(
            name="opB", f=self.pes.stable_interface_function, center=self.pes.state_B
        )

        stateA = paths.CVDefinedVolume(opA, 0.0, self.pes.state_boundary).named("StateA")
        stateB = paths.CVDefinedVolume(opB, 0.0, self.pes.state_boundary).named("StateB")

        xedges, yedges = self.create_x_y_edges(n_bins_2d=501)
        x_2d, y_2d = np.meshgrid(xedges, yedges)
        states_plot_A = np.vectorize(CallableVolume(stateA))(x_2d, y_2d)
        states_plot_B = np.vectorize(CallableVolume(stateB))(x_2d, y_2d)

        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.contour(x_2d, y_2d, states_plot_A, colors="red", linewidths=1.5)
        ax.contour(x_2d, y_2d, states_plot_B, colors="blue", linewidths=1.5)
        return ax

    def plot_potential_contour(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot PES contours."""
        if self.pes is None:
            raise ValueError("PES not set")
        if ax is None:
            _, ax = plt.subplots(1, 1)
        levels_U = self.beta * self.pes.levels
        ax.contour(self.x_2d, self.y_2d, self.U, levels=levels_U, colors="gray", linewidths=1.5)
        return ax

    # -------------------------
    # Theory overlays
    # -------------------------
    def load_theoretical_committor(
        self,
        theoretical_committor_path: str,
        n_x: int,
        n_y: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if n_y is None:
            n_y = n_x
        p_theory = np.load(theoretical_committor_path)
        xedges = np.linspace(self.dims_extent[0], self.dims_extent[1], n_x)
        yedges = np.linspace(self.dims_extent[2], self.dims_extent[3], n_y)
        q_theory = -np.log(1 / p_theory - 1)
        return p_theory, q_theory, xedges, yedges

    def plot_theoretical_committor(
        self,
        theoretical_committor_path: str,
        n_x: int,
        n_y: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
        colorbar: bool = True,
    ) -> plt.Axes:
        p_theory, _, xedges, yedges = self.load_theoretical_committor(
            theoretical_committor_path, n_x, n_y
        )
        if ax is None:
            _, ax = plt.subplots(1, 1)
        im = ax.imshow(
            p_theory,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
            alpha=0.8,
        )
        if colorbar and ax.figure is not None:
            cb = ax.figure.colorbar(im, ax=ax)
            cb.set_label(r"Theoretical $p_B$")
        return ax

    def plot_theoretical_q_contour(
        self,
        theoretical_committor_path: str,
        n_x: int,
        n_y: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
        levels: Optional[Iterable[float]] = None,
        colorbar: bool = True,
    ) -> plt.Axes:
        _, q_theory, xedges, yedges = self.load_theoretical_committor(
            theoretical_committor_path, n_x, n_y
        )
        X, Y = np.meshgrid(xedges, yedges)
        if ax is None:
            _, ax = plt.subplots(1, 1)
        contour = ax.contour(
            X,
            Y,
            q_theory,
            levels=levels or self.levels_theory,
            colors="black",
            linestyles="--",
            linewidths=1.0,
        )
        ax.clabel(contour, inline=1, fontsize=10)
        if colorbar and ax.figure is not None:
            ax.figure.colorbar(contour, ax=ax)
        return ax

    # -------------------------
    # Comparison with theory on the reweighted path ensemble
    # -------------------------
    def set_theory(self, theoretical_committor_path: Union[str, Path]) -> None:
        """Load an exact committor p_B on a grid over ``self.dims_extent``.

        Expects the layout of :meth:`load_theoretical_committor`: ``p[i_y, i_x]``
        on ``np.linspace`` grids spanning the extent.
        """
        from scipy.interpolate import RegularGridInterpolator

        p = np.load(theoretical_committor_path)
        n_y, n_x = p.shape
        self._theory_path = str(theoretical_committor_path)
        self._theory_n = (n_x, n_y)
        x = np.linspace(self.dims_extent[0], self.dims_extent[1], n_x)
        y = np.linspace(self.dims_extent[2], self.dims_extent[3], n_y)
        self._theory_interp = RegularGridInterpolator((y, x), p, bounds_error=False, fill_value=np.nan)

    def theory_committor_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Theory p_B at points (bilinear interpolation of the loaded grid)."""
        if getattr(self, "_theory_interp", None) is None:
            raise ValueError("set_theory() must be called first.")
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return self._theory_interp(np.column_stack([y.ravel(), x.ravel()])).reshape(x.shape)

    def _bin_centres(self, n_bins_2d):
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)
        xc = 0.5 * (xedges[1:] + xedges[:-1])
        yc = 0.5 * (yedges[1:] + yedges[:-1])
        Xc, Yc = np.meshgrid(xc, yc)  # (ny, nx)
        return xedges, yedges, Xc, Yc

    def _binned(self, values, n_bins_2d, descriptor_dims=None):
        """Weighted sum of ``values`` per 2-D bin, as (ny, nx) for imshow."""
        if self._desc is None:
            raise ValueError("load_trainset() must be called first.")
        dims = tuple(descriptor_dims) if descriptor_dims is not None else self.descriptor_dims
        desc = self._desc_phys if self._desc_phys is not None else self._desc
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)
        H, _, _ = np.histogram2d(desc[:, dims[0]], desc[:, dims[1]], bins=[xedges, yedges], weights=values)
        return H.T

    def free_energy_error(
        self,
        n_bins_2d: int = 100,
        min_count: int = 20,
        descriptor_dims: Optional[Iterable[int]] = None,
    ) -> Dict[str, object]:
        r"""Reweighted-path-ensemble free energy vs the true potential, in kT.

        :math:`\beta F_{RPE} = -\ln \rho_{RPE}` from the weighted histogram is
        compared with :math:`\beta V` at the bin centres, only on bins with at
        least ``min_count`` frames (the region the RPE samples). The free-energy
        constant is fixed by the mean difference over those bins.

        Returns ``diff`` (masked map, NaN outside the data region), ``F_rpe`` and
        ``F_true`` (aligned, in kT), ``mask``, ``rmse`` (unweighted over the data
        bins), ``rmse_weighted`` (weighted by the RPE density), ``coverage``
        (fraction of bins with data) and the bin edges.
        """
        xedges, yedges, Xc, Yc = self._bin_centres(n_bins_2d)
        H = self._binned(self._w, n_bins_2d, descriptor_dims)
        counts = self._binned(np.ones_like(self._w), n_bins_2d, descriptor_dims)
        mask = counts >= min_count
        with np.errstate(divide="ignore"):
            F_rpe = -np.log(H / H.sum())
        F_true = self.beta * np.vectorize(self.pes.pes_2d_pot)(Xc, Yc)
        delta = F_rpe - F_true
        offset = float(np.mean(delta[mask])) if mask.any() else np.nan
        diff = np.where(mask, delta - offset, np.nan)
        rho = np.where(mask, H, 0.0)
        return {
            "diff": diff,
            "F_rpe": F_rpe - offset,
            "F_true": F_true,
            "mask": mask,
            "rmse": float(np.sqrt(np.nanmean(diff[mask] ** 2))) if mask.any() else np.nan,
            "rmse_weighted": float(np.sqrt(np.sum(rho * np.nan_to_num(diff) ** 2) / np.sum(rho))) if mask.any() else np.nan,
            "coverage": float(mask.mean()),
            "xedges": xedges,
            "yedges": yedges,
        }

    def committor_projection_error(
        self,
        model=None,
        n_bins_2d: int = 100,
        p_clip: float = 1e-3,
        min_count: int = 20,
        descriptor_dims: Optional[Iterable[int]] = None,
    ) -> Dict[str, object]:
        r"""Committor projected on the RPE data vs the theory committor.

        With ``model`` the projection is the weighted mean of the model q (and
        p_B) over the RPE frames in each bin, as :meth:`plot_q_model_projection`.
        With ``model=None`` it is the data-only shot committor of
        :meth:`plot_shot_committor` (a model-free baseline for the RPE itself).

        ``rmse_logit`` compares q with logit p_B^theory on data bins where the
        theory committor lies in ``(p_clip, 1 - p_clip)`` (outside that range the
        logit is unbounded). ``rmse_pB`` compares p_B on all data bins.
        """
        xedges, yedges, Xc, Yc = self._bin_centres(n_bins_2d)
        w_sum = self._binned(self._w, n_bins_2d, descriptor_dims)
        counts = self._binned(np.ones_like(self._w), n_bins_2d, descriptor_dims)
        mask_data = counts >= min_count
        with np.errstate(divide="ignore", invalid="ignore"):
            if model is None:
                H_A = self._binned(self._w * self._shot[:, 0], n_bins_2d, descriptor_dims)
                H_B = self._binned(self._w * self._shot[:, 1], n_bins_2d, descriptor_dims)
                p_bin = H_B / (H_A + H_B)
                q_bin = np.log(p_bin / (1.0 - p_bin))
            else:
                p_B_arr, q_arr = self._model_output_rpe(model)
                q_bin = self._binned(self._w * np.asarray(q_arr).reshape(-1), n_bins_2d, descriptor_dims) / w_sum
                p_bin = self._binned(self._w * np.asarray(p_B_arr).reshape(-1), n_bins_2d, descriptor_dims) / w_sum
            p_theory = self.theory_committor_at(Xc, Yc)
            q_theory = np.log(p_theory / (1.0 - p_theory))
        mask_logit = mask_data & (p_theory > p_clip) & (p_theory < 1.0 - p_clip) & np.isfinite(q_bin)
        mask_pb = mask_data & np.isfinite(p_theory) & np.isfinite(p_bin)
        diff_q = np.where(mask_logit, q_bin - q_theory, np.nan)
        diff_p = np.where(mask_pb, p_bin - p_theory, np.nan)
        return {
            "q": np.where(mask_data, q_bin, np.nan),
            "p_B": np.where(mask_data, p_bin, np.nan),
            "q_theory": q_theory,
            "p_theory": p_theory,
            "diff_q": diff_q,
            "diff_p": diff_p,
            "mask_data": mask_data,
            "mask_logit": mask_logit,
            "rmse_logit": float(np.sqrt(np.mean(diff_q[mask_logit] ** 2))) if mask_logit.any() else np.nan,
            "rmse_pB": float(np.sqrt(np.mean(diff_p[mask_pb] ** 2))) if mask_pb.any() else np.nan,
            "xedges": xedges,
            "yedges": yedges,
        }

    def _error_image(self, diff, xedges, yedges, ax, vmax, label):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        im = ax.imshow(diff, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        if ax.figure is not None:
            ax.figure.colorbar(im, ax=ax, label=label)
        return im

    def plot_true_free_energy(self, ax: Optional[plt.Axes] = None, vmax: Optional[float] = None,
                              cmap: str = "Blues_r"):
        r"""True :math:`\beta V(x, y)` (minimum shifted to 0), as a heatmap."""
        if ax is None:
            _, ax = plt.subplots(1, 1)
        U = self.beta * (self.U - np.min(self.U))
        return ax.imshow(U, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                         extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])

    def plot_free_energy_error(self, ax: Optional[plt.Axes] = None, n_bins_2d: int = 100,
                               min_count: int = 20, vmax: float = 3.0):
        r""":math:`\beta F_{RPE} - \beta V` on the bins the RPE samples (see :meth:`free_energy_error`)."""
        res = self.free_energy_error(n_bins_2d=n_bins_2d, min_count=min_count)
        self._error_image(res["diff"], res["xedges"], res["yedges"], ax, vmax,
                          r"$\beta F_{RPE} - \beta V$ [kT]")
        return res

    def plot_committor_projection_error(self, model=None, ax: Optional[plt.Axes] = None,
                                        n_bins_2d: int = 100, p_clip: float = 1e-3,
                                        min_count: int = 20, vmax: float = 3.0):
        """Projected q minus theory logit p_B on the data bins (see :meth:`committor_projection_error`)."""
        res = self.committor_projection_error(model=model, n_bins_2d=n_bins_2d, p_clip=p_clip,
                                              min_count=min_count)
        self._error_image(res["diff_q"], res["xedges"], res["yedges"], ax, vmax,
                          r"$q - q_{theory}$" if model is not None else r"$q_{shots} - q_{theory}$")
        return res
