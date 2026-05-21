"""System-independent visualization base class.

Designed to reduce repeated computations (grids, histograms) and
provide a clean structure for adding future plotting methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


@dataclass
class PlotSettings:
    """Container for common plot settings."""

    alpha: float = 0.7
    linewidth: float = 2.0
    fontsize: int = 14
    cmap: str = "Spectral"


class BaseVisualizer:
    """System-independent visualization utilities.

    Parameters
    ----------
    temperature
        Simulation temperature for beta weighting.
    resolution
        Default grid resolution for 2D plots.
    descriptor_dims
        Which descriptors are plotted on x and y axes.
    total_num_descriptors
        Total descriptor count for model inputs.
    dims_extent
        Plot extent as [xmin, xmax, ymin, ymax].
    standard_value
        Default values for non-plotted descriptor dimensions.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        resolution: int = 501,
        descriptor_dims: Iterable[int] = (0, 1),
        total_num_descriptors: Optional[int] = None,
        dims_extent: Optional[Iterable[float]] = None,
        standard_value: Optional[Iterable[float]] = None,
    ) -> None:
        self.temperature = temperature
        self.beta = 1.0 / temperature
        self.resolution = resolution
        self.descriptor_dims = tuple(descriptor_dims)
        self.total_num_descriptors = total_num_descriptors
        self.dims_extent = list(dims_extent) if dims_extent is not None else [-10, 10, -10, 10]
        self.standard_value = list(standard_value) if standard_value is not None else None

        self.plot_settings = PlotSettings()
        self._cache: Dict[str, object] = {}
        self.RPE = None

    # -------------------------
    # Data management
    # -------------------------
    def set_rpe(self, rpe) -> None:
        """Attach RPE data container."""
        self.RPE = rpe

    def clear_cache(self) -> None:
        """Clear cached grids and histograms."""
        self._cache.clear()

    # -------------------------
    # Grid helpers
    # -------------------------
    def create_x_y_edges(
        self,
        n_bins_2d: int | Tuple[int, int] = 100,
        dims_extent: Optional[Iterable[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create x/y bin edges for 2D grids."""
        extent = list(dims_extent) if dims_extent is not None else self.dims_extent
        if np.shape(n_bins_2d) == ():
            n_bins_2d = (n_bins_2d, n_bins_2d)
        xedges = np.linspace(extent[0], extent[1], n_bins_2d[0])
        yedges = np.linspace(extent[2], extent[3], n_bins_2d[1])
        return xedges, yedges

    def create_2d_projection_coord(
        self,
        n_bins_2d: int = 100,
        descriptor_dims: Optional[Iterable[int]] = None,
        dims_extent: Optional[Iterable[float]] = None,
        standard_value: Optional[Iterable[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create coordinates for 2D model evaluation."""
        dims = tuple(descriptor_dims) if descriptor_dims is not None else self.descriptor_dims
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d, dims_extent=dims_extent)

        if standard_value is None:
            if self.standard_value is None:
                if self.total_num_descriptors is None:
                    raise ValueError("total_num_descriptors must be set for model projections")
                standard_value = [0] * self.total_num_descriptors
            else:
                standard_value = list(self.standard_value)

        if self.total_num_descriptors is None:
            self.total_num_descriptors = len(standard_value)

        if len(standard_value) != self.total_num_descriptors:
            raise ValueError(
                f"standard_value must have length {self.total_num_descriptors}, got {len(standard_value)}"
            )

        coord = []
        for yv in yedges:
            for xv in xedges:
                point = list(standard_value)
                point[dims[0]] = xv
                point[dims[1]] = yv
                coord.append(point)
        return np.array(coord, dtype=np.float32), xedges, yedges

    def compute_q_model_2d(
        self,
        model,
        n_bins_2d: int = 300,
        descriptor_dims: Optional[Iterable[int]] = None,
        dims_extent: Optional[Iterable[float]] = None,
        standard_value: Optional[Iterable[float]] = None,
        cache_key: str = "q_model_2d",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 2D q(x) model grid with caching."""
        if cache_key in self._cache:
            return self._cache[cache_key]

        coord, xedges, yedges = self.create_2d_projection_coord(
            n_bins_2d=n_bins_2d,
            descriptor_dims=descriptor_dims,
            dims_extent=dims_extent,
            standard_value=standard_value,
        )
        q = model.log_prob(coord, use_transform=False)
        q = q.reshape((len(yedges), len(xedges)))
        X, Y = np.meshgrid(xedges, yedges)

        self._cache[cache_key] = (q, X, Y)
        return q, X, Y

    # -------------------------
    # Histogram helpers
    # -------------------------
    def _weighted_histogram2d(
        self,
        data,
        weights,
        descriptor_dims: Optional[Iterable[int]] = None,
        n_bins_2d: int = 100,
        dims_extent: Optional[Iterable[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a weighted 2D histogram for a dataset."""
        dims = tuple(descriptor_dims) if descriptor_dims is not None else self.descriptor_dims
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d, dims_extent=dims_extent)
        extent = list(dims_extent) if dims_extent is not None else self.dims_extent

        H, _, _ = np.histogram2d(
            data[:, dims[0]],
            data[:, dims[1]],
            weights=weights,
            bins=(xedges, yedges),
            range=[[extent[0], extent[1]], [extent[2], extent[3]]],
            density=True,
        )
        return H, xedges, yedges

    def compute_histograms(
        self,
        descriptor_dims: Optional[Iterable[int]] = None,
        n_bins_2d: int = 100,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Compute and cache weighted histograms for RPE data."""
        if self.RPE is None:
            raise ValueError("RPE must be set before computing histograms")

        dims = tuple(descriptor_dims) if descriptor_dims is not None else self.descriptor_dims
        cache_key = f"hist_{dims}_{n_bins_2d}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        hist = {}
        forward_data, forward_weights = self.RPE.data_Forward[:2]
        hist["forward"] = self._weighted_histogram2d(
            forward_data, forward_weights, descriptor_dims=dims, n_bins_2d=n_bins_2d
        )

        backward_data, backward_weights = self.RPE.data_Backward[:2]
        hist["backward"] = self._weighted_histogram2d(
            backward_data, backward_weights, descriptor_dims=dims, n_bins_2d=n_bins_2d
        )

        if self.RPE.data_Stable is not None:
            stable_data, stable_weights = self.RPE.data_Stable[:2]
            hist["stable"] = self._weighted_histogram2d(
                stable_data, stable_weights, descriptor_dims=dims, n_bins_2d=n_bins_2d
            )

        self._cache[cache_key] = hist
        return hist

    # -------------------------
    # Plotting helpers
    # -------------------------
    def plot_q_contours(
        self,
        model,
        ax: Optional[plt.Axes] = None,
        levels: Optional[Iterable[float]] = None,
        n_bins_2d: int = 300,
        descriptor_dims: Optional[Iterable[int]] = None,
        dims_extent: Optional[Iterable[float]] = None,
        standard_value: Optional[Iterable[float]] = None,
        grid_to_plot_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot q(x) contour lines for a model."""
        dims = tuple(descriptor_dims) if descriptor_dims is not None else self.descriptor_dims
        q, X, Y = self.compute_q_model_2d(
            model,
            n_bins_2d=n_bins_2d,
            descriptor_dims=dims,
            dims_extent=dims_extent,
            standard_value=standard_value,
            cache_key=f"q_model_2d_{dims}_{n_bins_2d}_{tuple(dims_extent) if dims_extent is not None else tuple(self.dims_extent)}",
        )

        X_plot = X
        Y_plot = Y
        if grid_to_plot_transform is not None:
            if self.total_num_descriptors is None:
                raise ValueError("total_num_descriptors must be set when using grid_to_plot_transform")

            if standard_value is None:
                if self.standard_value is None:
                    standard_value = [0.0] * self.total_num_descriptors
                else:
                    standard_value = list(self.standard_value)

            points = np.tile(np.asarray(standard_value, dtype=float), (X.size, 1))
            points[:, dims[0]] = X.ravel()
            points[:, dims[1]] = Y.ravel()
            mapped = np.asarray(grid_to_plot_transform(points), dtype=float)
            X_plot = mapped[:, dims[0]].reshape(X.shape)
            Y_plot = mapped[:, dims[1]].reshape(Y.shape)

        if levels is not None:
            levels = np.sort(np.array(levels))
        if ax is None:
            _, ax = plt.subplots(1, 1)
        contour = ax.contour(X_plot, Y_plot, q, levels=levels, **kwargs)
        ax.clabel(contour, inline=1, fontsize=self.plot_settings.fontsize * 0.6)
        return ax

    def plot_q_heatmap(
        self,
        model,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot q(x) as a heatmap."""
        q, X, Y = self.compute_q_model_2d(model)
        if ax is None:
            _, ax = plt.subplots(1, 1)
        im = ax.imshow(
            q,
            origin="lower",
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            aspect="auto",
            **kwargs,
        )
        if ax.figure is not None:
            ax.figure.colorbar(im, ax=ax)
        return ax

    def plot_rpe_distribution(
        self,
        ax: Optional[plt.Axes] = None,
        n_bins_2d: int = 100,
        descriptor_dims: Optional[Iterable[int]] = None,
        cmap: str = "Blues",
        **kwargs,
    ) -> plt.Axes:
        """Plot the RPE distribution as a heatmap."""
        hist = self.compute_histograms(descriptor_dims=descriptor_dims, n_bins_2d=n_bins_2d)
        H, xedges, yedges = hist["forward"]
        X, Y = np.meshgrid(xedges, yedges)
        if ax is None:
            _, ax = plt.subplots(1, 1)
        im = ax.pcolormesh(X, Y, H.T, cmap=cmap, **kwargs)
        if ax.figure is not None:
            ax.figure.colorbar(im, ax=ax)
        return ax

    # -------------------------
    # Committor projections from RPE data
    # -------------------------
    def compute_committor_2d(
        self,
        descriptors: np.ndarray,
        weights: np.ndarray,
        shot_results: np.ndarray,
        descriptor_dims: Optional[Iterable[int]] = None,
        n_bins_2d: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute a 2D committor p_B histogram from RPE descriptors, weights, and shot results.

        The shot-result encoding is [[2,0]=AA, [1,1]=AB, [0,2]=BB] per snapshot.
        The weighted histogram of shot_results[:,1] (B-side count) divided by the
        total weighted histogram gives the per-bin committor estimate.

        Parameters
        ----------
        descriptors
            Array of shape (N, n_dims) — descriptor values for all snapshots.
        weights
            Array of shape (N,) — MBAR weights (already normalised).
        shot_results
            Array of shape (N, 2) — per-snapshot shot-result encoding.
        descriptor_dims
            Which two dimensions to project onto (x, y).  Defaults to self.descriptor_dims.
        n_bins_2d
            Number of bins along each axis.

        Returns
        -------
        p_B : np.ndarray, shape (n_bins, n_bins)
            Committor estimate; NaN where no data.
        xedges, yedges : np.ndarray
            Bin edge arrays.
        """
        dims = tuple(descriptor_dims) if descriptor_dims is not None else self.descriptor_dims
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)
        extent = self.dims_extent

        H_pB, _, _ = np.histogram2d(
            descriptors[:, dims[0]],
            descriptors[:, dims[1]],
            bins=(xedges, yedges),
            range=[[extent[0], extent[1]], [extent[2], extent[3]]],
            weights=weights * shot_results[:, 1],
        )
        H_pA, _, _ = np.histogram2d(
            descriptors[:, dims[0]],
            descriptors[:, dims[1]],
            bins=(xedges, yedges),
            range=[[extent[0], extent[1]], [extent[2], extent[3]]],
            weights=weights * shot_results[:, 0],
        )

        normalizing = H_pA.T + H_pB.T
        with np.errstate(invalid="ignore", divide="ignore"):
            p_B = np.where(normalizing > 0, H_pB.T / normalizing, np.nan)

        return p_B, xedges, yedges

    def plot_committor_rpe(
        self,
        descriptors: np.ndarray,
        weights: np.ndarray,
        shot_results: np.ndarray,
        ax: Optional[plt.Axes] = None,
        n_bins_2d: int = 100,
        descriptor_dims: Optional[Iterable[int]] = None,
        cmap: str = "Spectral",
        vmin: float = 0.0,
        vmax: float = 1.0,
        **kwargs,
    ) -> plt.Axes:
        """Plot the 2D committor p_B projection from RPE data.

        Parameters
        ----------
        descriptors, weights, shot_results
            Combined RPE arrays (e.g. from ``create_total_trainset``).
        cmap
            Colormap; "Spectral" maps 0→1 from blue (A) to red (B).
        vmin, vmax
            Colour scale limits (default 0..1).
        """
        p_B, xedges, yedges = self.compute_committor_2d(
            descriptors, weights, shot_results,
            descriptor_dims=descriptor_dims, n_bins_2d=n_bins_2d,
        )
        X, Y = np.meshgrid(xedges, yedges)
        if ax is None:
            _, ax = plt.subplots(1, 1)
        im = ax.pcolormesh(X, Y, p_B, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        if ax.figure is not None:
            ax.figure.colorbar(im, ax=ax, label=r"$p_B$")
        return ax

    def plot_logit_committor_rpe(
        self,
        descriptors: np.ndarray,
        weights: np.ndarray,
        shot_results: np.ndarray,
        ax: Optional[plt.Axes] = None,
        n_bins_2d: int = 100,
        descriptor_dims: Optional[Iterable[int]] = None,
        cmap: str = "Spectral",
        **kwargs,
    ) -> plt.Axes:
        r"""Plot the 2D logit-committor  q = log(p_B / (1 - p_B))  from RPE data.

        Bins with p_B ∈ {0, 1} or no data are shown as NaN (masked).
        """
        p_B, xedges, yedges = self.compute_committor_2d(
            descriptors, weights, shot_results,
            descriptor_dims=descriptor_dims, n_bins_2d=n_bins_2d,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            q = np.where((p_B > 0) & (p_B < 1), np.log(p_B / (1.0 - p_B)), np.nan)
        X, Y = np.meshgrid(xedges, yedges)
        if ax is None:
            _, ax = plt.subplots(1, 1)
        im = ax.pcolormesh(X, Y, q, cmap=cmap, **kwargs)
        if ax.figure is not None:
            ax.figure.colorbar(im, ax=ax, label=r"$q = \ln(p_B\,/\,(1-p_B))$")
        return ax
