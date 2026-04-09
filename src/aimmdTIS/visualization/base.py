"""System-independent visualization base class.

Designed to reduce repeated computations (grids, histograms) and
provide a clean structure for adding future plotting methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

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
        q = model.log_prob(torch.as_tensor(coord, device=model._device), use_transform=False)
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
        **kwargs,
    ) -> plt.Axes:
        """Plot q(x) contour lines for a model."""
        q, X, Y = self.compute_q_model_2d(model)
        if ax is None:
            _, ax = plt.subplots(1, 1)
        contour = ax.contour(X, Y, q, levels=levels, **kwargs)
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
