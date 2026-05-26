"""System-independent visualization base class.

Designed to reduce repeated computations (grids, histograms) and
provide a clean structure for adding future plotting methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import matplotlib
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
        desc_min: Optional[Iterable[float]] = None,
        desc_max: Optional[Iterable[float]] = None,
    ) -> None:
        self.temperature = temperature
        self.beta = 1.0 / temperature
        self.resolution = resolution
        self.descriptor_dims = tuple(descriptor_dims)
        self.total_num_descriptors = total_num_descriptors
        self.dims_extent = list(dims_extent) if dims_extent is not None else [-10, 10, -10, 10]
        self.standard_value = list(standard_value) if standard_value is not None else None
        # Descriptor min/max scaling: stored descriptors are in scaled [0,1] space;
        # physical values are recovered as  desc_phys = desc_scaled * (max - min) + min.
        self._desc_min: Optional[np.ndarray] = (
            np.asarray(desc_min, dtype=np.float64) if desc_min is not None else None
        )
        self._desc_max: Optional[np.ndarray] = (
            np.asarray(desc_max, dtype=np.float64) if desc_max is not None else None
        )

        self.plot_settings = PlotSettings()
        self._cache: Dict[str, object] = {}
        self._model_output_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.RPE = None
        # Flat-array trainset (new interface, set via load_trainset)
        self._desc: Optional[np.ndarray] = None       # scaled space (model input)
        self._desc_phys: Optional[np.ndarray] = None  # physical space (visualization)
        self._w: Optional[np.ndarray] = None
        self._shot: Optional[np.ndarray] = None

    # -------------------------
    # Data management
    # -------------------------
    def set_rpe(self, rpe) -> None:
        """Attach RPE data container (legacy interface)."""
        self.RPE = rpe

    def load_trainset(self, trainset: dict) -> None:
        """Load a flat-array trainset dict.

        Expected keys
        -------------
        descriptors : array-like, shape (N, n_desc)
        weights     : array-like, shape (N,)       -- MBAR weights (min==1)
        shot_results: array-like, shape (N, 2)     -- [n_A, n_B] shot counts
        """
        self._desc  = np.asarray(trainset["descriptors"],   dtype=np.float32)
        self._w     = np.asarray(trainset["weights"],       dtype=np.float64)
        self._shot  = np.asarray(trainset["shot_results"],  dtype=np.float64)
        self._make_desc_phys()
        # Invalidate any cached model output — data has changed.
        self._model_output_cache.clear()

    def _make_desc_phys(self) -> None:
        """Compute ``_desc_phys`` from ``_desc`` using the stored min/max scaling.

        If no scaling is set, ``_desc_phys`` is the same object as ``_desc``
        (no copy).
        """
        if self._desc is None:
            self._desc_phys = None
            return
        if self._desc_min is None or self._desc_max is None:
            self._desc_phys = self._desc
            return
        scale = (self._desc_max - self._desc_min).astype(np.float32)
        # Guard against zero-range dimensions (e.g. constant descriptor).
        scale = np.where(scale != 0, scale, 1.0).astype(np.float32)
        self._desc_phys = self._desc * scale + self._desc_min.astype(np.float32)

    def set_descriptor_scaling(
        self,
        desc_min: Iterable[float],
        desc_max: Iterable[float],
    ) -> None:
        """Set or update the min/max scaling used to recover physical descriptor values.

        Physical value = scaled_value * (max - min) + min.

        Clears the histogram cache because bin edges change.
        """
        self._desc_min = np.asarray(desc_min, dtype=np.float64)
        self._desc_max = np.asarray(desc_max, dtype=np.float64)
        self._make_desc_phys()
        self._cache.clear()  # invalidate: physical-space bin edges have changed

    def clear_cache(self) -> None:
        """Clear all cached grids, histograms, and model outputs."""
        self._cache.clear()
        self._model_output_cache.clear()

    def _model_output_rpe(self, model) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(p_B, q)`` for all stored RPE frames.

        The result is cached by ``id(model)``.  Calling this method a second
        time for the same model object is free; the forward pass is only run
        once per (model, trainset) combination.

        Returns
        -------
        p_B : ndarray, shape (N,)
            Predicted committor values.
        q : ndarray, shape (N,)
            Log-odds (q = log(p_B / p_A)).
        """
        if self._desc is None:
            raise ValueError("load_trainset() must be called before model evaluation.")
        key = id(model)
        if key not in self._model_output_cache:
            desc_t = torch.as_tensor(self._desc)
            with torch.no_grad():
                raw = model.log_prob(desc_t, use_transform=False)
            q = raw[:, 0] if raw.ndim == 2 else raw
            p_B = 1.0 / (1.0 + np.exp(-q))
            self._model_output_cache[key] = (p_B, q)
        return self._model_output_cache[key]

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
        """Create coordinates for 2D model evaluation.

        The returned *xedges* / *yedges* are in **physical** space (for display).
        The *coord* array that is fed to the model is in **scaled** space, obtained
        by inverting the min/max transform when scaling parameters are available.
        """
        dims = tuple(descriptor_dims) if descriptor_dims is not None else self.descriptor_dims
        # Physical-space edges — used for display (meshgrid, axis labels).
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d, dims_extent=dims_extent)

        # Scaled-space edges — what the model actually sees.
        if self._desc_min is not None and self._desc_max is not None:
            sx = self._desc_max[dims[0]] - self._desc_min[dims[0]]
            sy = self._desc_max[dims[1]] - self._desc_min[dims[1]]
            sx = sx if sx != 0 else 1.0
            sy = sy if sy != 0 else 1.0
            xedges_model = (xedges - self._desc_min[dims[0]]) / sx
            yedges_model = (yedges - self._desc_min[dims[1]]) / sy
        else:
            xedges_model, yedges_model = xedges, yedges

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
        for yv in yedges_model:
            for xv in xedges_model:
                point = list(standard_value)
                point[dims[0]] = xv
                point[dims[1]] = yv
                coord.append(point)
        # coord is in scaled space (model input); edges are in physical space (display).
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
        """Plot the RPE distribution as a heatmap (legacy interface, needs set_rpe)."""
        hist = self.compute_histograms(descriptor_dims=descriptor_dims, n_bins_2d=n_bins_2d)
        H, xedges, yedges = hist["forward"]
        X, Y = np.meshgrid(xedges, yedges)
        if ax is None:
            _, ax = plt.subplots(1, 1)
        im = ax.pcolormesh(X, Y, H.T, cmap=cmap, **kwargs)
        if ax.figure is not None:
            ax.figure.colorbar(im, ax=ax)
        return ax

    # ------------------------------------------------------------------
    # Flat-trainset plotting (requires load_trainset to be called first)
    # ------------------------------------------------------------------

    def plot_rpe_free_energy(
        self,
        descriptor_dims: Optional[Iterable[int]] = None,
        n_bins_2d: int | Tuple[int, int] = 100,
        ax: Optional[plt.Axes] = None,
        offset: bool = True,
        v_min_max: Optional[Tuple[float, float]] = None,
        cmap: str = "Blues_r",
    ):
        """Plot the MBAR-weighted free-energy surface $\\Delta F = -\\ln\\rho$.

        Uses the flat trainset loaded via :meth:`load_trainset`.  The
        histogram is cached under a key derived from *descriptor_dims* and
        *n_bins_2d* so repeated calls with the same parameters are cheap.

        Parameters
        ----------
        descriptor_dims
            Two descriptor indices for x and y axes.  Defaults to
            ``self.descriptor_dims``.
        n_bins_2d
            Number of histogram bins, scalar or ``(nx, ny)``.
        ax
            Axes to draw on.  A new figure is created if *None*.
        offset
            If *True* (default) shift so minimum $\\Delta F = 0$.
        v_min_max
            Colour scale limits ``(vmin, vmax)``.
        cmap
            Colour map.

        Returns
        -------
        im
            The ``AxesImage`` returned by :func:`~matplotlib.axes.Axes.imshow`.
        """
        if self._desc is None:
            raise ValueError("load_trainset() must be called before plot_rpe_free_energy.")
        dims = tuple(descriptor_dims) if descriptor_dims is not None else self.descriptor_dims
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)

        extent_key = tuple(self.dims_extent)
        cache_key = f"fe2d_{dims}_{np.shape(n_bins_2d)}_{extent_key}"
        if cache_key not in self._cache:
            # Use physical-space descriptors so axes reflect real units.
            desc_plot = self._desc_phys if self._desc_phys is not None else self._desc
            H, _, _ = np.histogram2d(
                desc_plot[:, dims[0]],
                desc_plot[:, dims[1]],
                bins=[xedges, yedges],
                weights=self._w,
                density=True,
            )
            self._cache[cache_key] = H
        H = self._cache[cache_key]

        with np.errstate(divide="ignore"):
            fe = np.where(H > 0, -np.log(H), np.nan)
        if offset:
            fe -= np.nanmin(fe)
        fe = fe.T  # (ny, nx) for imshow with origin='lower'

        if ax is None:
            _, ax = plt.subplots(1, 1)
        vmin = v_min_max[0] if v_min_max is not None else None
        vmax = v_min_max[1] if v_min_max is not None else None
        im = ax.imshow(
            fe,
            origin="lower",
            aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        return im

    def plot_q_committor_from_shots(
        self,
        descriptor_dims: Optional[Iterable[int]] = None,
        n_bins_2d: int | Tuple[int, int] = 100,
        ax: Optional[plt.Axes] = None,
        v_min_max: Optional[Tuple[float, float]] = None,
        cmap: str = "Spectral",
    ):
        """Plot the data-estimated committor $q = \\ln(p_B/p_A)$ from shot results.

        The committor is estimated as the MBAR-weighted ratio of B-shots to
        A-shots accumulated in each 2-D bin — no model evaluation required.

        Parameters
        ----------
        descriptor_dims
            Two descriptor indices for x and y axes.
        n_bins_2d
            Histogram bins, scalar or ``(nx, ny)``.
        ax
            Axes to draw on.
        v_min_max
            Colour scale limits ``(vmin, vmax)`` for q.  Defaults to
            ``(-10, 10)``.
        cmap
            Colour map.

        Returns
        -------
        im
            The ``AxesImage`` returned by :func:`~matplotlib.axes.Axes.imshow`.
        """
        if self._desc is None:
            raise ValueError("load_trainset() must be called before plot_q_committor_from_shots.")
        dims = tuple(descriptor_dims) if descriptor_dims is not None else self.descriptor_dims
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)

        n_B = self._shot[:, 1]
        n_A = self._shot[:, 0]
        # Use physical-space descriptors so axes reflect real units.
        desc_plot = self._desc_phys if self._desc_phys is not None else self._desc
        H_B, _, _ = np.histogram2d(
            desc_plot[:, dims[0]], desc_plot[:, dims[1]],
            bins=[xedges, yedges], weights=self._w * n_B, density=False,
        )
        H_A, _, _ = np.histogram2d(
            desc_plot[:, dims[0]], desc_plot[:, dims[1]],
            bins=[xedges, yedges], weights=self._w * n_A, density=False,
        )
        mask = (H_B + H_A) > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(
                mask,
                np.log(np.where(H_B > 0, H_B, 1e-20)) - np.log(np.where(H_A > 0, H_A, 1e-20)),
                np.nan,
            )
        q = q.T  # (ny, nx) for imshow with origin='lower'

        if ax is None:
            _, ax = plt.subplots(1, 1)
        vmin, vmax = (v_min_max[0], v_min_max[1]) if v_min_max is not None else (-10, 10)
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im = ax.imshow(
            q,
            origin="lower",
            aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap=cmap,
            norm=norm,
        )
        return im
