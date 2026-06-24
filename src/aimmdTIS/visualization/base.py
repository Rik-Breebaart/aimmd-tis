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
        self._gradient_output_cache: Dict[int, Dict[str, object]] = {}
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
        self._gradient_output_cache.clear()

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
    
    def _scale_desc_phys_to_desc_scaled(self, descriptors_phys: np.ndarray) -> np.ndarray:
        """Scale physical descriptor values to the model's input space."""
        if self._desc_min is None or self._desc_max is None:
            return descriptors_phys
        scale = (self._desc_max - self._desc_min).astype(np.float32)
        scale = np.where(scale != 0, scale, 1.0).astype(np.float32)
        return (descriptors_phys - self._desc_min.astype(np.float32)) / scale

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
        self._gradient_output_cache.clear()

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
        n_bins_2d_tuple = (n_bins_2d, n_bins_2d) if np.shape(n_bins_2d) == () else tuple(n_bins_2d)
        cache_key = f"hist_{dims}_{n_bins_2d_tuple[0]}_{n_bins_2d_tuple[1]}"
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

    def smooth_histogram(self, H: np.ndarray, sigma: float = 1.0) -> np.ndarray:

        from scipy.ndimage import gaussian_filter

        nan_mask = np.isnan(H)
        H_filled = np.where(nan_mask, 0.0, H)
        w_mask = (~nan_mask).astype(float)
        sm_num = gaussian_filter(H_filled, sigma=sigma)
        sm_den = gaussian_filter(w_mask,   sigma=sigma)
        with np.errstate(invalid="ignore"):
            H_smooth = np.where(sm_den > 1e-3, sm_num / sm_den, np.nan)
        return H_smooth
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
        if np.shape(n_bins_2d) == ():
            n_bins_2d_tuple = (n_bins_2d, n_bins_2d)
        else:
            n_bins_2d_tuple = tuple(n_bins_2d)
        q, X, Y = self.compute_q_model_2d(
            model,
            n_bins_2d=n_bins_2d,
            descriptor_dims=dims,
            dims_extent=dims_extent,
            standard_value=standard_value,
            cache_key=f"q_model_2d_{dims}_{n_bins_2d_tuple[0]}_{n_bins_2d_tuple[1]}_{tuple(dims_extent) if dims_extent is not None else tuple(self.dims_extent)}",
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
        smooth_sigma: Optional[float] = 0.0,
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
        n_bins_2d_tuple = (n_bins_2d, n_bins_2d) if np.shape(n_bins_2d) == () else tuple(n_bins_2d)
        extent_key = tuple(self.dims_extent)
        cache_key = f"fe2d_{dims}_{n_bins_2d_tuple[0]}_{n_bins_2d_tuple[1]}_{extent_key}"
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
        if smooth_sigma is not None and smooth_sigma > 0:
            fe = self.smooth_histogram(fe, sigma=smooth_sigma)

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
        smooth_sigma: Optional[float] = 0.0,
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
        P_B = np.where(mask, H_B / (H_B + H_A), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(mask, np.log(P_B / (1 - P_B)), np.nan)
        q = q.T  # (ny, nx) for imshow with origin='lower'
        if smooth_sigma is not None and smooth_sigma > 0:
            q = self.smooth_histogram(q, sigma=smooth_sigma)
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

    def plot_q_model_projection(
        self,
        model,
        descriptor_dims: Optional[Iterable[int]] = None,
        n_bins_2d: int | Tuple[int, int] = 100,
        ax: Optional[plt.Axes] = None,
        v_min_max: Optional[Tuple[float, float]] = None,
        cmap: str = "Spectral",
        logit: bool = True,
        smooth_sigma: Optional[float] = 0.0,
    ):
        """Plot the MBAR-weighted average model committor projected onto 2-D descriptor space.

        For every 2-D spatial bin the plotted value is

        .. math::
            \\langle q(x|\\theta) \\rangle_{\\text{bin}} =
                \\frac{\\sum_{i \\in \\text{bin}} w_i\\, q_i}{\\sum_{i \\in \\text{bin}} w_i}

        where :math:`q_i = \\log(p_B / p_A)` is the logit committor (``logit=True``,
        default) or the linear committor :math:`p_B` (``logit=False``).

        Unlike :meth:`plot_q_contours` this does **not** evaluate the model on a
        grid — it bins the *actual* RPE data points using the cached model output
        from :meth:`_model_output_rpe`, so the result reflects the distribution of
        visited configurations.

        The result is cached by ``(descriptor_dims, n_bins_2d, id(model), logit)``.

        Parameters
        ----------
        model
            Trained committor model.
        descriptor_dims
            Two descriptor indices for x and y axes.
        n_bins_2d
            Histogram bins, scalar or ``(nx, ny)``.
        ax
            Axes to draw on; created if *None*.
        v_min_max
            Colour-scale limits ``(vmin, vmax)``.  Defaults to ``(-15, 15)``
            for logit and ``(0, 1)`` for linear committor.
        cmap
            Colour map.  ``"Spectral"`` works well for signed logit q (diverging),
            ``"viridis"`` for linear :math:`p_B`.
        logit
            If *True* (default) plot the logit committor :math:`q = \\ln(p_B/p_A)`.
            If *False* plot the linear committor :math:`p_B \\in [0, 1]`.

        Returns
        -------
        im
            The ``AxesImage`` returned by :func:`~matplotlib.axes.Axes.imshow`.
        """
        if self._desc is None:
            raise ValueError("load_trainset() must be called before plot_q_model_projection.")

        dims = tuple(descriptor_dims) if descriptor_dims is not None else self.descriptor_dims
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)
        if np.shape(n_bins_2d) == ():
            n_bins_2d_tuple = (n_bins_2d, n_bins_2d)
        else:
            n_bins_2d_tuple = tuple(n_bins_2d)
        extent_key = tuple(self.dims_extent)
        cache_key = f"q_model_proj_{dims}_{n_bins_2d_tuple[0]}_{n_bins_2d_tuple[1]}_{extent_key}_{id(model)}_{'logit' if logit else 'pB'}"

        if cache_key not in self._cache:
            p_B_arr, q_arr = self._model_output_rpe(model)
            values = q_arr if logit else p_B_arr

            desc_plot = self._desc_phys if self._desc_phys is not None else self._desc
            x = desc_plot[:, dims[0]]
            y = desc_plot[:, dims[1]]

            # Weighted sum of committor values per bin
            H_val, _, _ = np.histogram2d(
                x, y, bins=[xedges, yedges], weights=self._w * values,
            )
            # Total weight per bin (normaliser)
            H_norm, _, _ = np.histogram2d(
                x, y, bins=[xedges, yedges], weights=self._w,
            )
            with np.errstate(invalid="ignore"):
                mean_val = np.where(H_norm > 0, H_val / H_norm, np.nan)
            self._cache[cache_key] = mean_val

        mean_val = self._cache[cache_key]
        img = mean_val.T  # (ny, nx) for imshow with origin='lower'
        if smooth_sigma > 0:
            img = self.smooth_histogram(img, sigma=smooth_sigma)

        if ax is None:
            _, ax = plt.subplots(1, 1)

        if v_min_max is not None:
            vmin, vmax = v_min_max
        elif logit:
            vmin, vmax = -15.0, 15.0
        else:
            vmin, vmax = 0.0, 1.0

        if logit:
            norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            im = ax.imshow(
                img,
                origin="lower",
                aspect="auto",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap=cmap,
                norm=norm,
            )
        else:
            im = ax.imshow(
                img,
                origin="lower",
                aspect="auto",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
        return im

    def plot_q_model_projection_contours(
        self,
        model,
        descriptor_dims: Optional[Iterable[int]] = None,
        n_bins_2d: int | Tuple[int, int] = 100,
        ax: Optional[plt.Axes] = None,
        levels: Optional[Iterable[float]] = None,
        logit: bool = True,
        colors: str = "black",
        linewidths: float = 0.8,
        alpha: float = 0.7,
        smooth_sigma: Optional[float] = 0.0,
        linestyles: str = "solid",
        clabel: bool = False,
        clabel_fontsize: int = 7,
    ):
        """Contour lines of the MBAR-weighted model committor projection.

        Reuses the grid already cached by :meth:`plot_q_model_projection` (same
        ``descriptor_dims``, ``n_bins_2d`` and ``logit`` arguments).  If the
        cache entry is missing the histogram is computed on the fly.

        Parameters
        ----------
        model
            Trained committor model.
        descriptor_dims
            Two descriptor indices for x and y axes.
        n_bins_2d
            Must match the value used for the background heatmap.
        ax
            Axes to draw on; uses current axes if *None*.
        levels
            Contour levels.  Defaults to ``[-15,-10,-5,-3,-1,0,1,3,5,10,15]``
            for logit q and ``[0.1,0.2,0.3,0.5,0.7,0.8,0.9]`` for linear pB.
        logit
            Must match the value used for the background heatmap.
        colors, linewidths, alpha, linestyles
            Passed directly to :func:`~matplotlib.axes.Axes.contour`.
        clabel
            If *True*, inline contour labels are drawn.
        clabel_fontsize
            Font size for inline labels.

        Returns
        -------
        cs
            The :class:`~matplotlib.contour.QuadContourSet` instance.
        """
        dims = tuple(descriptor_dims) if descriptor_dims is not None else self.descriptor_dims
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)

        extent_key = tuple(self.dims_extent)
        if np.shape(n_bins_2d) == ():
            n_bins_2d_tuple = (n_bins_2d, n_bins_2d)
        else:
            n_bins_2d_tuple = tuple(n_bins_2d)
        cache_key = (
            f"q_model_proj_{dims}_{n_bins_2d_tuple[0]}_{n_bins_2d_tuple[1]}_{extent_key}"
            f"_{id(model)}_{'logit' if logit else 'pB'}"
        )

        if cache_key not in self._cache:
            # Populate the cache via the heatmap method (draw into a temp axes)
            import matplotlib
            fig_tmp, ax_tmp = plt.subplots(1, 1)
            self.plot_q_model_projection(
                model, descriptor_dims=dims, n_bins_2d=n_bins_2d, ax=ax_tmp, logit=logit
            )
            plt.close(fig_tmp)

        mean_val = self._cache[cache_key]  # (nx, ny)

        xc = (xedges[:-1] + xedges[1:]) / 2
        yc = (yedges[:-1] + yedges[1:]) / 2
        Xc, Yc = np.meshgrid(xc, yc)
        
        Z = np.ma.masked_invalid(mean_val.T)  # (ny, nx)
        if smooth_sigma > 0:
            Z = self.smooth_histogram(Z, sigma=smooth_sigma)

        if levels is None:
            if logit:
                levels = [-15, -10, -5, -3, -1, 0, 1, 3, 5, 10, 15]
            else:
                levels = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]

        if ax is None:
            ax = plt.gca()

        cs = ax.contour(
            Xc, Yc, Z,
            levels=levels,
            colors=colors,
            linewidths=linewidths,
            alpha=alpha,
            linestyles=linestyles,
        )
        if clabel:
            ax.clabel(cs, inline=True, fontsize=clabel_fontsize)
        return cs

    def plot_shot_committor_contours(
        self,
        descriptor_dims: Optional[Iterable[int]] = None,
        n_bins_2d: int | Tuple[int, int] = 100,
        ax: Optional[plt.Axes] = None,
        levels: Optional[Iterable[float]] = None,
        colors: str = "dimgray",
        linewidths: float = 0.9,
        alpha: float = 0.8,
        linestyles: str = "dashed",
        smooth_sigma: Optional[float] = 1.5,
        clabel: bool = False,
        clabel_fontsize: int = 7,
    ):
        """Contour lines of the data-estimated committor from shot results.

        Uses the same MBAR-weighted histogram computation as
        :meth:`plot_q_committor_from_shots` but draws isolines instead of a
        filled heatmap.  Optional Gaussian smoothing reduces contouring artefacts
        in sparsely populated bins.

        Parameters
        ----------
        descriptor_dims
            Two descriptor indices for x and y axes.
        n_bins_2d
            Histogram bin count, scalar or ``(nx, ny)``.
        ax
            Axes to draw on.
        levels
            Contour levels in logit :math:`q = \\ln(p_B/p_A)`.
            Default: ``[-5, -3, -1, 0, 1, 3, 5]``.
        colors, linewidths, alpha, linestyles
            Passed to :func:`~matplotlib.axes.Axes.contour`.
        smooth_sigma
            Standard deviation (in bins) for Gaussian smoothing applied before
            contouring.  Set to ``0`` to skip smoothing.
        clabel
            If *True*, inline contour labels are drawn.
        clabel_fontsize
            Font size for inline labels.

        Returns
        -------
        cs
            The :class:`~matplotlib.contour.QuadContourSet` instance.
        """
        if self._desc is None:
            raise ValueError("load_trainset() must be called before plot_shot_committor_contours.")

        dims = tuple(descriptor_dims) if descriptor_dims is not None else self.descriptor_dims
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)

        n_B = self._shot[:, 1]
        n_A = self._shot[:, 0]
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

        q_plot = q.T  # (ny, nx) to match meshgrid convention

        if smooth_sigma > 0:
            q_plot = self.smooth_histogram(q_plot, sigma=smooth_sigma)

        xc = (xedges[:-1] + xedges[1:]) / 2
        yc = (yedges[:-1] + yedges[1:]) / 2
        Xc, Yc = np.meshgrid(xc, yc)
        Z = np.ma.masked_invalid(q_plot)

        if levels is None:
            levels = [-5, -3, -1, 0, 1, 3, 5]

        if ax is None:
            ax = plt.gca()

        cs = ax.contour(
            Xc, Yc, Z,
            levels=levels,
            colors=colors,
            linewidths=linewidths,
            alpha=alpha,
            linestyles=linestyles,
        )
        if clabel:
            ax.clabel(cs, inline=True, fontsize=clabel_fontsize)
        return cs

    def plot_rpe_free_energy_contours(
        self,
        descriptor_dims: Optional[Iterable[int]] = None,
        n_bins_2d: int | Tuple[int, int] = 100,
        ax: Optional[plt.Axes] = None,
        offset: bool = True,
        levels: Optional[Iterable[float]] = None,
        colors: str = "black",
        linewidths: float = 0.8,
        alpha: float = 0.7,
        linestyles: str = "solid",
        smooth_sigma: float = 1.5,
        clabel: bool = False,
        clabel_fontsize: int = 7,
    ):
        """Contour lines of the MBAR-weighted free-energy surface.

        Uses the same MBAR-weighted histogram computation as
        :meth:`plot_rpe_free_energy` but draws isolines instead of a filled heatmap.

        Parameters
        ----------
        descriptor_dims
            Two descriptor indices for x and y axes.
        n_bins_2d
            Histogram bin count, scalar or ``(nx, ny)``.
        ax
            Axes to draw on; uses current axes if *None*.
        levels
            Contour levels in free energy units (default: ``[0, 1, 2, 3, 4, 5]``).
        colors, linewidths, alpha, linestyles
            Passed to :func:`~matplotlib.axes.Axes.contour`.
        clabel
            If *True*, inline contour labels are drawn.
        clabel_fontsize
            Font size for inline labels.

        Returns
        -------
        cs
            The :class:`~matplotlib.contour.QuadContourSet` instance.
        """
        if self._desc is None:
            raise ValueError("load_trainset() must be called before plot_rpe_free_energy_contours.")

        dims = tuple(descriptor_dims) if descriptor_dims is not None else self.descriptor_dims
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)

        extent_key = tuple(self.dims_extent)
        if np.shape(n_bins_2d) == ():
            n_bins_2d_tuple = (n_bins_2d, n_bins_2d)
        else:
            n_bins_2d_tuple = tuple(n_bins_2d)
        cache_key = f"fe2d_{dims}_{n_bins_2d_tuple[0]}_{n_bins_2d_tuple[1]}_{extent_key}"

        if cache_key not in self._cache:
            # Populate the cache via the heatmap method (draw into a temp axes)
            import matplotlib
            fig_tmp, ax_tmp = plt.subplots(1, 1)
            self.plot_rpe_free_energy(descriptor_dims=dims, n_bins_2d=n_bins_2d, ax=ax_tmp, offset=offset)
            plt.close(fig_tmp)

        H = self._cache[cache_key]

        with np.errstate(divide="ignore"):
            fe = np.where(H > 0, -np.log(H), np.nan)
        if offset:
            fe -= np.nanmin(fe)

        xc = (xedges[:-1] + xedges[1:]) / 2
        yc = (yedges[:-1] + yedges[1:]) / 2
        Xc, Yc = np.meshgrid(xc, yc)
        
        Z = np.ma.masked_invalid(fe.T)  # (ny, nx)
        if smooth_sigma > 0:
            Z = self.smooth_histogram(Z, sigma=smooth_sigma)


        if levels is None:
            levels = [0, 1, 2, 3, 4, 5]

        if ax is None:
            ax = plt.gca()

        cs = ax.contour(
            Xc, Yc, Z,
            levels=levels,
            colors=colors,
            linewidths=linewidths,
            alpha=alpha,
            linestyles=linestyles,
        )
        if clabel:
            ax.clabel(cs, inline=True, fontsize=clabel_fontsize)
        return cs