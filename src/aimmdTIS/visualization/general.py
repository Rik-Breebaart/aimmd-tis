"""General visualization utilities for non-toy AIMMD-TIS systems."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .base import BaseVisualizer
from .model_analysis import ModelAnalysisMixin


class SystemVisualizer(BaseVisualizer, ModelAnalysisMixin):
    """General-purpose visualizer for descriptor-space analysis.

    This class provides reusable plotting helpers for real molecular systems
    (for example T4L), where toy PES-specific overlays are not available.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        resolution: int = 501,
        descriptor_dims: Iterable[int] = (0, 1),
        total_num_descriptors: Optional[int] = None,
        dims_extent: Optional[Iterable[float]] = None,
        standard_value: Optional[Iterable[float]] = None,
        descriptor_labels: Optional[Sequence[str]] = None,
        desc_min: Optional[Iterable[float]] = None,
        desc_max: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__(
            temperature=temperature,
            resolution=resolution,
            descriptor_dims=descriptor_dims,
            total_num_descriptors=total_num_descriptors,
            dims_extent=dims_extent,
            standard_value=standard_value,
            desc_min=desc_min,
            desc_max=desc_max,
        )
        self.descriptor_labels = list(descriptor_labels) if descriptor_labels is not None else None

    def plot_trajectories(
        self,
        trajectories: Sequence[np.ndarray],
        ax: Optional[plt.Axes] = None,
        alpha: float = 0.7,
        linewidth: float = 1.0,
        color: str = "tab:blue",
    ) -> plt.Axes:
        """Plot one or more trajectories in descriptor space."""
        dims = self.descriptor_dims
        if ax is None:
            _, ax = plt.subplots(1, 1)

        for traj in trajectories:
            if traj is None or len(traj) == 0:
                continue
            arr = np.asarray(traj)
            ax.plot(arr[:, dims[0]], arr[:, dims[1]], alpha=alpha, linewidth=linewidth, color=color)

        return ax

    def plot_state_rectangles(
        self,
        state_definitions: Dict[str, Dict[str, object]],
        ax: Optional[plt.Axes] = None,
        edgecolor: str = "black",
        alpha: float = 0.18,
    ) -> plt.Axes:
        """Plot rectangular state bounds from a TPS setup state_definitions dict.

        Expected shape per state:
            state_definitions[state_name]["bounds"] = [(xmin, xmax), (ymin, ymax)]
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)

        palette = ["tab:red", "tab:green", "tab:blue", "tab:orange"]
        for idx, (state_name, state_def) in enumerate(state_definitions.items()):
            bounds = state_def.get("bounds")
            if bounds is None or len(bounds) < 2:
                continue
            x_rng = bounds[0]
            y_rng = bounds[1]
            rect = plt.Rectangle(
                (x_rng[0], y_rng[0]),
                x_rng[1] - x_rng[0],
                y_rng[1] - y_rng[0],
                edgecolor=edgecolor,
                facecolor=palette[idx % len(palette)],
                alpha=alpha,
                linestyle="-",
                label=f"{state_name} bounds",
            )
            ax.add_patch(rect)

        return ax

    def plot_interface_line(
        self,
        interface_value: float,
        ax: Optional[plt.Axes] = None,
        axis: str = "x",
        **kwargs,
    ) -> plt.Axes:
        """Plot an interface as a vertical or horizontal line."""
        if ax is None:
            _, ax = plt.subplots(1, 1)

        line_kwargs = {"color": "black", "linestyle": "--", "linewidth": 1.5}
        line_kwargs.update(kwargs)

        if axis == "x":
            ax.axvline(interface_value, **line_kwargs)
        elif axis == "y":
            ax.axhline(interface_value, **line_kwargs)
        else:
            raise ValueError("axis must be 'x' or 'y'")

        return ax
