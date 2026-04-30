"""Toy-system visualization utilities.

Extends BaseVisualizer with PES-dependent plotting and theory overlays.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import openpathsampling as paths

from .base import BaseVisualizer
from ..Tools import create_discrete_cmap
from ..Tools import CallableVolume


class ToyVisualizer(BaseVisualizer):
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
