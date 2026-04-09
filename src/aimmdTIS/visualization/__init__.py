"""Visualization utilities for AIMMD-TIS.

This module provides a clean separation between:
- BaseVisualizer: system-independent plotting and data utilities.
- ToyVisualizer: toy-system-specific extensions (PES, theory overlays).
"""

from .base import BaseVisualizer
from .toy import ToyVisualizer

__all__ = ["BaseVisualizer", "ToyVisualizer"]
