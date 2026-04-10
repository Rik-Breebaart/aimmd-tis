"""Training utilities for AIMMD-TIS.

This package groups committor training loops, snapshot losses,
and dataset helper utilities in one discoverable location.
"""

from .loops import (
    train_one_stage,
    _apply_stage_hparams,
    _make_optimizer,
    TorchRCModelLite,
)
from .losses import (
    q_histogram_plot,
    snapshot_loss_original,
    snapshot_loss_smoothness,
    snapshot_loss_normalized_q,
    snapshot_lnP,
    snapshot_loss_low_q_scaled,
    snapshot_loss_sqrt_rho_weight,
)
from .dataset import (
    train_test_split,
    create_train_test_split,
    SyntheticDataGenerator,
    q_normalized_trainset,
)
from .diagnostics import (
    save_training_results,
    load_training_results,
    summarize_training,
    plot_training_diagnostics,
)

# Backward-compatible aliases that resolve to the single canonical loop.
train_function = train_one_stage
pretraining_train_function = train_one_stage
combined_train_function = train_one_stage
combined_train_function_l1_regularized = train_one_stage

__all__ = [
    "train_one_stage",
    "_apply_stage_hparams",
    "_make_optimizer",
    "TorchRCModelLite",
    "train_function",
    "pretraining_train_function",
    "combined_train_function",
    "combined_train_function_l1_regularized",
    "q_histogram_plot",
    "snapshot_loss_original",
    "snapshot_loss_smoothness",
    "snapshot_loss_normalized_q",
    "snapshot_lnP",
    "snapshot_loss_low_q_scaled",
    "snapshot_loss_sqrt_rho_weight",
    "train_test_split",
    "create_train_test_split",
    "SyntheticDataGenerator",
    "q_normalized_trainset",
    "save_training_results",
    "load_training_results",
    "summarize_training",
    "plot_training_diagnostics",
]
