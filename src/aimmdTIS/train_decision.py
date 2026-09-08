"""Training-decision policies owned by AIMMD-TIS."""

import numpy as np


EE_SCALE_CUTOFF_DEFAULTS = {
    "efficiency_factor": False,
    "lr_0": 1e-3,
    "lr_min": 1e-4,
    "epochs_per_train": 1,
    "interval": 5,
    "window": 100,
    "batch_size": None,
    "cut_off": 30,
    "cut_off_scaling": 0.99,
    "max_clipping_norm": None,
    "smoothness_penalty_weight": 0.05,
    "l1_regularization": 0.001,
    "stochastic_gate_regularization": 0.0,
}


EE_SCALE_CUTOFF_DOC = """
    Controls training by multiplying lr with the expected-efficiency factor and
    reduces lr when model q-values exceed the configured cutoff.

    ee_params contains the standard expected-efficiency parameters plus:
        efficiency_factor - whether to recalculate lr from expected efficiency
        cut_off - absolute q limit that triggers lr scaling
        cut_off_scaling - multiplicative lr scaling factor
"""


def train_decision_ee_scale_cutoff(model, trainset):
    """Train periodically while the cutoff-adjusted learning rate is viable."""
    params = model.ee_params
    if params["cut_off_scaling"] <= 0:
        raise ValueError("cut_off_scaling must be positive")

    if params["efficiency_factor"]:
        model.lr = params["lr_0"] * model.train_expected_efficiency_factor(
            trainset, params["window"]
        )

    descriptors = getattr(trainset, "descriptors", None)
    if descriptors is not None and len(descriptors) > 0:
        q_model = model._log_prob(descriptors, batch_size=None)
        if np.min(q_model) < -params["cut_off"] or np.max(q_model) > params["cut_off"]:
            model.lr *= params["cut_off_scaling"]
        else:
            model.lr = min(model.lr / params["cut_off_scaling"], params["lr_0"])

    train = (
        model._count_train_hook % params["interval"] == 0
        and model.lr >= params["lr_min"]
    )
    return train, model.lr, params["epochs_per_train"], params["batch_size"]