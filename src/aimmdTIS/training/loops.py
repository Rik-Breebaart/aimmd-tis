from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TorchRCModelLite:
    """CPU state_dict wrapper that is safe to store in AIMMD storage shelves."""

    def __init__(self, state_dict, meta=None):
        self.state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
        self.meta = dict(meta or {})

    def object_for_pickle(self, group, **kwargs):
        return self

    def complete_from_h5py_group(self, group):
        return self


def _set_lr(optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _make_optimizer(opt_cfg: Dict[str, Any], params):
    typ = opt_cfg.get("type", "adamw").lower()
    lr = float(opt_cfg.get("lr", 1e-4))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.95]))
    eps = float(opt_cfg.get("eps", 1e-8))
    wd = float(opt_cfg.get("weight_decay", 1e-4))
    if typ == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)


def _apply_stage_hparams(model, stage_cfg: Dict[str, Any], ee_params: Optional[Dict[str, Any]] = None):
    """Apply stage-specific regularization/clipping values to model.ee_params."""
    if ee_params is not None:
        smoothness = float(
            stage_cfg.get("smoothness_penalty_weight", ee_params.get("smoothness_penalty_weight", 0.0))
        )
        l1 = float(stage_cfg.get("l1_regularization", ee_params.get("l1_regularization", 0.0)))
        clip = stage_cfg.get("max_clipping_norm", ee_params.get("max_clipping_norm", None))
    else:
        smoothness = float(
            stage_cfg.get("smoothness_penalty_weight", model.ee_params.get("smoothness_penalty_weight", 0.0))
        )
        l1 = float(stage_cfg.get("l1_regularization", model.ee_params.get("l1_regularization", 0.0)))
        clip = stage_cfg.get("max_clipping_norm", model.ee_params.get("max_clipping_norm", None))

    model.ee_params["smoothness_penalty_weight"] = smoothness
    model.ee_params["l1_regularization"] = l1
    model.ee_params["max_clipping_norm"] = None if clip is None else float(clip)
    return model


def train_one_stage(
    aimmd_store,
    model,
    trainset,
    testset,
    stage_name: str,
    epochs: int,
    batch_size: int,
    base_lr: float,
    optimizer_cfg: Dict[str, Any],
    plateau_patience: int = 20,
    plateau_factor: float = 0.5,
    min_lr: float = 1e-5,
    warmup_epochs: int = 0,
    warmup_init: float = 1e-6,
    normalization: bool = False,
    early_stop_patience: int = 100,
    nan_rescue: bool = True,
    rescue_shrink: float = 0.2,
    clip_warmup_epochs: int = 0,
    clip_init: float = None,
):
    """Single canonical training loop for AIMMD-TIS committor model stages."""
    model.optimizer = _make_optimizer({**optimizer_cfg, "lr": base_lr}, model.nnet.parameters())
    scheduler = ReduceLROnPlateau(
        model.optimizer,
        mode="min",
        patience=int(plateau_patience),
        factor=float(plateau_factor),
        min_lr=float(min_lr),
    )

    train_total = []
    test_total = []
    train_model = []
    test_model = []
    train_smooth = []
    test_smooth = []
    train_l1 = []
    test_l1 = []
    lr_log = []

    best_loss = None
    best_state = None
    no_improve = 0
    ee_params_init = deepcopy(model.ee_params)

    for epoch in range(1, int(epochs) + 1):
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            lr_now = warmup_init + (base_lr - warmup_init) * (epoch / warmup_epochs)
            _set_lr(model.optimizer, lr_now)
        else:
            _set_lr(model.optimizer, base_lr)

        if clip_warmup_epochs > 0 and epoch <= clip_warmup_epochs:
            clipping = clip_init
        else:
            clipping = ee_params_init.get("max_clipping_norm", None)
        model.ee_params["max_clipping_norm"] = clipping

        train_losses = model.train_epoch_smoothness(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            normalization=normalization,
        )
        test_losses = model.test_loss_smoothness(
            testset,
            batch_size=batch_size,
            normalization=normalization,
        )

        if nan_rescue and (
            not np.isfinite(train_losses["total_loss"]) or not np.isfinite(test_losses["total_loss"])
        ):
            print(f"[{stage_name}] NaN/Inf at epoch {epoch}. Restoring best state and reducing LR.")
            if best_state is not None:
                model.nnet.load_state_dict(best_state)
                for param in model.nnet.parameters():
                    param.grad = None
            base_lr = max(float(min_lr), float(base_lr) * float(rescue_shrink))
            model.optimizer = _make_optimizer({**optimizer_cfg, "lr": base_lr}, model.nnet.parameters())
            no_improve += 1
            continue

        train_total.append(train_losses["total_loss"])
        test_total.append(test_losses["total_loss"])
        train_model.append(train_losses.get("model_loss", np.nan))
        test_model.append(test_losses.get("model_loss", np.nan))
        train_smooth.append(train_losses.get("smoothness_loss", np.nan))
        test_smooth.append(test_losses.get("smoothness_loss", np.nan))
        train_l1.append(train_losses.get("l1_regularization", np.nan))
        test_l1.append(test_losses.get("l1_regularization", np.nan))
        lr_log.append(model.optimizer.param_groups[0]["lr"])

        print(
            f"[{stage_name}] Epoch {epoch}/{epochs} | "
            f"Train={train_total[-1]:.4e} Test={test_total[-1]:.4e} "
            f"LR={lr_log[-1]:.2e} Batch={batch_size}"
        )
        aimmd_store.rcmodels[f"{stage_name}_state_dict_most_recent"] = TorchRCModelLite(
            model.nnet.state_dict(),
            meta={"stage": stage_name, "epoch": epoch, "lr": float(lr_log[-1])},
        )
        aimmd_store.rcmodels[f"{stage_name}_model_most_recent"] = model

        current_val = float(test_losses["total_loss"])
        if best_loss is None or current_val < best_loss:
            best_loss = current_val
            best_state = deepcopy(model.nnet.state_dict())
            no_improve = 0
            aimmd_store.rcmodels[f"{stage_name}_state_dict_best"] = TorchRCModelLite(
                best_state,
                meta={"epoch": epoch, "val_loss": best_loss},
            )
            aimmd_store.rcmodels[f"{stage_name}_model_best"] = model

        else:
            no_improve += 1

        scheduler.step(current_val)

        if no_improve >= int(early_stop_patience):
            print(f"[{stage_name}] Early stopping at epoch {epoch} (patience {early_stop_patience}).")
            break

    if best_state is not None:
        model.nnet.load_state_dict(best_state)
        for param in model.nnet.parameters():
            param.grad = None
        aimmd_store.rcmodels[f"most_recent"] = model


    return {
        "train_total": train_total,
        "test_total": test_total,
        "train_model": train_model,
        "test_model": test_model,
        "train_smooth": train_smooth,
        "test_smooth": test_smooth,
        "train_l1": train_l1,
        "test_l1": test_l1,
        "lr": lr_log,
        "best_val": best_loss,
    }
