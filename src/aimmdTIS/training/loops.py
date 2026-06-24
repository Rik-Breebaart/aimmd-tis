from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from aimmd.base import Properties


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
    typ = opt_cfg.get("type", "Adamw").lower()
    lr = float(opt_cfg.get("lr", 1e-4))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.95]))
    eps = float(opt_cfg.get("eps", 1e-8))
    wd = float(opt_cfg.get("weight_decay", 1e-4))
    if typ == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)

    elif typ == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    elif typ == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    raise ValueError(f"Unknown optimizer type: {typ}. Choose from either 'Adam', 'Adamw', or 'SGD'.")


def _epoch_batch_diagnostics(
    model,
    trainset,
    batch_size: int,
    max_batches: int,
    tail_quantile: float,
    normalization: bool=True,
):
    """Compute inexpensive diagnostics on a subset of train batches.

    Returns the batch with the largest weighted model loss among sampled batches,
    and how much of that batch belongs to the top-weight tail.
    """
    max_batches = int(max_batches)
    if max_batches <= 0:
        return {
            "sampled_batches": 0,
            "worst_batch_loss": np.nan,
            "worst_batch_top_weight_frac": np.nan,
            "worst_batch_weight_sum": np.nan,
            "tail_weight_threshold": np.nan,
        }

    q = float(np.clip(tail_quantile, 0.0, 100.0))
    tail_threshold = float(np.percentile(trainset.weights, q))
    worst_batch_loss = -np.inf
    worst_batch_top_weight_frac = np.nan
    worst_batch_weight_sum = np.nan
    sampled_batches = 0

    was_training = model.nnet.training
    model.nnet.eval()
    full_shot_counts = torch.sum(
        torch.as_tensor(trainset.shot_results, device=model._device, dtype=torch.float64),
        dim=-1
    )
    full_weights = torch.as_tensor(trainset.weights, device=model._device, dtype=torch.float64)

    effective_mass_full_mean = float(torch.mean(full_weights * full_shot_counts))
    try:
        with torch.no_grad():
            for batch_idx, target in enumerate(trainset.iter_batch(batch_size, True)):
                if batch_idx >= max_batches:
                    break

                targ = {
                    key: torch.as_tensor(val, device=model._device, dtype=model._dtype)
                    for key, val in target.items()
                }
                targ[Properties.q] = model.nnet(targ[Properties.descriptors])
                batch_loss = float(model.loss(targ).detach().cpu().item())

                if normalization:
                    shot_counts = torch.sum(targ[Properties.shot_results], dim=-1).to(torch.float64)
                    weights64 = targ[Properties.weights].to(torch.float64)
                    
                    effective_weight_batch = torch.sum(shot_counts * weights64)
                    batch_norm = effective_weight_batch/effective_mass_full_mean
                    batch_norm = batch_norm.to(targ[Properties.q].dtype)
                    batch_loss = batch_loss / batch_norm
                else:
                    batch_loss = batch_loss
                    batch_norm = None


                w = np.asarray(target[Properties.weights])
                top_frac = float(np.mean(w >= tail_threshold)) if w.size else np.nan
                w_sum = float(np.sum(w)) if w.size else np.nan

                sampled_batches += 1
                if batch_loss > worst_batch_loss:
                    worst_batch_loss = batch_loss
                    worst_batch_top_weight_frac = top_frac
                    worst_batch_weight_sum = w_sum
    finally:
        if was_training:
            model.nnet.train()

    if sampled_batches == 0:
        worst_batch_loss = np.nan

    return {
        "sampled_batches": sampled_batches,
        "worst_batch_loss": worst_batch_loss,
        "worst_batch_top_weight_frac": worst_batch_top_weight_frac,
        "worst_batch_weight_sum": worst_batch_weight_sum,
        "tail_weight_threshold": tail_threshold,
    }


def _flatten_grads_like_params(grads, params):
    chunks = []
    for grad, param in zip(grads, params):
        if grad is None:
            chunks.append(torch.zeros_like(param, device=param.device).reshape(-1))
        else:
            chunks.append(grad.detach().reshape(-1))
    if not chunks:
        return None
    return torch.cat(chunks)


def epoch_projected_gradient_contributions(
    model,
    trainset,
    batch_size: int,
    max_batches: int = 2,
    normalization: bool = True,
):
    """Estimate projected per-term gradient contributions on sampled batches.

    For each sampled batch this computes:
    C_i = (g_i dot g_total) / ||g_total||^2,
    where i in {model, smoothness, l1}.
    """
    max_batches = int(max_batches)
    if max_batches <= 0:
        return {
            "sampled_batches": 0,
            "proj_model_mean": np.nan,
            "proj_smooth_mean": np.nan,
            "proj_l1_mean": np.nan,
            "proj_sum_mean": np.nan,
            "total_grad_l2_mean": np.nan,
        }

    params = [p for p in model.nnet.parameters() if p.requires_grad]
    if len(params) == 0:
        return {
            "sampled_batches": 0,
            "proj_model_mean": np.nan,
            "proj_smooth_mean": np.nan,
            "proj_l1_mean": np.nan,
            "proj_sum_mean": np.nan,
            "total_grad_l2_mean": np.nan,
        }

    proj_model = []
    proj_smooth = []
    proj_l1 = []
    total_grad_l2 = []

    smooth_w = float(model.ee_params.get("smoothness_penalty_weight", 0.0) or 0.0)
    l1_w = float(model.ee_params.get("l1_regularization", 0.0) or 0.0)
    full_shot_counts = torch.sum(
    torch.as_tensor(trainset.shot_results, device=model._device, dtype=torch.float64),
    dim=-1
    )
    full_weights = torch.as_tensor(trainset.weights, device=model._device, dtype=torch.float64)

    effective_mass_full_mean = torch.mean(full_weights * full_shot_counts)
    was_training = model.nnet.training
    model.nnet.eval()

    try:
        for batch_idx, target in enumerate(trainset.iter_batch(batch_size, True)):
            if batch_idx >= max_batches:
                break

            targ = {
                key: torch.as_tensor(val, device=model._device, dtype=model._dtype)
                for key, val in target.items()
            }
            targ[Properties.descriptors].requires_grad = True
            q_pred = model.nnet(targ[Properties.descriptors])
            targ[Properties.q] = q_pred

            model_term = model.loss(targ)
            if normalization:
                shot_counts = torch.sum(targ[Properties.shot_results], dim=-1).to(torch.float64)
                weights64 = targ[Properties.weights].to(torch.float64)
                effective_weight_batch = torch.sum(shot_counts * weights64)
                batch_norm = effective_weight_batch/effective_mass_full_mean
                batch_norm = batch_norm.to(model_term.dtype)
                model_term = model_term / batch_norm
            else:
                batch_norm = None

            smooth_term = model_term.new_zeros(())
            if smooth_w != 0.0:
                q_grad = torch.autograd.grad(
                    outputs=q_pred.sum(),
                    inputs=targ[Properties.descriptors],
                    create_graph=True,
                    retain_graph=True,
                )[0]
                smoothness_loss = (torch.abs(q_grad) ** 2).mean()
                smooth_term = smooth_w * smoothness_loss

            l1_term = model_term.new_zeros(())
            if l1_w != 0.0:
                l1_term = l1_w * sum(p.abs().sum() for p in params)

            total_term = model_term + smooth_term + l1_term

            g_model = torch.autograd.grad(model_term, params, retain_graph=True, allow_unused=True)
            if smooth_w != 0.0:
                g_smooth = torch.autograd.grad(smooth_term, params, retain_graph=True, allow_unused=True)
            else:
                g_smooth = [None] * len(params)
            if l1_w != 0.0:
                g_l1 = torch.autograd.grad(l1_term, params, retain_graph=True, allow_unused=True)
            else:
                g_l1 = [None] * len(params)
            g_total = torch.autograd.grad(total_term, params, retain_graph=False, allow_unused=True)

            v_model = _flatten_grads_like_params(g_model, params)
            v_smooth = _flatten_grads_like_params(g_smooth, params)
            v_l1 = _flatten_grads_like_params(g_l1, params)
            v_total = _flatten_grads_like_params(g_total, params)

            if v_total is None:
                continue

            denom = float(torch.dot(v_total, v_total).item())
            if not np.isfinite(denom) or denom <= 0.0:
                continue

            proj_model.append(float(torch.dot(v_model, v_total).item()) / denom)
            proj_smooth.append(float(torch.dot(v_smooth, v_total).item()) / denom)
            proj_l1.append(float(torch.dot(v_l1, v_total).item()) / denom)
            total_grad_l2.append(float(np.sqrt(denom)))
    finally:
        if was_training:
            model.nnet.train()

    sampled = len(proj_model)
    if sampled == 0:
        return {
            "sampled_batches": 0,
            "proj_model_mean": np.nan,
            "proj_smooth_mean": np.nan,
            "proj_l1_mean": np.nan,
            "proj_sum_mean": np.nan,
            "total_grad_l2_mean": np.nan,
        }

    proj_model_mean = float(np.mean(proj_model))
    proj_smooth_mean = float(np.mean(proj_smooth))
    proj_l1_mean = float(np.mean(proj_l1))
    return {
        "sampled_batches": sampled,
        "proj_model_mean": proj_model_mean,
        "proj_smooth_mean": proj_smooth_mean,
        "proj_l1_mean": proj_l1_mean,
        "proj_sum_mean": float(proj_model_mean + proj_smooth_mean + proj_l1_mean),
        "total_grad_l2_mean": float(np.mean(total_grad_l2)),
    }


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
    weighted_smoothness: bool = False,
    plateau_patience: int = 5,
    plateau_factor: float = 0.5,
    min_lr: float = 1e-5,
    warmup_epochs: int = 0,
    warmup_init: float = 1e-6,
    normalization: bool = True,
    early_stop_patience: int = 5,
    nan_rescue: bool = True,
    rescue_shrink: float = 0.2,
    clip_warmup_epochs: int = 0,
    clip_init: float = None,
    diagnostics: bool = False,
    diag_max_batches: int = 64,
    diag_tail_quantile: float = 99.9,
    grad_diagnostics: bool = False,
    grad_diag_max_batches: int = 2,
    ema_beta: float = 0.5,
    min_delta: float = 1e-4,
    early_stop_start_epoch: int = 20,
    train_explosion_factor: float = 2.0
):
    """Training loop for AIMMD-TIS committor model stages."""
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
    train_smooth_model_ratio = []
    test_smooth_model_ratio = []
    train_smooth_total_frac = []
    test_smooth_total_frac = []
    train_l1_total_frac = []
    test_l1_total_frac = []
    train_reg_total_frac = []
    test_reg_total_frac = []
    lr_log = []
    diag_sampled_batches = []
    diag_worst_batch_loss = []
    diag_worst_batch_top_weight_frac = []
    diag_worst_batch_weight_sum = []
    diag_tail_weight_threshold = []
    grad_proj_sampled_batches = []
    grad_proj_model = []
    grad_proj_smooth = []
    grad_proj_l1 = []
    grad_proj_sum = []
    grad_total_l2 = []

    best_loss = np.inf
    best_state = None
    no_improve = 0
    ema_val = None
    best_train_loss = np.inf

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
        

        
        _ = model.train_epoch_smoothness(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            normalization=normalization,
            weighted_smoothness=weighted_smoothness
        )

        # report diagnostics on train/test sets at the end of the epoch
        train_losses = model.test_loss_smoothness(
            trainset,
            batch_size=batch_size,
            normalization=normalization,
            weighted_smoothness=weighted_smoothness
        )
        test_losses = model.test_loss_smoothness(
            testset,
            batch_size=batch_size,
            normalization=normalization,
            weighted_smoothness=weighted_smoothness
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
        train_smooth_model_ratio.append(
            train_smooth[-1] / train_model[-1]
            if np.isfinite(train_model[-1]) and train_model[-1] != 0.0
            else np.nan
        )
        test_smooth_model_ratio.append(
            test_smooth[-1] / test_model[-1]
            if np.isfinite(test_model[-1]) and test_model[-1] != 0.0
            else np.nan
        )
        train_smooth_total_frac.append(
            (train_smooth[-1]) / train_total[-1]
            if np.isfinite(train_total[-1]) and train_total[-1] != 0.0
            else np.nan
        )
        test_smooth_total_frac.append(
            (test_smooth[-1]) / test_total[-1]
            if np.isfinite(test_total[-1]) and test_total[-1] != 0.0
            else np.nan
        )
        train_l1_total_frac.append(
            (train_l1[-1]) / train_total[-1]
            if np.isfinite(train_total[-1]) and train_total[-1] != 0.0
            else np.nan
        )
        test_l1_total_frac.append(
            (test_l1[-1]) / test_total[-1]
            if np.isfinite(test_total[-1]) and test_total[-1] != 0.0
            else np.nan
        )
        train_reg_total_frac.append(
            train_smooth_total_frac[-1] + train_l1_total_frac[-1]
            if np.isfinite(train_smooth_total_frac[-1]) and np.isfinite(train_l1_total_frac[-1])
            else np.nan
        )
        test_reg_total_frac.append(
            test_smooth_total_frac[-1] + test_l1_total_frac[-1]
            if np.isfinite(test_smooth_total_frac[-1]) and np.isfinite(test_l1_total_frac[-1])
            else np.nan
        )
        lr_log.append(model.optimizer.param_groups[0]["lr"])

        if diagnostics:
            diag = _epoch_batch_diagnostics(
                model=model,
                trainset=trainset,
                batch_size=batch_size,
                max_batches=diag_max_batches,
                tail_quantile=diag_tail_quantile,
                normalization=normalization
            )
            diag_sampled_batches.append(diag["sampled_batches"])
            diag_worst_batch_loss.append(diag["worst_batch_loss"])
            diag_worst_batch_top_weight_frac.append(diag["worst_batch_top_weight_frac"])
            diag_worst_batch_weight_sum.append(diag["worst_batch_weight_sum"])
            diag_tail_weight_threshold.append(diag["tail_weight_threshold"])
            if grad_diagnostics:
                grad_diag = epoch_projected_gradient_contributions(
                    model=model,
                    trainset=trainset,
                    batch_size=batch_size,
                    max_batches=min(int(grad_diag_max_batches), int(diag_max_batches)),
                    normalization=normalization
                )
                grad_proj_sampled_batches.append(grad_diag["sampled_batches"])
                grad_proj_model.append(grad_diag["proj_model_mean"])
                grad_proj_smooth.append(grad_diag["proj_smooth_mean"])
                grad_proj_l1.append(grad_diag["proj_l1_mean"])
                grad_proj_sum.append(grad_diag["proj_sum_mean"])
                grad_total_l2.append(grad_diag["total_grad_l2_mean"])

        print(
            f"[{stage_name}] Epoch {epoch}/{epochs} | "
            f"Train={train_total[-1]:.4e} Test={test_total[-1]:.4e} "
            f"LR={lr_log[-1]:.2e} Batch={batch_size}"
        )
        if diagnostics:
            print(
                f"[{stage_name}] Diag epoch {epoch} | "
                f"WorstBatchLoss={diag_worst_batch_loss[-1]:.4e} "
                f"TopWFrac={diag_worst_batch_top_weight_frac[-1]:.4f} "
                f"TopWThr={diag_tail_weight_threshold[-1]:.4e} "
                f"WSum={diag_worst_batch_weight_sum[-1]:.4e} "
                f"Sampled={diag_sampled_batches[-1]}"
            )
            print(
                f"[{stage_name}] Loss ratio epoch {epoch} | "
                f"Smooth/Model train={train_smooth_model_ratio[-1]:.4e} "
                f"test={test_smooth_model_ratio[-1]:.4e}"
            )
            print(
                f"[{stage_name}] Loss share epoch {epoch} | "
                f"Smooth/Total train={train_smooth_total_frac[-1]:.4e} "
                f"test={test_smooth_total_frac[-1]:.4e} "
                f"L1/Total train={train_l1_total_frac[-1]:.4e} "
                f"test={test_l1_total_frac[-1]:.4e}"
            )
            print(
                f"[{stage_name}] Reg share epoch {epoch} | "
                f"train={train_reg_total_frac[-1]:.4e} test={test_reg_total_frac[-1]:.4e}"
            )
            if grad_diagnostics:
                print(
                    f"[{stage_name}] Grad proj epoch {epoch} | "
                    f"Model={grad_proj_model[-1]:.4e} "
                    f"Smooth={grad_proj_smooth[-1]:.4e} "
                    f"L1={grad_proj_l1[-1]:.4e} "
                    f"Sum={grad_proj_sum[-1]:.4e} "
                    f"TotalGradL2={grad_total_l2[-1]:.4e} "
                    f"Sampled={grad_proj_sampled_batches[-1]}"
                )
        aimmd_store.rcmodels[f"{stage_name}_state_dict_most_recent"] = TorchRCModelLite(
            model.nnet.state_dict(),
            meta={"stage": stage_name, "epoch": epoch, "lr": float(lr_log[-1])},
        )
        aimmd_store.rcmodels[f"{stage_name}_model_most_recent"] = model

        current_val_raw = float(test_losses["total_loss"])
        current_train = float(train_losses["total_loss"])

        if epoch >= early_stop_start_epoch:
            if ema_val is None:
                ema_val = current_val_raw
            else:
                ema_val = float(ema_beta) * ema_val + (1.0 - float(ema_beta)) * current_val_raw
            current_val = ema_val
        else:
            print(f"[{stage_name}] Early stopping not active until epoch {early_stop_start_epoch}.")
            current_val = current_val_raw

        if current_train < best_train_loss:
            best_train_loss = current_train
        

        improved = current_val < best_loss - float(min_delta)

        if improved:
            best_loss = current_val
            best_state = deepcopy(model.nnet.state_dict())
            no_improve = 0
            aimmd_store.rcmodels[f"{stage_name}_state_dict_best"] = TorchRCModelLite(
                best_state,
                meta={
                    "epoch": epoch,
                    "val_loss_ema": best_loss,
                    "val_loss_raw": current_val_raw,
                    "train_loss": current_train,
                },
            )
            aimmd_store.rcmodels[f"{stage_name}_model_best"] = model

        else:
            no_improve += 1

        scheduler.step(current_val_raw)
        print(
            f"[{stage_name}] EMA_Test={current_val:.4e} BestEMA={best_loss:.4e} "
            f"NoImprove={no_improve}/{early_stop_patience} "
        )

        if (
            np.isfinite(current_train)
            and np.isfinite(best_train_loss)
            and current_train > float(train_explosion_factor) * best_train_loss
        ):
            print(
                f"[{stage_name}] Train-loss guard triggered at epoch {epoch}: "
                f"train={current_train:.4e}, best_train={best_train_loss:.4e}, "
                f"factor={train_explosion_factor:.2f}. Restoring best state."
            )
            break

        if no_improve >= int(early_stop_patience):
            print(
                f"[{stage_name}] Early stopping at epoch {epoch} "
                f"(patience {early_stop_patience}, best EMA val={best_loss:.4e})."
            )
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
        "train_smooth_model_ratio": train_smooth_model_ratio,
        "test_smooth_model_ratio": test_smooth_model_ratio,
        "train_smooth_total_frac": train_smooth_total_frac,
        "test_smooth_total_frac": test_smooth_total_frac,
        "train_l1_total_frac": train_l1_total_frac,
        "test_l1_total_frac": test_l1_total_frac,
        "train_reg_total_frac": train_reg_total_frac,
        "test_reg_total_frac": test_reg_total_frac,
        "lr": lr_log,
        "diag_sampled_batches": diag_sampled_batches,
        "diag_worst_batch_loss": diag_worst_batch_loss,
        "diag_worst_batch_top_weight_frac": diag_worst_batch_top_weight_frac,
        "diag_worst_batch_weight_sum": diag_worst_batch_weight_sum,
        "diag_tail_weight_threshold": diag_tail_weight_threshold,
        "diag_grad_proj_sampled_batches": grad_proj_sampled_batches,
        "diag_grad_proj_model": grad_proj_model,
        "diag_grad_proj_smooth": grad_proj_smooth,
        "diag_grad_proj_l1": grad_proj_l1,
        "diag_grad_proj_sum": grad_proj_sum,
        "diag_grad_total_l2": grad_total_l2,
        "best_val": best_loss,
    }
