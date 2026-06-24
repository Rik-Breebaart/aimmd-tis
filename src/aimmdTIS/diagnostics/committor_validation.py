"""Reusable committor-validation utilities.

This module provides a clean API for the committor validation workflow that was
previously implemented ad-hoc in notebooks.

Main use cases
--------------
1) Given per-frame model logits and shooting-based committor estimates,
   compute validation statistics and cache them to disk.
2) Recreate the classic two-panel validation plot:
   - left: q_ref vs q_model
   - right: p_B_ref vs p_B_model
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

def load_committor_frame_data(
    descriptors_npy: Path | str,
    shots_csv: Path | str,
    bound_col: str = "bound",
    unbound_col: str = "unbound",
) -> Dict[str, np.ndarray]:
    """Load committor-analysis frame data from descriptors + shot-result CSV.

    Parameters
    ----------
    descriptors_npy
        Path to ``descriptors_all_frames.npy`` with shape ``(N, D)``.
    shots_csv
        Path to CSV containing per-frame shot outcomes.
    bound_col
        Column name for counts that end in state A (bound by default).
    unbound_col
        Column name for counts that end in state B (unbound by default).

    Returns
    -------
    out : dict
        Keys:
        - ``descriptors``: ndarray (N, D)
        - ``shot_results``: ndarray (N, 2), ordered as [n_A, n_B]
        - ``weights``: ndarray (N,), currently uniform 1.0
    """
    descriptors = np.asarray(np.load(Path(descriptors_npy)), dtype=float)
    if descriptors.ndim != 2:
        raise ValueError("descriptors array must be 2D with shape (N, D).")

    table = np.genfromtxt(
        Path(shots_csv),
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    names = list(table.dtype.names or [])
    if not names:
        raise ValueError("shots_csv appears empty or has no header row.")

    name_lut = {n.lower(): n for n in names}
    if bound_col.lower() not in name_lut or unbound_col.lower() not in name_lut:
        raise ValueError(
            f"shots_csv must contain columns '{bound_col}' and '{unbound_col}'. Found: {names}"
        )

    a_col = name_lut[bound_col.lower()]
    b_col = name_lut[unbound_col.lower()]
    n_a = np.asarray(table[a_col], dtype=float).reshape(-1)
    n_b = np.asarray(table[b_col], dtype=float).reshape(-1)
    shot_results = np.column_stack([n_a, n_b])

    if descriptors.shape[0] != shot_results.shape[0]:
        raise ValueError(
            "Row-count mismatch between descriptors and shot CSV: "
            f"{descriptors.shape[0]} vs {shot_results.shape[0]}"
        )

    weights = np.ones(descriptors.shape[0], dtype=float)
    return {
        "descriptors": descriptors,
        "shot_results": shot_results,
        "weights": weights,
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _safe_logit(p: np.ndarray, eps: float = 5e-3) -> np.ndarray:
    p_clip = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p_clip / (1.0 - p_clip))


def estimate_pb_from_shots(shot_results: np.ndarray, eps: float = 5e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate per-frame committor from shot outcomes.

    Parameters
    ----------
    shot_results
        Array of shape (N, 2), with columns [n_A, n_B].
    eps
        Clipping used for a stable logit transform.

    Returns
    -------
    p_b : ndarray, shape (N,)
        Shooting-based p_B estimate.
    q_ref : ndarray, shape (N,)
        Logit committor estimate from p_B.
    """
    shot = np.asarray(shot_results, dtype=float)
    if shot.ndim != 2 or shot.shape[1] != 2:
        raise ValueError("shot_results must have shape (N, 2) with [n_A, n_B].")

    total = shot.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_b = np.where(total > 0, shot[:, 1] / total, np.nan)
    q_ref = _safe_logit(p_b, eps=eps)
    return p_b, q_ref


def bin_stats(
    x_values: np.ndarray,
    y_values: np.ndarray,
    bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean/std/count of x_values binned by y_values.

    This matches the original plotting logic:
        - y_values define the bin assignment, i.e. the vertical plot coordinate.
        - x_values are averaged inside each y-bin, i.e. the horizontal mean value.

    Use together with make_step_plot() and ax.fill_betweenx().
    """
    x_values = np.asarray(x_values, dtype=float).reshape(-1)
    y_values = np.asarray(y_values, dtype=float).reshape(-1)
    bins = np.asarray(bins, dtype=float)

    if bins.ndim != 1 or bins.size < 2:
        raise ValueError("bins must be a 1D array with at least 2 edges.")
    if x_values.shape[0] != y_values.shape[0]:
        raise ValueError("x_values and y_values must have the same length.")

    n_bins = bins.size - 1
    means = np.full(n_bins, np.nan, dtype=float)
    stds = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    bin_indices = np.digitize(y_values, bins) - 1

    valid = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & (bin_indices >= 0)
        & (bin_indices < n_bins)
    )

    for i in range(n_bins):
        vals = x_values[valid & (bin_indices == i)]
        if vals.size == 0:
            continue

        means[i] = float(np.mean(vals))
        stds[i] = float(np.std(vals))
        counts[i] = int(vals.size)

    return means, stds, counts


def make_step_plot(
    mean: np.ndarray,
    std: np.ndarray,
    bins: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create vertical step vectors for a mean +/- std envelope.

    Returns arrays for:
        ax.plot(step_x, step_y)
        ax.fill_betweenx(fill_y, lower_x, upper_x)

    The bin coordinate is on the y-axis. The mean and std envelope are on the x-axis.
    """
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    bins = np.asarray(bins, dtype=float)

    if bins.ndim != 1 or bins.size != mean.size + 1:
        raise ValueError("bins must be a 1D array with length len(mean) + 1.")
    if std.shape != mean.shape:
        raise ValueError("std must have the same shape as mean.")

    if mask is None:
        mask_arr = np.ones(mean.shape, dtype=bool)
    else:
        mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
        if mask_arr.shape != mean.shape:
            raise ValueError("mask must have the same shape as mean.")

    valid = mask_arr & np.isfinite(mean) & np.isfinite(std)

    step_x = []
    step_y = []
    lower_x = []
    upper_x = []
    fill_y = []

    for i in np.where(valid)[0]:
        lower = mean[i] - std[i]
        upper = mean[i] + std[i]

        step_x.extend([mean[i], mean[i]])
        step_y.extend([bins[i], bins[i + 1]])
        lower_x.extend([lower, lower])
        upper_x.extend([upper, upper])
        fill_y.extend([bins[i], bins[i + 1]])

    return (
        np.asarray(step_x, dtype=float),
        np.asarray(step_y, dtype=float),
        np.asarray(lower_x, dtype=float),
        np.asarray(upper_x, dtype=float),
        np.asarray(fill_y, dtype=float),
    )


def compute_validation(
    q_model: np.ndarray,
    p_b_ref: np.ndarray,
    bins_q: Optional[Iterable[float]] = None,
    eps: float = 5e-3,
) -> Dict[str, np.ndarray]:
    """Compute validation arrays and binned statistics.

    Parameters
    ----------
    q_model
        Model logit committor values, shape (N,) or (N,1).
    p_b_ref
        Reference committor estimate in probability space, shape (N,).
        Typically derived from shooting analysis.
    bins_q
        Bin edges in q-space for step-statistics.
    eps
        Clipping for reference p_B before logit transform.
    """
    q_model = np.asarray(q_model, dtype=float).reshape(-1)
    p_b_ref = np.asarray(p_b_ref, dtype=float).reshape(-1)

    if q_model.shape[0] != p_b_ref.shape[0]:
        raise ValueError("q_model and p_b_ref must have the same length.")

    q_ref = _safe_logit(p_b_ref, eps=eps)
    p_b_model = _sigmoid(q_model)

    valid = np.isfinite(q_ref) & np.isfinite(q_model) & np.isfinite(p_b_ref) & np.isfinite(p_b_model)
    q_ref_v = q_ref[valid]
    q_model_v = q_model[valid]
    p_b_ref_v = p_b_ref[valid]
    p_b_model_v = p_b_model[valid]

    if bins_q is None:
        bins_q = np.arange(-6.0, 6.5, 0.5)
    bins_q = np.asarray(list(bins_q), dtype=float)

    mean_q, std_q, count_q = bin_stats(q_ref_v, q_model_v, bins_q)
    bins_pb = _sigmoid(bins_q)
    mean_pb, std_pb, count_pb = bin_stats(p_b_ref_v, p_b_model_v, bins_pb)

    rmse_q = float(np.sqrt(np.mean((q_model_v - q_ref_v) ** 2))) if q_ref_v.size else np.nan
    rmse_pb = float(np.sqrt(np.mean((p_b_model_v - p_b_ref_v) ** 2))) if p_b_ref_v.size else np.nan

    return {
        "q_ref": q_ref,
        "q_model": q_model,
        "p_b_ref": p_b_ref,
        "p_b_model": p_b_model,
        "valid_mask": valid,
        "bins_q": bins_q,
        "bins_pb": bins_pb,
        "mean_q": mean_q,
        "std_q": std_q,
        "count_q": count_q,
        "mean_pb": mean_pb,
        "std_pb": std_pb,
        "count_pb": count_pb,
        "rmse_q": np.array([rmse_q]),
        "rmse_pb": np.array([rmse_pb]),
    }


def save_validation_cache(cache_file: Path | str, stats: Dict[str, np.ndarray]) -> Path:
    """Persist computed validation stats to an .npz cache."""
    out = Path(cache_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **stats)
    return out


def load_validation_cache(cache_file: Path | str) -> Dict[str, np.ndarray]:
    """Load validation stats from an .npz cache."""
    path = Path(cache_file)
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def run_validation_with_cache(
    q_model: np.ndarray,
    p_b_ref: np.ndarray,
    cache_file: Optional[Path | str] = None,
    force_recompute: bool = False,
    bins_q: Optional[Iterable[float]] = None,
    eps: float = 5e-3,
) -> Dict[str, np.ndarray]:
    """Compute validation stats, optionally reading/writing a cache file."""
    if cache_file is not None:
        cpath = Path(cache_file)
        if cpath.exists() and not force_recompute:
            return load_validation_cache(cpath)

    stats = compute_validation(q_model=q_model, p_b_ref=p_b_ref, bins_q=bins_q, eps=eps)
    if cache_file is not None:
        save_validation_cache(cache_file, stats)
    return stats


def plot_validation_panels(
    stats: Dict[str, np.ndarray],
    colors: Optional[np.ndarray] = None,
    alpha: float = 0.8,
    lim_q: float = 4.0,
    title: str = "Committor Validation",
):
    """Create the classic two-panel validation figure.

    Left panel: q_ref vs q_model with ideal line and binned mean/std.
    Right panel: p_B_ref vs p_B_model with ideal line and binned mean/std.
    """
    valid = stats["valid_mask"].astype(bool)

    q_ref = stats["q_ref"][valid]
    q_model = stats["q_model"][valid]
    p_b_ref = stats["p_b_ref"][valid]
    p_b_model = stats["p_b_model"][valid]

    step_x_q, step_y_q, lower_x_q, upper_x_q, fill_y_q = make_step_plot(
        stats["mean_q"],
        stats["std_q"],
        stats["bins_q"],
        mask=stats["count_q"] > 0,
    )
    step_x_pb, step_y_pb, lower_x_pb, upper_x_pb, fill_y_pb = make_step_plot(
        stats["mean_pb"],
        stats["std_pb"],
        stats["bins_pb"],
        mask=stats["count_pb"] > 0,
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot([-lim_q, lim_q], [-lim_q, lim_q], color="red", label="Ideal")
    ax[0].fill_betweenx(fill_y_q, lower_x_q, upper_x_q, color="blue", alpha=0.2, label="+-1 std dev")
    ax[0].plot(step_x_q, step_y_q, color="blue", label="Mean q")
    if colors is None:
        ax[0].scatter(q_ref, q_model, alpha=alpha, color="black", s=10, label="Samples")
    else:
        ax[0].scatter(q_ref, q_model, alpha=alpha, c=colors[valid], s=10, label="Samples")
    ax[0].set_xlabel("Committor analysis q")
    ax[0].set_ylabel("Learned q")
    ax[0].set_xlim([-lim_q, lim_q])
    ax[0].set_ylim([-lim_q, lim_q])
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot([0.0, 1.0], [0.0, 1.0], color="red", label="Ideal")
    ax[1].fill_betweenx(fill_y_pb, lower_x_pb, upper_x_pb, color="blue", alpha=0.2, label="+-1 std dev")
    ax[1].plot(step_x_pb, step_y_pb, color="blue", label="Mean p_B")
    if colors is None:
        ax[1].scatter(p_b_ref, p_b_model, alpha=alpha, color="black", s=10, label="Samples")
    else:
        ax[1].scatter(p_b_ref, p_b_model, alpha=alpha, c=colors[valid], s=10, label="Samples")
    ax[1].set_xlabel("Committor analysis p_B")
    ax[1].set_ylabel("Learned p_B")
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.0])
    ax[1].grid(True)
    ax[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig, ax
