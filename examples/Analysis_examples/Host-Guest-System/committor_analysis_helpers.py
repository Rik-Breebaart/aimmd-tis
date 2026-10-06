"""Shared numerical and plotting helpers for the committor analysis notebooks.

The notebooks deliberately keep their own imports, configuration, data/model
loading, and ``SystemVisualizer`` construction.  This module contains only
reusable analysis operations so that their implementations have one source of
truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
import aimmdTIS
__all__ = [
    "predict_q",
    "sigmoid_stable",
    "aimmd_softplus_loss_terms",
    "weighted_model_loss",
    "descriptor_names",
    "weighted_mean",
    "subsample_arrays",
    "weighted_pearson_corr",
    "top_matrix_pairs",
    "weighted_gradient_importance",
    "weighted_permutation_importance",
    "conditional_permutation_importance",
    "permutation_influence_correlation",
    "descriptor_swap_loss_matrix",
    "plot_square_matrix",
    "plot_correlated_pair",
    "run_descriptor_correlation_tools",
]


def _as_vector(values, name, length=None, dtype=float):
    values = np.asarray(values, dtype=dtype).reshape(-1)
    if length is not None and len(values) != length:
        raise ValueError(f"{name} has length {len(values)}; expected {length}.")
    return values


def predict_q(model, X, batch_size=65_536):
    """Return one logit-committor value per descriptor row."""
    q = np.asarray(
        model.log_prob(np.asarray(X), use_transform=False, batch_size=batch_size)
    )
    if q.ndim == 2 and q.shape[1] == 1:
        q = q[:, 0]
    elif q.ndim != 1:
        raise ValueError(f"Expected scalar model output, got shape {q.shape}.")
    return q.astype(float, copy=False)


def sigmoid_stable(q):
    """Numerically stable conversion from logit committor to ``p_B``."""
    q = np.asarray(q, dtype=float)
    out = np.empty_like(q)
    positive = q >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-q[positive]))
    exp_q = np.exp(q[~positive])
    out[~positive] = exp_q / (1.0 + exp_q)
    return out


def aimmd_softplus_loss_terms(q, shots, weights=None):
    """Return per-frame AIMMD negative log-likelihood terms.

    ``shots[:, 0]`` contains the number of shots ending in A and
    ``shots[:, 1]`` the number ending in B.  Optional weights are applied
    exactly once.
    """
    q = torch.as_tensor(q, dtype=torch.float32) if not isinstance(q, torch.Tensor) else q
    shots = torch.as_tensor(shots, dtype=torch.float32) if not isinstance(shots, torch.Tensor) else shots
    if weights is None:
        weights = np.zeros(len(q), dtype=float)
    weights = torch.as_tensor(weights, dtype=torch.float32) if weights is not None and not isinstance(weights, torch.Tensor) else weights
    terms = aimmdTIS.training.losses.snapshot_loss_softplus(q.view(-1, 1), weights, shots)
    terms = terms.detach().cpu().numpy().reshape(-1)
    return terms

def aimmd_smoothness_loss_terms(model, descriptors, weights=None, batch_size=4096):
    """Return per-frame AIMMD smoothness loss terms.

    the smoothness loss is computed as the squared norm of the gradient of the model output with respect to the input descriptors. Optional weights are applied
    the model is evaluated in batches to avoid memory issues, and the loss terms are returned as a numpy array.


    exactly once.
    """
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    descriptors = torch.as_tensor(descriptors, dtype=torch.float32, device=device) 
    if weights is None:
        weights = np.zeros(len(descriptors), dtype=float)
    weights = torch.as_tensor(weights, dtype=torch.float32, device=device) if weights is not None and not isinstance(weights, torch.Tensor) else weights
    terms = torch.empty(len(descriptors), dtype=torch.float32, device=device)
    for batch in range(0, len(descriptors), batch_size):
        descriptors_batch = descriptors[batch:batch + batch_size]
        weights_batch = weights[batch:batch + batch_size]
        terms[batch:batch + batch_size] = aimmdTIS.training.losses.snapshot_loss_smoothness(model.nnet, descriptors_batch)
    terms = terms.detach().cpu().numpy().reshape(-1)
    return terms


def weighted_mean(values, weights=None):
    """Arithmetic mean, or a mean normalized by the sum of finite weights."""
    values = _as_vector(values, "values")
    if weights is None:
        return float(np.mean(values))
    weights = _as_vector(weights, "weights", len(values))
    if np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("Weights must be finite and non-negative.")
    weight_sum = weights.sum(dtype=float)
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("Weights must have a positive finite sum.")
    return float(np.dot(weights / weight_sum, values))


def weighted_model_loss(model, X, shots, weights=None, batch_size=65_536):
    """Return ``(mean_loss, q, unweighted_per_frame_loss)``."""
    q = predict_q(model, X, batch_size=batch_size)
    terms = aimmd_softplus_loss_terms(q, shots, weights=weights)
    return np.mean(terms), q, terms


def descriptor_names(n_features, labels=None, units=None):
    """Construct readable descriptor names, adding units without duplication."""
    names = []
    for i in range(n_features):
        label = str(labels[i]) if labels is not None and i < len(labels) else f"desc_{i}"
        unit = str(units[i]).strip() if units is not None and i < len(units) else ""
        if unit:
            label += f" {unit}" if unit.startswith("[") else f" [{unit}]"
        names.append(label)
    return names


def subsample_arrays(X, weights=None, shots=None, max_samples=250_000, random_state=2026):
    """Reproducibly subsample aligned arrays without replacement."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got shape {X.shape}.")
    n_samples = len(X)
    weights = None if weights is None else _as_vector(weights, "weights", n_samples)
    shots = None if shots is None else np.asarray(shots)
    if shots is not None and len(shots) != n_samples:
        raise ValueError("shots and X must contain the same number of rows.")
    if max_samples is None or n_samples <= max_samples:
        indices = np.arange(n_samples)
    else:
        rng = np.random.default_rng(random_state)
        indices = np.sort(rng.choice(n_samples, size=max_samples, replace=False))
        print(f"Subsampled {max_samples:,} / {n_samples:,} frames.")
    return (
        X[indices],
        None if weights is None else weights[indices],
        None if shots is None else shots[indices],
        indices,
    )


def weighted_pearson_corr(X, weights=None):
    """Return a finite weighted Pearson correlation matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got shape {X.shape}.")
    if weights is None:
        corr = np.corrcoef(X, rowvar=False)
    else:
        weights = _as_vector(weights, "weights", len(X))
        if np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError("Weights must be finite and non-negative.")
        weight_sum = weights.sum(dtype=float)
        if weight_sum <= 0 or not np.isfinite(weight_sum):
            raise ValueError("Weights must have a positive finite sum.")
        weights = weights / weight_sum
        centered = X - np.sum(weights[:, None] * X, axis=0)
        covariance = centered.T @ (weights[:, None] * centered)
        denominator = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = covariance / denominator
    corr = np.asarray(corr, dtype=float)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def top_matrix_pairs(matrix, top_n=15, absolute=True, threshold=None):
    """Rank the upper-triangular pairs in a square DataFrame."""
    if not isinstance(matrix, pd.DataFrame) or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square pandas DataFrame.")
    row, col = np.triu_indices_from(matrix.to_numpy(), k=1)
    values = matrix.to_numpy()[row, col]
    scores = np.abs(values) if absolute else values
    records = []
    for index in np.argsort(scores)[::-1]:
        if threshold is not None and scores[index] < threshold:
            continue
        records.append({
            "feature_1": matrix.index[row[index]],
            "feature_2": matrix.columns[col[index]],
            "value": float(values[index]),
            "abs_value": float(abs(values[index])),
        })
        if len(records) == top_n:
            break
    return pd.DataFrame(records)


def weighted_gradient_importance(
    model,
    vis,
    X,
    weights=None,
    descriptor_names_list=None,
    batch_size=8192,
    q_center=None,
    q_width=None,
):
    """RPE-weighted global or transition-state-gated ``|dq/dx|`` importance."""
    X = np.asarray(X, dtype=np.float32)
    n_samples, n_features = X.shape
    weights = np.ones(n_samples) if weights is None else _as_vector(weights, "weights", n_samples)
    names = descriptor_names_list or [f"desc_{i}" for i in range(n_features)]
    absolute_sum = np.zeros(n_features)
    square_sum = np.zeros(n_features)
    total_weight = 0.0
    if (q_center is None) != (q_width is None):
        raise ValueError("q_center and q_width must be supplied together.")
    if q_width is not None and q_width <= 0:
        raise ValueError("q_width must be positive.")
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        batch = X[start:stop]
        batch_weights = weights[start:stop].copy()
        gradients = np.asarray(vis._compute_gradients_raw(model, batch), dtype=float)
        if q_center is not None:
            q = predict_q(model, batch, batch_size=batch_size)
            batch_weights *= np.exp(-0.5 * ((q - q_center) / q_width) ** 2)
        absolute_sum += np.sum(batch_weights[:, None] * np.abs(gradients), axis=0)
        square_sum += np.sum(batch_weights[:, None] * gradients**2, axis=0)
        total_weight += batch_weights.sum(dtype=float)
    if total_weight <= 0 or not np.isfinite(total_weight):
        raise ValueError("Gradient analysis has no positive finite weight.")
    result = pd.DataFrame({
        "descriptor": names,
        "weighted_mean_abs_gradient": absolute_sum / total_weight,
        "weighted_rms_gradient": np.sqrt(square_sum / total_weight),
    })
    result["gradient_rank"] = result["weighted_mean_abs_gradient"].rank(
        ascending=False, method="min"
    ).astype(int)
    return result.sort_values("gradient_rank").reset_index(drop=True)


def weighted_permutation_importance(
    model,
    X,
    shots,
    weights=None,
    descriptor_names_list=None,
    n_repeats=5,
    batch_size=65_536,
    random_state=123,
    metric="loss",
):
    """Measure the loss or model-output change after shuffling each descriptor."""
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    weights = np.ones(n_samples) if weights is None else _as_vector(weights, "weights", n_samples)
    names = descriptor_names_list or [f"desc_{i}" for i in range(n_features)]
    base_loss, q_reference, _ = weighted_model_loss(model, X, shots, weights, batch_size)
    rng = np.random.default_rng(random_state)
    records = []
    for feature in range(n_features):
        scores = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[:, feature] = shuffled[rng.permutation(n_samples), feature]
            q = predict_q(model, shuffled, batch_size)
            if metric == "loss":
                score = np.mean(aimmd_softplus_loss_terms(q, shots, weights)) - base_loss
            elif metric == "q_mse":
                score = weighted_mean((q - q_reference) ** 2, weights)
            else:
                raise ValueError("metric must be 'loss' or 'q_mse'.")
            scores.append(score)
        records.append({
            "descriptor": names[feature],
            "permutation_importance_mean": float(np.mean(scores)),
            "permutation_importance_std": float(np.std(scores, ddof=1)) if n_repeats > 1 else 0.0,
        })
    result = pd.DataFrame(records)
    result["permutation_rank"] = result["permutation_importance_mean"].rank(
        ascending=False, method="min"
    ).astype(int)
    return result.sort_values("permutation_rank").reset_index(drop=True)


def conditional_permutation_importance(
    model,
    X,
    shots,
    weights=None,
    descriptor_names_list=None,
    condition_on=None,
    n_bins=20,
    n_repeats=5,
    batch_size=65_536,
    random_state=123,
    metric="loss",
):
    """Shuffle each descriptor within bins of correlated descriptors."""
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    weights = np.ones(n_samples) if weights is None else _as_vector(weights, "weights", n_samples)
    names = descriptor_names_list or [f"desc_{i}" for i in range(n_features)]
    base_loss, q_reference, _ = weighted_model_loss(model, X, shots, weights, batch_size)
    if condition_on is None:
        correlations = weighted_pearson_corr(X, weights)
        np.fill_diagonal(correlations, 0.0)
        condition_on = {
            feature: [int(np.argmax(np.abs(correlations[feature])))]
            for feature in range(n_features)
        }
    rng = np.random.default_rng(random_state)
    records = []
    for feature in range(n_features):
        conditioning_features = condition_on.get(feature, [])
        components = []
        for condition in conditioning_features:
            edges = np.unique(np.quantile(X[:, condition], np.linspace(0, 1, n_bins + 1)))
            components.append(
                np.zeros(n_samples, dtype=int)
                if len(edges) <= 2
                else np.digitize(X[:, condition], edges[1:-1])
            )
        if components:
            _, bin_ids = np.unique(np.column_stack(components), axis=0, return_inverse=True)
        else:
            bin_ids = np.zeros(n_samples, dtype=int)
        scores = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            for bin_id in np.unique(bin_ids):
                indices = np.flatnonzero(bin_ids == bin_id)
                if len(indices) > 2:
                    shuffled[indices, feature] = X[rng.permutation(indices), feature]
            q = predict_q(model, shuffled, batch_size)
            if metric == "loss":
                score = np.mean(aimmd_softplus_loss_terms(q, shots, weights)) - base_loss
            elif metric == "q_mse":
                score = weighted_mean((q - q_reference) ** 2, weights)
            else:
                raise ValueError("metric must be 'loss' or 'q_mse'.")
            scores.append(score)
        records.append({
            "descriptor": names[feature],
            "conditioned_on": [names[index] for index in conditioning_features],
            "conditional_importance_mean": float(np.mean(scores)),
            "conditional_importance_std": float(np.std(scores, ddof=1)) if n_repeats > 1 else 0.0,
        })
    result = pd.DataFrame(records)
    result["conditional_rank"] = result["conditional_importance_mean"].rank(
        ascending=False, method="min"
    ).astype(int)
    return result.sort_values("conditional_rank").reset_index(drop=True)


def permutation_influence_correlation(
    model,
    X,
    shots,
    weights=None,
    descriptor_names_list=None,
    n_repeats=3,
    batch_size=65_536,
    random_state=31,
):
    """Correlate per-frame loss-change profiles caused by descriptor shuffles."""
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    names = descriptor_names_list or [f"desc_{i}" for i in range(n_features)]
    rng = np.random.default_rng(random_state)
    reference = aimmd_softplus_loss_terms(predict_q(model, X, batch_size), shots, weights=weights)
    influence = np.zeros((n_features, n_samples), dtype=float)
    for feature in range(n_features):
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[:, feature] = shuffled[rng.permutation(n_samples), feature]
            terms = aimmd_softplus_loss_terms(predict_q(model, shuffled, batch_size), shots, weights=weights)
            influence[feature] += np.abs(terms - reference)
        influence[feature] /= n_repeats
    correlation = np.corrcoef(influence)
    correlation[~np.isfinite(correlation)] = 0.0
    np.fill_diagonal(correlation, 1.0)
    return pd.DataFrame(correlation, index=names, columns=names), influence


def descriptor_swap_loss_matrix(
    model,
    X,
    shots,
    weights=None,
    descriptor_names_list=None,
    batch_size=65_536,
):
    """Return the weighted-loss increase after swapping every feature pair."""
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]
    names = descriptor_names_list or [f"desc_{i}" for i in range(n_features)]
    base_loss, _, _ = weighted_model_loss(model, X, shots, weights, batch_size)
    matrix = np.zeros((n_features, n_features), dtype=float)
    for first in range(n_features):
        for second in range(first + 1, n_features):
            swapped = X.copy()
            swapped[:, [first, second]] = swapped[:, [second, first]]
            swap_loss, _, _ = weighted_model_loss(model, swapped, shots, weights, batch_size)
            matrix[first, second] = matrix[second, first] = swap_loss - base_loss
    return pd.DataFrame(matrix, index=names, columns=names)


def plot_square_matrix(
    matrix,
    title,
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    cbar_label=None,
    figsize=(9, 8),
    log_scale=False,
    fontsize = 20
):
    """Plot a labeled square DataFrame and return ``(figure, axes)``."""
    values = matrix.to_numpy(dtype=float)
    figure, axes = plt.subplots(figsize=figsize)
    norm = None
    if log_scale:
        positive = values[values > 0]
        if positive.size:
            norm = LogNorm(vmin=vmin or positive.min(), vmax=vmax or positive.max())
    image = axes.imshow(values, cmap=cmap, vmin=None if norm else vmin,
                        vmax=None if norm else vmax, norm=norm, aspect="auto")
    axes.set_xticks(np.arange(len(matrix.columns)))
    axes.set_yticks(np.arange(len(matrix.index)))
    axes.set_xticklabels(matrix.columns, rotation=90, fontsize=fontsize,ha='right')
    axes.set_yticklabels(matrix.index, fontsize=fontsize,ha='right', va='center')
    axes.set_title(title)
    colorbar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    if cbar_label:
        colorbar.set_label(cbar_label, fontsize=fontsize)
    figure.tight_layout()
    return figure, axes


def plot_correlated_pair(correlation_results, pair_rank=0, gridsize=100):
    """Plot one ranked descriptor pair from ``run_descriptor_correlation_tools``."""
    pairs = correlation_results["top_pairs"]
    if not 0 <= pair_rank < len(pairs):
        raise IndexError(f"pair_rank must be between 0 and {len(pairs) - 1}.")
    row = pairs.iloc[pair_rank]
    first, second = row["feature_1"], row["feature_2"]
    frame = correlation_results["dataframe"]
    figure, axes = plt.subplots(figsize=(7, 6))
    artist = axes.hexbin(frame[first], frame[second], gridsize=gridsize,
                         mincnt=1, cmap="viridis")
    figure.colorbar(artist, ax=axes, label="Frame count")
    axes.set(xlabel=first, ylabel=second,
             title=f"Rank {pair_rank + 1}: r = {row['value']:.3f}")
    figure.tight_layout()
    return figure, axes


def run_descriptor_correlation_tools(
    descriptors,
    descriptor_labels=None,
    descriptor_units=None,
    weights=None,
    max_points=50_000,
    top_n=12,
    random_state=7,
):
    """Prepare a sample, weighted correlation matrix, and ranked pair table."""
    descriptors = np.asarray(descriptors)
    sampled, sampled_weights, _, indices = subsample_arrays(
        descriptors,
        weights=weights,
        max_samples=max_points,
        random_state=random_state,
    )
    names = descriptor_names(sampled.shape[1], descriptor_labels, descriptor_units)
    frame = pd.DataFrame(sampled, columns=names)
    correlation = weighted_pearson_corr(sampled, sampled_weights)
    correlation_frame = pd.DataFrame(correlation, index=names, columns=names)
    return {
        "dataframe": frame,
        "weights": sampled_weights,
        "sample_index": indices,
        "corr_df": correlation_frame,
        "top_pairs": top_matrix_pairs(correlation_frame, top_n=top_n),
    }
