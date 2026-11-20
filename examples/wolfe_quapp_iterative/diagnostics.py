"""Per-iteration analysis of the iterative AIMMD-TIS run on Wolfe-Quapp.

After iteration k has produced the reweighted path ensemble RPE_k and the model
q_{k+1} trained on it, :func:`analyse_iteration` compares both q_k and q_{k+1}
with the exact (theory) committor and the true potential, on the region the
RPE samples, and writes figures to ``<out>/analysis/it{k}/`` and numbers to
``<out>/rpe_it{k}/analysis.json``.

Metrics (all on bins of the (x, y) plane holding at least ``min_count`` RPE frames):

* ``fe_rmse_kT``            RPE free energy -ln rho_RPE vs beta V (constant aligned)
* ``fe_coverage``           fraction of bins with RPE data
* ``shot_proj_rmse_logit``  data-only shot committor vs theory (RPE quality baseline)
* ``{m}_proj_rmse_logit``  model q projected on the RPE frames vs logit p_B^theory
* ``{m}_proj_rmse_pB``     the same in p_B
* ``{m}_calib_mean/median`` |logit p_B^RPE(q) - q| along q (binned shot results)
* ``{m}_grad_mag_transition`` weighted mean |grad q| for |q| < 2
* ``delta_q_rms``           weighted RMS of q_next - q_k on the RPE frames with
                            |q_k| < run.convergence.q_window (theory-free convergence measure)

with ``{m}`` = ``q_k`` (the model that placed this iteration's interfaces) and
``q_next`` (the model trained on RPE_k).
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import rpe_mbar
from aimmdTIS import ToyVisualizer


def p_clip_of(acfg):
    """Theory p_B range for the logit comparison: |q_exact| < analysis.q_max_logit
    (p_clip = 1/(1+e^q_max)), or an explicit analysis.p_clip in older configs."""
    if "q_max_logit" in acfg:
        return float(1.0 / (1.0 + np.exp(float(acfg["q_max_logit"]))))
    return float(acfg["p_clip"])


def _visualizer(wf, k):
    theory_path, _ = wf.theory_committor()
    vis = ToyVisualizer(pes=wf.pes, temperature=wf.cfg["system"]["integrator_params"]["temperature"])
    # Analysis RPE: TIS frames only beyond the joined first interface (no double counting
    # with the stable-state MD); the training set keeps all frames.
    ts = np.load(wf.rpe_dir(k) / "rpe_analysis.npz")
    vis.load_trainset({"descriptors": ts["descriptors"], "weights": ts["weights"],
                       "shot_results": ts["shot_results"]})
    vis.set_theory(theory_path)
    return vis, theory_path


def _calibration(vis, model, n_bins):
    """|logit p_B^RPE(q) - q| over bins along q that hold both shot outcomes."""
    q, w, shot = vis._q_and_weights(model)
    rho_A, edges = np.histogram(q, bins=n_bins, weights=w * shot[:, 0])
    rho_B, _ = np.histogram(q, bins=edges, weights=w * shot[:, 1])
    centres = 0.5 * (edges[1:] + edges[:-1])
    ok = (rho_A > 0) & (rho_B > 0)
    dev = np.abs(np.log(rho_B[ok] / rho_A[ok]) - centres[ok])
    return float(np.mean(dev)), float(np.median(dev))


def _gradients(vis, model, q_range, dq):
    """Gradient statistics along q with physical (unscaled) descriptor gradients."""
    q_bins = np.arange(q_range[0], q_range[1] + dq, dq)
    stats = vis.compute_gradient_stats_per_q(
        model, norm_factors=np.ones(vis._desc.shape[1]), q_bins=q_bins, use_cache=True)
    centres, counts = stats["bin_centers"], stats["counts"]
    sel = (np.abs(centres) < 2.0) & (counts > 0)
    mag = float(np.average(stats["gradient_magnitude_per_q_bin"][sel], weights=counts[sel])) if sel.any() else np.nan
    return stats, mag


def model_change(vis, model_old, model_new, q_window=6.0):
    """Weighted RMS of q_new - q_old over the RPE frames with |q_old| < q_window.

    A theory-free convergence measure: how much retraining on the latest
    reweighted path ensemble still changes the model in the transition region.
    """
    _, q_old = vis._model_output_rpe(model_old)
    _, q_new = vis._model_output_rpe(model_new)
    sel = np.abs(q_old) < q_window
    if not sel.any():
        return float("nan")
    return float(np.sqrt(np.average((q_new[sel] - q_old[sel]) ** 2, weights=vis._w[sel])))


def _decorate(vis, ax):
    vis.plot_potential_contour(ax=ax)
    vis.plot_states(ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _figure_free_energy(vis, acfg, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    vmax = 12
    im = vis.plot_rpe_free_energy(n_bins_2d=acfg["n_bins_2d"], ax=axes[0], v_min_max=(0, vmax))
    fig.colorbar(im, ax=axes[0], label=r"$\beta F_{RPE}$ [kT]")
    axes[0].set_title("RPE free energy")
    im = vis.plot_true_free_energy(ax=axes[1], vmax=vmax)
    fig.colorbar(im, ax=axes[1], label=r"$\beta V$ [kT]")
    axes[1].set_title("true potential")
    res = vis.plot_free_energy_error(ax=axes[2], n_bins_2d=acfg["n_bins_2d"], min_count=acfg["min_count"])
    axes[2].set_title(f"RPE - true, RMSE = {res['rmse']:.2f} kT (coverage {res['coverage']:.0%})")
    for ax in axes:
        _decorate(vis, ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return res


def _figure_committor_projection(vis, model, label, theory_path, acfg, path):
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    lim = (-10, 10)
    im = vis.plot_shot_committor(n_bins_2d=acfg["n_bins_2d"], ax=axes[0], v_min_max=lim)
    fig.colorbar(im, ax=axes[0], label=r"$q_{shots}$")
    shot = vis.committor_projection_error(None, n_bins_2d=acfg["n_bins_2d"], p_clip=p_clip_of(acfg),
                                          min_count=acfg["min_count"])
    axes[0].set_title(f"shot committor on RPE (RMSE {shot['rmse_logit']:.2f})")
    im = vis.plot_q_model_projection(model, n_bins_2d=acfg["n_bins_2d"], ax=axes[1], v_min_max=lim)
    fig.colorbar(im, ax=axes[1], label=f"${label}$")
    axes[1].set_title(f"${label}$ projected on RPE")
    p_th, q_th, xe, ye = vis.load_theoretical_committor(theory_path, *np.load(theory_path).shape[::-1])
    im = axes[2].imshow(np.clip(q_th, *lim), origin="lower", aspect="auto", cmap="Spectral",
                        norm=matplotlib.colors.TwoSlopeNorm(vmin=lim[0], vcenter=0, vmax=lim[1]),
                        extent=[xe[0], xe[-1], ye[0], ye[-1]])
    fig.colorbar(im, ax=axes[2], label=r"$q_{theory}$")
    axes[2].set_title("theory committor")
    res = vis.plot_committor_projection_error(model, ax=axes[3], n_bins_2d=acfg["n_bins_2d"],
                                              p_clip=p_clip_of(acfg), min_count=acfg["min_count"])
    axes[3].set_title(f"${label}$ - theory, RMSE {res['rmse_logit']:.2f} (p_B: {res['rmse_pB']:.3f})")
    for ax in axes:
        _decorate(vis, ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return res, shot


def _figure_committor_slice(vis, models, ifaces_k, theory_path, k, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    lim = (-10, 10)
    levels = np.arange(-8, 9, 2)
    _, q_th, xe, ye = vis.load_theoretical_committor(theory_path, *np.load(theory_path).shape[::-1])
    Xt, Yt = np.meshgrid(xe, ye)
    for ax, (label, model) in zip(axes, models.items()):
        im = vis.plot_q_slice_heatmap(model, ax=ax, n_bins_2d=200, v_min_max=lim)
        fig.colorbar(im, ax=ax, label=f"${label}$ (harmonic dims = 0)")
        with np.errstate(invalid="ignore"):
            ax.contour(Xt, Yt, q_th, levels=levels, colors="k", linestyles="--", linewidths=0.8)
        if label == f"q_{k}":
            vis.plot_q_contours(model, ax=ax, levels=ifaces_k["forward"], colors="r", linewidths=1.3)
            vis.plot_q_contours(model, ax=ax, levels=ifaces_k["backward"], colors="b", linewidths=1.3)
            ax.set_title(f"${label}$ with iteration-{k} interfaces (red fwd, blue bwd)")
        else:
            vis.plot_q_contours(model, ax=ax, levels=levels, colors="w", linewidths=1.0)
            ax.set_title(f"${label}$ (white) vs theory (dashed)")
        _decorate(vis, ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _figure_along_q(vis, model, label, n_bins_q, path):
    _, q_all = vis._model_output_rpe(model)
    q_min, q_max = np.floor(np.min(q_all)), np.ceil(np.max(q_all))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    vis.plot_loss_along_q(model, ax=ax, n_bins=n_bins_q, color="black")
    vis.plot_plnp_along_q(model, ax=ax, n_bins=n_bins_q, color="steelblue")
    vis.plot_loss_normalized_along_q(model, ax=ax, n_bins=n_bins_q, color="orange")
    vis.plot_weight_along_q(model, ax=ax, n_bins=n_bins_q, color="seagreen")
    q_theory = np.linspace(q_min, q_max, 100)
    p_B_theory = 1.0 / (1.0 + np.exp(-q_theory))
    information = -(p_B_theory * np.log(p_B_theory) + (1 - p_B_theory) * np.log(1 - p_B_theory))
    ax.plot(q_theory, information, color="darkblue", linestyle="--", lw=2, label=r"Ideal $I(q')$")
    ax.set_yscale("log")
    ax.legend(loc="lower left", fontsize=9, frameon=False, ncol=3)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_xlabel(r"$q'$", fontsize=vis.plot_settings.fontsize)
    ax.set_ylim(bottom=1 / (vis._w.max() * vis._w.max()))

    ax_right = axes[1]
    edges, _, _ = vis.plot_pA_pB_along_q(model, ax=ax_right, n_bins=n_bins_q)
    q_space = edges[1:]
    ax_right.plot(q_space, 1 / (1 + np.exp(q_space)), label=r"$p_A$ theory", linestyle="--", lw=2, color="C0")
    ax_right.plot(q_space, 1 / (1 + np.exp(-q_space)), label=r"$p_B$ theory", linestyle="--", lw=2, color="C1")
    ax_right.set_yscale("log")
    ax_right.set_xlim([q_min, q_max])
    ax_right.set_xlabel(r"$q'$", fontsize=vis.plot_settings.fontsize)
    ax_right.set_ylabel(r"$p_A$, $p_B$", fontsize=vis.plot_settings.fontsize)
    ax_right.legend(loc="lower right", fontsize=9, frameon=False, ncol=2)
    ax_right.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_right.set_ylim(bottom=1 / (vis._w.max() * vis._w.max()))
    fig.suptitle(f"${label}$ along $q'$ on the reweighted path ensemble")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _figure_gradients(vis, model, label, q_range, path):
    q_lo, q_hi = q_range
    vis.plot_gradient_unit_and_magnitude_along_q(
        model,
        q_ranges=[q_lo, -2, 2, q_hi],
        q_spacing=[2, 1, 2],
        descriptor_labels=["x", "y"] + [f"h{i}" for i in range(vis._desc.shape[1] - 2)],
        figsize=(14, 8),
    )
    fig = plt.gcf()
    fig.suptitle(rf"${label}$: $\partial q/\partial d_i$ direction and $|\nabla q|$ along $q$")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def analyse_iteration(wf, k):
    """Metrics and figures for RPE_k with q_k (sampling model) and q_{k+1} (trained on it)."""
    acfg = wf.cfg["analysis"]
    fig_dir = wf.out / "analysis" / f"it{k}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    vis, theory_path = _visualizer(wf, k)
    models = {f"q_{k}": wf.load_model(k), f"q_{k + 1}": wf.load_model(k + 1)}
    new_label, new_model = f"q_{k + 1}", models[f"q_{k + 1}"]

    fe = _figure_free_energy(vis, acfg, fig_dir / "free_energy.png")
    metrics = {"iteration": k, "fe_rmse_kT": fe["rmse"], "fe_rmse_weighted_kT": fe["rmse_weighted"],
               "fe_coverage": fe["coverage"]}

    proj_new, shot = _figure_committor_projection(vis, new_model, new_label, theory_path, acfg,
                                                  fig_dir / "committor_projection.png")
    metrics["shot_proj_rmse_logit"] = shot["rmse_logit"]
    metrics["shot_proj_rmse_pB"] = shot["rmse_pB"]
    for (label, model), key in zip(models.items(), ("q_k", "q_next")):
        proj = proj_new if label == new_label else vis.committor_projection_error(
            model, n_bins_2d=acfg["n_bins_2d"], p_clip=p_clip_of(acfg), min_count=acfg["min_count"])
        metrics[f"{key}_proj_rmse_logit"] = proj["rmse_logit"]
        metrics[f"{key}_proj_rmse_pB"] = proj["rmse_pB"]
        metrics[f"{key}_calib_mean"], metrics[f"{key}_calib_median"] = _calibration(vis, model, acfg["n_bins_q"])
        _, metrics[f"{key}_grad_mag_transition"] = _gradients(vis, model, acfg["grad_q_range"], acfg["grad_dq"])

    metrics["delta_q_rms"] = model_change(vis, models[f"q_{k}"], new_model,
                                          wf.run_cfg.get("convergence", {}).get("q_window", 6.0))

    metrics.update(channel_fractions(wf, k))

    _figure_committor_slice(vis, models, wf.interfaces(k), theory_path, k, fig_dir / "committor_slice.png")
    _figure_along_q(vis, new_model, new_label, int(acfg["n_bins_q"]), fig_dir / "along_q.png")
    _figure_gradients(vis, new_model, new_label, acfg["grad_q_range"], fig_dir / "gradients.png")

    (wf.rpe_dir(k) / "analysis.json").write_text(json.dumps(metrics, indent=2))
    print(f"[it{k}] analysis: " + ", ".join(f"{key}={value:.3g}" for key, value in metrics.items()
                                          if key != "iteration"))
    return metrics


# ----------------------------------------------------------------------------
# Bootstrap of the reactive fraction (used by monitor.py and extras/convergence.py)
# ----------------------------------------------------------------------------
def _without_indices(data):
    return {o: {f: {t: v for t, v in data[o][f].items() if t != "origin_indices"} for f in data[o]} for o in data}


def reactive_fraction_bootstrap(wf, k, direction, n_bootstrap, block_size, seed):
    """P(lambda_B | joined lambda_0) and its block-bootstrap distribution."""
    data, dataset, weights, _ = wf._load_direction(k, direction)
    reactive = wf._path_is_reactive(data, dataset)
    blocks, i = {}, 0
    for origin in data:
        for iface in data[origin]:
            n = len(data[origin][iface][origin])
            blocks[(origin, iface)] = reactive[i:i + n]
            i += n
    p = float(np.sum(weights[reactive]) / np.sum(weights))
    samples = []
    for b in range(n_bootstrap):
        np.random.seed(seed + b)
        boot = rpe_mbar.bootstrap_data(data, block_size)
        clean = _without_indices(boot)
        w_b, _, _ = rpe_mbar.solve_weights_multiset(clean, direction=direction)
        r_b = np.concatenate([blocks[(o, f)][boot[o][f]["origin_indices"]] for o in clean for f in clean[o]])
        samples.append(float(np.sum(w_b[r_b]) / np.sum(w_b)))
    return p, np.asarray(samples)


# ============================================================================
# Walkthrough diagnostics (used by aimmd_tis_walkthrough.ipynb)
#
# Each function draws one diagnostic for one iteration k of a finished (or
# running) run. Only saved outputs are read; nothing is re-sampled.
# ============================================================================
import pandas as pd

import run_iterative as ri


def quiet():
    """Silence setup/solver output of the run code (for notebooks)."""
    for name in ("aimmd", "asyncmd", "openpathsampling"):
        logging.getLogger(name).setLevel(logging.ERROR)
    return contextlib.redirect_stdout(io.StringIO())


def load_iteration(wf, k):
    """Visualizer on RPE_k (analysis set, theory loaded) and the models q_k, q_{k+1}.

    q_{k+1} is ``None`` when it has not been trained yet.
    """
    vis, _ = _visualizer(wf, k)
    q_next = wf.load_model(k + 1) if wf.model_path(k + 1).exists() else None
    return vis, wf.load_model(k), q_next


def iteration_metrics(out):
    """One row per finished iteration: sampling/MBAR diagnostics and analysis metrics."""
    out = Path(out)
    rows, k = [], 0
    while (out / f"rpe_it{k}" / "diagnostics.json").exists():
        d = json.loads((out / f"rpe_it{k}" / "diagnostics.json").read_text())
        ifaces = json.loads((out / f"it{k}" / "interfaces.json").read_text())
        row = {"iteration": k,
               "n_ifaces": len(ifaces["forward"]) + len(ifaces["backward"]),
               "n_paths": d["directions"]["forward"]["n_paths"] + d["directions"]["backward"]["n_paths"],
               "P_forward": d["directions"]["forward"]["p_reactive"],
               "P_backward": d["directions"]["backward"]["p_reactive"],
               "k_AB": d["k_AB"], "k_BA": d["k_BA"], "c_AB": d["c_AB"]}
        for name in ("analysis.json", "p_bootstrap.json"):
            f = out / f"rpe_it{k}" / name
            if f.exists():
                row.update({key: v for key, v in json.loads(f.read_text()).items() if key != "iteration"})
        rows.append(row)
        k += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 1. interfaces
def plot_interfaces(wf, k, axes=None):
    """Where TIS samples in iteration k.

    Left: potential, states and the iteration's interfaces as q_k contours
    (red: from A, blue: from B). Right: q_k of the stable-state MD frames in A and
    B; interfaces are placed between max q in A and min q in B, so no interface
    cuts into a state.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    model = wf.load_model(k)
    ifaces = wf.interfaces(k)
    vis = ToyVisualizer(pes=wf.pes, temperature=wf.cfg["system"]["integrator_params"]["temperature"])
    ax = axes[0]
    vis.plot_potential_contour(ax=ax)
    vis.plot_states(ax=ax)
    vis.plot_q_contours(model, ax=ax, levels=ifaces["forward"], colors="r", linewidths=1.3)
    vis.plot_q_contours(model, ax=ax, levels=ifaces["backward"], colors="b", linewidths=1.3)
    ax.set_title(f"iteration {k}: interfaces on $q_{{{k}}}$")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    _, desc, in_state = wf.stable_data()
    q = ri.q_of(model, desc)
    ax = axes[1]
    bins = np.linspace(np.percentile(q, 0.1), np.percentile(q, 99.9), 120)
    ax.hist(q[in_state[:, 0]], bins=bins, color="C3", alpha=0.6, label="stable MD in A")
    ax.hist(q[in_state[:, 1]], bins=bins, color="C0", alpha=0.6, label="stable MD in B")
    for v in ifaces["forward"]:
        ax.axvline(v, color="r", lw=0.8)
    for v in ifaces["backward"]:
        ax.axvline(v, color="b", lw=0.8, ls="--")
    ax.set_yscale("log"); ax.set_xlabel(f"$q_{{{k}}}$"); ax.set_ylabel("frames")
    ax.set_title(f"{len(ifaces['forward'])} forward / {len(ifaces['backward'])} backward interfaces")
    ax.legend(fontsize=8)
    return axes


# ---------------------------------------------------------------- 2. crossing probability vs ideal
def path_extrema(wf, k, model, direction):
    """Per-path extremum of ``model`` and multi-set MBAR weight for iteration k's paths.

    Paths are the thermalised TIS paths of all interface sets q_0..q_k of one
    direction (from the descriptor storages). The extremum is taken outside the
    initial state, along -q for the backward direction, so a larger value always
    means "further towards the other state".
    """
    with contextlib.redirect_stdout(io.StringIO()):  # MBAR solver / loader progress lines
        data, dataset, weights, _ = wf._load_direction(k, direction)
    sign = 1.0 if direction == "forward" else -1.0
    in_origin = wf.origin_in_state_fn(direction)
    trajs = [(o, f, i) for o in data for f in data[o] for i in range(len(data[o][f][o]))]
    descs = [dataset.get_trajectory_descriptors(o, f, i) for o, f, i in trajs]
    lengths = np.array([len(d) for d in descs])
    allx = np.concatenate(descs)
    q = sign * ri.q_of(model, allx)
    q = np.where(in_origin(allx), -np.inf, q)
    q_ext = np.maximum.reduceat(q, np.concatenate([[0], np.cumsum(lengths)[:-1]]))
    return q_ext, np.asarray(weights)


def plot_crossing_vs_ideal(wf, k, model, direction, ax=None, q_refs=None, label="q"):
    r"""RPE crossing probability in the learned q against the ideal-committor relation.

    For reference interfaces q_i, P(q|q_i) = sum_x w[x] theta(q_max[x]-q) / sum_x w[x]
    over paths crossing q_i (JCP 165, 124125, Eq. 13). For an exact committor it
    equals (1+e^-q)/(1+e^-q_i) (dashed). Agreement tests the model globally;
    dips or jumps reveal non-monotonic "bubbles" in q.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    q_max, w = path_extrema(wf, k, model, direction)
    ok = np.isfinite(q_max)
    q_max, w = q_max[ok], w[ok]
    # Paths ending in the other state plateau at P(other state | q_i); stop at the
    # edge of that state's stable-state q range, where the plateau starts.
    _, desc, in_state = wf.stable_data()
    q_stable = (1.0 if direction == "forward" else -1.0) * ri.q_of(model, desc)
    q_end = np.min(q_stable[in_state[:, 1 if direction == "forward" else 0]])
    if q_refs is None:
        lo = max(np.percentile(q_max, 5), -10.0)
        q_refs = np.linspace(lo, 0.0, 5)
    cmap = plt.get_cmap("viridis", len(q_refs) + 1)
    p_min = 1.0
    for i, qi in enumerate(q_refs):
        crossed = q_max >= qi
        if crossed.sum() < 5:
            continue
        m, wc = q_max[crossed], w[crossed]
        x = np.linspace(qi, max(qi + 1.0, q_end), 200)
        emp = np.array([wc[m >= v].sum() for v in x]) / wc.sum()
        ideal = (1 + np.exp(-x)) / (1 + np.exp(-qi))  # = p_B(q_i) / p_B(q)
        ax.plot(x, emp, color=cmap(i), lw=2, label="RPE" if i == 0 else None)
        ax.plot(x, ideal, "--", color=cmap(i), lw=1.5, label="ideal committor" if i == 0 else None)
        p_min = min(p_min, emp[emp > 0].min())
    ax.set_yscale("log"); ax.set_ylim(max(p_min / 3, 1e-12), 2)
    ax.set_xlabel(f"${label}$" if direction == "forward" else f"$-{label}$")
    ax.set_ylabel(r"$P(q\,|\,q_i)$")
    ax.set_title(f"{direction} (from {'A' if direction == 'forward' else 'B'})")
    ax.grid(alpha=0.4, which="both"); ax.legend(fontsize=8)
    return ax


# ---------------------------------------------------------------- 3. chevron
def plot_chevron(vis, model, axes=None, n_bins_q=400, label="q"):
    r"""Training balance along q (the chevron plot, JCP 165, 124125 Sec. IV).

    Left: weighted loss L(q), its per-point form, RPE density and the information
    I(q) against the ideal entropy-like I(q). Right: p_A, p_B from the RPE shot
    results binned along q against the logistic relation.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4))
    _, q_all = vis._model_output_rpe(model)
    q_min, q_max = np.floor(np.min(q_all)), np.ceil(np.max(q_all))
    ax = axes[0]
    vis.plot_loss_along_q(model, ax=ax, n_bins=n_bins_q, color="black")
    vis.plot_plnp_along_q(model, ax=ax, n_bins=n_bins_q, color="steelblue")
    vis.plot_loss_normalized_along_q(model, ax=ax, n_bins=n_bins_q, color="orange")
    vis.plot_weight_along_q(model, ax=ax, n_bins=n_bins_q, color="seagreen")
    qt = np.linspace(q_min, q_max, 200)
    pb = 1.0 / (1.0 + np.exp(-qt))
    ax.plot(qt, -(pb * np.log(pb) + (1 - pb) * np.log(1 - pb)), "--", color="darkblue", lw=2, label="ideal $I(q)$")
    ax.set_yscale("log"); ax.set_ylim(bottom=1 / (vis._w.max() ** 2))
    ax.set_xlabel(f"${label}$"); ax.grid(alpha=0.4, which="both"); ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax = axes[1]
    edges, _, _ = vis.plot_pA_pB_along_q(model, ax=ax, n_bins=n_bins_q)
    qs = edges[1:]
    ax.plot(qs, 1 / (1 + np.exp(qs)), "--", lw=2, color="C0", label="$p_A$ ideal")
    ax.plot(qs, 1 / (1 + np.exp(-qs)), "--", lw=2, color="C1", label="$p_B$ ideal")
    ax.set_yscale("log"); ax.set_xlim(q_min, q_max); ax.set_ylim(bottom=1 / (vis._w.max() ** 2))
    ax.set_xlabel(f"${label}$"); ax.set_ylabel("$p_A$, $p_B$"); ax.grid(alpha=0.4, which="both"); ax.legend(fontsize=8)
    return axes


# ---------------------------------------------------------------- 4. calibration
def plot_calibration(vis, model, axes=None, n_bins=60, label="q"):
    """Model committor against the committor measured from the RPE shot results.

    Left: logit of the binned RPE p_B against q (diagonal = calibrated).
    Right: predicted against measured p_B per bin, coloured by frames per bin.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    p_model, q = vis._model_output_rpe(model)
    w, shot = vis._w, vis._shot
    edges = np.linspace(np.percentile(q, 0.5), np.percentile(q, 99.5), n_bins + 1)
    idx = np.digitize(q, edges) - 1
    centres, emp, pred, support = [], [], [], []
    for b in range(n_bins):
        sel = idx == b
        tot = np.sum(w[sel] * shot[sel].sum(axis=1))
        if not sel.any() or tot <= 0:
            continue
        centres.append(0.5 * (edges[b] + edges[b + 1]))
        emp.append(np.sum(w[sel] * shot[sel, 1]) / tot)
        pred.append(np.average(p_model[sel], weights=w[sel] * shot[sel].sum(axis=1)))
        support.append(sel.sum())
    centres, emp, pred, support = map(np.asarray, (centres, emp, pred, support))
    ok = (emp > 0) & (emp < 1)
    ax = axes[0]
    ax.plot(centres[ok], np.log(emp[ok] / (1 - emp[ok])), "o", ms=4, label="RPE shot results")
    ax.plot(centres, centres, "k--", label="calibrated")
    ax.set_xlabel(f"model ${label}$"); ax.set_ylabel(r"logit $p_B^{RPE}$"); ax.grid(alpha=0.4); ax.legend(fontsize=8)
    ax = axes[1]
    sc = ax.scatter(pred, emp, c=np.log10(support), s=22, cmap="viridis")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("model $p_B$"); ax.set_ylabel("RPE $p_B$"); ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="log10 frames per bin")
    return axes


# ---------------------------------------------------------------- 5. landscapes
def plot_landscape(vis, model, theory_path, axes=None, n_bins_2d=80, label="q"):
    """Four views of the (x, y) plane, as in the T4L mechanistic analysis.

    RPE free energy, committor measured from the RPE shot results, the model
    committor projected on the RPE frames, and the exact committor.
    """
    if axes is None:
        _, axes = plt.subplots(1, 4, figsize=(22, 4.6))
    lim = (-10, 10)
    levels = np.arange(-8, 9, 2)
    im = vis.plot_rpe_free_energy(n_bins_2d=n_bins_2d, ax=axes[0], v_min_max=(0, 12))
    vis.plot_rpe_free_energy_contours(n_bins_2d=n_bins_2d, ax=axes[0], levels=np.arange(0, 12, 2), smooth_sigma=1.0)
    plt.colorbar(im, ax=axes[0], label=r"$\beta F_{RPE}$")
    axes[0].set_title("RPE free energy")
    im = vis.plot_shot_committor(n_bins_2d=n_bins_2d, ax=axes[1], v_min_max=lim)
    plt.colorbar(im, ax=axes[1], label=r"$\ln(n_B/n_A)$")
    axes[1].set_title("committor from RPE shot results")
    im = vis.plot_q_model_projection(model, n_bins_2d=n_bins_2d, ax=axes[2], v_min_max=lim)
    vis.plot_q_model_projection_contours(model, n_bins_2d=n_bins_2d, ax=axes[2], levels=levels)
    plt.colorbar(im, ax=axes[2], label=f"${label}$")
    axes[2].set_title(f"model ${label}$ projected on the RPE")
    p_th, q_th, xe, ye = vis.load_theoretical_committor(theory_path, *np.load(theory_path).shape[::-1])
    im = axes[3].imshow(np.clip(q_th, *lim), origin="lower", aspect="auto", cmap="Spectral",
                        norm=matplotlib.colors.TwoSlopeNorm(vmin=lim[0], vcenter=0, vmax=lim[1]),
                        extent=[xe[0], xe[-1], ye[0], ye[-1]])
    X, Y = np.meshgrid(xe, ye)
    with np.errstate(invalid="ignore"):
        axes[3].contour(X, Y, q_th, levels=levels, colors="k", linewidths=0.8)
    plt.colorbar(im, ax=axes[3], label=r"$q_{exact}$")
    axes[3].set_title("exact committor")
    for ax in axes:
        _decorate(vis, ax)
        ax.set_xlim(vis.dims_extent[:2]); ax.set_ylim(vis.dims_extent[2:])
    return axes


def plot_errors_vs_theory(vis, model, acfg, axes=None, label="q"):
    """Free-energy error and committor-projection error on the bins the RPE samples."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fe = vis.plot_free_energy_error(ax=axes[0], n_bins_2d=acfg["n_bins_2d"], min_count=acfg["min_count"])
    axes[0].set_title(f"RPE free energy − true: RMSE {fe['rmse']:.2f} kT")
    pr = vis.plot_committor_projection_error(model, ax=axes[1], n_bins_2d=acfg["n_bins_2d"],
                                             p_clip=p_clip_of(acfg), min_count=acfg["min_count"])
    q_lim = np.log((1 - p_clip_of(acfg)) / p_clip_of(acfg))
    axes[1].set_title(f"${label}$ − exact (logit, $|q|<{q_lim:.0f}$): RMSE {pr['rmse_logit']:.2f}")
    for ax in axes:
        _decorate(vis, ax)
    return axes


# ---------------------------------------------------------------- 6. mechanism along q
def plot_descriptor_density(vis, model, axes=None, labels=("x", "y"), label="q"):
    r"""ln rho(xi | q): how each descriptor evolves along the committor (row-normalised)."""
    if axes is None:
        _, axes = plt.subplots(len(labels), 1, figsize=(11, 3.2 * len(labels)), sharex=True)
    _, q = vis._model_output_rpe(model)
    desc = vis._desc_phys if vis._desc_phys is not None else vis._desc
    xedges = np.linspace(np.percentile(q, 0.2), np.percentile(q, 99.8), 150)
    for i, (ax, name) in enumerate(zip(np.atleast_1d(axes), labels)):
        yedges = np.linspace(np.percentile(desc[:, i], 0.1), np.percentile(desc[:, i], 99.9), 80)
        H, _, _ = np.histogram2d(q, desc[:, i], bins=[xedges, yedges], weights=vis._w)
        rows = H.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.log(np.where(rows > 0, H / rows, np.nan))
        im = ax.imshow(logp.T, origin="lower", aspect="auto", cmap="Blues", vmin=-8, vmax=-1,
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.colorbar(im, ax=ax, label=rf"$\ln\rho({name}|{label})$")
        ax.set_ylabel(name); ax.grid(alpha=0.3)
    np.atleast_1d(axes)[-1].set_xlabel(f"${label}$")
    return axes


def plot_gradients(vis, model, q_range=(-12, 12), labels=("x", "y")):
    """Gradient magnitude and unit-gradient components along q (descriptor importance)."""
    vis.plot_gradient_unit_and_magnitude_along_q(
        model, q_ranges=[q_range[0], -2, 2, q_range[1]], q_spacing=[2, 1, 2],
        descriptor_labels=list(labels) + [f"h{i}" for i in range(vis._desc.shape[1] - len(labels))],
        figsize=(14, 7))
    return plt.gcf()


# ---------------------------------------------------------------- 7. convergence
def plot_channel_trend(df, wf=None, ax=None):
    """Right-channel fraction of the reactive paths per iteration (exact value dashed)."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 4))
    for col, lab in (("channel_right_frac_fwd", "forward"), ("channel_right_frac_bwd", "backward")):
        if col in df:
            ax.plot(df["iteration"], df[col], "o-", label=lab)
    if wf is not None:
        ax.axhline(theory_channel_fraction(wf), color="k", ls="--", label="exact reactive flux")
    ax.set_xlabel("iteration"); ax.set_ylabel("right-channel fraction")
    ax.set_title("reaction channels sampled"); ax.grid(alpha=0.4); ax.legend(fontsize=8)
    return ax


def plot_convergence(df, cfg, axes=None):
    """Trends over iterations: free energy, committor, calibration, model change, rates."""
    if axes is None:
        _, axes = plt.subplots(1, 5, figsize=(25, 4))
    it = df["iteration"]
    ax = axes[0]
    for col, lab, st in (("fe_rmse_kT", "unweighted", "o-"), ("fe_rmse_weighted_kT", "density weighted", "s--")):
        if col in df:
            ax.plot(it, df[col], st, label=lab)
    ax.set_title("RPE free energy vs true [kT]")
    ax = axes[1]
    for col, lab in (("q_k_proj_rmse_logit", "$q_k$ (placed interfaces)"),
                     ("q_next_proj_rmse_logit", "$q_{k+1}$ (trained on RPE$_k$)"),
                     ("shot_proj_rmse_logit", "shot committor (no model)")):
        if col in df:
            ax.plot(it, df[col], "o-", label=lab)
    ax.set_title("committor vs exact (logit RMSE)")
    ax = axes[2]
    for col, lab in (("q_k_calib_median", "$q_k$"), ("q_next_calib_median", "$q_{k+1}$")):
        if col in df:
            ax.plot(it, df[col], "o-", label=lab)
    ax.set_title(r"calibration: median $|\mathrm{logit}\,p_B^{RPE}(q)-q|$")
    ax = axes[3]
    if "delta_q_rms" in df:
        ax.semilogy(it, df["delta_q_rms"], "o-", label=r"$\Delta q$")
    tol = cfg["run"].get("convergence", {}).get("delta_q_tol")
    if tol is not None:
        ax.axhline(tol, color="k", ls="--", label=f"stop tolerance {tol}")
    ax.set_title(r"model change $q_{k+1}-q_k$")
    ax = axes[4]
    for rate, c in (("k_AB", "C0"), ("k_BA", "C1")):
        ax.errorbar(it, df[rate], yerr=df.get(f"{rate}_err"), fmt="o-", capsize=3, color=c, label=rate)
    ax.set_yscale("log"); ax.set_title("rates [1/time]")
    for ax in axes:
        ax.set_xlabel("iteration"); ax.grid(alpha=0.4, which="both")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    return axes


# ---------------------------------------------------------------- channels (Wolfe–Quapp)
def channel_crossings(wf, k, direction):
    """x where each reactive RPE path of iteration k first crosses y = 0, with its MBAR weight.

    Wolfe–Quapp has two reaction channels between A (upper left) and B (lower
    right): the left one runs down along x ≈ -1, the right one along x ≈ +1. The
    sign of the y = 0 crossing position tells which channel a path used.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        data, dataset, weights, _ = wf._load_direction(k, direction)
    xs, ws = [], []
    i = 0
    for o in data:
        for f in data[o]:
            for j in range(len(data[o][f][o])):
                shot = dataset.get_trajectory_shot_results(o, f, j)[0]
                if shot[0] == 1 and shot[1] == 1:
                    d = dataset.get_trajectory_descriptors(o, f, j)
                    cross = np.nonzero(np.diff(np.sign(d[:, 1])) != 0)[0]
                    if len(cross):
                        a, b = d[cross[0]], d[cross[0] + 1]
                        t = a[1] / (a[1] - b[1]) if a[1] != b[1] else 0.0
                        xs.append(a[0] + t * (b[0] - a[0]))
                        ws.append(weights[i])
                i += 1
    return np.asarray(xs), np.asarray(ws)


def channel_fractions(wf, k):
    """MBAR-weighted fraction of reactive paths through the right channel (x > 0 at y = 0)."""
    out = {}
    for direction, tag in (("forward", "fwd"), ("backward", "bwd")):
        x, w = channel_crossings(wf, k, direction)
        out[f"channel_right_frac_{tag}"] = float(np.sum(w[x > 0]) / np.sum(w)) if len(w) else float("nan")
        out[f"n_reactive_{tag}"] = int(len(w))
    return out


def plot_channels(wf, k, ax=None):
    """Weighted histogram of where reactive paths cross y = 0 (left vs right channel)."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5.5, 3.6))
    bins = np.linspace(-2, 2, 41)
    for direction, c in (("forward", "C3"), ("backward", "C0")):
        x, w = channel_crossings(wf, k, direction)
        if len(w) == 0:
            continue
        right = np.sum(w[x > 0]) / np.sum(w)
        ax.hist(x, bins=bins, weights=w / np.sum(w), histtype="step", lw=2, color=c,
                label=f"{direction}: right channel {right:.0%} ({len(w)} paths)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("x at first crossing of y = 0"); ax.set_ylabel("reweighted fraction")
    ax.set_title(f"iteration {k}: reaction channels (exact: right {theory_channel_fraction(wf):.0%})")
    ax.legend(fontsize=8); ax.grid(alpha=0.4)
    return ax


def theory_channel_fraction(wf):
    """Exact fraction of the reactive flux through the right channel at y = 0.

    Reactive flux density J ∝ exp(-βV) ∇p_B (overdamped dynamics, constant
    diffusion), from the exact committor in theory/; integrated over x < 0 and
    x > 0 along the line y = 0.
    """
    p = np.load(wf.theory_path())
    e = wf.pes.extent
    x = np.linspace(e[0], e[1], p.shape[1])
    y = np.linspace(e[2], e[3], p.shape[0])
    iy = int(np.argmin(np.abs(y)))
    V = np.array([wf.pes.pes_2d_pot(xi, y[iy]) for xi in x])
    jy = np.exp(-wf.beta * (V - V.min())) * np.gradient(p, y, axis=0)[iy]
    return float(jy[x > 0].sum() / jy.sum())
