#!/usr/bin/env python
"""
Convergence analysis of an iterative AIMMD-TIS run (output of run_iterative.py).

For every iteration k (multi-set MBAR over the interface sets q_0..q_k):

* P(lambda_B | joined lambda_0) per direction with a block-bootstrap error: every
  interface ensemble is block-resampled, the multi-set MBAR is re-solved and the
  reweighted fraction of reactive paths recomputed;
* rates k_AB, k_BA = flux * P, with the flux error from Poisson statistics of the
  first crossings through the joined interface in the stable-state MD;
* the crossing probability P(lambda | lambda_0) expressed in one common coordinate,
  the last sampling model q_K, so the curves of all iterations are comparable,
  with bootstrap error bands (rpe_mbar.bootstrap_crossing_probs);
* the logged per-iteration metrics (free-energy and committor errors vs theory,
  calibration, model change delta_q) from summary.csv.

Writes ``<out>/analysis/convergence/`` (figures, convergence.csv).

Usage:  python extras/convergence.py --out results_it_100_steps [--n-bootstrap 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# The example folder (one level up) holds run_iterative.py and diagnostics.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rpe_mbar
import run_iterative as ri
from diagnostics import reactive_fraction_bootstrap


def extrema_in_common_q(wf, k, direction, K):
    """Thermalised extrema dict of origin sets q_0..q_k with targets q_0..q_k and q_K."""
    targets = sorted(set(range(k + 1)) | {K})
    specs = [rpe_mbar.InterfaceFunctionSpec(filename=ri.origin_name(j), cv_column_base=ri.cv_column(j), iteration=j)
             for j in targets]
    data = {}
    for j in range(k + 1):
        name = ri.origin_name(j)
        rows = wf.usable_rows(j, direction)  # same ensembles as the reweighting
        data_j, _, _ = rpe_mbar.load_extrema_as_dict_iterative(
            run_folder=wf.it_dir(j, direction), direction=direction, interface_functions=specs,
            interfaces_by_origin={name: [float(r["interface_value"]) for r in rows]},
            n_thermalization=0, iterations=[j], strict=True)
        for r in rows:
            skip = wf.thermalization_step(r)
            block = data_j[name][f"{float(r['interface_value'])}"]
            for t in block:
                block[t] = np.asarray(block[t])[skip:]
        data[name] = data_j[name]
    return data


def crossing_in_common_q(wf, k, direction, K, n_bootstrap, block_size, seed, n_bins=300):
    data = extrema_in_common_q(wf, k, direction, K)
    lam0 = wf.interfaces(K)[direction][0]
    lam0 = lam0 if direction == "forward" else -lam0
    labeled = np.concatenate([data[o][f][ri.origin_name(K)] for o in data for f in data[o]])
    top = np.nanmax(labeled if direction == "forward" else -labeled)
    bins = np.linspace(lam0, top, n_bins)
    res = rpe_mbar.bootstrap_crossing_probs(
        data=data, interface_0=lam0, n_bootstrap=n_bootstrap, block_size=block_size,
        set_analysis=ri.origin_name(K), direction=direction, bins=bins, seed=seed)
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ri.HERE / "config.json")
    parser.add_argument("--n-bootstrap", type=int, default=50)
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    plt.switch_backend("Agg")  # script use only; importing this module keeps the caller's backend

    cfg = ri.load_config(args.config)
    wf = ri.IterativeAimmdTIS(cfg, args.out, n_workers=1)
    summary = pd.read_csv(wf.out / "summary.csv")
    iterations = [int(k) for k in summary["iteration"]]
    K = iterations[-1]
    block = int(cfg["run"]["block_size"])
    seed = int(cfg["system"].get("seed", 0))
    fig_dir = wf.out / "analysis" / "convergence"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_stable, _, _ = wf.stable_data()
    frame_time = float(cfg["system"]["integrator_params"]["dt"]) * int(getattr(wf.engine, "n_steps_per_frame", 1))

    rows, crossing = [], {}
    for k in iterations:
        print(f"[convergence] iteration {k}")
        diag = json.loads((wf.rpe_dir(k) / "diagnostics.json").read_text())
        row = {"iteration": k}
        for idx, (direction, state, rate) in enumerate((("forward", "A", "k_AB"), ("backward", "B", "k_BA"))):
            p, samples = reactive_fraction_bootstrap(wf, k, direction, args.n_bootstrap, block, seed)
            p_err = float(np.std(samples))
            t_total = len(data_stable["descriptors"][idx]) * frame_time
            flux = diag[f"flux_{state}"]
            n_cross = flux * t_total
            rel = np.sqrt(1.0 / max(n_cross, 1.0) + (p_err / p) ** 2 if p > 0 else np.nan)
            row.update({f"P_{direction}": p, f"P_{direction}_err": p_err,
                        f"P_{direction}_log10_err": float(np.std(np.log10(np.clip(samples, 1e-300, None)))),
                        f"flux_{state}": flux, f"n_first_crossings_{state}": n_cross,
                        rate: flux * p, f"{rate}_err": flux * p * rel})
            crossing[(k, direction)] = crossing_in_common_q(wf, k, direction, K, args.n_bootstrap, block, seed)
        rows.append(row)
    conv = pd.DataFrame(rows).merge(summary, on="iteration", how="left", suffixes=("", "_summary"))
    conv.to_csv(fig_dir / "convergence.csv", index=False)

    # ---------------- figure 1: metrics vs iteration ----------------
    tol = cfg["run"].get("convergence", {}).get("delta_q_tol", 0.3)
    fig, axes = plt.subplots(1, 4, figsize=(21, 4))
    it = conv["iteration"]
    axes[0].plot(it, conv["fe_rmse_kT"], "o-", label="unweighted")
    axes[0].plot(it, conv["fe_rmse_weighted_kT"], "s--", label="RPE-density weighted")
    axes[0].set_ylabel("RPE free energy vs true [kT]")
    for col, lab in (("q_k_proj_rmse_logit", "$q_k$ (sampling)"), ("q_next_proj_rmse_logit", "$q_{k+1}$ (trained)"),
                     ("shot_proj_rmse_logit", "shot committor")):
        axes[1].plot(it, conv[col], "o-", label=lab)
    axes[1].set_ylabel("committor projection vs theory (logit RMSE)")
    for col, lab in (("q_k_calib_median", "$q_k$"), ("q_next_calib_median", "$q_{k+1}$")):
        axes[2].plot(it, conv[col], "o-", label=lab)
    axes[2].set_ylabel(r"median $|\mathrm{logit}\,p_B^{RPE}(q) - q|$")
    if "delta_q_rms" in conv:
        axes[3].semilogy(it, conv["delta_q_rms"], "o-", label=r"$\Delta q$ (weighted RMS, $|q_k|<$window)")
        axes[3].axhline(tol, color="k", ls="--", label=f"tolerance {tol}")
    axes[3].set_ylabel(r"model change $q_{k+1}-q_k$")
    for ax in axes:
        ax.set_xlabel("iteration"); ax.grid(alpha=0.4); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "metrics.png", dpi=110); plt.close(fig)

    # ---------------- figure 2: P(lambda_B) and rates with bootstrap errors ----------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for direction, lab, c in (("forward", r"$P(\lambda_B|\lambda_0^A)$", "C0"), ("backward", r"$P(\lambda_A|\lambda_0^B)$", "C1")):
        axes[0].errorbar(it, conv[f"P_{direction}"], yerr=conv[f"P_{direction}_err"], fmt="o-", capsize=3, color=c, label=lab)
    axes[0].set_yscale("log"); axes[0].set_ylabel("reactive fraction (joined interface)")
    for rate, c in (("k_AB", "C0"), ("k_BA", "C1")):
        axes[1].errorbar(it, conv[rate], yerr=conv[f"{rate}_err"], fmt="o-", capsize=3, color=c, label=f"${rate[0]}_{{{rate[2:]}}}$")
    axes[1].set_yscale("log"); axes[1].set_ylabel("rate [1/time]")
    for direction, c in (("forward", "C0"), ("backward", "C1")):
        axes[2].plot(it, conv[f"P_{direction}"].pipe(lambda s: conv[f"P_{direction}_err"] / s), "o-", color=c, label=direction)
    axes[2].set_ylabel("relative bootstrap error of P")
    for ax in axes:
        ax.set_xlabel("iteration"); ax.grid(alpha=0.4, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "rates.png", dpi=110); plt.close(fig)

    # ---------------- figure 3: crossing probabilities in the common coordinate q_K ----------------
    cmap = plt.get_cmap("viridis", len(iterations))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
    for col, direction in enumerate(("forward", "backward")):
        for i, k in enumerate(iterations):
            res = crossing[(k, direction)]
            mean = np.where(res["bootstrap_mean"] > 0, res["bootstrap_mean"], np.nan)  # empty bins
            std = res["bootstrap_std"]
            axes[0, col].plot(res["bins"], mean, color=cmap(i), label=f"it {k}")
            axes[0, col].fill_between(res["bins"], np.clip(mean - std, np.nanmin(mean) / 10, None), mean + std,
                                      color=cmap(i), alpha=0.2)
            with np.errstate(invalid="ignore", divide="ignore"):
                axes[1, col].plot(res["bins"], std / mean, color=cmap(i))
            p_min = np.nanmin(mean) if i == 0 else min(p_min, np.nanmin(mean))
        axes[0, col].set_yscale("log")
        axes[0, col].set_ylim(p_min / 5, 2)
        axes[1, col].set_ylim(1e-3, 3)
        axes[0, col].set_title(f"{direction}: $P(\\lambda|\\lambda_0)$ in $q_{{{K}}}$ (MBAR over sets $q_0..q_k$)")
        axes[0, col].legend(fontsize=8, ncol=2)
        axes[1, col].set_yscale("log")
        axes[1, col].set_ylabel("relative bootstrap error")
        axes[1, col].set_xlabel(f"$q_{{{K}}}$" if direction == "forward" else f"$-q_{{{K}}}$")
        for ax in axes[:, col]:
            ax.grid(alpha=0.4, which="both")
    axes[0, 0].set_ylabel(r"$P(\lambda | \lambda_0)$")
    fig.tight_layout(); fig.savefig(fig_dir / "crossing_probabilities.png", dpi=110); plt.close(fig)

    print(conv[["iteration", "P_forward", "P_forward_err", "P_backward", "P_backward_err",
                "k_AB", "k_AB_err", "k_BA", "k_BA_err"]].to_string(index=False))
    print(f"figures and convergence.csv in {fig_dir}")


if __name__ == "__main__":
    main()
