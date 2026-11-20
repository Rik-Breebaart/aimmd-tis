"""Live monitoring of a running (or finished) run_iterative.py output folder.

Reads only files that each stage writes when it finishes (``.done`` markers,
``diagnostics.json``, ``analysis.json``, ``crossing_*.npz``, figures, the run log),
so it can be used while the run is in progress. Used by ``monitor_run.ipynb``.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import run_iterative as ri

STAGES = ("tis", "rpe", "train", "analysis")


def is_running(out: Path) -> bool:
    """True while a run_iterative.py process writes to this output folder."""
    pattern = f"run_iterative.py.*--out [^ ]*{Path(out).name}( |$)"
    res = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return bool(res.stdout.strip())


def iteration_status(out: Path) -> pd.DataFrame:
    """Stage completion per iteration, plus TIS ensembles finished."""
    out = Path(out)
    rows = []
    k = 0
    while (out / f"it{k}" / "interfaces.json").exists():
        ifaces = json.loads((out / f"it{k}" / "interfaces.json").read_text())
        n_ens = sum(len(v) for v in ifaces.values())
        n_done = sum(len(list((out / f"it{k}" / d).glob(".tis_i*.done"))) for d in ri.DIRECTIONS)
        rows.append({
            "iteration": k,
            "ensembles": f"{n_done}/{n_ens}",
            "tis": all((out / f"it{k}" / d / ".tis.done").exists() for d in ri.DIRECTIONS),
            "rpe": (out / f"rpe_it{k}" / ".rpe.done").exists(),
            "train": (out / "models" / f".train_it{k + 1}.done").exists(),
            "analysis": (out / f"rpe_it{k}" / ".analysis.done").exists(),
        })
        k += 1
    return pd.DataFrame(rows)


def latest_log_lines(out: Path, n: int = 3) -> list[str]:
    """Most recent stage / epoch lines of the run log."""
    log = Path(out) / "run.log"
    if not log.exists():
        return []
    lines = log.read_text(errors="ignore").splitlines()
    keep = [l for l in lines if re.match(r"^\[(it\d+|tps|stable)\]|^=====|converged", l)
            and "EMA_Test" not in l and "__getattribute__" not in l]
    epochs = [l for l in keep if " Epoch " in l]
    other = [l for l in keep if " Epoch " not in l]
    return other[-n:] + epochs[-1:]


def collect(out: Path) -> pd.DataFrame:
    """Per-iteration diagnostics + analysis metrics available so far."""
    out = Path(out)
    rows = []
    k = 0
    while (out / f"rpe_it{k}" / "diagnostics.json").exists():
        d = json.loads((out / f"rpe_it{k}" / "diagnostics.json").read_text())
        row = {"iteration": k, "k_AB": d["k_AB"], "k_BA": d["k_BA"], "c_AB": d["c_AB"],
               "P_forward": d["directions"]["forward"]["p_reactive"],
               "P_backward": d["directions"]["backward"]["p_reactive"],
               "tis_weight_before_joined_fwd": d.get("tis_weight_before_joined_fwd"),
               "tis_weight_before_joined_bwd": d.get("tis_weight_before_joined_bwd")}
        a = out / f"rpe_it{k}" / "analysis.json"
        if a.exists():
            row.update({key: v for key, v in json.loads(a.read_text()).items() if key != "iteration"})
        b = out / f"rpe_it{k}" / "p_bootstrap.json"
        if b.exists():
            row.update(json.loads(b.read_text()))
        rows.append(row)
        k += 1
    return pd.DataFrame(rows)


def update_bootstrap(wf, n_bootstrap: int = 30) -> None:
    """Bootstrap errors of P(lambda_B | joined lambda_0) and rates for finished iterations (cached)."""
    from diagnostics import reactive_fraction_bootstrap

    data_stable = None
    frame_time = float(wf.cfg["system"]["integrator_params"]["dt"]) * int(getattr(wf.engine, "n_steps_per_frame", 1))
    k = 0
    while (wf.rpe_dir(k) / ".rpe.done").exists():
        target = wf.rpe_dir(k) / "p_bootstrap.json"
        if not target.exists():
            if data_stable is None:
                data_stable, _, _ = wf.stable_data()
            diag = json.loads((wf.rpe_dir(k) / "diagnostics.json").read_text())
            res = {}
            for idx, (direction, state, rate) in enumerate((("forward", "A", "k_AB"), ("backward", "B", "k_BA"))):
                p, samples = reactive_fraction_bootstrap(wf, k, direction, n_bootstrap,
                                                         int(wf.run_cfg["block_size"]), int(wf.cfg["system"].get("seed", 0)))
                p_err = float(np.std(samples))
                n_cross = diag[f"flux_{state}"] * len(data_stable["descriptors"][idx]) * frame_time
                rel = float(np.sqrt(1.0 / max(n_cross, 1.0) + (p_err / p) ** 2)) if p > 0 else float("nan")
                res.update({f"P_{direction}_err": p_err, f"{rate}_err": diag[rate] * rel})
            target.write_text(json.dumps(res, indent=2))
        k += 1


def dashboard(out: Path, cfg: dict, figsize=(25, 8.5)):
    """Convergence trends (diagnostics.plot_convergence), reactive fraction and crossing probabilities."""
    from diagnostics import plot_convergence

    out = Path(out)
    df = collect(out)
    if df.empty:
        return None, df
    fig, axes = plt.subplots(2, 5, figsize=figsize)
    plot_convergence(df, cfg, axes=axes[0])
    it = df["iteration"]

    ax = axes[1, 0]
    for direction, lab, c in (("forward", r"$P(\lambda_B|\lambda_0^A)$", "C0"), ("backward", r"$P(\lambda_A|\lambda_0^B)$", "C1")):
        ax.errorbar(it, df[f"P_{direction}"], yerr=df.get(f"P_{direction}_err"), fmt="o-", capsize=3, color=c, label=lab)
    ax.set_yscale("log"); ax.set_title("reactive fraction (joined interface)"); ax.set_xlabel("iteration")

    cmap = plt.get_cmap("viridis", max(len(df), 2))
    for ax, direction in ((axes[1, 1], "forward"), (axes[1, 2], "backward")):
        p_min = 1.0
        for i, k in enumerate(it):
            f = out / f"rpe_it{k}" / f"crossing_{direction}.npz"
            if not f.exists():
                continue
            cp = np.load(f)
            mean = np.where(cp["mean"] > 0, cp["mean"], np.nan)
            ax.plot(cp["bins"], mean, color=cmap(i), label=f"it {k}")
            ax.fill_between(cp["bins"], np.clip(mean - cp["std"], np.nanmin(mean) / 10, None), mean + cp["std"],
                            color=cmap(i), alpha=0.2)
            p_min = min(p_min, np.nanmin(mean))
        ax.set_yscale("log"); ax.set_ylim(p_min / 5, 2)
        ax.set_title(f"{direction}: $P(\\lambda|\\lambda_0)$ in $q_k$ (bootstrap bands)")
        ax.set_xlabel("$q_k$" if direction == "forward" else "$-q_k$")
    # Statistical precision over iterations: relative bootstrap errors.
    ax = axes[1, 3]
    for direction, c in (("forward", "C0"), ("backward", "C1")):
        if f"P_{direction}_err" in df:
            ax.semilogy(it, df[f"P_{direction}_err"] / df[f"P_{direction}"], "o-", color=c, label=direction)
    ax.set_title(r"relative bootstrap error of $P(\lambda_B|\lambda_0)$"); ax.set_xlabel("iteration")

    ax = axes[1, 4]
    for direction, c in (("forward", "C0"), ("backward", "C1")):
        its, med = [], []
        for k in it:
            f = out / f"rpe_it{k}" / f"crossing_{direction}.npz"
            if not f.exists():
                continue
            cp = np.load(f)
            ok = cp["mean"] > 0
            if ok.any():
                its.append(k)
                med.append(np.median(cp["std"][ok] / cp["mean"][ok]))
        ax.semilogy(its, med, "o-", color=c, label=direction)
    ax.set_title(r"median relative error along $P(\lambda|\lambda_0)$"); ax.set_xlabel("iteration")

    for ax in axes[1]:
        ax.grid(alpha=0.4, which="both")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, df


def latest_figures(out: Path, names=("free_energy", "committor_projection", "committor_slice", "along_q")):
    """Paths of the newest iteration's analysis figures."""
    out = Path(out)
    its = sorted((int(p.name[2:]) for p in (out / "analysis").glob("it*") if p.name[2:].isdigit()), reverse=True)
    for k in its:
        paths = [out / "analysis" / f"it{k}" / f"{n}.png" for n in names]
        if all(p.exists() for p in paths):
            return k, paths
    return None, []
