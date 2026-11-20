#!/usr/bin/env python
"""
Offline training control for an iterative AIMMD-TIS run.

Retrains q_{k+1} on the existing reweighted training sets ``rpe_it{k}/trainset.npz``
(no new sampling) with a chosen initialisation and training-parameter variant,
and evaluates every model on the same analysis set ``rpe_it{k}/rpe_analysis.npz``
against the theory committor. The original run's models (warm-started from q_k)
are evaluated the same way, so the comparison is like for like.

    python extras/retrain_control.py --out results_it_100_steps --iterations 1-10 --init scratch
    python extras/retrain_control.py --out results_it_100_steps --iterations 2,9 --init warm --variant fast_lr

Results: ``<out>/control/<variant>_<init>/`` (models, training histories, metrics.csv).
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# The example folder (one level up) holds run_iterative.py and diagnostics.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Training-parameter variants (overrides of config["training"]).
VARIANTS = {
    "default": {},
    # only the learning rate raised (cold-restart test)
    "lr_1e-3": {"base_lr": 1e-3},
    # higher learning rate, LR decay that actually triggers before early stopping
    "fast_lr": {"base_lr": 1e-3, "plateau_patience": 10, "plateau_factor": 0.5,
                "early_stop_patience": 40, "epochs": 600},
    # intermediate learning rate, longer budget
    "mid_lr_long": {"base_lr": 3e-4, "plateau_patience": 15, "plateau_factor": 0.5,
                    "early_stop_patience": 50, "epochs": 800},
    # smoothness penalty scan (ee_params.smoothness_penalty_weight; the config uses 1e-6)
    "smooth_0": {"_ee": {"smoothness_penalty_weight": 0.0}},
    "smooth_1e-5": {"_ee": {"smoothness_penalty_weight": 1e-5}},
    "smooth_1e-4": {"_ee": {"smoothness_penalty_weight": 1e-4}},
}


def _parse_iterations(text):
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def model_metrics(wf, k, model):
    """Committor projection vs theory and calibration of ``model`` on RPE_k (analysis set)."""
    from aimmdTIS import ToyVisualizer
    from diagnostics import _calibration, p_clip_of

    acfg = wf.cfg["analysis"]
    vis = ToyVisualizer(pes=wf.pes, temperature=wf.cfg["system"]["integrator_params"]["temperature"])
    ts = np.load(wf.rpe_dir(k) / "rpe_analysis.npz")
    vis.load_trainset({"descriptors": ts["descriptors"], "weights": ts["weights"], "shot_results": ts["shot_results"]})
    vis.set_theory(wf.theory_committor()[0])
    proj = vis.committor_projection_error(model, n_bins_2d=acfg["n_bins_2d"], p_clip=p_clip_of(acfg),
                                          min_count=acfg["min_count"])
    calib_mean, calib_median = _calibration(vis, model, acfg["n_bins_q"])
    return {"proj_rmse_logit": proj["rmse_logit"], "proj_rmse_pB": proj["rmse_pB"],
            "calib_mean": calib_mean, "calib_median": calib_median,
            **smoothness_metrics(wf, vis, model, proj["mask_logit"])}


def smoothness_metrics(wf, vis, model, mask):
    """Smoothness of q on the analysis grid, over the sampled transition bins ``mask``.

    * grad_mag_mean      mean |grad q| (model) over the bins
    * grad_err_rms       RMS of |grad q_model| - |grad q_theory|
    * laplacian_rms      RMS of the discrete Laplacian of q_model (roughness)
    * laplacian_err_rms  RMS of lap q_model - lap q_theory
    """
    import run_iterative as ri

    xedges, yedges, Xc, Yc = vis._bin_centres(wf.cfg["analysis"]["n_bins_2d"])
    q = ri.q_of(model, wf.grid_descriptors(Xc, Yc)).reshape(Xc.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = vis.theory_committor_at(Xc, Yc)
        q_th = np.log(p / (1.0 - p))
    dx, dy = xedges[1] - xedges[0], yedges[1] - yedges[0]

    def grad_lap(f):
        gy, gx = np.gradient(f, dy, dx)
        lap = np.gradient(gx, dx, axis=1) + np.gradient(gy, dy, axis=0)
        return np.hypot(gx, gy), lap

    g_m, lap_m = grad_lap(q)
    g_t, lap_t = grad_lap(q_th)
    inner = mask.copy()
    inner[[0, -1], :] = False
    inner[:, [0, -1]] = False
    ok = inner & np.isfinite(g_t) & np.isfinite(lap_t)
    if not ok.any():
        return {}
    return {"grad_mag_mean": float(np.mean(g_m[ok])),
            "grad_err_rms": float(np.sqrt(np.mean((g_m[ok] - g_t[ok]) ** 2))),
            "laplacian_rms": float(np.sqrt(np.mean(lap_m[ok] ** 2))),
            "laplacian_err_rms": float(np.sqrt(np.mean((lap_m[ok] - lap_t[ok]) ** 2)))}


def train_job(out, k, init, variant, threads):
    """Train q_{k+1} on rpe_it{k} with the given init/variant; return metrics."""
    import torch
    torch.set_num_threads(threads)
    import aimmd
    import aimmdTIS
    import run_iterative as ri
    from aimmdTIS.training import save_training_results

    cfg = ri.load_run_config(out)
    wf = ri.IterativeAimmdTIS(cfg, out, n_workers=1)
    dest = wf.out / "control" / f"{variant}_{init}"
    dest.mkdir(parents=True, exist_ok=True)
    metrics_path = dest / f"metrics_it{k}.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())

    seed = int(cfg["system"].get("seed", 0))
    torch.manual_seed(seed + 100 + k)
    np.random.seed(seed + 100 + k)
    train_cfg = {key: v for key, v in copy.deepcopy(cfg["training"]).items() if not key.startswith("_")}
    overrides = copy.deepcopy(VARIANTS[variant])
    ee_overrides = overrides.pop("_ee", {})
    train_cfg.update(overrides)
    optimizer_cfg = {**train_cfg.pop("optimizer"), "lr": train_cfg["base_lr"]}

    ts = np.load(wf.rpe_dir(k) / "trainset.npz")
    # Same split as the original run (run_training uses seed + k).
    train_data, test_data = aimmdTIS.create_train_test_split(
        ts["descriptors"], ts["shot_results"], weights=ts["weights"], split=[4, 1], seed=seed + k)
    n_states = len(wf.states)
    train_set = aimmd.TrainSet(n_states=n_states, descriptors=train_data[0], shot_results=train_data[1], weights=train_data[2])
    test_set = aimmd.TrainSet(n_states=n_states, descriptors=test_data[0], shot_results=test_data[1], weights=test_data[2])

    store = aimmd.Storage(str(dest / f"aimmd_it{k + 1}.h5"), "w")
    warm = wf.model_path(k) if init == "warm" else None
    model = wf.aimmd_setup.setup_RCModel(store, load_model_path=warm)
    model.ee_params.update(ee_overrides)  # e.g. smoothness_penalty_weight
    t0 = time.time()
    results = aimmdTIS.train_one_stage(store, model, train_set, test_set, stage_name=f"it{k + 1}",
                                       optimizer_cfg=optimizer_cfg, **train_cfg)
    elapsed = time.time() - t0
    store.close()
    save_training_results(results, dest / f"training_it{k + 1}.json")
    model.nnet.eval()
    row = {"iteration": k, "init": init, "variant": variant, "train_frames": int(len(ts["weights"])),
           "smoothness_penalty_weight": float(model.ee_params.get("smoothness_penalty_weight", 0.0)),
           "epochs": len(results["test_total"]), "best_test_loss": float(np.min(results["test_total"])),
           "train_minutes": elapsed / 60, **model_metrics(wf, k, model)}
    metrics_path.write_text(json.dumps(row, indent=2))
    return row


def original_row(out, k):
    """The run's own (warm-started) q_{k+1}, evaluated the same way."""
    import run_iterative as ri

    wf = ri.IterativeAimmdTIS(ri.load_run_config(out), out, n_workers=1)
    hist = json.loads((wf.out / "models" / f"training_it{k + 1}.json").read_text())
    return {"iteration": k, "init": "warm (run)", "variant": "default", "train_frames": None,
            "epochs": len(hist["test_total"]), "best_test_loss": float(np.min(hist["test_total"])),
            "train_minutes": None, **model_metrics(wf, k, wf.load_model(k + 1))}


def rescore(out, variant, init, its):
    """Recompute the metrics of already trained control models (and the run's own)."""
    import aimmd
    import run_iterative as ri

    wf = ri.IterativeAimmdTIS(ri.load_run_config(out), out, n_workers=1)
    dest = wf.out / "control" / f"{variant}_{init}"
    rows = []
    for k in its:
        path = dest / f"metrics_it{k}.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        store = aimmd.Storage(str(dest / f"aimmd_it{k + 1}.h5"), "r")
        model = store.rcmodels["most_recent"]
        store.close()
        model.nnet.to("cpu").eval()
        row.update(model_metrics(wf, k, model))
        path.write_text(json.dumps(row, indent=2))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", default="1-10", help="trainsets rpe_it{k}, e.g. 1-10 or 2,9")
    parser.add_argument("--init", choices=["scratch", "warm"], default="scratch")
    parser.add_argument("--variant", choices=list(VARIANTS), default="default")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=3, help="torch threads per training process")
    parser.add_argument("--with-original", action="store_true", help="also evaluate the run's own models")
    parser.add_argument("--rescore", action="store_true", help="only recompute metrics of trained models")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    its = _parse_iterations(args.iterations)
    dest = Path(args.out) / "control" / f"{args.variant}_{args.init}"
    if args.rescore:
        rows = rescore(str(args.out), args.variant, args.init, its)
        if args.with_original:
            rows += [original_row(str(args.out), k) for k in its]
        df = pd.DataFrame(rows).sort_values(["iteration", "init"])
        df.to_csv(dest / "metrics.csv", index=False)
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        return
    rows = []
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(train_job, str(args.out), k, args.init, args.variant, args.threads): k for k in its}
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(f"[control] {args.variant}/{args.init} it{row['iteration']}: "
                  f"logit {row['proj_rmse_logit']:.3f} pB {row['proj_rmse_pB']:.4f} "
                  f"calib {row['calib_median']:.3f} test {row['best_test_loss']:.4f} "
                  f"epochs {row['epochs']} ({row['train_minutes']:.1f} min)")
    if args.with_original:
        rows += [original_row(str(args.out), k) for k in its]
    df = pd.DataFrame(rows).sort_values(["iteration", "init"])
    df.to_csv(dest / "metrics.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
