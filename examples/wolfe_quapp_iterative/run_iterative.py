#!/usr/bin/env python
"""
End-to-end iterative AIMMD-TIS on the Wolfe-Quapp toy potential.

Pipeline
--------
1. Seed TPS with an AIMMD selector and online training  -> model q_0
2. Stable-state MD in A and B                            -> flux / interface placement data
3. For every iteration k = 0 .. N-1:
     a. place forward/backward interfaces on q_k from the stable-state q range
     b. run AIMMD-TIS (forward and backward, one process per interface) on q_k. Each
        path's descriptors and shot results go to a per-interface descriptor storage
        and its extrema to the extrema CSV while sampling (index = MC step)
     c. back-fill the extrema of every model q_0..q_k into every iteration's extrema
        CSVs from those descriptor storages (no OPS storage is read back)
     d. multi-set MBAR over all origin sets q_it0..q_itk (rpe_mbar) -> reweighted path
        ensemble, with the RpeDataset built from the descriptor storages
     e. train q_{k+1} on the reweighted ensemble with aimmdTIS.train_one_stage
     f. analysis (analysis.py): RPE free energy vs the true potential, the committor
        projected on the RPE vs the exact committor (solved once, saved in theory/),
        calibration and gradients along q, rates and crossing probabilities

Every stage writes a ``.done`` marker and is skipped when rerun, so an interrupted
run can be resumed with the same command.

Usage
-----
    python run_iterative.py                       # full run, results in ./results
    python run_iterative.py --quick --out quick   # small smoke-test run (minutes on CPU)
    python run_iterative.py --stage tps           # run a single stage
    python run_iterative.py --workers 4           # TIS interface ensembles run in parallel processes

The companion notebook ``wolfe_quapp_iterative.ipynb`` imports this module for plotting.
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# aimmd.Storage still calls np.string_, which NumPy 2 removed.
if not hasattr(np, "string_"):
    np.string_ = np.bytes_

import openpathsampling as paths
import aimmd
import aimmdTIS
import rpe_mbar
from aimmdTIS.Tools import interface_indicator
from aimmdTIS.TIS_AIMMD_setup import _coordinates_descriptor_transform
from aimmdTIS.aimmd_setup import AIMMDSetup
from aimmdTIS.diagnostics.interface_placement import check_interfaces
from aimmdTIS.training import save_training_results, plot_training_diagnostics
from ops_setup.systems.examples.toy_systems import ToyTPSSetup
from ops_setup.descriptor_storage import build_stable_state_storage, load_stable_state_payload

HERE = Path(__file__).resolve().parent
THEORY_DIR = HERE / "theory"
SUMMARY_PATH_KEYS = ("storage_path", "cv_extrema_file", "descriptor_storage")
STATE_LABELS = ["A", "B"]
DIRECTIONS = ("forward", "backward")


# ----------------------------------------------------------------------------
# Config and small helpers
# ----------------------------------------------------------------------------
def _deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None, quick=False):
    with open(path or HERE / "config.json") as f:
        cfg = json.load(f)
    quick_cfg = cfg.pop("quick", {})
    if quick:
        _deep_update(cfg, quick_cfg)
    return cfg


def load_run_config(out, path=None, quick=False):
    """Config a run was started with (``<out>/config_used.json``); falls back to load_config.

    Use this for analysis of an existing run folder, so that e.g. a 22-D run is
    analysed with its own settings and not with the default config.json.
    """
    saved = Path(out) / "config_used.json"
    if saved.exists():
        return json.loads(saved.read_text())
    return load_config(path, quick=quick)


def trajectory_descriptors(trajectory):
    """Descriptors of a stored trajectory, read snapshot by snapshot (no OPS CV cache)."""
    return np.asarray([snapshot.coordinates[0] for snapshot in trajectory])


def origin_name(iteration):
    return f"q_it{iteration}"


def cv_column(iteration):
    return f"cv_q_it{iteration}"


def q_of(model, descriptors):
    """Committor log-odds q(x) of an AIMMD model on a (n, d) descriptor array."""
    q = model.log_prob(np.atleast_2d(descriptors), use_transform=False, batch_size=None)
    return np.asarray(q).reshape(-1)


class _Stage:
    """Context manager that skips a stage whose ``.done`` marker exists."""

    def __init__(self, directory, name):
        self.marker = Path(directory) / f".{name}.done"
        self.name = name

    @property
    def done(self):
        return self.marker.exists()

    def finish(self):
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        self.marker.touch()


# ----------------------------------------------------------------------------
# Workflow
# ----------------------------------------------------------------------------
def _tis_worker(cfg, out_dir, k, direction, interface_value, seed):
    """Process entry point: one interface ensemble, with its own RNG seed."""
    torch.set_num_threads(1)
    workflow = IterativeAimmdTIS(cfg, out_dir, seed=seed)
    try:
        return workflow.run_tis_interface(k, direction, interface_value)
    finally:
        workflow.close()


class IterativeAimmdTIS:
    def __init__(self, cfg, out_dir, seed=None, n_workers=None):
        self.cfg = cfg
        self.run_cfg = cfg["run"]
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.n_workers = int(n_workers or self.run_cfg.get("n_workers") or os.cpu_count() or 1)

        seed = int(cfg["system"].get("seed", 0)) if seed is None else int(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        sys_cfg = cfg["system"]
        self.system = ToyTPSSetup(
            potential_name=sys_cfg["potential_name"],
            potential_kwargs=sys_cfg["potential_kwargs"],
            integrator_params=sys_cfg["integrator_params"],
        )
        self.pes = self.system.pes
        self.states = self.system.states
        self.engine = self.system.md_engine
        self.beta = 1.0 / float(sys_cfg["integrator_params"]["temperature"])
        # The function must live in an importable module: OPS stores it by
        # reference, and a function from this script would be recorded as
        # __main__._coordinates, so the TIS storages could only be reopened from
        # this script and not from a notebook or another analysis script.
        self.descriptor_transform = paths.FunctionCV(
            "descriptor_transform", _coordinates_descriptor_transform, cv_wrap_numpy_array=True
        ).with_diskcache()
        # Descriptors are the full coordinate vector: the 2 PES dimensions plus
        # any harmonic (noise) dimensions.
        self.descriptor_dim = int(self.pes.n_dims_pot + self.pes.n_harmonics)
        self.aimmd_setup = AIMMDSetup(
            cfg,
            descriptor_dim=self.descriptor_dim,
            states=self.states,
            descriptor_transform=self.descriptor_transform,
        )
        self._models = {}
        self._thermalization = {}
        self._excluded = set()
        self._seed_storages = []
        self._last_skips = {}
        self._tps_storage = None
        self._stable = None

    # ---------------- paths ----------------
    def model_path(self, k):
        return self.out / "tps" / "aimmd.h5" if k == 0 else self.out / "models" / f"aimmd_it{k}.h5"

    def it_dir(self, k, direction=None):
        d = self.out / f"it{k}"
        return d / direction if direction else d

    def rpe_dir(self, k):
        return self.out / f"rpe_it{k}"

    def descriptor_storage_path(self, k, direction, interface_value):
        """One descriptor storage per interface ensemble: parallel workers cannot share a file."""
        return self.it_dir(k, direction) / f"descriptors_i{interface_indicator(interface_value)}"

    # ---------------- state functions (descriptor space) ----------------
    def state_function_A(self, desc):
        return np.sqrt(np.sum((np.atleast_2d(desc)[:, :2] - self.pes.state_A[:2]) ** 2, axis=1))

    def state_function_B(self, desc):
        return -np.sqrt(np.sum((np.atleast_2d(desc)[:, :2] - self.pes.state_B[:2]) ** 2, axis=1))

    def in_state(self, desc, state):
        """Frames inside state "A" or "B", as the CVDefinedVolume states (distance < boundary)."""
        dist = self.state_function_A(desc) if state == "A" else -self.state_function_B(desc)
        return dist < self.pes.state_boundary

    def origin_in_state_fn(self, direction):
        """In-state test for the initial state of a forward (A) or backward (B) ensemble."""
        state = "A" if direction == "forward" else "B"
        return lambda desc: self.in_state(desc, state)

    def grid_descriptors(self, x, y):
        """Descriptors for (x, y) points, with harmonic dimensions at their minimum (0)."""
        x = np.ravel(x)
        desc = np.zeros((x.size, self.descriptor_dim))
        desc[:, 0] = x
        desc[:, 1] = np.ravel(y)
        return desc

    # ---------------- models ----------------
    def load_model(self, k):
        if k not in self._models:
            store = aimmd.Storage(str(self.model_path(k)), "r")
            model = store.rcmodels["most_recent"]
            store.close()
            model.nnet.to("cpu")
            model.nnet.eval()
            model.descriptor_transform = self.descriptor_transform
            self._models[k] = model
        return self._models[k]

    # ------------------------------------------------------------------
    # Stage 1: seed TPS
    # ------------------------------------------------------------------
    def initial_trajectory(self):
        n = int(self.run_cfg["initial_path_length"])
        return paths.Trajectory(self.pes.simple_initial_path(n, self.engine))

    def run_tps(self):
        stage = _Stage(self.out / "tps", "tps")
        if stage.done:
            print("[tps] done, skipping")
            return
        (self.out / "tps").mkdir(parents=True, exist_ok=True)
        paths.PathMover.engine = self.engine
        network = self.system.create_network()

        aimmd_store = aimmd.Storage(str(self.model_path(0)), "w")
        model = self.aimmd_setup.setup_RCModel(aimmd_store)
        selector = self.aimmd_setup.setup_selector(model)
        trainset = aimmd.TrainSet(n_states=len(self.states))

        move_scheme = self.system.create_move_scheme(network, "TwoWay", selector=selector)
        initial_conditions = move_scheme.initial_conditions_from_trajectories(self.initial_trajectory())
        initial_conditions.sanity_check()

        storage = paths.Storage(str(self.out / "tps" / "tps.nc"), "w", template=self.system.template)
        storage.save(self.system.template)
        storage.save(self.engine)
        storage.save(move_scheme)
        storage.save(network)

        sampler = paths.PathSampling(storage, move_scheme, initial_conditions)
        sampler.attach_hook(aimmd.ops.TrainingHook(model, trainset))
        sampler.attach_hook(aimmd.ops.AimmdStorageHook(aimmd_store, model, trainset))
        sampler.attach_hook(aimmd.ops.DensityCollectionHook(model))
        print(f"[tps] running {self.run_cfg['tps_steps']} AIMMD-TPS steps")
        sampler.run(int(self.run_cfg["tps_steps"]))
        storage.close()
        aimmd_store.close()
        stage.finish()

    def initial_path(self):
        """Last TPS path: an A->B path that is valid for every interface ensemble."""
        if self._tps_storage is None:
            self._tps_storage = paths.Storage(str(self.out / "tps" / "tps.nc"), "r")
        return self._tps_storage.steps[-1].active[0].trajectory

    def seed_path(self, k, direction, interface_value):
        """Initial path for one interface ensemble of iteration k, and where it came from.

        With ``run.reseed = "previous_iteration"`` (default) the chains continue
        from iteration k-1: the candidates are the last active path of each
        k-1 ensemble of the same direction; the valid one (starts in the origin
        state and crosses the new interface in q_k) with the smallest margin is
        used. Otherwise, or without a valid candidate, the last TPS path.
        """
        if k == 0 or self.run_cfg.get("reseed", "previous_iteration") != "previous_iteration":
            return self.initial_path(), "tps"
        # Load the TPS fallback first, and keep every scanned storage open until
        # close(): OPS shares objects by UUID across storages (the iteration-0
        # storages contain the TPS path as step 0), and a snapshot handed out from
        # a closed storage can no longer be saved into the new one.
        tps_path = self.initial_path()
        model = self.load_model(k)
        sign = 1.0 if direction == "forward" else -1.0
        in_origin = self.origin_in_state_fn(direction)
        best = None
        for row in self.run_summaries(k - 1, direction):
            storage = paths.Storage(row["storage_path"], "r")
            self._seed_storages.append(storage)
            traj = storage.steps[-1].active[0].trajectory
            desc = trajectory_descriptors(traj)
            outside = ~in_origin(desc)
            if in_origin(desc[:1])[0] and outside.any():
                margin = np.max(sign * q_of(model, desc[outside])) - sign * interface_value
                if margin >= 0 and (best is None or margin < best[0]):
                    best = (margin, traj, f"it{k - 1} {direction} {row['interface_value']}")
        if best is None:
            print(f"[it{k}] WARNING no previous-iteration path crosses {direction} interface "
                  f"{interface_value}; seeding from the TPS path")
            return tps_path, "tps (fallback)"
        return best[1], best[2]

    # ------------------------------------------------------------------
    # Stage 2: stable-state MD
    # ------------------------------------------------------------------
    def stable_storage_paths(self):
        return [self.out / "stable" / f"state{label}.nc" for label in STATE_LABELS]

    def run_stable(self):
        stage = _Stage(self.out / "stable", "stable")
        if stage.done:
            print("[stable] done, skipping")
            return
        (self.out / "stable").mkdir(parents=True, exist_ok=True)
        init_traj = self.initial_trajectory()
        for state, storage_path in zip(self.states, self.stable_storage_paths()):
            start = next(snap for snap in init_traj if state(snap))
            storage = paths.Storage(str(storage_path), "w", template=self.system.template)
            storage.save(self.engine)
            storage.tag["template"] = start
            sim = paths.DirectSimulation(
                storage=storage,
                engine=self.engine,
                states=list(self.states),
                initial_snapshot=start,
            )
            print(f"[stable] {self.run_cfg['stable_steps']} MD steps in {state.name}")
            sim.run(int(self.run_cfg["stable_steps"]))
            storage.close()

        build_stable_state_storage(
            storage_paths=self.stable_storage_paths(),
            descriptor_transform=self.descriptor_transform,
            output_path=self.out / "stable" / "stable_descriptor_storage",
            state_volumes=self.states,
            state_labels=STATE_LABELS,
            descriptor_name="descriptors",
            set_name="stable",
        )
        stage.finish()

    def stable_data(self):
        """Stable-state payload in the dict form rpe_mbar expects."""
        if self._stable is None:
            payload = load_stable_state_payload(
                self.out / "stable" / "stable_descriptor_storage",
                descriptor_name="descriptors",
                set_name="stable",
                state_labels=STATE_LABELS,
            )
            data_stable = {
                "descriptors": payload["descriptors"],
                "shot_results": payload["shot_results"],
                "in_state_flags": payload["in_state_flags"],
            }
            desc_total = np.concatenate(data_stable["descriptors"], axis=0)
            in_state = np.zeros((desc_total.shape[0], len(STATE_LABELS)), dtype=bool)
            offset = 0
            for idx, flags in enumerate(data_stable["in_state_flags"]):
                in_state[offset:offset + len(flags), idx] = flags
                offset += len(flags)
            self._stable = (data_stable, desc_total, in_state)
        return self._stable

    # ------------------------------------------------------------------
    # Stage 3a: interface placement
    # ------------------------------------------------------------------
    def interfaces(self, k):
        path = self.it_dir(k) / "interfaces.json"
        if path.exists():
            return json.loads(path.read_text())
        _, desc_total, in_state = self.stable_data()
        model = self.load_model(k)
        info = check_interfaces(
            model,
            STATE_LABELS,
            desc_total,
            in_state=in_state,
            overlap=float(self.run_cfg["overlap"]),
        )
        if info["forward_interfaces"] is None or info["backward_interfaces"] is None:
            raise RuntimeError(f"Interface placement failed for q_{k}: {info}")
        # An interface inside the opposite state's q range can never be crossed by
        # a path that ends in that state, so TIS cannot start there. Keep only
        # interfaces in front of the opposite state.
        q_stable = q_of(model, desc_total)
        q_B_min = float(np.min(q_stable[in_state[:, 1]]))
        q_A_max = float(np.max(q_stable[in_state[:, 0]]))
        forward = [float(v) for v in info["forward_interfaces"] if v < q_B_min]
        backward = [float(v) for v in info["backward_interfaces"] if v > q_A_max]
        min_gap = float(self.run_cfg.get("min_state_q_gap", 2.0))
        if q_B_min - q_A_max < min_gap:
            raise RuntimeError(
                f"q_{k} does not separate the states: q_B_min - q_A_max = {q_B_min - q_A_max:.2f} < {min_gap} "
                f"(q_A_max={q_A_max:.2f}, q_B_min={q_B_min:.2f}). The model is (nearly) flat; for q_0 run a "
                "longer TPS seed (run.tps_steps) or check the online-training settings (AIMMD_settings.ee_params).")
        # The overlap rule assumes q is a calibrated log-odds; an early model is
        # not, and gives too few interfaces to connect A and B. Fall back to
        # evenly spaced interfaces between the two stable-state q ranges.
        n_min = int(self.run_cfg.get("min_interfaces", 1))
        if len(forward) < n_min or len(backward) < n_min:
            grid = np.round(np.linspace(q_A_max, q_B_min, n_min + 1), 2)
            forward = [float(v) for v in grid[:-1]]
            backward = [float(v) for v in grid[1:][::-1]]
            print(f"[it{k}] overlap rule gave too few interfaces; using {n_min} evenly spaced ones")
        # Interfaces stop at the transition state: the last forward interface lies
        # at q = 0 or at most run.interface_overshoot (default 1) beyond it, and
        # mirrored for backward. Beyond the TS the paths of an ensemble mostly
        # commit to the far state, so shooting from there rarely gives a valid
        # path, and the far side is covered by the other direction anyway.
        overshoot = float(self.run_cfg.get("interface_overshoot", 1.0))
        forward = [v for v in forward if v <= overshoot]
        backward = [v for v in backward if v >= -overshoot]
        if not forward or max(forward) < 0.0:
            forward.append(0.0)
        if not backward or min(backward) > 0.0:
            backward.append(0.0)
        # Order interfaces the way rpe_mbar's solver does (forward ascending,
        # backward descending), so dict order and solver order agree.
        ifaces = {"forward": sorted(forward), "backward": sorted(backward, reverse=True)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ifaces, indent=2))
        print(f"[it{k}] interfaces forward={ifaces['forward']} backward={ifaces['backward']}")
        return ifaces

    # ------------------------------------------------------------------
    # Stage 3b: AIMMD-TIS
    # ------------------------------------------------------------------
    def run_summaries(self, k, direction):
        """Run summaries of iteration k, with file paths resolved against the output folder."""
        rows = json.loads((self.it_dir(k, direction) / "run_summary.json").read_text())
        for row in rows:
            for key in SUMMARY_PATH_KEYS:
                if key in row and not Path(row[key]).is_absolute():
                    row[key] = str(self.out / row[key])
        return rows

    def run_tis_interface(self, k, direction, interface_value):
        """Run AIMMD-TIS for a single interface ensemble (one worker's job)."""
        directory = self.it_dir(k, direction)
        tag = f"i{interface_indicator(interface_value)}"
        summary_path = directory / f"run_summary_{tag}.json"
        stage = _Stage(directory, f"tis_{tag}")
        if stage.done:
            return json.loads(summary_path.read_text())
        tis = aimmdTIS.AIMMD_TIS(
            engine=self.engine,
            AIMMD_model=self.load_model(k),
            stateA=self.states[0],
            stateB=self.states[1],
            descriptor_transform=self.descriptor_transform,
            directory=directory,
            cvq_type="pkl",
            cv_label=cv_column(k),
        )
        # A previous, interrupted attempt may have left a partial descriptor storage;
        # the TIS storage is overwritten, so start the descriptor storage fresh too
        # (DescriptorStorage would otherwise append to it).
        ds_path = self.descriptor_storage_path(k, direction, interface_value)
        for leftover in ds_path.parent.glob(ds_path.name + ".*"):
            leftover.unlink()
        seed, seed_source = self.seed_path(k, direction, interface_value)
        print(f"[it{k}] {direction} {interface_value}: seeded from {seed_source}")
        summary = tis.run_TIS_sequentially(
            n_mc_steps=int(self.run_cfg["tis_steps"]),
            initial_path=seed,
            interface_values=[interface_value],
            template=self.system.template,
            scheme_move="TwoWay",
            scheme_selector=self.run_cfg["selector"],
            scheme_modifier="RandomVelocities",
            gaussian_parameter_width=float(self.run_cfg["gaussian_width"]),
            gaussian_parameter_shift=float(self.run_cfg["gaussian_shift"]),
            direction=direction,
            directory=directory,
            overwrite=True,
            origin_label=origin_name(k),
            iteration=k,
            # Path descriptors and shot results are stored while sampling (index =
            # MC step) and path extrema are recorded on the fly, so the OPS storage
            # is never read back.
            descriptor_storage_path=self.descriptor_storage_path(k, direction, interface_value),
            descriptor_set_name=origin_name(k),
            descriptor_name="descriptors",
            descriptor_flush_every=int(self.run_cfg.get("descriptor_flush_every", 25)),
            extrema_source="on_the_fly",
        )[0]
        summary["descriptor_storage"] = str(self.descriptor_storage_path(k, direction, interface_value))
        summary["seed_source"] = seed_source
        # Store paths relative to the output folder, so the folder can be moved.
        for key in SUMMARY_PATH_KEYS:
            summary[key] = os.path.relpath(Path(summary[key]).resolve(), self.out.resolve())
        summary_path.write_text(json.dumps(summary, indent=2))
        stage.finish()
        return summary

    def run_tis(self, k):
        """Run all interface ensembles of iteration k, in parallel worker processes.

        The ensembles are independent: each has its own OPS storage and extrema
        CSV and starts from the same A->B TPS path.
        """
        ifaces = self.interfaces(k)
        jobs = [
            (direction, iface)
            for direction in DIRECTIONS
            if not _Stage(self.it_dir(k, direction), "tis").done
            for iface in ifaces[direction]
        ]
        if not jobs:
            print(f"[it{k}] TIS done, skipping")
            return
        for direction in DIRECTIONS:
            self.it_dir(k, direction).mkdir(parents=True, exist_ok=True)

        n_workers = min(len(jobs), self.n_workers)
        print(f"[it{k}] running {len(jobs)} TIS ensembles on {n_workers} worker(s)")
        base_seed = int(self.cfg["system"].get("seed", 0))
        results = {}
        if n_workers <= 1:
            for direction, iface in jobs:
                results[(direction, iface)] = self.run_tis_interface(k, direction, iface)
        else:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futures = {
                    pool.submit(_tis_worker, self.cfg, str(self.out), k, direction, iface,
                                base_seed + 1000 * (k + 1) + idx): (direction, iface)
                    for idx, (direction, iface) in enumerate(jobs)
                }
                for future in as_completed(futures):
                    direction, iface = futures[future]
                    results[(direction, iface)] = future.result()
                    print(f"[it{k}] TIS {direction} interface {iface} finished")

        for direction in DIRECTIONS:
            stage = _Stage(self.it_dir(k, direction), "tis")
            if stage.done:
                continue
            summaries = [results[(direction, iface)] for iface in ifaces[direction]]
            (self.it_dir(k, direction) / "run_summary.json").write_text(json.dumps(summaries, indent=2))
            stage.finish()

    # ------------------------------------------------------------------
    # Stage 3c: extrema of every model on every iteration's paths
    # ------------------------------------------------------------------
    def update_extrema(self, k):
        """Make sure each extrema CSV of iterations 0..k holds min/max of q_0..q_k.

        Multi-set MBAR needs every path's extremum in every interface set, also
        for models trained after the path was sampled. The missing columns are
        computed from the descriptor storages written during sampling (one
        batched pass per model and interface), with frames inside the ensemble's
        initial state excluded, as in the on-the-fly extrema of run_TIS.
        """
        models = {i: self.load_model(i) for i in range(k + 1)}
        for j in range(k + 1):
            for direction in DIRECTIONS:
                for row in self.run_summaries(j, direction):
                    csv_path = Path(row["cv_extrema_file"])
                    df = pd.read_csv(csv_path)
                    missing = [i for i in models if f"max_{cv_column(i)}" not in df.columns]
                    if not missing:
                        continue
                    new = aimmdTIS.extrema_from_descriptor_storage(
                        row["descriptor_storage"],
                        origin_name(j),
                        float(row["interface_value"]),
                        {cv_column(i): (lambda desc, m=models[i]: q_of(m, desc)) for i in missing},
                        in_state_fn=self.origin_in_state_fn(direction),
                        direction=direction,
                    )
                    merged = df.merge(new, on="step", how="left")
                    if merged.isna().any().any():
                        raise RuntimeError(f"Descriptor storage does not cover all steps of {csv_path.name}")
                    merged.to_csv(csv_path, index=False)
                    print(f"[it{k}] added q_{missing} extrema to {csv_path.name}")

    def thermalization_step(self, row):
        """First MC step used for analysis in one interface ensemble.

        ``run.thermalization = "first_accepted"`` (default): the step of the first
        accepted path. With two-way shooting an accepted move replaces the whole
        path, so from there on the ensemble no longer holds the shared initial
        (TPS) path. Found from the descriptor storage as the first stored path
        that differs from step 0. ``"fixed"`` uses ``run.n_skip`` for every ensemble.
        """
        if self.run_cfg.get("thermalization", "first_accepted") == "fixed":
            return int(self.run_cfg["n_skip"])
        from ops_setup.descriptor_storage import DescriptorStorage

        key = (row["descriptor_storage"], float(row["interface_value"]))
        if key not in self._thermalization:
            store = DescriptorStorage.load(row["descriptor_storage"], mode="r")
            trajectories = store.sets[row["origin_label"]].interfaces[f"{float(row['interface_value']):.2f}"].trajectories
            desc = store.descriptors["descriptors"]
            first = desc[trajectories[0].descriptor_start:trajectories[0].descriptor_end]
            step = None
            for t in trajectories[1:]:
                d = desc[t.descriptor_start:t.descriptor_end]
                if d.shape != first.shape or not np.array_equal(d, first):
                    step = int(t.storage_index)
                    break
            if step is None:
                print(f"[thermalization] WARNING no accepted path in {row['descriptor_storage']}; ensemble unused")
                step = int(trajectories[-1].storage_index) + 1
            self._thermalization[key] = step
        return self._thermalization[key]

    def usable_rows(self, j, direction):
        """Run summaries of iteration j's ensembles that have enough paths to analyse.

        An ensemble whose chain was accepted too late (fewer than
        ``run.min_usable_paths`` paths after thermalization, default the bootstrap
        block size) is left out of the reweighting with a warning. A chain that
        never accepts usually means the interface lies where the model q is
        miscalibrated, so shots from it never commit to the origin state.
        """
        n_min = int(self.run_cfg.get("min_usable_paths", self.run_cfg.get("block_size", 10)))
        rows = []
        for r in self.run_summaries(j, direction):
            n_used = int(r["n_mc_steps"]) + 1 - self.thermalization_step(r)
            if n_used >= n_min:
                rows.append(r)
            else:
                key = (j, direction, float(r["interface_value"]))
                if key not in self._excluded:
                    self._excluded.add(key)
                    print(f"[it{j}] WARNING {direction} ensemble {r['interface_value']} has {max(n_used, 0)} "
                          f"usable paths (< {n_min}); left out of the reweighting")
        return rows

    def rpe_dataset(self, k, direction, n_each_set, skips):
        """RpeDataset for the origin sets q_0..q_k, built from the descriptor storages.

        The storages already hold each path's descriptors and shot results (one
        entry per MC step), so nothing is read from the OPS storages.
        """
        from ops_setup.descriptor_storage import DescriptorStorage

        dataset = rpe_mbar.RpeDataset(metadata={"thermalization_steps": skips, "descriptor_dim": self.descriptor_dim})
        descriptors, shot_results, offset = [], [], 0
        for j in range(k + 1):
            name = origin_name(j)
            set_data = rpe_mbar.SetData(set_name=name)
            for row, n_expected, n_skip in zip(self.usable_rows(j, direction), n_each_set[name], skips[name]):
                iface = float(row["interface_value"])
                store = DescriptorStorage.load(row["descriptor_storage"], mode="r")
                trajectories = store.sets[name].interfaces[f"{iface:.2f}"].trajectories
                selected = [(idx, t) for idx, t in enumerate(trajectories) if t.storage_index >= n_skip]
                if len(selected) != n_expected:
                    raise RuntimeError(f"{row['descriptor_storage']}: {len(selected)} stored paths after "
                                       f"thermalization, extrema CSV has {n_expected}")
                iface_data = rpe_mbar.InterfaceData(interface_value=iface)
                for rel, (idx, t) in enumerate(selected):
                    desc = store.get_trajectory_descriptors(name, iface, idx, ["descriptors"])
                    shots = store.get_trajectory_shot_results(name, iface, idx)
                    if shots is None:  # storage written without shot results
                        ends = desc[[0, -1]]
                        shots = np.tile([float(np.sum(self.in_state(ends, "A"))),
                                         float(np.sum(self.in_state(ends, "B")))], (len(desc), 1))
                    descriptors.append(desc)
                    shot_results.append(shots)
                    iface_data.trajectories.append(rpe_mbar.TrajectoryInfo(
                        storage_index=int(t.storage_index), relative_index=rel, n_points=len(desc),
                        descriptor_start=offset, descriptor_end=offset + len(desc)))
                    offset += len(desc)
                set_data.interfaces[f"{iface}"] = iface_data
            dataset.sets[name] = set_data
        dataset.descriptors = np.concatenate(descriptors, axis=0)
        dataset.shot_results = np.concatenate(shot_results, axis=0)
        return dataset

    # ------------------------------------------------------------------
    # Stage 3d: RPE via multi-set MBAR
    # ------------------------------------------------------------------
    def _load_direction(self, k, direction):
        """Extrema dict, RPE dataset and per-path MBAR weights for one direction.

        Each interface ensemble is used from its own thermalization step on
        (see :meth:`thermalization_step`).
        """
        specs = [
            rpe_mbar.InterfaceFunctionSpec(filename=origin_name(j), cv_column_base=cv_column(j), iteration=j)
            for j in range(k + 1)
        ]
        data = {}
        interface_values = {}
        n_each_set = {}
        skips = {}
        for j in range(k + 1):
            name = origin_name(j)
            rows = self.usable_rows(j, direction)
            ifaces = [float(r["interface_value"]) for r in rows]
            # One loader call per origin with only that origin's iteration: the
            # path resolver otherwise falls back to iteration-less legacy files.
            data_j, _, _ = rpe_mbar.load_extrema_as_dict_iterative(
                run_folder=self.it_dir(j, direction),
                direction=direction,
                interface_functions=specs,
                interfaces_by_origin={name: ifaces},
                n_thermalization=0,
                iterations=[j],
                strict=True,
            )

            counts, skips[name] = [], []
            for r, iface in zip(rows, ifaces):
                steps = pd.read_csv(r["cv_extrema_file"])["step"].to_numpy()
                # CSV row i, descriptor-storage index i and OPS step i are the same
                # path, so extrema line up with the RpeDataset paths.
                if not np.array_equal(steps, np.arange(len(steps))):
                    raise RuntimeError(f"Extrema steps are not contiguous in {r['cv_extrema_file']}")
                n_skip = self.thermalization_step(r)
                block = data_j[name][f"{iface}"]
                for target in block:
                    block[target] = np.asarray(block[target])[n_skip:]
                counts.append(len(steps) - n_skip)
                skips[name].append(n_skip)
            data[name] = data_j[name]
            interface_values[name] = ifaces
            n_each_set[name] = counts

        dataset = self.rpe_dataset(k, direction, n_each_set, skips)
        self._last_skips[direction] = skips

        # MBAR chains interfaces through paths that reach the next interface of
        # their own set; an ensemble where none do leaves the chain unconstrained.
        for name, ifaces in interface_values.items():
            sign = 1.0 if direction == "forward" else -1.0
            for this, nxt in zip(ifaces[:-1], ifaces[1:]):
                extrema = sign * data[name][f"{this}"][name]
                if not np.any(extrema >= sign * nxt):
                    print(f"[it{k}] WARNING {direction} {name}: no path at interface {this} reaches {nxt}; "
                          "crossing probabilities will be unreliable (more tis_steps or closer interfaces)")

        weights, max_labeled, _ = rpe_mbar.solve_weights_multiset(data, direction=direction, log=True)
        return data, dataset, weights, max_labeled

    def _path_is_reactive(self, data, dataset):
        """Per-path A<->B flag, in the same order as the MBAR weights."""
        flags = []
        for origin in data:
            for iface in data[origin]:
                n = len(data[origin][iface][origin])
                for idx in range(n):
                    shot = dataset.get_trajectory_shot_results(origin, iface, idx)[0]
                    flags.append(shot[0] == 1 and shot[1] == 1)
        return np.asarray(flags, dtype=bool)

    def run_rpe(self, k):
        rpe_dir = self.rpe_dir(k)
        stage = _Stage(rpe_dir, "rpe")
        if stage.done:
            print(f"[it{k}] RPE done, skipping")
            return
        rpe_dir.mkdir(parents=True, exist_ok=True)
        self.update_extrema(k)
        data_stable, _, _ = self.stable_data()
        ifaces = {j: self.interfaces(j) for j in range(k + 1)}
        models = [self.load_model(j) for j in range(k + 1)]
        q_functions = [lambda desc, m=m: q_of(m, desc) for m in models]

        streams = {}
        diag = {}
        for direction in DIRECTIONS:
            data, dataset, weights, max_labeled = self._load_direction(k, direction)
            desc, w, shots, counts = rpe_mbar.get_descriptors_and_weights(data, weights, dataset)
            streams[direction] = (desc, w, shots, counts, weights)

            # Crossing probability in q_k, conditioned on its first interface.
            lambda_0 = ifaces[k][direction][0]
            lambda_0 = lambda_0 if direction == "forward" else -lambda_0
            bins = np.linspace(lambda_0, lambda_0 + 25.0, 400)
            boot = rpe_mbar.bootstrap_crossing_probs(
                data=data,
                interface_0=lambda_0,
                n_bootstrap=int(self.run_cfg["n_bootstrap"]),
                block_size=int(self.run_cfg["block_size"]),
                set_analysis=origin_name(k),
                direction=direction,
                bins=bins,
                seed=int(self.cfg["system"].get("seed", 0)),
            )
            np.savez(rpe_dir / f"crossing_{direction}.npz", bins=boot["bins"],
                     mean=boot["bootstrap_mean"], std=boot["bootstrap_std"])

            # P(reach the other state | joined first interface). Every TIS path crosses
            # its own set's first interface and so the joined one: P is the reweighted
            # fraction of reactive paths.
            reactive = self._path_is_reactive(data, dataset)
            if len(reactive) != len(weights):
                raise RuntimeError("Path ordering mismatch between weights and dataset")
            diag[direction] = {
                "p_reactive": float(np.sum(weights[reactive]) / np.sum(weights)),
                "n_paths": int(len(weights)),
                "thermalization_steps": self._last_skips[direction],
            }

        desc_f, w_f, shot_f, counts_f, weights_f = streams["forward"]
        desc_b, w_b, shot_b, counts_b, weights_b = streams["backward"]
        # counts = (n_AA, n_AB, n_BB). Without reactive paths in both directions
        # c_AB is 0 or undefined and the training set loses one of the states.
        if counts_f[1] == 0 or counts_b[1] == 0:
            raise RuntimeError(
                f"[it{k}] no reactive paths (forward AB={counts_f[1]}, backward AB={counts_b[1]}); "
                "increase run.tis_steps or run.min_interfaces before training on this ensemble")

        def reweighted(cutoff):
            return rpe_mbar.build_reweighted_rpe_trainset(
                desc_f=desc_f, w_f=w_f, shot_f=shot_f, counts_f=counts_f,
                desc_b=desc_b, w_b=w_b, shot_b=shot_b, counts_b=counts_b,
                weights_per_path_f=weights_f,
                weights_per_path_b=weights_b,
                data_stable=data_stable,
                interface_functions=q_functions,
                first_interfaces_forward=[ifaces[j]["forward"][0] for j in range(k + 1)],
                first_interfaces_backward=[ifaces[j]["backward"][0] for j in range(k + 1)],
                state_function_A=self.state_function_A,
                state_function_B=self.state_function_B,
                state_threshold_A=self.pes.state_boundary,
                state_threshold_B=-self.pes.state_boundary,
                cutoff_beyond_joined_interface=cutoff,
                name_lambda_functions=[origin_name(j) for j in range(k + 1)],
            )

        # Training keeps every TIS frame, also those before the joined first
        # interface (a region that shrinks as the model improves). The analysis
        # RPE keeps TIS frames only beyond the joined interface, where the stable-
        # state MD does not already cover them, so no region is counted twice.
        rpe_train = reweighted(cutoff=False)
        rpe_analysis = reweighted(cutoff=True)
        for name, rpe in (("trainset.npz", rpe_train), ("rpe_analysis.npz", rpe_analysis)):
            np.savez(rpe_dir / name, descriptors=rpe["descriptors"], weights=rpe["weights"],
                     shot_results=rpe["shot_results"])
        before_joined = {
            direction: float(1.0 - np.sum(w[np.asarray(idx).ravel()]) / np.sum(w))
            for direction, w, idx in (("forward", w_f, rpe_analysis["idx_beyond_forward"]),
                                      ("backward", w_b, rpe_analysis["idx_beyond_backward"]))
        }

        rates = self._rates(rpe_analysis["flux"], data_stable, diag)
        (rpe_dir / "diagnostics.json").write_text(json.dumps(
            {"c_AB": rpe_train["c_AB"], **rates,
             "tis_weight_before_joined_fwd": before_joined["forward"],
             "tis_weight_before_joined_bwd": before_joined["backward"],
             "directions": diag}, indent=2))
        stage.finish()

    def _rates(self, flux, data_stable, diag):
        """TIS rates k_AB = flux(A -> joined lambda_0) * P(B | joined lambda_0), in the toy's time units.

        ``flux`` is rpe_mbar's first-crossing count through the joined first
        interface of all interface sets (from build_reweighted_rpe_trainset).
        """
        frame_time = float(self.cfg["system"]["integrator_params"]["dt"]) * int(
            getattr(self.engine, "n_steps_per_frame", 1))
        out = {}
        for idx, (direction, key) in enumerate([("forward", "k_AB"), ("backward", "k_BA")]):
            n_cross = len(flux["first_crossings"][idx])
            total_time = len(data_stable["descriptors"][idx]) * frame_time
            out[f"flux_{STATE_LABELS[idx]}"] = n_cross / total_time
            out[key] = out[f"flux_{STATE_LABELS[idx]}"] * diag[direction]["p_reactive"]
        return out

    # ------------------------------------------------------------------
    # Stage 3e: train q_{k+1}
    # ------------------------------------------------------------------
    def run_training(self, k):
        stage = _Stage(self.out / "models", f"train_it{k + 1}")
        if stage.done:
            print(f"[it{k}] training q_{k + 1} done, skipping")
            return
        (self.out / "models").mkdir(parents=True, exist_ok=True)
        train_cfg = {key: value for key, value in copy.deepcopy(self.cfg["training"]).items()
                     if not key.startswith("_")}
        optimizer_cfg = {**train_cfg.pop("optimizer"), "lr": train_cfg["base_lr"]}

        ts = np.load(self.rpe_dir(k) / "trainset.npz")
        train_data, test_data = aimmdTIS.create_train_test_split(
            ts["descriptors"], ts["shot_results"], weights=ts["weights"], split=[4, 1],
            seed=int(self.cfg["system"].get("seed", 0)) + k,
        )
        n_states = len(self.states)
        train_set = aimmd.TrainSet(n_states=n_states, descriptors=train_data[0],
                                   shot_results=train_data[1], weights=train_data[2])
        test_set = aimmd.TrainSet(n_states=n_states, descriptors=test_data[0],
                                  shot_results=test_data[1], weights=test_data[2])

        aimmd_store = aimmd.Storage(str(self.model_path(k + 1)), "w")
        warm = self.model_path(k) if self.run_cfg.get("warm_start", True) else None
        model = self.aimmd_setup.setup_RCModel(aimmd_store, load_model_path=warm)
        aimmd_store.save_trainset(aimmd.TrainSet(n_states=n_states, descriptors=ts["descriptors"],
                                                 shot_results=ts["shot_results"], weights=ts["weights"]))
        print(f"[it{k}] training q_{k + 1} on {len(ts['weights'])} weighted frames")
        results = aimmdTIS.train_one_stage(
            aimmd_store, model, train_set, test_set,
            stage_name=f"it{k + 1}",
            optimizer_cfg=optimizer_cfg,
            **train_cfg,
        )
        aimmd_store.close()
        save_training_results(results, self.out / "models" / f"training_it{k + 1}.json")
        plot_training_diagnostics(results, stage_name=f"it{k + 1}",
                                  save_path=self.out / "models" / f"training_it{k + 1}.png", show=False)
        stage.finish()

    # ------------------------------------------------------------------
    # Theory reference committor (saved next to this script)
    # ------------------------------------------------------------------
    def potential_parameters(self):
        """Effective potential parameters: class defaults overridden by the config."""
        signature = inspect.signature(type(self.pes).__init__)
        params = {name: p.default for name, p in signature.parameters.items()
                  if p.default is not inspect.Parameter.empty}
        params.update(self.cfg["system"].get("potential_kwargs", {}))
        return params

    def theory_path(self):
        """Theory committor file for the current potential, temperature and grid."""
        params = self.potential_parameters()
        n = int(self.cfg["analysis"]["theory_grid"])
        temperature = self.cfg["system"]["integrator_params"]["temperature"]
        name = (f"{self.cfg['system']['potential_name'].replace('-', '_')}"
                f"_scale{params.get('scale')}_rot{params.get('rotation_degrees')}_d{params.get('d')}"
                f"_T{temperature}_n{n}_committor.npy")
        return THEORY_DIR / name

    def theory_committor(self):
        """Exact 2-D committor p_B on an n x n grid over pes.extent (rows y, columns x).

        Solved once by relaxation of the backward Kolmogorov equation for the
        overdamped dynamics on the PES (aimmdTIS.solve_committor_by_relaxation)
        and saved as .npy (the format ToyVisualizer.load_theoretical_committor
        reads) with a .meta.json describing the settings.
        """
        path = self.theory_path()
        if path.exists():
            return path, np.load(path)
        n = int(self.cfg["analysis"]["theory_grid"])
        ext = self.pes.extent
        X, Y, Fx, Fy = self.pes.get_2d_pes_F(np.linspace(ext[0], ext[1], n), np.linspace(ext[2], ext[3], n))
        grid = self.grid_descriptors(X, Y)
        in_A = self.in_state(grid, "A").reshape(X.shape)
        in_B = self.in_state(grid, "B").reshape(X.shape)
        progress = [5, 4, 2, 1]
        print(f"[theory] solving the reference committor on a {n}x{n} grid (one time)")
        p = aimmdTIS.solve_committor_by_relaxation(
            X, Y, self.beta * Fx, self.beta * Fy, in_A, in_B, np.full(X.shape, 0.5), progress=progress)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, p)
        meta = {
            "potential_name": self.cfg["system"]["potential_name"],
            "potential_parameters": self.potential_parameters(),
            "temperature": self.cfg["system"]["integrator_params"]["temperature"],
            "extent": [float(v) for v in ext],
            "grid": n,
            "layout": "p_B[i_y, i_x] on np.linspace(extent[0], extent[1], n) x np.linspace(extent[2], extent[3], n)",
            "state_A": [float(v) for v in self.pes.state_A[:2]],
            "state_B": [float(v) for v in self.pes.state_B[:2]],
            "state_boundary": float(self.pes.state_boundary),
            "solver": "aimmdTIS.solve_committor_by_relaxation",
            "solver_progress": progress,
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        return path, p

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def run_analysis(self, k):
        """Theory comparison, logged metrics and figures for RPE_k, q_k and q_{k+1}."""
        stage = _Stage(self.rpe_dir(k), "analysis")
        if stage.done:
            print(f"[it{k}] analysis done, skipping")
            return
        from diagnostics import analyse_iteration
        analyse_iteration(self, k)
        stage.finish()

    def write_summary(self, n_iterations):
        """One row per iteration: sampling/MBAR diagnostics and the analysis metrics."""
        rows = []
        for k in range(n_iterations):
            diag_path = self.rpe_dir(k) / "diagnostics.json"
            if not diag_path.exists():
                break
            d = json.loads(diag_path.read_text())
            ifaces = self.interfaces(k)
            row = {
                "iteration": k,
                "n_ifaces_fwd": len(ifaces["forward"]),
                "n_ifaces_bwd": len(ifaces["backward"]),
                "n_paths_fwd": d["directions"]["forward"]["n_paths"],
                "n_paths_bwd": d["directions"]["backward"]["n_paths"],
                "P_B_given_lambda0A": d["directions"]["forward"]["p_reactive"],
                "P_A_given_lambda0B": d["directions"]["backward"]["p_reactive"],
                "k_AB": d["k_AB"],
                "k_BA": d["k_BA"],
                "c_AB": d["c_AB"],
                "tis_weight_before_joined_fwd": d.get("tis_weight_before_joined_fwd"),
                "tis_weight_before_joined_bwd": d.get("tis_weight_before_joined_bwd"),
            }
            analysis_path = self.rpe_dir(k) / "analysis.json"
            if analysis_path.exists():
                metrics = json.loads(analysis_path.read_text())
                metrics.pop("iteration", None)
                row.update(metrics)
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary.to_csv(self.out / "summary.csv", index=False)
        print(summary.to_string(index=False))
        return summary

    # ------------------------------------------------------------------
    def converged(self, k):
        """Theory-free stopping rule after iteration k.

        Converged when the model change ``delta_q_rms`` (weighted RMS of
        q_{k+1} - q_k over the reweighted frames with |q_k| < q_window, logged by
        the analysis) stayed below ``run.convergence.delta_q_tol`` for
        ``run.convergence.patience`` consecutive iterations.
        """
        conv = self.run_cfg.get("convergence", {})
        tol, patience = float(conv.get("delta_q_tol", 0.3)), int(conv.get("patience", 2))
        if k + 1 < patience:
            return False
        changes = [json.loads((self.rpe_dir(j) / "analysis.json").read_text()).get("delta_q_rms", np.inf)
                   for j in range(k + 1 - patience, k + 1)]
        return all(c < tol for c in changes)

    def run(self, stage="all", n_iterations=None, until_converged=False):
        """Run the pipeline.

        With ``until_converged`` iterations continue until :meth:`converged`, up to
        ``run.convergence.max_iterations``; otherwise exactly ``n_iterations``.
        """
        # Record the settings of this run next to its results (see load_run_config).
        (self.out / "config_used.json").write_text(json.dumps(self.cfg, indent=2))
        if until_converged:
            n_iterations = int(self.run_cfg.get("convergence", {}).get("max_iterations", 12))
        n_iterations = int(n_iterations or self.run_cfg["n_iterations"])
        if stage in ("all", "tps"):
            self.run_tps()
        if stage in ("all", "stable"):
            self.run_stable()
        if stage in ("all", "iterate"):
            done = 0
            for k in range(n_iterations):
                print(f"\n========== iteration {k} ==========")
                self.run_tis(k)
                self.run_rpe(k)
                self.run_training(k)
                self.run_analysis(k)
                done = k + 1
                if until_converged and self.converged(k):
                    print(f"[it{k}] converged: model change below tolerance for "
                          f"{self.run_cfg.get('convergence', {}).get('patience', 2)} iterations")
                    break
            self.write_summary(done)
        self.close()

    def close(self):
        for storage in self._seed_storages:
            storage.close()
        self._seed_storages = []
        if self._tps_storage is not None:
            self._tps_storage.close()
            self._tps_storage = None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--stage", choices=["all", "tps", "stable", "iterate"], default="all")
    parser.add_argument("--iterations", type=int, default=None, help="number of TIS iterations")
    parser.add_argument("--quick", action="store_true", help="apply the config's 'quick' overrides")
    parser.add_argument("--tis-steps", type=int, default=None, help="override run.tis_steps (MC steps per interface)")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="override run.convergence.max_iterations (with --until-converged)")
    parser.add_argument("--until-converged", action="store_true",
                        help="iterate until the model change converges (run.convergence), ignoring --iterations")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel TIS processes (default: run.n_workers or all CPUs)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up in redirected logs

    cfg = load_config(args.config, quick=args.quick)
    if args.tis_steps is not None:
        cfg["run"]["tis_steps"] = int(args.tis_steps)
    if args.max_iterations is not None:
        cfg["run"].setdefault("convergence", {})["max_iterations"] = int(args.max_iterations)
    IterativeAimmdTIS(cfg, args.out, n_workers=args.workers).run(
        stage=args.stage, n_iterations=args.iterations, until_converged=args.until_converged)


if __name__ == "__main__":
    main()
