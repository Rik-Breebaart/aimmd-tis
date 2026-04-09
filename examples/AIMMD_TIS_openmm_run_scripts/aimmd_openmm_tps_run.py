#!/usr/bin/env python
"""
Simplified AIMMD TPS example for OpenMM systems using ops-setup integration.

This script runs AIMMD TPS without the boilerplate found in original examples
by leveraging the refactored setup utilities and ops-setup classes.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import openpathsampling as paths
import torch
from openpathsampling.experimental.storage import Storage
from simtk import unit

from aimmdTIS import AIMMDSetup, TPS_setup, load_initial_trajectory


def run_aimmd_tps(
    config_path: Path,
    n_steps: int = 10,
    traj_path: Path = None,
    output_path: Path = None,
    resource_directory: Path = None,
    previous_model_file: Path = None,
    system_resource_directory: Path = None,
    descriptor_transform=None,
    cv_function=None,
):
    """
    Run AIMMD-enhanced TPS for a single system without multiprocessing.

    Parameters
    ----------
    config_path : Path
        Path to TPS JSON configuration file.
    n_steps : int
        Number of MC steps.
    traj_path : Path, optional
        Path to initial trajectory storage.
    output_path : Path, optional
        Output directory.
    resource_directory : Path, optional
        System resource directory.
    previous_model_file : Path, optional
        Pre-trained AIMMD model.
    system_resource_directory : Path, optional
        Alternative name for resource_directory.
    descriptor_transform : callable, optional
        Custom descriptor transform.
    cv_function : callable, optional
        Custom CV function for defining states.
    """

    config_path = Path(config_path)
    output_path = Path(output_path) if output_path else Path(".")
    resource_directory = Path(resource_directory or system_resource_directory or "")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Setup MD engine using ops-setup
    print("Setting up OpenMM TPS...")
    tps_setup = TPS_setup(config_path, resource_directory, print_config=True,
                          cv_function=cv_function)
    paths.PathMover.engine = tps_setup.md_engine

    # Load initial trajectory
    print(f"Loading initial trajectory from {traj_path}...")
    init_traj = load_initial_trajectory(traj_path)

    # Compute descriptor dimension
    if descriptor_transform:
        test_descriptor = descriptor_transform(tps_setup.template)
        descriptor_dim = len(test_descriptor)
    else:
        descriptor_dim = 1  # Default

    # Setup AIMMD model
    print("Setting up AIMMD RCModel...")
    aimmd_setup = AIMMDSetup(
        tps_setup.config,
        descriptor_dim,
        tps_setup.states,
        descriptor_transform=descriptor_transform,
        print_config=False
    )

    # Prepare storage paths
    aimmd_store_path = output_path / f"aimmd_{aimmd_setup.distribution}_tps_{tps_setup.system_name}.h5"
    ops_store_path = output_path / f"aimmd_tps_{tps_setup.system_name}.db"

    # Create AIMMD storage and model
    aimmd_store = __import__("aimmd").Storage(str(aimmd_store_path), "w")
    model = aimmd_setup.setup_RCModel(aimmd_store, load_model_path=previous_model_file)

    # Get GPU device
    if torch.cuda.is_available() and aimmd_setup.use_GPU:
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available() and aimmd_setup.use_GPU:
        print("Using MPS GPU")

    # Setup hooks for AIMMD
    aimmd = __import__("aimmd").aimmd
    trainset = aimmd.TrainSet(n_states=2)
    trainhook = aimmd.ops.TrainingHook(model, trainset)
    storehook = aimmd.ops.AimmdStorageHook(aimmd_store, model, trainset)
    densityhook = aimmd.ops.DensityCollectionHook(model)

    # Setup selector
    selector = aimmd_setup.setup_selector(model)

    # Create move scheme
    beta = 1.0 / (tps_setup.integrator.getTemperature() * unit.BOLTZMANN_CONSTANT_kB)
    modifier = paths.RandomVelocities(beta=beta, engine=tps_setup.md_engine)

    tw_strategy = paths.strategies.TwoWayShootingStrategy(
        modifier=modifier,
        selector=selector,
        engine=tps_setup.md_engine,
        group="TwoWayShooting"
    )

    network = paths.TPSNetwork(tps_setup.states[0], tps_setup.states[1])
    move_scheme = paths.MoveScheme(network=network)
    move_scheme.append(tw_strategy)
    move_scheme.append(paths.strategies.OrganizeByMoveGroupStrategy())
    move_scheme.build_move_decision_tree()

    # Initial conditions
    print("Preparing initial conditions...")
    initial_conditions = move_scheme.initial_conditions_from_trajectories(init_traj)
    initial_conditions.sanity_check()

    # Create storage
    print(f"Creating storage: {ops_store_path}")
    storage = Storage(str(ops_store_path), "w")
    storage.save(tps_setup.template)
    storage.save(initial_conditions)
    storage.save(move_scheme)

    # Run sampler
    print(f"Running AIMMD TPS for {n_steps} MC steps...")
    sampler = paths.PathSampling(storage, move_scheme, initial_conditions).named("TPS_sampler")
    sampler.attach_hook(trainhook)
    sampler.attach_hook(storehook)
    sampler.attach_hook(densityhook)
    sampler.run(n_steps)

    # Summary
    print(f"\n{'='*60}")
    print(f"AIMMD TPS Simulation Complete!")
    print(f"{'='*60}")
    print(f"  Total MC steps: {len(storage.steps)}")
    print(f"  OPS Storage: {ops_store_path}")
    print(f"  AIMMD Storage: {aimmd_store_path}")

    storage.close()
    aimmd_store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AIMMD-enhanced TPS for OpenMM systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="Path to TPS configuration JSON file")
    parser.add_argument("n_steps", type=int, nargs="?", default=10,
                        help="Number of MC steps (default: 10)")
    parser.add_argument("-t", "--trajectory", type=Path, required=True,
                        help="Path to initial trajectory (.nc or .db)")
    parser.add_argument("-o", "--output", type=Path,
                        help="Output directory (default: current directory)")
    parser.add_argument("-r", "--resources", type=Path,
                        help="System resource directory")
    parser.add_argument("-m", "--model", type=Path,
                        help="Previous AIMMD model to load")

    args = parser.parse_args()

    try:
        run_aimmd_tps(
            config_path=args.config,
            n_steps=args.n_steps,
            traj_path=args.trajectory,
            output_path=args.output,
            resource_directory=args.resources,
            previous_model_file=args.model,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
