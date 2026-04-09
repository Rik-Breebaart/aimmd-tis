#!/usr/bin/env python
"""
TPS with AIMMD enhancement for toy systems.

This script demonstrates TPS sampling with AIMMD-based shooting point selection
on toy potential systems.

Usage:
    python toy_tps_aimmd.py [potential_name] [n_steps] [options]

Examples:
    python toy_tps_aimmd.py potential_1 10 -t trajectory.nc -o results/
    python toy_tps_aimmd.py potential_2 50 -m pretrained_model.h5 -o results/
"""

import sys
import argparse
import numpy as np
from pathlib import Path

import openpathsampling as paths
import torch

from ops_setup.systems.examples.toy_systems import (
    ToyTPSSetup,
    potential_0,
    potential_1,
    potential_2,
)
from aimmdTIS import AIMMDSetup, load_initial_trajectory


def run_toy_tps_aimmd(
    potential_name="potential_1",
    n_steps=10,
    traj_path=None,
    output_path="./tps_aimmd_toy_results",
    previous_model=None,
    save_final_traj=False,
    descriptor_transform=None,
):
    """
    Run TPS with AIMMD for toy systems.

    Parameters
    ----------
    potential_name : str
        Name of toy potential: 'potential_0', 'potential_1', 'potential_2'
    n_steps : int
        Number of MC steps to run
    traj_path : Path, optional
        Path to initial trajectory
    output_path : Path or str
        Output directory for results
    previous_model : Path, optional
        Path to pre-trained AIMMD model
    save_final_traj : bool
        Whether to save final trajectory separately
    descriptor_transform : callable, optional
        Custom descriptor transformation function
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"TPS with AIMMD - Toy System: {potential_name}")
    print(f"{'='*70}\n")

    # Validate potential
    valid_potentials = ["potential_0", "potential_1", "potential_2"]
    if potential_name not in valid_potentials:
        raise ValueError(
            f"Unknown potential: {potential_name}. "
            f"Valid options: {valid_potentials}"
        )

    # Step 1: Setup TPS via ops-setup
    print("[1/5] Setting up TPS engine...")
    tps_setup = ToyTPSSetup(
        potential_name=potential_name,
        potential_kwargs={},
        integrator_params={"dt": 0.05, "temperature": 1.0, "gamma": 2.5},
    )
    print(f"  Potential: {tps_setup.system_name}")
    print(f"  Engine: {tps_setup.md_engine.name}")

    # Step 2: Load or create initial trajectory
    print("\n[2/5] Loading initial trajectory...")
    paths.PathMover.engine = tps_setup.md_engine

    if traj_path is not None and Path(traj_path).exists():
        init_traj = load_initial_trajectory(traj_path)
        print(f"  Loaded: {len(init_traj)} snapshots from {traj_path}")
    else:
        n_snapshots = 150 if potential_name == "potential_0" else 200
        init_traj = paths.Trajectory(
            tps_setup.pes.simple_initial_path(n_snapshots, tps_setup.md_engine)
        )
        print(f"  Generated: {len(init_traj)} snapshots")

    # Step 3: Setup AIMMD model
    print("\n[3/5] Setting up AIMMD model...")

    # Extract descriptor dimension from potential
    descriptor_dim = 2 if potential_name != "potential_0" else 1

    aimmd_config = {
        "AIMMD_settings": {
            "activation": "ReLU",
            "layers": {"0": 16, "1": 8},
            "dropout": {"0": 0.1, "1": 0.05},
            "distribution": "lorentzian",
            "scale": 1.5,
            "use_GPU": torch.cuda.is_available(),
            "ee_params": {"lr_0": 1e-3},
        }
    }

    aimmd_setup = AIMMDSetup(
        config=aimmd_config,
        descriptor_dim=descriptor_dim,
        states=(tps_setup.stateA, tps_setup.stateB),
        descriptor_transform=descriptor_transform,
    )

    # Create storage for AIMMD
    aimmd_storage_path = output_path / f"aimmd_tps_{potential_name}.h5"
    try:
        import aimmd
        aimmd_storage = aimmd.Storage(str(aimmd_storage_path), "w")
    except Exception as e:
        print(f"  Warning: Could not create AIMMD storage: {e}")
        aimmd_storage = None

    # Load or create RC model
    if previous_model is not None and Path(previous_model).exists():
        print(f"  Loading model: {previous_model}")
        model = aimmd_setup.load_RCModel(previous_model)
    else:
        print(f"  Creating new RC model...")
        if aimmd_storage is not None:
            model = aimmd_setup.setup_RCModel(aimmd_storage)
        else:
            print("  Warning: Could not create RC model without AIMMD storage")
            model = None

    # Create selector if model exists
    if model is not None:
        selector = aimmd_setup.setup_selector(model)
        print(f"  Selector: {type(selector).__name__}")
    else:
        selector = None

    # Step 4: Create TPS network and move scheme
    print("\n[4/5] Setting up TPS network and move scheme...")
    network = tps_setup.create_network("TPS")
    move_scheme = tps_setup.create_move_scheme(network, "OneWay", selector=selector)
    print(f"  Network: {type(network).__name__}")
    print(f"  Move scheme: {type(move_scheme).__name__}")

    # Step 5: Run TPS simulation
    print(f"\n[5/5] Running TPS for {n_steps} MC steps...")

    storage_path = output_path / f"tps_aimmd_{potential_name}.nc"
    storage = paths.Storage(str(storage_path), "w", template=tps_setup.template)
    storage.save(tps_setup.md_engine)
    storage.save(move_scheme)
    storage.save(network)

    # Prepare initial conditions
    initial_conditions = move_scheme.initial_conditions_from_trajectories(init_traj)
    initial_conditions.sanity_check()

    # Run sampling
    sampler = paths.PathSampling(storage, move_scheme, initial_conditions)
    sampler.run(n_steps)

    # Summary
    print(f"\n{'='*70}")
    print(f"TPS Simulation Complete!")
    print(f"{'='*70}")
    print(f"  Total MC steps: {len(storage.steps)}")
    print(f"  OPS Storage: {storage_path}")
    if aimmd_storage is not None:
        print(f"  AIMMD Storage: {aimmd_storage_path}")
    print(f"  Acceptance rate: {move_scheme.acceptance_rate:.1%}")

    n_accepted = sum(1 for step in storage.steps if step.active[0].is_accepted)
    print(f"  Accepted trajectories: {n_accepted}/{len(storage.steps)}")

    traj_lengths = [len(step.active[0].trajectory) for step in storage.steps]
    print(
        f"  Trajectory lengths: "
        f"min={min(traj_lengths)}, max={max(traj_lengths)}, "
        f"avg={np.mean(traj_lengths):.1f}"
    )

    # Optional: Save final trajectory
    if save_final_traj and len(storage.steps) > 0:
        final_traj_path = output_path / f"final_traj_aimmd_{potential_name}.nc"
        final_storage = paths.Storage(str(final_traj_path), "w")
        final_traj = storage.steps[-1].active[0].trajectory
        final_storage.save(final_traj)
        final_storage.close()
        print(f"  Final trajectory saved: {final_traj_path}")

    storage.close()
    if aimmd_storage is not None:
        aimmd_storage.close()

    print(f"\nResults saved to: {output_path}\n")


def main():
    """Parse command line arguments and run TPS with AIMMD."""
    parser = argparse.ArgumentParser(
        description="Run TPS with AIMMD on toy systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "potential",
        nargs="?",
        default="potential_1",
        help="Toy potential: potential_0 (1D), potential_1 (2D), potential_2 (2D shifted)",
    )

    parser.add_argument(
        "n_steps",
        nargs="?",
        type=int,
        default=10,
        help="Number of TPS MC steps",
    )

    parser.add_argument(
        "-t",
        "--trajectory",
        type=Path,
        help="Path to initial trajectory",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="./tps_aimmd_toy_results",
        help="Output directory",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        help="Pre-trained AIMMD model",
    )

    parser.add_argument(
        "--save-final",
        action="store_true",
        help="Save final trajectory separately",
    )

    args = parser.parse_args()

    try:
        run_toy_tps_aimmd(
            potential_name=args.potential,
            n_steps=args.n_steps,
            traj_path=args.trajectory,
            output_path=args.output,
            previous_model=args.model,
            save_final_traj=args.save_final,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
