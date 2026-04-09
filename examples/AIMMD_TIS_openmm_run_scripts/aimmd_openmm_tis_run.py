#!/usr/bin/env python
"""
Simplified AIMMD TIS example for OpenMM systems using ops-setup integration.

This script runs AIMMD TIS for a single interface without boilerplate
by leveraging the refactored setup utilities and ops-setup classes.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import openpathsampling as paths
import torch
from openpathsampling.experimental.storage import Storage

from aimmdTIS import AIMMDSetup, TIS_setup, load_initial_trajectory


def run_aimmd_tis_single_interface(
    tps_config_path: Path,
    tis_config_path: Path,
    interface_value: float,
    n_steps: int = 10,
    traj_path: Path = None,
    output_path: Path = None,
    resource_directory: Path = None,
    direction: str = "forward",
    previous_model_file: Path = None,
    system_resource_directory: Path = None,
    descriptor_transform=None,
    cv_function=None,
    save_final_traj: bool = False,
):
    """
    Run AIMMD TIS for a single interface (multiple runs needed for multiple interfaces).

    Parameters
    ----------
    tps_config_path : Path
        Path to TPS JSON configuration file.
    tis_config_path : Path
        Path to TIS JSON configuration file.
    interface_value : float
        Single interface value for this run.
    n_steps : int
        Number of MC steps.
    traj_path : Path, optional
        Path to initial trajectory storage.
    output_path : Path, optional
        Output directory.
    resource_directory : Path, optional
        System resource directory.
    direction : str
        'forward' or 'backward' TIS.
    previous_model_file : Path, optional
        Pre-trained AIMMD model.
    system_resource_directory : Path, optional
        Alternative name for resource_directory.
    descriptor_transform : callable, optional
        Custom descriptor transform.
    cv_function : callable, optional
        Custom CV function for defining states.
    save_final_traj : bool
        Whether to save final trajectory separately.
    """

    tps_config_path = Path(tps_config_path)
    tis_config_path = Path(tis_config_path)
    output_path = Path(output_path) if output_path else Path(".")
    resource_directory = Path(resource_directory or system_resource_directory or "")

    if not tps_config_path.exists() or not tis_config_path.exists():
        raise FileNotFoundError("Config files not found")

    # Setup MD engine and TIS configuration using ops-setup
    print(f"Setting up OpenMM TIS for interface {interface_value}...")
    tis_setup = TIS_setup(tps_config_path, tis_config_path, resource_directory,
                          print_config=True, cv_function=cv_function)
    paths.PathMover.engine = tis_setup.md_engine

    # Load initial trajectory
    print(f"Loading initial trajectory from {traj_path}...")
    init_traj = load_initial_trajectory(traj_path)

    # Compute descriptor dimension
    if descriptor_transform:
        test_descriptor = descriptor_transform(tis_setup.template)
        descriptor_dim = len(test_descriptor)
    else:
        descriptor_dim = 1

    # Setup AIMMD model
    print("Setting up AIMMD RCModel...")
    aimmd_setup = AIMMDSetup(
        tis_setup.config,
        descriptor_dim,
        tis_setup.states,
        descriptor_transform=descriptor_transform,
        print_config=False
    )

    # Prepare storage paths
    interface_indicator = int(round(interface_value * 100))
    aimmd_store_path = (
        output_path / 
        f"aimmd_{direction}_interface_{interface_indicator}_{aimmd_setup.distribution}.h5"
    )
    ops_store_path = (
        output_path /
        f"aimmd_tis_{tis_setup.system_name}_{direction}_int_{interface_indicator}.db"
    )

    # Create AIMMD storage and model
    aimmd = __import__("aimmd").aimmd
    aimmd_store = aimmd.Storage(str(aimmd_store_path), "w")
    model = aimmd_setup.setup_RCModel(aimmd_store, load_model_path=previous_model_file)

    # Setup AIMMD TIS framework
    from aimmdTIS import TIS_AIMMD
    tis_framework = TIS_AIMMD(
        tis_setup.md_engine,
        model,
        tis_setup.states[0],
        tis_setup.states[1],
        descriptor_transform=descriptor_transform
    )

    # Create storage
    print(f"Creating storage: {ops_store_path}")
    storage = Storage(str(ops_store_path), "w")
    storage.save(tis_setup.template)
    storage.save(tis_setup.md_engine)

    # Run TIS for single interface
    print(f"Running AIMMD TIS for interface {interface_value}, {n_steps} MC steps...")
    tis_framework.run_single_TIS(
        n_mc_steps=n_steps,
        storage=storage,
        initial_path=init_traj,
        interface_value=interface_value,
        scheme_move=tis_setup.shooting_move,
        scheme_selector=tis_setup.TIS_selector,
        scheme_modifier=tis_setup.modification_method,
        gaussian_parameter_width=tis_setup.gaussian_width,
        gausssian_parameter_shift=tis_setup.gaussian_origin_shift,
        direction=direction,
        directory=output_path
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"AIMMD TIS Complete (Interface {interface_value})!")
    print(f"{'='*60}")
    print(f"  Direction: {direction}")
    print(f"  Interface value: {interface_value}")
    print(f"  Total MC steps: {len(storage.steps)}")
    print(f"  OPS Storage: {ops_store_path}")
    print(f"  AIMMD Storage: {aimmd_store_path}")

    # Save final trajectory if requested
    if save_final_traj:
        final_traj_path = (
            output_path /
            f"aimmd_tis_final_traj_{direction}_int_{interface_indicator}.db"
        )
        print(f"Saving final trajectory to {final_traj_path}...")
        final_storage = Storage(str(final_traj_path), "w")
        final_traj = storage.steps[-1].active[0].trajectory
        final_storage.save(final_traj)
        final_storage.close()

    storage.close()
    aimmd_store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AIMMD-enhanced TIS for OpenMM systems (single interface)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("tps_config", help="Path to TPS configuration JSON file")
    parser.add_argument("tis_config", help="Path to TIS configuration JSON file")
    parser.add_argument("interface_value", type=float, help="Interface value for this run")
    parser.add_argument("n_steps", type=int, nargs="?", default=10,
                        help="Number of MC steps (default: 10)")
    parser.add_argument("-t", "--trajectory", type=Path, required=True,
                        help="Path to initial trajectory (.nc or .db)")
    parser.add_argument("-o", "--output", type=Path,
                        help="Output directory (default: current directory)")
    parser.add_argument("-r", "--resources", type=Path,
                        help="System resource directory")
    parser.add_argument("-d", "--direction", choices=["forward", "backward"], default="forward",
                        help="TIS direction (default: forward)")
    parser.add_argument("-m", "--model", type=Path,
                        help="Previous AIMMD model to load")
    parser.add_argument("--save-final", action="store_true",
                        help="Save final trajectory separately")

    args = parser.parse_args()

    try:
        run_aimmd_tis_single_interface(
            tps_config_path=args.tps_config,
            tis_config_path=args.tis_config,
            interface_value=args.interface_value,
            n_steps=args.n_steps,
            traj_path=args.trajectory,
            output_path=args.output,
            resource_directory=args.resources,
            direction=args.direction,
            previous_model_file=args.model,
            save_final_traj=args.save_final,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
