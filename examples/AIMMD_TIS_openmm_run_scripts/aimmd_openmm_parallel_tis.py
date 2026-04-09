#!/usr/bin/env python
"""
Parallel AIMMD TIS runner for multiple interfaces.

This script runs AIMMD TIS for multiple interface values in parallel using multiprocessing,
reusing the single-interface TIS runner.
"""

import argparse
import sys
from multiprocessing import Process
from pathlib import Path
from time import sleep

from aimmd_openmm_tis_run import run_aimmd_tis_single_interface


def run_parallel_tis(
    tps_config: Path,
    tis_config: Path,
    interface_values: list,
    n_steps: int,
    traj_path: Path,
    output_path: Path,
    resource_directory: Path = None,
    direction: str = "forward",
    previous_model: Path = None,
    save_final_traj: bool = False,
    descriptor_transform=None,
    cv_function=None,
    process_delay: float = 10.0,
):
    """
    Run AIMMD TIS in parallel for multiple interface values.

    Parameters
    ----------
    tps_config : Path
        Path to TPS configuration.
    tis_config : Path
        Path to TIS configuration.
    interface_values : list of float
        List of interface values to sample.
    n_steps : int
        MC steps per interface.
    traj_path : Path
        Initial trajectory path.
    output_path : Path
        Output directory.
    resource_directory : Path, optional
        System resource directory.
    direction : str
        'forward' or 'backward'.
    previous_model : Path, optional
        Previous AIMMD model.
    save_final_traj : bool
        Save final trajectory.
    descriptor_transform : callable, optional
        Custom descriptor transform.
    cv_function : callable, optional
        Custom CV function.
    process_delay : float
        Delay between process starts (seconds).
    """

    processes = []
    for i, interface_value in enumerate(interface_values):
        print(f"Starting process {i} for interface {interface_value}...")
        process = Process(
            target=run_aimmd_tis_single_interface,
            args=(
                tps_config,
                tis_config,
                interface_value,
                n_steps,
                traj_path,
                output_path,
                resource_directory,
                direction,
                previous_model,
                resource_directory,
                descriptor_transform,
                cv_function,
                save_final_traj,
            ),
            name=f"TIS_Interface_{interface_value}",
        )
        processes.append(process)
        process.start()

        if i < len(interface_values) - 1:
            sleep(process_delay)

    # Wait for all processes
    print(f"\nWaiting for {len(processes)} TIS processes to complete...")
    for i, process in enumerate(processes):
        process.join()
        print(f"Process {i} completed")

    print("\nAll parallel TIS runs completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel AIMMD TIS for multiple interfaces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("tps_config", help="Path to TPS configuration JSON")
    parser.add_argument("tis_config", help="Path to TIS configuration JSON")
    parser.add_argument("n_steps", type=int, help="MC steps per interface")
    parser.add_argument("-i", "--interfaces", type=float, nargs="+", required=True,
                        help="Interface values to sample")
    parser.add_argument("-t", "--trajectory", type=Path, required=True,
                        help="Path to initial trajectory")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output directory")
    parser.add_argument("-r", "--resources", type=Path,
                        help="System resource directory")
    parser.add_argument("-d", "--direction", choices=["forward", "backward"],
                        default="forward", help="TIS direction")
    parser.add_argument("-m", "--model", type=Path,
                        help="Previous AIMMD model to load")
    parser.add_argument("--save-final", action="store_true",
                        help="Save final trajectories")
    parser.add_argument("--delay", type=float, default=10.0,
                        help="Delay between process starts (seconds)")

    args = parser.parse_args()

    try:
        run_parallel_tis(
            tps_config=Path(args.tps_config),
            tis_config=Path(args.tis_config),
            interface_values=args.interfaces,
            n_steps=args.n_steps,
            traj_path=Path(args.trajectory),
            output_path=Path(args.output),
            resource_directory=Path(args.resources) if args.resources else None,
            direction=args.direction,
            previous_model=Path(args.model) if args.model else None,
            save_final_traj=args.save_final,
            process_delay=args.delay,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
