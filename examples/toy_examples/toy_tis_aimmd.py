#!/usr/bin/env python
"""
AIMMD-TIS on a 2D toy potential (step 2 of the AIMMD-TIS workflow).

Uses the committor model and the last path from ``toy_tps_aimmd.py``. It runs
TIS for a set of interfaces placed on the learned committor q, in one direction
(``forward``: A -> B, ``backward``: B -> A). Each interface gets its own OPS
storage, so runs stay short. Descriptors are written on the fly to an ops-setup
``DescriptorStorage`` and can be loaded for RPE/MBAR reweighting and training
(see ``examples/aimmd_with_toy_systems.ipynb``).

Interfaces are values of the model log-committor q. To place them from
stable-state data, use ``aimmdTIS.diagnostics.check_interfaces`` as in the
notebook.

Usage:
    python toy_tis_aimmd.py TPS_DIR -i Q1 Q2 ... [options]

Examples:
    python toy_tis_aimmd.py results/ -i -3 -2 -1 -n 50
    python toy_tis_aimmd.py results/ -i 3 2 1 -d backward -n 50
"""

import argparse
import sys
from pathlib import Path

import aimmd
import openpathsampling as paths

import aimmdTIS
from toy_tps_aimmd import make_toy_setup, tps_storage_paths


def run_toy_tis_aimmd(
    tps_dir,
    interface_values,
    n_steps=50,
    direction="forward",
    output_path=None,
    n_thermalize=0,
):
    """
    Run AIMMD-TIS on the toy potential for a list of interfaces.

    Parameters
    ----------
    tps_dir : Path or str
        Output directory of ``toy_tps_aimmd.py``.
    interface_values : list of float
        Interface values on the model log-committor q.
    n_steps : int
        Number of MC steps per interface.
    direction : {"forward", "backward"}
        TIS direction.
    output_path : Path or str, optional
        Output directory (default: ``<tps_dir>/aimmd_tis``).
    n_thermalize : int
        MC steps skipped when computing CV extrema.

    Returns
    -------
    list of dict
        One run summary per interface, including the storage path.
    """
    tps_dir = Path(tps_dir)
    output_path = Path(output_path) if output_path else tps_dir / "aimmd_tis"

    tps_setup = make_toy_setup()
    paths.PathMover.engine = tps_setup.md_engine
    aimmd_store_path, ops_store_path = tps_storage_paths(tps_dir, tps_setup.system_name)

    # Trained committor model and last TPS path
    aimmd_storage = aimmd.Storage(str(aimmd_store_path), "r")
    model = aimmd_storage.rcmodels["most_recent"]
    model.nnet.to(model._device)
    ops_storage = paths.Storage(str(ops_store_path), "r")
    initial_path = ops_storage.steps[-1].active[0].trajectory

    tis = aimmdTIS.AIMMD_TIS(
        engine=tps_setup.md_engine,
        AIMMD_model=model,
        stateA=tps_setup.states[0],
        stateB=tps_setup.states[1],
        directory=output_path,
        use_transform=False,
        model_pkl=False,
        cvq_type="pkl",
    )
    summaries = tis.run_TIS_sequentially(
        n_mc_steps=n_steps,
        initial_path=initial_path,
        interface_values=interface_values,
        template=tps_setup.template,
        scheme_move="TwoWay",
        scheme_selector="Gaussian",
        scheme_modifier="RandomVelocities",
        gaussian_parameter_width=1.0,
        gaussian_parameter_shift=0.2,
        direction=direction,
        directory=output_path,
        overwrite=True,
        n_thermalize=n_thermalize,
        descriptor_storage_path=output_path / f"tis_descriptors_{direction}",
    )

    print("\nAIMMD-TIS run summary:")
    for row in summaries:
        print(f"  q = {row['interface_value']:>6}: {row['storage_path']}")

    ops_storage.close()
    aimmd_storage.close()
    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="AIMMD-TIS on the Wolfe-Quapp toy potential",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("tps_dir", type=Path,
                        help="Output directory of toy_tps_aimmd.py")
    parser.add_argument("-i", "--interfaces", type=float, nargs="+", required=True,
                        help="Interface values on the model log-committor q")
    parser.add_argument("-n", "--n-steps", type=int, default=50,
                        help="MC steps per interface (default: 50)")
    parser.add_argument("-d", "--direction", choices=["forward", "backward"],
                        default="forward", help="TIS direction (default: forward)")
    parser.add_argument("-o", "--output", type=Path,
                        help="Output directory (default: TPS_DIR/aimmd_tis)")
    parser.add_argument("--n-thermalize", type=int, default=0,
                        help="MC steps skipped for CV extrema (default: 0)")
    args = parser.parse_args()

    try:
        run_toy_tis_aimmd(
            tps_dir=args.tps_dir,
            interface_values=args.interfaces,
            n_steps=args.n_steps,
            direction=args.direction,
            output_path=args.output,
            n_thermalize=args.n_thermalize,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
