#!/usr/bin/env python
"""
AIMMD-TPS on a 2D toy potential (step 1 of the AIMMD-TIS workflow).

Runs TPS with an AIMMD shooting-point selector. The committor model is trained
on the fly from the shooting results. The trained model (aimmd ``.h5`` storage)
and the TPS paths (OPS ``.nc`` storage) are the inputs for ``toy_tis_aimmd.py``.

The toy system (potential, engine, states) comes from ops-setup; the AIMMD
model and selector come from aimmdTIS.

Usage:
    python toy_tps_aimmd.py [n_steps] [options]

Examples:
    python toy_tps_aimmd.py 100 -o results/
    python toy_tps_aimmd.py 100 -o results/ -m results/aimmd_tps_wolfe-quapp.h5
"""

import argparse
import sys
from pathlib import Path

import aimmd
import numpy as np
import openpathsampling as paths
from ops_setup.engines.openmm_sampling import load_initial_trajectory
from ops_setup.systems.examples.toy_systems import ToyTPSSetup

from aimmdTIS import AIMMDSetup
from aimmdTIS.Tools import read_config

EXAMPLES_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = EXAMPLES_DIR / "config_aimmd.json"

# Same toy system as examples/aimmd_with_toy_systems.ipynb
POTENTIAL_NAME = "wolfe-quapp"
POTENTIAL_KWARGS = {"n_harmonics": 0, "rotation_degrees": 0, "scale": 2}
INTEGRATOR_PARAMS = {"dt": 0.05, "temperature": 1, "gamma": 10}


def make_toy_setup():
    """Create the ops-setup toy system shared by the TPS and TIS scripts."""
    return ToyTPSSetup(
        potential_name=POTENTIAL_NAME,
        potential_kwargs=POTENTIAL_KWARGS,
        integrator_params=INTEGRATOR_PARAMS,
    )


def tps_storage_paths(output_path, system_name):
    """Return the (aimmd, ops) storage paths written by this script."""
    output_path = Path(output_path)
    return (
        output_path / f"aimmd_tps_{system_name}.h5",
        output_path / f"tps_{system_name}.nc",
    )


def run_toy_tps_aimmd(
    n_steps=100,
    output_path="./toy_results",
    config_path=DEFAULT_CONFIG,
    traj_path=None,
    previous_model=None,
):
    """
    Run AIMMD-TPS on the toy potential.

    Parameters
    ----------
    n_steps : int
        Number of TPS Monte Carlo steps.
    output_path : Path or str
        Directory for the aimmd and OPS storages.
    config_path : Path or str
        AIMMD configuration JSON (``AIMMD_settings`` block).
    traj_path : Path, optional
        Initial A->B trajectory. If omitted, a straight path is generated.
    previous_model : Path, optional
        aimmd storage to continue training from.

    Returns
    -------
    tuple of Path
        Paths of the aimmd storage and the OPS storage.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Toy system from ops-setup
    tps_setup = make_toy_setup()
    paths.PathMover.engine = tps_setup.md_engine
    print(f"System: {tps_setup.system_name}, engine: {tps_setup.md_engine.name}")

    if traj_path is not None:
        init_traj = load_initial_trajectory(traj_path)
    else:
        init_traj = paths.Trajectory(
            tps_setup.pes.simple_initial_path(200, tps_setup.md_engine)
        )
    print(f"Initial trajectory: {len(init_traj)} frames")

    # 2. AIMMD model and selector from aimmdTIS
    aimmd_set = AIMMDSetup(
        read_config(config_path),
        descriptor_dim=tps_setup.pes.n_dims_pot,
        states=tps_setup.states,
    )
    aimmd_store_path, ops_store_path = tps_storage_paths(
        output_path, tps_setup.system_name
    )
    aimmd_storage = aimmd.Storage(str(aimmd_store_path), "w")
    model = aimmd_set.setup_RCModel(aimmd_storage, load_model_path=previous_model)
    selector = aimmd_set.setup_selector(model)

    # Hooks: train the model on the shooting results and store it
    trainset = aimmd.TrainSet(n_states=2)
    hooks = [
        aimmd.ops.TrainingHook(model, trainset),
        aimmd.ops.AimmdStorageHook(aimmd_storage, model, trainset),
        aimmd.ops.DensityCollectionHook(model),
    ]

    # 3. TPS network, move scheme and storage
    network = tps_setup.create_network()
    move_scheme = tps_setup.create_move_scheme(network, "TwoWay", selector=selector)
    initial_conditions = move_scheme.initial_conditions_from_trajectories(init_traj)
    initial_conditions.sanity_check()

    storage = paths.Storage(str(ops_store_path), "w", template=tps_setup.template)
    storage.save(tps_setup.template)
    storage.save(tps_setup.md_engine)
    storage.save(move_scheme)
    storage.save(network)

    # 4. Run
    sampler = paths.PathSampling(storage, move_scheme, initial_conditions)
    for hook in hooks:
        sampler.attach_hook(hook)
    print(f"Running AIMMD-TPS for {n_steps} MC steps...")
    sampler.run(n_steps)

    lengths = [len(step.active[0].trajectory) for step in storage.steps]
    print(f"\nDone: {len(storage.steps)} steps, "
          f"path length min/mean/max = {min(lengths)}/{np.mean(lengths):.1f}/{max(lengths)}")
    print(f"  aimmd storage: {aimmd_store_path}")
    print(f"  OPS storage:   {ops_store_path}")

    storage.close()
    aimmd_storage.close()
    return aimmd_store_path, ops_store_path


def main():
    parser = argparse.ArgumentParser(
        description="AIMMD-TPS on the Wolfe-Quapp toy potential",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("n_steps", nargs="?", type=int, default=100,
                        help="Number of TPS MC steps (default: 100)")
    parser.add_argument("-o", "--output", type=Path, default=Path("./toy_results"),
                        help="Output directory (default: ./toy_results)")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG,
                        help="AIMMD config JSON (default: examples/config_aimmd.json)")
    parser.add_argument("-t", "--trajectory", type=Path,
                        help="Initial trajectory (.nc or .db); generated if omitted")
    parser.add_argument("-m", "--model", type=Path,
                        help="aimmd storage (.h5) to continue training from")
    args = parser.parse_args()

    try:
        run_toy_tps_aimmd(
            n_steps=args.n_steps,
            output_path=args.output,
            config_path=args.config,
            traj_path=args.trajectory,
            previous_model=args.model,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
