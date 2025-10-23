import numpy as np
import matplotlib.pyplot as plt
import openpathsampling as paths
import torch
import sys
import os
import argparse
from pathlib import Path
from multiprocessing import Process
from functools import reduce
from copy import deepcopy
from simtk import unit
import openpathsampling.engines.openmm as ops_openmm
from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage.collective_variables import MDTrajFunctionCV
from openpathsampling.experimental.storage import Storage

# Monkey patch OpenPathSampling to use experimental features
paths = monkey_patch_all(paths)

# Add necessary directories to sys.path
# current_directory = os.path.dirname(os.path.abspath(os.getcwd()))
current_directory = "/home/rbreeba/Projects/-RE-TIS-AIMMD/TIS_AIMMD_biosystems/Host-Guest-System"

sys.path.append(current_directory)
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
parent_parent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir))
sys.path.append(parent_directory)
sys.path.append(parent_parent_directory)

# Custom imports
from aimmd import aimmd
import TIS_AIMMD_toy_framework as TAI
from TIS_AIMMD_toy_framework import TIS_AIMMD_setup, read_config
from AIMMD_TIS_openmm_run_scripts.setup_utilities import TPS_setup, AIMMD_setup, create_parser

def run_MD(input_path: Path, config_path=None, traj_path=None, system_resource_directory=None, n_steps=None, output_path=None, 
           ops_storage_path=None, state_sampled="A"):
    
    if system_resource_directory is None:
        system_resource_directory= ""
    config_path = str(input_path.joinpath(config_path))
    TPS_utils = TPS_setup(config_path, resource_directory=system_resource_directory)

    # Select the OpenMM engine
    paths.EngineMover.engine = TPS_utils.md_engine

    # Load or initialize initial trajectory
    if traj_path:
        # Check the file extension
        if traj_path.suffix == ".nc":
            old_store = paths.Storage(traj_path)
            init_traj = old_store.tag["trajectory"]
            mdtraj_traj = init_traj.to_mdtraj()
            old_store.close()
            init_traj = ops_openmm.trajectory_from_mdtraj(mdtraj_traj)
        elif traj_path.suffix == ".db":
            old_store = Storage(traj_path)
            init_traj = old_store.tag["trajectory"]
            mdtraj_traj = init_traj.to_mdtraj()
            old_store.close()
            init_traj = ops_openmm.trajectory_from_mdtraj(mdtraj_traj)
        else:
            raise ValueError("Unsupported file type. Use .nc or .db for trajectory storage.")
    else:
        raise ValueError("traj_path is required to load initial trajectory.")
    # Define states and interfaces
    def distance_function(snapshot, receptor_atoms, ligand_atoms):
        receptor_com = snapshot.xyz[receptor_atoms, :].mean(0)
        ligand_com = snapshot.xyz[ligand_atoms, :].mean(0)
        return np.sqrt(((receptor_com - ligand_com) ** 2).sum())

    distance_cv = paths.CoordinateFunctionCV(
        name="distance",
        f=distance_function,
        receptor_atoms=np.arange(0, 126),
        ligand_atoms=np.arange(126, 156)
    ).with_diskcache()

    bound = paths.CVDefinedVolume(distance_cv, lambda_min=0.0, lambda_max=0.05).named('bound')
    unbound = paths.CVDefinedVolume(distance_cv, lambda_min=0.6, lambda_max=float("inf")).named('unbound')

    states = (bound, unbound)

    # Set initial state based on the sampled state
    state_index = 0 if state_sampled == "A" else 1
    init_index = int(state_index * -1)
    initial_snapshot = init_traj.snapshots[init_index]
    check = states[state_index](initial_snapshot)
    print(f"Start in {state_sampled}: {check}")

    # Set storage path
    if ops_storage_path is None:
        ops_storage_path = f"stable_storage_tps_storage_{TPS_utils.system_name}_{state_sampled}"
    ops_storage_path = Path(output_path / ops_storage_path).with_suffix(".db")

    # Initialize storage and save initial state
    print(f"Attempting to create storage at: {ops_storage_path}")
    storage = Storage(ops_storage_path, "w")
    storage.save(initial_snapshot)

    # Define the TPS network and set up the simulation
    tps_network = paths.TPSNetwork(states[0], states[1])
    states_set = set(tps_network.initial_states + tps_network.final_states)
    sim = paths.DirectSimulation(
        storage=storage,
        engine=TPS_utils.md_engine,
        states=states_set,
        initial_snapshot=storage.snapshots[0]
    )

    # Run the MD simulation
    print(f"Start stable state MD run: {n_steps}")
    sim.run(n_steps)
    storage.close()
    print("Finished Stable MD simulation.")

if __name__ == "__main__":


    # Define the list of argument configurations
    arguments_list = [
        {"flags": ["-dir", "--directory"], "type": Path, "required": True, "help": "Input directory path"},
        {"flags": ["-cfg", "--config_file"], "type": str, "required": True, "help": "Configuration file"},
        {"flags": ["-nr", "--n_steps"], "type": int, "required": True, "help": "Number of MD steps"},
        {"flags": ["-out", "--output_path"], "type": Path, "required": False, "help": "Output directory"},
        {"flags": ["-ops_store", "--ops_store"], "type": Path, "required": False, "help": "OPS storage name", "default": Path("default_storage_path_stable")},
        {"flags": ["-state", "--state"], "type": str, "required": True, "help": "Starting state ('A' or 'B')", "deault": "A"},
        {"flags": ["-traj", "--trajectory_path"], "type": Path, "required": True, "help": "Trajectory file path (.nc or .db)"},
        {"flags": ["-sys_source_dir", "--system_resource_directory"], "type": Path, "required": False, "help": "system specific folder path ","default":""},
    ]

    parser= create_parser(arguments_list)
    args = parser.parse_args()
    # Start the process, passing the arguments namespace directly to the function
    in_path = args.directory
    configfile = args.config_file
    nsteps = args.n_steps
    outpath = args.output_path or in_path
    ops_storage_path = args.ops_store
    state_to_sample = args.state
    input_traj_path = args.trajectory_path
    system_resource_directory = args.system_resource_directory or ""

    process = Process(target=run_MD, args=(in_path, configfile, input_traj_path, system_resource_directory, nsteps, outpath, ops_storage_path, state_to_sample))
    process.start()