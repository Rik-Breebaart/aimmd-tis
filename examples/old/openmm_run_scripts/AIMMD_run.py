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
current_directory = "/home/rbreeba/Projects/-RE-TIS-AIMMD/TIS_AIMMD_biosystems/Host-Guest-System-NPT"

sys.path.append(current_directory)
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
parent_parent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir))
sys.path.append(parent_directory)
sys.path.append(parent_parent_directory)

# Custom imports
from aimmd import aimmd
import TIS_AIMMD_toy_framework as TAI
from TIS_AIMMD_toy_framework import TIS_AIMMD_setup, read_config
from examples.AIMMD_TIS_openmm_run_scripts.HostGuest.setup_utilities import TPS_setup, AIMMD_setup, create_parser, global_arguments
from examples.AIMMD_TIS_openmm_run_scripts.HostGuest.transform_functions import cv_hg_distance
from examples.AIMMD_TIS_openmm_run_scripts.HostGuest.transform_functions import descriptor_transform_HG_new_symmetric_scaled as descriptor_transform


# Define the main function to run AIMMD TPS
def run_AIMMD_TPS(input_path: Path = None, config_path=None, n_steps=None, traj_path=None, output_path=None, 
                  previous_model_file=None, aimmd_store_path=None, ops_storage_path=None, save_final_traj=True, 
                  final_traj_storage_path=None, system_resource_directory=None):
    
    # Construct the full config path
    if system_resource_directory is None:
        system_resource_directory= ""
    config_path = Path(input_path / config_path).with_suffix(".json")
    TPS_utils = TPS_setup(config_path, resource_directory=system_resource_directory, print_config=True)
    paths.PathMover.engine = TPS_utils.md_engine
    
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
            init_traj = old_store.trajectories[-1]
            mdtraj_traj = init_traj.to_mdtraj()
            old_store.close()
            init_traj = ops_openmm.trajectory_from_mdtraj(mdtraj_traj)
        else:
            raise ValueError("Unsupported file type. Use .nc or .db for trajectory storage.")
    else:
        raise ValueError("traj_path is required to load initial trajectory.")
    
    # test the descriptor
    test_descriptor = descriptor_transform(TPS_utils.template)
    descriptor_dim = len(test_descriptor)

    # AIMMD setup
    AIMMD_set = AIMMD_setup(config_path, descriptor_dim, TPS_utils.states, descriptor_transform=descriptor_transform)
    
    # Create storage paths
    if ops_storage_path is None:
        ops_storage_path = "aimmd_{}_tps_storage_{}".format(AIMMD_set.distribution, TPS_utils.system_name)
    ops_storage_path = Path(output_path / ops_storage_path).with_suffix(".db")
    
    if aimmd_store_path is None:
        aimmd_store_path = "aimmd_{}_run_storage_{}".format(AIMMD_set.distribution, TPS_utils.system_name)
    aimmd_store_path = Path(output_path / aimmd_store_path).with_suffix(".h5")

    if previous_model_file is not None:
        previous_model_file = Path(input_path / previous_model_file).with_suffix(".h5")

    # Create AIMMD storage
    aimmd_store = aimmd.Storage(aimmd_store_path, "w")
    model = AIMMD_set.setup_RCModel(aimmd_store, load_model_path=previous_model_file)

    # Initialize AIMMD hooks
    trainset = aimmd.TrainSet(n_states=2)
    trainhook = aimmd.ops.TrainingHook(model, trainset)
    storehook = aimmd.ops.AimmdStorageHook(aimmd_store, model, trainset)
    densityhook = aimmd.ops.DensityCollectionHook(model)

    # Setup the selector
    selector = AIMMD_set.setup_selector(model)

    # Velocity randomizer setup
    beta =  1. / (TPS_utils.integrator.getTemperature()* unit.BOLTZMANN_CONSTANT_kB)
    modifier = paths.RandomVelocities(beta=beta, engine=TPS_utils.md_engine)
    
    # Shooting strategy: TwoWayShooting
    tw_strategy = paths.strategies.TwoWayShootingStrategy(modifier=modifier, selector=selector, engine=TPS_utils.md_engine, group='TwoWayShooting')
    
    # Transition network to sample transitions
    network = paths.TPSNetwork.from_states_all_to_all(TPS_utils.states)
    
    # Move scheme for sampling
    move_scheme = paths.MoveScheme(network=network)
    move_scheme.append(tw_strategy)
    move_scheme.append(paths.strategies.OrganizeByMoveGroupStrategy())
    move_scheme.build_move_decision_tree()

    fig, ax = plt.subplots(1,1)
    ax.plot(cv_hg_distance(init_traj))
    ax.set_xlabel("frames")
    ax.set_ylabel("distance (nm)")
    TAI.Tools.save_fig_pdf_and_png(fig, "Initial_trajectory",output_path=output_path)

    # Initial conditions
    initial_conditions = move_scheme.initial_conditions_from_trajectories(init_traj)
    initial_conditions.sanity_check()

    # Create storage
    storage = Storage(ops_storage_path, "w")
    storage.save(TPS_utils.template)
    storage.save(initial_conditions)
    storage.save(move_scheme)

    # Create sampler and attach AIMMD components
    sampler = paths.PathSampling(storage, move_scheme, initial_conditions).named("TPS_sampler")
    sampler.attach_hook(trainhook)
    sampler.attach_hook(storehook)
    sampler.attach_hook(densityhook)

    # Run the sampler
    """
    psuedo code example of saving descriptors while running
        n_part = 5
    for i in range(n_steps/n_part):
        sampler.run(n_part)
        #save descriptors for last n_part steps
        for step in range(storage.steps[-n_part:]):
            array[step_index] = descriptor_transform(traj_step)
    
    """
    sampler.run(n_steps)


    
    # Store the final trajectory in a separate file
    if save_final_traj:
        if final_traj_storage_path is None:
            final_traj_storage_path = "aimmd_tps_storage_final_traj{}.db".format(TPS_utils.system_name)
        final_traj_storage_path = Path(output_path / final_traj_storage_path).with_suffix(".db")
        final_storage = Storage(final_traj_storage_path, "w")
        final_traj = storage.steps[-1].active[0].trajectory
        final_storage.save(final_traj)
        final_storage.close()
    
    # Save trainset and close storages
    aimmd_store.save_trainset(trainset)
    storage.close()
    aimmd_store.close()

# Main entry point for the script
if __name__ == "__main__":
    # Parse command-line arguments

    arguments_list = [
        global_arguments["directory"],
        global_arguments["config_file"],
        global_arguments["n_steps"],
        global_arguments["trajectory_path"],
        global_arguments["output_path"],
        global_arguments["previous_model"],
        global_arguments["aimmd_store"],
        global_arguments["ops_store"],
        global_arguments["final_storage"],
        global_arguments["system_resource_directory"]
    ]
    # Generate the parser
    parser = create_parser(arguments_list)
    args = parser.parse_args()
    
    # Retrieve command-line argument values
    in_path = args.directory
    configfile = args.config_file
    nsteps = args.n_steps
    outpath = args.output_path
    traj_path = args.trajectory_path
    old_aimmd = args.old_aimmd
    aimmd_store_path = args.aimmd_store
    ops_storage_path = args.ops_store 
    final_traj_storage_path = args.final_storage
    system_resource_directory = args.system_resource_directory
    
    # Determine whether to save the final trajectory
    save_final_traj = False
    if final_traj_storage_path is not None:
        save_final_traj = True
    if outpath is None:
        outpath = in_path
    # Run the process with the specified arguments
    process = Process(target=run_AIMMD_TPS, args=(in_path, configfile, nsteps, traj_path,  outpath, 
                                                  old_aimmd, aimmd_store_path, ops_storage_path, save_final_traj, final_traj_storage_path, system_resource_directory))
    process.start()
