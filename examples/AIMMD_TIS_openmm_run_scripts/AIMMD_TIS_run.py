import numpy as np 
import matplotlib.pyplot as plt
import openpathsampling as paths
import torch
import sys
import os
import argparse
from pathlib import Path
from multiprocessing import Process
from time import sleep

from functools import reduce
from copy import deepcopy
# current_directory = os.path.dirname(os.path.abspath(os.getcwd()))
current_directory = "/home/rbreeba/Projects/-RE-TIS-AIMMD/TIS_AIMMD_biosystems/Host-Guest-System"
# Get the current directory and add it to the system path
sys.path.append(current_directory)

# Get the parent and grandparent directories and add them to the system path
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
parent_parent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir))
sys.path.append(parent_directory)
sys.path.append(parent_parent_directory)

# Custom imports from the aimmd module and TIS_AIMMD_toy_framework
from aimmd import aimmd
import TIS_AIMMD_toy_framework as TAI
from TIS_AIMMD_toy_framework import TIS_AIMMD_setup, read_config, save_fig_pdf_and_png
from AIMMD_TIS_openmm_run_scripts.setup_utilities import TPS_setup, AIMMD_setup, TIS_setup, create_parser, global_arguments
from AIMMD_TIS_openmm_run_scripts.transform_functions import descriptor_transform_HG_simple_symmetriced as descriptor_transform

from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage.collective_variables import MDTrajFunctionCV
from openpathsampling.experimental.storage import Storage
from simtk import unit
import openpathsampling.engines.openmm as ops_openmm
paths = monkey_patch_all(paths)

orig_settings = np.seterr(all='ignore') 


def run_TIS_AIMMD_interfaces(input_path: Path = None, TPS_config_path=None, TIS_config_path=None, n_steps=None, traj_path=None, output_path=None, 
                  interface_model_file=None, interface_value=None, direction=None, ops_storage_path=None, save_final_traj=True, 
                  final_traj_storage_path=None, system_resource_directory=None, i=None):
    # Construct the full config path
    print("Process {}: started".format(i))
    TPS_config_path = Path(input_path/ TPS_config_path).with_suffix(".json")
    TIS_config_path = Path(input_path / TIS_config_path).with_suffix(".json")
    # Setup TPS utilities
    print_config = (i == 0 or i is None)
    TIS_utils = TIS_setup(TPS_config_path, TIS_config_path, system_resource_directory, print_config=print_config)
    print("Process {}: loaded config".format(i))

    paths.PathMover.engine = TIS_utils.md_engine
    
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
    test_descriptor = descriptor_transform(TIS_utils.template)
    descriptor_dim = len(test_descriptor)

    # AIMMD setup
    AIMMD_set = AIMMD_setup(TPS_config_path, descriptor_dim, TIS_utils.states, descriptor_transform=descriptor_transform)
    

    interface_indicator = int(np.floor(interface_value*100))
    if interface_model_file is None:
        interface_model_file = "aimmd_{}_run_storage_{}".format(AIMMD_set.distribution, TIS_utils.pes.__repr__())
    load_model_path = Path(input_path/ interface_model_file).with_suffix(".h5")
    #temperorary simulation model for parallel TIS runs


    # Ensure the file has a .h5 extension using pathlib's with_suffix
    aimmd_store_temp_path = (output_path/ f"AIMMD_interface_model_{direction}_temp_{direction,interface_indicator}").with_suffix(".h5")
    aimmd_store = aimmd.Storage(aimmd_store_temp_path, "w")
    model = AIMMD_set.setup_RCModel(aimmd_store,load_model_path=load_model_path)

    # move model to GPU if CUDA is available
    use_cuda = AIMMD_set.use_GPU
    if torch.cuda.is_available() and use_cuda:
        model = TAI.Tools.model_to(model, "cuda")
        print("using cuda")
    elif torch.backends.mps.is_available() and use_cuda:
        model = TAI.Tools.model_to(model, "mps")
        print("using mps")
    else: 
        model = TAI.Tools.model_to(model, "cpu")


    TIS_Framework = TAI.AIMMD_TIS(TIS_utils.md_engine, model, TIS_utils.states[0], TIS_utils.states[1], descriptor_transform=descriptor_transform)

    print("Running TIS for q-interface {}".format(interface_value))
    print("Running for {} MC steps".format(n_steps))
        # Create storage paths
    if ops_storage_path is None:
        ops_storage_path = "aimmd_tis_{}".format(TIS_utils.system_name)
    ops_storage_path = f"{ops_storage_path}_{direction}_{interface_indicator}.db"
    ops_storage_path = Path(output_path / ops_storage_path).with_suffix(".db")

    storage = Storage(ops_storage_path, "w")
    storage.save(TIS_utils.template)
    storage.save(TIS_utils.md_engine)
    TIS_Framework.run_single_TIS(n_steps ,storage, init_traj, interface_value, 
                                scheme_move=TIS_utils.shooting_move,
                                scheme_selector=TIS_utils.TIS_selector,
                                scheme_modifier=TIS_utils.modification_method,
                                gaussian_parameter_width=TIS_utils.gaussian_width,
                                gausssian_parameter_shift=TIS_utils.gaussian_origin_shift,
                                direction=direction, directory=output_path)
    # Store the final trajectory in a separate file
    if save_final_traj:
        if final_traj_storage_path is None:
            final_traj_storage_path = "TIS_storage_final_traj_interface"
        final_traj_storage_path = final_traj_storage_path+"_{}_{}_{}.db".format(direction,interface_indicator,TIS_utils.system_name)
        final_traj_storage_path = Path(output_path / final_traj_storage_path).with_suffix(".db")
        final_storage = Storage(final_traj_storage_path, "w")
        final_traj = storage.steps[-1].active[0].trajectory
        final_storage.save(final_traj)
        final_storage.close()
    storage.close()
    aimmd_store.close()
    os.remove(aimmd_store_temp_path)
    # if initial_path_ops_storage_path is not None:
    #     temp_init_storage.close()
    #     os.remove(initial_path_temp_dir)

if __name__ == "__main__":
    # Parse command-line arguments

    arguments_list = [
        global_arguments["directory"],
        global_arguments["config_file_TPS"],
        global_arguments["config_file_TIS"],
        global_arguments["n_steps"],
        global_arguments["trajectory_path"],
        global_arguments["output_path"],
        global_arguments["interface_model"],
        global_arguments["ops_store"],
        global_arguments["final_storage"],
        global_arguments["interface_values"],
        global_arguments["direction"],
        global_arguments["system_resource_directory"]
    ]

    # Generate the parser and parse arguments
    parser = create_parser(arguments_list)
    args = parser.parse_args()

    # Assign parsed arguments directly to variables 
    input_path = args.directory
    TPS_config_path = args.config_file_TPS
    TIS_config_path = args.config_file_TIS
    n_steps = args.n_steps
    traj_path = args.trajectory_path
    output_path = args.output_path or input_path  # Default to input path if output path is None
    interface_model_file = args.interface_model
    ops_storage_path = args.ops_store
    final_traj_storage_path = args.final_storage
    interface_values = args.interface_values
    direction = args.direction
    system_resource_directory = args.system_resource_directory
    save_final_traj = final_traj_storage_path is not None  # Determine if final trajectory should be saved

    # Initialize and start processes for each interface value
    processes = []
    for i, interface_value in enumerate(interface_values):
        process = Process(
            target=run_TIS_AIMMD_interfaces,
            args=(
                input_path, TPS_config_path, TIS_config_path, n_steps, traj_path, output_path,
                interface_model_file, interface_value, direction, ops_storage_path,
                save_final_traj, final_traj_storage_path, system_resource_directory, i
            )
        )
        processes.append(process)
        process.start()
        sleep(10)  # Delay between process starts
        
    # Wait for all processes to complete
    for process in processes:
        process.join()
