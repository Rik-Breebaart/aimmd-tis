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
from examples.AIMMD_TIS_openmm_run_scripts.HostGuest.setup_utilities import TPS_setup, AIMMD_setup, TIS_setup, create_parser, global_arguments
from examples.AIMMD_TIS_openmm_run_scripts.HostGuest.transform_functions import descriptor_transform_HG_simple_symmetriced as descriptor_transform
from examples.AIMMD_TIS_openmm_run_scripts.HostGuest.transform_functions import descriptor_transform_HG_distance_one_hot_hydrogen as descriptor_transform_analysis

from AIMMD_TIS_openmm_run_scripts.AIMMD_TIS_RPE_class import TIS_analysis

from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage.collective_variables import MDTrajFunctionCV
from openpathsampling.experimental.storage import Storage
from simtk import unit
import openpathsampling.engines.openmm as ops_openmm
paths = monkey_patch_all(paths)

orig_settings = np.seterr(all='ignore') 

def run_TIS_AIMMD_analysis(input_path: Path = None, TPS_config_path=None, TIS_config_path=None, output_path=None, interface_model_file=None, 
                           ops_storage_path=None, RPE_storage_path=None, stable_storage_path=None, system_resource_directory=None, 
                           number_traj_used=None, n_thermalization=None, cutoff=None, interface_value_given=None, direction=None,
                           max_min_filename=None
                           ):
    
    # Base storage folders
    storage_base_folder = Path("/home/rbreeba/Projects/-RE-TIS-AIMMD/TIS_AIMMD_biosystems/Host-Guest-System/CarbonJobs/Run_bound_005_unbound_100_production")

    # Interface values
    Interfaces_TIS_itteration_broad = [-40.125, -38.52, -36.91, -35.3, -33.69, -32.08, -30.47, -28.86, -27.25, -25.64, -24.03, -22.42, -20.81, -19.2, -17.59, -15.98, -14.37, -12.76, -11.15,  -9.54, -7.93, -6.32] 
    Interfaces_TIS_itteration_narrow = [ -4.7, -3.05, -1.23, 0.0]
    folder_broad = Path(storage_base_folder/ "TIS_itteration_2_run_3_broader_selectors" )
    folder_narrow = Path(storage_base_folder/ "TIS_itteration_2_run_2_interface_definition_fixed")

    # Load config files
    TPS_config_path = Path(input_path / TPS_config_path).with_suffix(".json")
    TIS_config_path = Path(input_path / TIS_config_path).with_suffix(".json")
    TIS_utils = TIS_setup(TPS_config_path, TIS_config_path, print_config=False, resource_directory=system_resource_directory)
    
    Interfaces_TIS_itteration_forward = TIS_utils.interface_values_forward
    # Generate storage path list for forward iteration
    storage_path_forward = [
        Path(folder_broad if interface in Interfaces_TIS_itteration_broad else folder_narrow )/ 
        f"TIS_itteration_2_forward_{TAI.Tools.interface_indicator(interface)}.db"
        for interface in Interfaces_TIS_itteration_forward
    ]

    Interfaces_TIS_itteration_backward = TIS_utils.interface_values_backward
    storage_path_backward = [Path(folder_narrow /
        f"TIS_itteration_2_backward_{TAI.Tools.interface_indicator(interface)}.db")
        for interface in Interfaces_TIS_itteration_backward
    ]
  
    # Determine interface list based on direction
    if direction == "forward":
        interfaces = TIS_utils.interface_values_forward
    elif direction == "backward":
        interfaces = TIS_utils.interface_values_backward
    else:
        print("Error: direction not recognized. Use 'forward' or 'backward'.")
        return

    # Find the index corresponding to the provided interface value
    try:
        interface_index = interfaces.index(interface_value_given[0])
    except ValueError:
        print(f"Error: Provided interface value {interface_value_given} not found in {direction} interfaces.")
        return
    print(max_min_filename)

    analysis_class= TIS_analysis(input_path,TPS_config_path, TIS_config_path,output_path,interface_model_file,
                                     ops_storage_path,RPE_storage_path,stable_storage_path,system_resource_directory=system_resource_directory, 
                                     n_thermalization=n_thermalization, cutoff=cutoff, RPE_already_stored=False, 
                                     storage_path_list_forward=storage_path_forward if direction=="forward" else None, 
                                     storage_path_list_backward=storage_path_backward if direction=="backward" else None,
                                     load_forward=True if direction=="forward" else False, load_backward=True if direction=="backward" else False,
                                     max_min_filename=max_min_filename)

    if direction =="forward":
        crprw = analysis_class.CrPrW_forward
    else:
        crprw = analysis_class.CrPrW_backward

    test_descriptor_analysis = descriptor_transform_analysis(TIS_utils.template)
    descriptor_dim_analysis = len(test_descriptor_analysis)

    descriptors_interface, weights_interface, shot_results_interface= analysis_class.RPE.Create_RPE_trainset_for_interface(
        crprw, descriptor_transform_analysis, descriptor_dim=descriptor_dim_analysis, states=analysis_class.TIS_utils.states, 
        interface_index=interface_index, number_traj_used=number_traj_used, start_traj=n_thermalization, n_jump=1)
    filepath = Path(output_path / "RPE_{}_interface_{}.pkl".format(direction, TAI.Tools.interface_indicator(interface_value_given[0])))
    analysis_class.RPE.save_RPE_for_interface(filepath, interface_index, descriptors_interface, weights_interface, shot_results_interface, mode=direction)
    print(f"Saved results to {filepath}")

if __name__ == "__main__":
    # Parse command-line arguments

    arguments_list = [
        global_arguments["directory"],
        global_arguments["config_file_TPS"],
        global_arguments["config_file_TIS"],
        global_arguments["output_path"],
        global_arguments["interface_model"],
        global_arguments["ops_store"],
        global_arguments["RPE_name"],
        global_arguments["stable_ops_storages"],
        global_arguments["number_trajectories"],
        global_arguments["n_thermalization"],
        global_arguments["cutoff"],
        global_arguments["system_resource_directory"],
        global_arguments["interface_values"],
        global_arguments["direction"],
        global_arguments["max_min_filename"]
    ]

    # Generate the parser and parse arguments
    parser = create_parser(arguments_list)
    args = parser.parse_args()
    
    # Retrieve command-line argument values
    in_path = args.directory
    configfile_TPS = args.config_file_TPS
    configfile_TIS = args.config_file_TIS
    outpath = args.output_path
    interface_model = args.interface_model
    ops_storage_path = args.ops_store 
    RPE_storage_path = args.RPE_name
    stable_storage_path = args.stable_ops_store
    n_thermalization = args.n_therm
    cutoff = args.cutoff
    n_traj = args.n_traj
    system_resource_directory = args.system_resource_directory
    interface_value = args.interface_values
    direction = args.direction
    max_min_filename = args.max_min
    if outpath is None:
        outpath = in_path

    process = Process(target=run_TIS_AIMMD_analysis, args=(in_path, configfile_TPS, configfile_TIS, outpath, interface_model, 
                                                           ops_storage_path, RPE_storage_path, stable_storage_path, system_resource_directory,
                                                           n_traj, n_thermalization, cutoff, interface_value, direction, max_min_filename))
    process.start()