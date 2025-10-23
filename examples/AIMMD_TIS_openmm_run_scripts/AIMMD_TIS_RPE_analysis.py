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
from AIMMD_TIS_openmm_run_scripts.AIMMD_TIS_RPE_class import TIS_analysis

from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage.collective_variables import MDTrajFunctionCV
from openpathsampling.experimental.storage import Storage
from simtk import unit
import openpathsampling.engines.openmm as ops_openmm
paths = monkey_patch_all(paths)

orig_settings = np.seterr(all='ignore') 

    
def run_TIS_AIMMD_analysis(input_path: Path = None, TPS_config_path=None, TIS_config_path=None, output_path=None, interface_model_file=None, 
                           ops_storage_path=None, RPE_storage_path=None, stable_storage_path=None, system_resource_directory=None, number_traj_used=None, n_thermalization=None, cutoff=None):
    
    analysis_class= TIS_analysis(input_path,TPS_config_path, TIS_config_path,output_path,interface_model_file,
                                     ops_storage_path,RPE_storage_path,stable_storage_path,system_resource_directory=system_resource_directory, 
                                     n_thermalization=n_thermalization, cutoff=cutoff)
    if number_traj_used is not None:    
        analysis_class.load_and_save_RPE(number_traj_used=number_traj_used, start_traj=n_thermalization)
    else :
        analysis_class.load_and_save_RPE()

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
        global_arguments["system_resource_directory"]
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
    if outpath is None:
        outpath = in_path

    process = Process(target=run_TIS_AIMMD_analysis, args=(in_path, configfile_TPS, configfile_TIS, outpath, interface_model, 
                                                           ops_storage_path, RPE_storage_path, stable_storage_path, system_resource_directory,
                                                           n_traj, n_thermalization, cutoff))
    process.start()