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
from examples.AIMMD_TIS_openmm_run_scripts.HostGuest.transform_functions import descriptor_transform_HG_new_symmetric_continuous_waters_scaled as descriptor_transform


def Stable_training(input_path: Path = None, config_path=None, n_epochs=None, output_path=None, 
                    previous_model_file=None, aimmd_store_path=None, ops_storage_path=None, system_resource_directory=None):

    
    if system_resource_directory is None:
        system_resource_directory= ""
    config_path = Path(input_path / config_path).with_suffix(".json")
    TPS_utils = TPS_setup(config_path, resource_directory=system_resource_directory, print_config=True)
    paths.PathMover.engine = TPS_utils.md_engine

    # test the descriptor
    test_descriptor = descriptor_transform(TPS_utils.template)
    descriptor_dim = len(test_descriptor)
    states = TPS_utils.states

    AIMMD_set = AIMMD_setup(config_path, descriptor_dim, TPS_utils.states, descriptor_transform=descriptor_transform)
    
    if aimmd_store_path is None:
        aimmd_store_path = "aimmd_{}_stable_trained_{}".format(AIMMD_set.distribution, TPS_utils.system_name)
    aimmd_store_path = Path(output_path / aimmd_store_path).with_suffix(".h5")

    if previous_model_file is not None:
        previous_model_file = Path(input_path / previous_model_file).with_suffix(".h5")

    aimmd_store = aimmd.Storage(aimmd_store_path, "w")
    model = AIMMD_set.setup_RCModel(aimmd_store, load_model_path=previous_model_file)

    #TODO Import class function
    stable_states = [states[0].name, states[1].name]
    storage_stable = []

    for i, stable_state in enumerate(stable_states):
        if ops_storage_path is None:
            stable_path = "stable_{}_{}_store.db".format(stable_state, TPS_utils.system_name)
        else:
            stable_path = ops_storage_path[i]
        stable_path = Path(str(input_path.joinpath(stable_path))).with_suffix(".db")
        print("Loaded stable run in stable state {}".format(stable_state))
        storage_stable.append(Storage(stable_path,"r"))
    
    #TODO: store stable data as pickle
    Stable_data = TAI.Create_Stable_trainset(storage_stable, descriptor_transform, descriptor_dim=descriptor_dim, states=states)
    in_state = Stable_data[1][:]

    print("Start stable training")

    stable_data_length = np.size(Stable_data[1])
    Total_snapshots = stable_data_length
    weights_total = np.zeros(Total_snapshots)
    descriptors_total = np.zeros((Total_snapshots,descriptor_dim))
    shot_results_total = np.zeros((Total_snapshots,2))

    Weight_factor = [1/np.sum(in_state[0]), 1/np.sum(in_state[1])]
    start = 0
    for i, state in enumerate(stable_states):
        snapshots_interface = np.shape(Stable_data[0][i])[0]
        weights_total[start:start + snapshots_interface] = Stable_data[1][i] * Weight_factor[i]*snapshots_interface
        descriptors_total[start:start + snapshots_interface, :] = Stable_data[0][i]
        shot_results_total[start:start + snapshots_interface, :] = Stable_data[2][i]
        start += snapshots_interface
    weights_total = weights_total[:start]
    descriptors_total = descriptors_total[:start]
    shot_results_total = shot_results_total[:start]
    n_states = len(stable_states)
    trainset = aimmd.TrainSet(n_states=n_states,
                        descriptors=descriptors_total,
                        shot_results=shot_results_total,
                        weights=weights_total)

    aimmd_store.rcmodels["initial"] = model

    los_train,los_test,lr_used= TAI.Tools.combined_train_function_l1_regularized(aimmd_store, model, trainset, trainset, max_epochs=n_epochs, max_epochs_sans_improvement=20, stopping_criteria=0.001)
    
    fig, ax = plt.subplots()
    ax.plot(lr_used)
    ax.set_xlabel("epoch")
    ax.set_ylabel("lr")
    fig_path = Path(output_path / "learning_rate_stable_training").with_suffix(".png")
    fig.savefig(fig_path)

    fig, ax = plt.subplots()
    ax.plot(los_train,label="train loss")
    ax.plot(los_test,label="test loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    fig_path = Path(output_path / "loss_stable_training").with_suffix(".png")
    fig.savefig(fig_path)
    out = TAI.Tools.check_interfaces(model, stable_states, descriptors_total, weights=weights_total, shot_results=shot_results_total)

    aimmd_store.save_trainset(trainset)
    aimmd_store.rcmodels["most_recent"] = model
    aimmd_store.close()
    for state in range(len(storage_stable)):
        storage_stable[state].close()

if __name__ == "__main__":

    arguments_list = [
        global_arguments["directory"],
        global_arguments["config_file"],
        global_arguments["n_epoch"],
        global_arguments["output_path"],
        global_arguments["stable_ops_storages"],
        global_arguments["previous_model"],
        global_arguments["aimmd_store"],
        global_arguments["system_resource_directory"]
    ]
    parser= create_parser(arguments_list)
    args = parser.parse_args()
    args = parser.parse_args()
    in_path = args.directory
    configfile = args.config_file
    n_epoch=args.n_epoch
    outpath = args.output_path
    if outpath is None:
        outpath = in_path
    prev_model = args.old_aimmd
    aimmd_store = args.aimmd_store
    ops_storage = args.stable_ops_store
    system_resource_directory = args.system_resource_directory or ""

    process = Process(target=Stable_training, args=(in_path, configfile, n_epoch, outpath, prev_model, aimmd_store, ops_storage, system_resource_directory))
    process.start()