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

# Custom imports
import aimmd
import aimmdTIS

from .HostGuest.setup_utilities import TPS_setup, AIMMD_setup, create_parser, global_arguments
from .HostGuest.transform_functions import descriptor_transform_HG_continuous_waters_7_descriptors_symmetriced_or_scaled as descriptor_transform


def run_initialize_model(in_path: Path=None, config_path=None, aimmd_store_path=None, output_path=None, system_resource_path=None):
    config_path = Path(in_path / config_path).with_suffix(".json")
    TPS_utils= TPS_setup(config_path, system_resource_path, print_config=True)

    # test the descriptor
    test_descriptor = descriptor_transform(TPS_utils.template)
    descriptor_dim = len(test_descriptor)


    AIMMD_set = AIMMD_setup(config_path, descriptor_dim, TPS_utils.states, descriptor_transform=descriptor_transform)
    # create aimmd storage
    if aimmd_store_path is None:
        aimmd_store_path = "aimmd_{}_initialized_storage_{}".format(AIMMD_set.distribution, TPS_utils.system_name)
    aimmd_store_path = Path(output_path / aimmd_store_path).with_suffix(".h5")
    aimmd_store = aimmd.Storage(aimmd_store_path, "w")

    model = AIMMD_set.setup_RCModel(aimmd_store)
    aimmd_store.close()


if __name__ == "__main__":

    arguments_list = [
        global_arguments["directory"],
        global_arguments["config_file"],
        global_arguments["aimmd_store"],
        global_arguments["output_path"],
        global_arguments["system_resource_directory"]
    ]
    parser = create_parser(arguments_list)
    
    args = parser.parse_args()

    in_path = args.directory  # -dir <input/directory/path>
    configfile = args.config_file
    aimmd_store_path = args.aimmd_store
    outpath = args.output_path  # -out </path/to/output/directory>
    system_resource_directory = args.system_resource_directory

    if outpath is None:
        outpath = in_path

    process = Process(target=run_initialize_model, args=(in_path, configfile, aimmd_store_path, outpath, system_resource_directory))
    process.start()
