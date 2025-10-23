import numpy as np
import matplotlib.pyplot as plt
import openpathsampling as paths
import torch
import sys
import os
from pathlib import Path
import openpathsampling.engines.toy as toys
import argparse
from functools import reduce
from multiprocessing import Process

# Add directories to sys.path for module imports
# current_directory = os.path.dirname(os.path.abspath(os.getcwd()))
current_directory = "/home/rbreeba/Projects/-RE-TIS-AIMMD/TIS_AIMMD_biosystems/Host-Guest-System"
print(current_directory)
sys.path.append(current_directory)
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
parent_parent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir))
print(parent_directory)
print(parent_parent_directory)
sys.path.append(parent_directory)
sys.path.append(parent_parent_directory)

# Custom imports
from aimmd import aimmd
import TIS_AIMMD_toy_framework as TAI
from run_scripts.setup_utilities import TPS_setup, AIMMD_setup



def Stable_interface_analysis(input_path: Path = None, config_path=None, output_path=None, 
                    previous_model_file=None, ops_storage_path=None, stable_model=False, model_key=None, overlap=0.2):
     
    config_path = Path(input_path / config_path).with_suffix(".json")
    TPS_utils = TPS_setup(config_path, print_config=True)
    paths.PathMover.engine = TPS_utils.md_engine

    # collective variable to transform OPS snapshots to model descriptor space,
    # i.e. the space in which the model learns
    def transform_func(snapshot, receptor_atoms, ligand_atoms):
        import numpy as np
        dims = np.shape(snapshot.xyz)
        receptor_com = snapshot.xyz[:,receptor_atoms,:].mean(1)
        ligand_com = snapshot.xyz[:,ligand_atoms,:].mean(1)
        coordinates = snapshot.xyz[:,0:156,:]
        distances = np.sqrt(((receptor_com - ligand_com)**2).sum(1))
        return np.concatenate((coordinates.reshape(dims[0],156*3),distances.reshape(dims[0],1)) , axis=1)

    # wrap the function
    descriptor_transform = paths.MDTrajFunctionCV('descriptor_transform',  # name in OPS
                                                transform_func,  # the function we just defined
                                                TPS_utils.topology,
                                                receptor_atoms=np.arange(0,126),
                                                ligand_atoms=np.arange(126,156),
                                                cv_scalarize_numpy_singletons=False,  # to make sure it always returns a 2d array, even if called on trajectories with a single frame
                                                ).with_diskcache()  # enable caching of values
    
    test_descriptor = descriptor_transform(TPS_utils.template)
    descriptor_dim = len(test_descriptor)
    
    AIMMD_set = AIMMD_setup(config_path, descriptor_dim, TPS_utils.states, descriptor_transform=descriptor_transform)
    
    if previous_model_file is not None:
        previous_model_file = Path(input_path / previous_model_file).with_suffix(".h5")

    stable_states = ["A", "B"]

    aimmd_store = aimmd.Storage(previous_model_file, "r")
    if model_key is not None:
        model = aimmd_store.rcmodels[model_key]
    else:
        model = aimmd_store.rcmodels["most_recent"]


    print(model.nnet)
    # move model to GPU if CUDA is available
    use_cuda = AIMMD_set.use_GPU
    if torch.cuda.is_available() and use_cuda:
        model = TAI.Tools.model_to(model, "cuda")
        print("using cuda")
    elif torch.backends.mps.is_available() and use_cuda:
        model = TAI.Tools.model_to(model, "mps")
    else: 
        model = TAI.Tools.model_to(model, "cpu")

    if stable_model:
        trainset = aimmd_store.load_trainset() 
        descriptors = trainset.descriptors
        weights = trainset.weights
        shot_results = trainset.shot_results
    else: 
        storage_stable = []

        for i, stable_state in enumerate(stable_states):
            if ops_storage_path is None:
                stable_path = "stable_{}_{}_store.db".format(stable_state, TPS_utils.system_name)
            else:
                stable_path = ops_storage_path[i]
            stable_path = Path(str(input_path.joinpath(stable_path))).with_suffix(".nc")
            print("Loaded stable run in stable state {}".format(stable_state))
            storage_stable.append(paths.AnalysisStorage(stable_path))
        
        #TODO: store stable data as pickle
        
        descriptors, weights, shot_results = TAI.Create_Stable_trainset(storage_stable, descriptor_transform, descriptor_dim=descriptor_dim, states=TPS_utils.states)
        descriptors = np.concatenate((descriptors[0],descriptors[1]),axis=0)
        weights= np.concatenate((weights[0],weights[1]),axis=0)
        shot_results = np.concatenate((shot_results[0],shot_results[1]),axis=0)

    out = TAI.Tools.check_interfaces(model, stable_states, descriptors, weights=weights, shot_results=shot_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Directory containing initial trajectory and configuration files.')
    parser.add_argument('-cfg', '--config_file', type=str, required=True,
                        help='File in python configparser format with simulation settings.')
    parser.add_argument('-out', '--output_path', type=Path, required=False,
                        help='Directory for storing TPS output files.')
    parser.add_argument('-prev_model', '--previous_model_file', type=str, required=False,
                        help='Path to the previous model file to load.')
    parser.add_argument('-ops_store', '--ops_storage_path', nargs='+', type=str, required=False,
                        help='Paths to OPS storage files for stable states A and B.')
    parser.add_argument('-stable_model', action='store_true', 
                          help="Boolean indicator if a stable trained model is given")    
    parser.add_argument('-model_key','--model_key', required=False,
                        help="model key of model to be analysed.")                  
    parser.add_argument('-overlap','--overlap',type=float, required=False,
                        help="Desired overlap between q-interfaces to construct interface values, Default 0.2" )
    args = parser.parse_args()
    in_path = args.directory
    configfile = args.config_file
    outpath = args.output_path
    if outpath is None:
        outpath = in_path
    prev_model = args.previous_model_file
    ops_storage_path = args.ops_storage_path
    stable_model = args.stable_model
    model_key = args.model_key
    overlap = args.overlap

    process = Process(target=Stable_interface_analysis, args=(in_path, configfile, outpath, prev_model, ops_storage_path, stable_model, model_key, overlap))
    process.start()