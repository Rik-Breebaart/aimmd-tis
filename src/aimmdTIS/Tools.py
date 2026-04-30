""" 
This script will for now include all tools which are usefull but do not have a specific place.
Such as:

Looking through the directory to find specific files 
Determining from a stbale state run what the q-values inside the state are such that the next interfaces can be choosen.
"""
import numpy as np 
import json
import matplotlib.pyplot as plt
import openpathsampling.engines.toy as toys
from pathlib import Path
from .diagnostics.interface_placement import check_interfaces as diagnostics_check_interfaces
from .training import (
    train_one_stage,
    _apply_stage_hparams,
    _make_optimizer,
    TorchRCModelLite,
    train_test_split,
    create_train_test_split,
    SyntheticDataGenerator,
    q_normalized_trainset,
)

train_function = train_one_stage
combined_train_function = train_one_stage
combined_train_function_l1_regularized = train_one_stage

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def print_config(config, prefix=''):
    if isinstance(config, dict):
        for key, value in config.items():
            new_prefix = f"{prefix}{key}: "
            if isinstance(value, (dict, list)):
                print(new_prefix)
                print_config(value, prefix + '    ')
            else:
                print(f"{new_prefix}{value}")
    elif isinstance(config, list):
        for index, item in enumerate(config):
            new_prefix = f"{prefix}[{index}]: "
            if isinstance(item, (dict, list)):
                print(new_prefix)
                print_config(item, prefix + '    ')
            else:
                print(f"{new_prefix}{item}")

def interface_indicator(interface_value):
    return int(np.floor(interface_value*100))

def figure_storage_path(filename, suffix=".pdf",output_path=None):
    return Path(output_path / filename).with_suffix(suffix)


def save_fig_pdf_and_png(fig, filename, output_path=None, *args, **kwargs):
    fig.savefig(figure_storage_path(filename, suffix=".pdf", output_path=output_path),*args, **kwargs)
    fig.savefig(figure_storage_path(filename, suffix=".png", output_path=output_path),*args, **kwargs)   

def create_discrete_cmap(n_color_steps=20, cmap=plt.cm.Spectral):
    # extract all colors from the .jet map
    cmap_jumps = np.linspace(0,cmap.N,n_color_steps, dtype=int)
    cmaplist = [cmap(i) for i in cmap_jumps]

    # create the new map
    import matplotlib as mpl
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, n_color_steps)
    return cmap

def count_sign_changes(arr):
    # Calculate the differences between consecutive elements
    sgn_array = np.sign(arr)
    
    # Count the number of non-zero differences
    differences =  [sgn_array[i+1]-sgn_array[i] for i in range(len(arr)-1)]
    count = np.count_nonzero(differences)
    
    return count


def compute_flux(cv_values, in_state, lambda_i, timestep_fs):
    """
    Compute the flux through an interface lambda_i using first forward crossings
    from stable state defined by lambda_0.

    Parameters:
    -----------
    cv_values : array-like
        Time series of CV values (1D array).
    lambda_0 : float
        Value defining the stable state.
    lambda_i : float
        Interface value for which to compute flux.
    timestep_fs : float
        Time per step in femtoseconds (e.g., 0.002 fs).

    Returns:
    --------
    float
        Flux through the interface in 1/fs (crossings per femtosecond).
    int
        Number of first crossings.
    """
    first_crossings = find_first_crossings(cv_values, in_state, lambda_i)
    total_time_fs = len(cv_values) * timestep_fs
    flux = len(first_crossings) / total_time_fs
    return flux, len(first_crossings)


def find_first_crossings(cv_values, in_state, lambda_i):
    """
    Detect first forward crossings through interface lambda_i
    after returning to the stable state defined by lambda_0.

    Parameters:
    -----------
    cv_values : array-like
        Time series of CV values.
    lambda_0 : float
        Value defining the stable state (e.g., basin A).
        Crossing below this value resets the search.
    lambda_i : float
        Interface value for detecting forward crossings.

    Returns:
    --------
    list of int
        Indices (time steps) where first forward crossings occurred.
    """
    in_A = in_state
    above_interface = cv_values > lambda_i
    crossings = []
    crossed = False

    for t in range(1, len(cv_values)):
        if in_A[t]:
            crossed = False  # reset on return to stable state

        if not in_A[t] and not crossed and not above_interface[t - 1] and above_interface[t]:
            crossings.append(t)
            crossed = True  # suppress additional crossings until return to A

    return crossings

def count_forward_crossings(arr):
    """
    Count the number of forward crossings through an interface.

    A forward crossing is defined as the collective variable (CV)
    moving from below the interface to equal or above it.

    Parameters:
    -----------
    cv_values : array-like
        Time series of the CV values (1D array).
    interface_value : float
        Value of the interface.

    Returns:
    --------
    int
        Number of forward crossings.
    """
    return sum(1 for i in range(1, len(arr))
               if arr[i - 1] < 0 and arr[i] >= 0)


def ceil_decimal(x, decimals=1):
    multiply = 10^decimals
    return np.ceil(multiply*x)/multiply

def floor_decimal(x, decimals=1):
    multiply = 10^decimals
    return np.floor(multiply*x)/multiply

def interfaces_q_space(q_0,overlap, direction="forward"):
    interfaces = [q_0]
    if direction == "forward":
        s = -1
    elif direction == "backward":
        s = 1
    else:
        raise ValueError("Invalid direction")
    while np.isnan(interfaces[-1])==0:
        interfaces.append(round(s*np.log(overlap*(1+np.exp(s*interfaces[-1]))-1),2))
    return np.array(interfaces[:-1])


def check_interfaces(model, stable_states, descriptors, weights=None, shot_results=None,in_state=None, overlap=0.2):
    return diagnostics_check_interfaces(
        model=model,
        stable_states=stable_states,
        descriptors=descriptors,
        weights=weights,
        shot_results=shot_results,
        in_state=in_state,
        overlap=overlap,
    )

def model_to(model,mode="cuda"):
    model.nnet = model.nnet.to(mode)
    model._device = mode
    model.device= mode 
    return model

class CallableVolume(object):
    def __init__(self, vol):
        self.vol = vol

    def __call__(self, x, y):
        snapshot = toys.Snapshot(coordinates=np.array([[x,y,0.0]]))
        return 1.0 if self.vol(snapshot) else 0.0

