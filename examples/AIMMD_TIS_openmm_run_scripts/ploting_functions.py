import numpy as np 
import matplotlib.pyplot as plt
import openpathsampling as paths
from pathlib import Path
import sys
import os
import math
from simtk import unit
# current_directory = os.path.dirname(os.path.abspath(os.getcwd()))
current_directory = "/home/rbreeba/Projects/-RE-TIS-AIMMD/TIS_AIMMD_biosystems/Host-Guest-System-NPT"
# Get the current directory and add it to the system path
sys.path.append(current_directory)

# Get the parent and grandparent directories and add them to the system path
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
parent_parent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir))
sys.path.append(parent_directory)
sys.path.append(parent_parent_directory)
from examples.AIMMD_TIS_openmm_run_scripts.HostGuest.transform_functions import cv_volume_square_box


def descriptor_transform_analysis_plot(trajectory, descriptor_transform, dim_labels, max_columns=4, system_info=False):
    descriptors_trajectory = descriptor_transform(trajectory)
    n_descriptors = descriptors_trajectory.shape[1]
    
    if len(dim_labels) < n_descriptors:
        raise ValueError("dim_labels must have at least as many entries as there are descriptors.")

    total_plots = n_descriptors + system_info*2  # +2 for temperature and volume

    # Determine layout: use up to max_columns per row
    n_cols = min(max_columns, total_plots)
    n_rows = math.ceil(total_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    # Plot descriptors
    for i in range(n_descriptors):
        axes[i].plot(descriptors_trajectory[:, i])
        axes[i].set_title(f"Descriptor {i+1}")
        axes[i].set_ylabel(dim_labels[i])
        axes[i].set_xlabel("Frame")

    if system_info:
        # Plot temperature
        temperature = trajectory.instantaneous_temperature
        kelvin_values = [q.value_in_unit(unit.kelvin) for q in temperature]
        axes[n_descriptors].plot(kelvin_values)
        axes[n_descriptors].set_title("Instantaneous Temperature")
        axes[n_descriptors].set_ylabel("Temperature [K]")
        axes[n_descriptors].set_xlabel("Frame")

        # Plot volume
        volume = cv_volume_square_box(trajectory)
        volume_values = [q.value_in_unit(unit.meters**3) * 1e27 for q in volume]
        axes[n_descriptors + 1].plot(volume_values)
        axes[n_descriptors + 1].set_title("Simulation Box Volume")
        axes[n_descriptors + 1].set_ylabel(r"Volume [$nm^3$]")
        axes[n_descriptors + 1].set_xlabel("Frame")

    # Hide any extra unused axes
    for ax in axes[total_plots:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    if system_info:
        return descriptors_trajectory, kelvin_values, volume_values
    else:
        return descriptors_trajectory

