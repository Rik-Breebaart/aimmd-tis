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

class TIS_analysis():
    def __init__(self, input_path=None, TPS_config_path=None, TIS_config_path=None, output_path=None, interface_model_file=None,
                 ops_storage_path=None, RPE_storage_path=None, stable_storage_path=None, system_resource_directory=None,
                 RPE_already_stored=False, n_thermalization=200, cutoff=0.01, storage_path_list_forward=None, 
                 storage_path_list_backward=None, max_min_filename=None, load_forward=True, load_backward=True):
        
        self.input_path = input_path
        self.output_path = output_path
        self.RPE_storage_path = RPE_storage_path
        self.stable_storage_path = stable_storage_path
        self.set_plot_settings()

        # Load config files
        TPS_config_path = Path(input_path / TPS_config_path).with_suffix(".json")
        TIS_config_path = Path(input_path / TIS_config_path).with_suffix(".json")
        self.TIS_utils = TIS_setup(TPS_config_path, TIS_config_path, print_config=True, resource_directory=system_resource_directory)
        
        self.interface_values_forward = self.TIS_utils.interface_values_forward
        self.interface_values_backward = self.TIS_utils.interface_values_backward

        # Load storage files if provided otherwise use directory and ops_storage_path
        storage_list_forward = self.load_storage(storage_path_list=storage_path_list_forward) if storage_path_list_forward is not None else False
        storage_list_backward = self.load_storage(storage_path_list=storage_path_list_backward) if storage_path_list_backward is not None else False

        # Descriptor Transform
        self.descriptor_transform = descriptor_transform
        test_descriptor = descriptor_transform(self.TIS_utils.template)
        self.descriptor_dim = len(test_descriptor)

        # AIMMD setup
        self.AIMMD_set = AIMMD_setup(TPS_config_path, self.descriptor_dim, self.TIS_utils.states, descriptor_transform=self.descriptor_transform)
        interface_model_file = Path(input_path / interface_model_file).with_suffix(".h5")
        self.model = self.AIMMD_set.load_RCModel(interface_model_file)
        self.training_model = None
        self.q_bins_WHAM = self.set_q_bins_WHAM()

        self.CrPrW_forward = TAI.Crossing_Probability_and_weights(
            self.interface_values_forward, "forward", 
            storage_folder=input_path, 
            storage_filename=None if storage_list_forward else ops_storage_path, 
            storage_list=storage_list_forward,
            RPE_already_stored=True if RPE_already_stored else not(load_forward)
        )

        # Plot forward crossing probabilities
        fig, ax = plt.subplots(figsize=(20,10))
        ax.set_title("TIS crossing probabilities forward")
        cros_prob_forward, wham_weights_forward, path_weights_forward = self.CrPrW_forward.Compute_crossing_prob_and_wham_path_weights(
            self.q_bins_WHAM, max_min_filename=max_min_filename, n_thermalization=n_thermalization, cutoff=cutoff, ax=ax
        )
        save_fig_pdf_and_png(fig, "crossing_per_interface_forward", output_path=self.output_path)
        self.plot_crossing_probabilities(cros_prob_forward=cros_prob_forward)

        self.CrPrW_backward = TAI.Crossing_Probability_and_weights(
            self.interface_values_backward, "backward", 
            storage_folder= input_path, 
            storage_filename=None if storage_list_backward else ops_storage_path, 
            storage_list=storage_list_backward,
            RPE_already_stored= True if RPE_already_stored else not(load_backward)
        )

        # Plot backward crossing probabilities
        fig, ax = plt.subplots(figsize=(20,10))
        ax.set_title("TIS crossing probabilities backward")
        cros_prob_backward, wham_weights_backward, path_weights_backward = self.CrPrW_backward.Compute_crossing_prob_and_wham_path_weights(
            self.q_bins_WHAM, max_min_filename=max_min_filename, n_thermalization=n_thermalization, cutoff=cutoff, ax=ax
        )
        save_fig_pdf_and_png(fig, "crossing_per_interface_backward", output_path=output_path)
        self.plot_crossing_probabilities(cros_prob_backward=cros_prob_backward)

        # Compute RPE
        self.RPE = TAI.RPE_toy(self.CrPrW_forward, self.CrPrW_backward)

        # Define default RPE storage path
        if self.RPE_storage_path is None:
            self.RPE_storage_path = "RPE_storage"
        self.RPE_storage_path = Path(self.output_path / self.RPE_storage_path).with_suffix(".pkl")

    def load_stable_storage(self, stable_storage_path, input_path, type="db"):
        stable_states = ["A", "B"]
        storage_stable = []

        for i, stable_state in enumerate(stable_states):
            if stable_storage_path is None:
                stable_path = "stable_{}_{}_store.nc".format(stable_state, self.TIS_utils.pes.__repr__())
            else:
                stable_path = stable_storage_path[i]
                stable_path = Path(str(input_path.joinpath(stable_path))).with_suffix(f".{type}")
                print("Loaded stable run in stable state {}".format(stable_state))
                if type == "nc":
                    storage_stable.append(paths.AnalysisStorage(stable_path))
                elif type == "db":
                    storage_stable.append(Storage(stable_path))
                else:
                    raise TypeError("Incorrect storage type given. Should be either 'db' or 'nc' storage")
        return storage_stable

    def load_storage(self,storage_path_list,type="db"):
        storage_list = []
        for storage_path in storage_path_list:
            storage_path = Path(storage_path)
            print("loading storage: {}".format(storage_path))
            if type=="nc":
                storage_list.append(paths.AnalysisStorage(storage_path))
            elif type=="db":
                storage_list.append(Storage(storage_path))
            else:
                raise TypeError("incorrect storage type given. should be either 'db' or 'nc' storage")
        return storage_list

    def load_and_save_RPE(self, number_traj_used=1000, n_jump =1, start_traj=200):
        if self.stable_storage_path is None:
            stable_storages = None
            self.load_stable = False
        else: 
            stable_storages = self.load_stable_storage(self.stable_storage_path, self.input_path)
            self.load_stable = True
        
        self.RPE.load_RPE(self.TIS_utils.states, self.descriptor_transform, self.descriptor_dim, number_traj_used=number_traj_used, n_jump=n_jump, stable_storages=stable_storages)
        self.RPE.save_RPE(self.RPE_storage_path)
    
    def load_pickle(self, load_stable=True):
        self.load_stable = load_stable
        self.RPE.load_RPE(data_pickle=self.RPE_storage_path, load_stable=load_stable)
        if load_stable:
            self.RPE.compute_flux_compensation_stable_states(interface_model=self.model)
        
    def set_q_bins_WHAM(self):
        # upscale = 1.5
        # q_A = np.floor(np.min(self.interface_values_forward))
        # q_B = np.ceil(np.max(self.interface_values_backward))
        return np.arange(-80, 80, 0.01)
    
    def return_storage_list(self):
        return self.storage_lists

    def check_interfaces_states(self):
        pass

    def set_plot_settings(self):
        colors = []
        colors.append([33,104,108])
        colors.append([1,82,54])
        colors.append([107, 196,  166])
        colors.append([254,152, 42])
        colors.append([187,79,12])
        colors.append([110,41,12])
        colors.append([158,158,158])
        self.colors = [[c[0]/254,c[1]/254,c[2]/254] for c in colors]
        import matplotlib.pylab as pylab
        params = {'legend.fontsize': 'x-large',
            'figure.figsize': (15, 5),
            'axes.labelsize': 'x-large',
            'axes.titlesize':'x-large',
            'xtick.labelsize':'x-large',
            'ytick.labelsize':'x-large'}
        pylab.rcParams.update(params)
        self.linewidth = 2
        self.color_forward = self.colors[2]
        self.color_backward = self.colors[4]
        self.color_prob = self.colors[-2]
        self.alpha = 0.7
        self.fontsize = 20

    def plot_crossing_probabilities_V1(self, cros_prob_forward, cros_prob_backward, ax1=None,ax2=None,fig=None,xlim=[-20,20],save_fig=False):
        if ax1 is None or ax2 is None:
            fig, (ax1, ax2) = plt.subplots(2, sharex = True)
        linewidth = 4
        color_forward = self.colors[2]
        color_backward = self.colors[4]
        color_prob_forward = self.colors[1]
        color_prob_backward = self.colors[-2]
        if fig is not None:
            fig.suptitle(r"$A\rightarrow B$ and $B\rightarrow A$ crossing probability", fontsize=self.fontsize)
        ax1.set_yscale('log')
        ax1.plot(cros_prob_forward, '-o', color=color_prob_forward)
        for interface_val in self.interface_values_forward:
            ax1.axvline((interface_val),alpha=1, color=color_forward,linewidth=linewidth)
        ax1.set_ylabel(r'$P(q(x)|\lambda_{q1})$',fontsize=self.fontsize)
        ax2.plot(-cros_prob_backward.index,cros_prob_backward, '-o',color=color_prob_backward)
        for interface_val in self.interface_values_backward:
            ax2.axvline((interface_val),alpha=1,color=color_backward, linewidth=linewidth)

        ax2.set_xlabel("q(x)",fontsize=self.fontsize)
        ax2.set_yscale('log')
        ax2.set_ylabel(r'$P(q(x)|\lambda_{qn})$',fontsize=self.fontsize)
        ax1.set_xlim(xlim)
        # ax1.set_xlim([self.q_bins_WHAM[0],self.q_bins_WHAM[-1]])
        ax1.grid()
        ax2.grid()
        if save_fig:
            save_fig_pdf_and_png(fig, "full_crossing_probability", output_path=self.output_path)


    def plot_crossing_probabilities(self, cros_prob_forward=None, cros_prob_backward=None, fig=None, axes=None):
        # Determine plot layout based on input probabilities
        plot_forward = cros_prob_forward is not None
        plot_backward = cros_prob_backward is not None

        if not plot_forward and not plot_backward:
            raise ValueError("At least one of cros_prob_forward or cros_prob_backward must be provided.")

        # Create figure and axes if not provided
        if fig is None or axes is None:
            if plot_forward and plot_backward:
                fig, axes = plt.subplots(2, sharex=True, figsize=(10, 5))
            else:
                fig, axes = plt.subplots(1, figsize=(5, 3))
                axes = [axes]  # Ensure axes is a list for consistent handling
        elif not isinstance(axes, list):
            axes = [axes]

        linewidth = 4
        color_forward = self.colors[2]
        color_backward = self.colors[4]
        color_prob_forward = self.colors[1]
        color_prob_backward = self.colors[-2]

        # Plot forward crossing probability if provided
        if plot_forward:
            ax = axes[0] if len(axes) > 1 or not plot_backward else axes[0]
            ax.set_yscale('log')
            ax.plot(cros_prob_forward, '-o', color=color_prob_forward)
            for interface_val in self.interface_values_forward:
                ax.axvline(interface_val, alpha=0.5, color=color_forward, linewidth=linewidth/4)
            ax.set_ylabel(r'$P(q(x)|A)$', fontsize=self.fontsize)
            ax.grid()

        # Plot backward crossing probability if provided
        if plot_backward:
            if len(axes) > 1:
                ax = axes[1]
            else:
                ax = axes[0] if not plot_forward else fig.add_subplot(212, sharex=axes[0])
                axes.append(ax)

            ax.plot(-cros_prob_backward.index, cros_prob_backward, '-o', color=color_prob_backward)
            for interface_val in self.interface_values_backward:
                ax.axvline(interface_val, alpha=0.5, color=color_backward, linewidth=linewidth/4)
            ax.set_xlabel("q(x)", fontsize=self.fontsize)
            ax.set_yscale('log')
            ax.set_ylabel(r'$P(q(x)|B)$', fontsize=self.fontsize)
            ax.grid()

        # Set title and x-limits if axes is shared
        if plot_forward and plot_backward:
            fig.suptitle(r"$A\rightarrow B$ and $B\rightarrow A$ crossing probability", fontsize=self.fontsize)
        if plot_forward or plot_backward:
            axes[0].set_xlim([self.interface_values_forward[0]-4, self.interface_values_backward[0]+4])

        # Save the figure with appropriate naming
        if plot_forward and plot_backward:
            save_fig_pdf_and_png(fig, "full_crossing_probability", output_path=self.output_path)
        elif plot_forward:
            save_fig_pdf_and_png(fig, "forward_crossing_probability", output_path=self.output_path)
        elif plot_backward:
            save_fig_pdf_and_png(fig, "backward_crossing_probability", output_path=self.output_path)



    def initialize_plotter(self):
        self.plotter= TAI.ToyAimmdVisualizer(self.TIS_utils.pes, temperature=self.TIS_utils.temperature)
        self.plotter.load_RPE_data(self.RPE)

    def plot_RPE_subplots(self, save=False, label=""):
        
        fig, ax = plt.subplots(1,3,figsize=(15,5))
        self.plotter.RPE_2d(ax=ax[0])
        self.plotter.committor_2d_RPE_data(ax=ax[1])
        self.plotter.q_2d_RPE_data(ax=ax[2])
        fig.tight_layout()
        if save:
            save_fig_pdf_and_png(fig, "RPE_results"+label, output_path=self.output_path)

    def plot_model_projection(self,plot_model=None,save=False, label=""):
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        if plot_model is None:
            plot_model = self.model
        self.plotter.committor_2d_projection(plot_model,ax=ax[0])
        self.plotter.q_space_2d_projection(plot_model,fig=fig, ax=ax[1])
        fig.tight_layout()
        if save:
            save_fig_pdf_and_png(fig, "Model_projection"+label, output_path=self.output_path)
    
    def plot_model_on_RPE(self,plot_model=None, save=False, label=""):
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        if plot_model is None:
            plot_model = self.model
        self.plotter.committor_model_2d_RPE_data(plot_model,ax=ax[0])
        self.plotter.q_model_2d_RPE_data(plot_model,fig=fig, ax=ax[1])
        fig.tight_layout()
        if save:
            save_fig_pdf_and_png(fig, "model_on_RPE"+label, output_path=self.output_path)
    
    def plot_model_on_RPE_with_scatter(self, plot_model=None,save=False, label=""):
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        if plot_model is None:
            plot_model = self.model
        self.plotter.committor_model_2d_RPE_data(plot_model,ax=ax[0])
        self.plotter.q_model_2d_RPE_data(plot_model,fig=fig, ax=ax[1])
        self.plotter.scatter_rc_path(plot_model,ax=ax[1])
        fig.tight_layout()
        if save:
            save_fig_pdf_and_png(fig, "model_on_RPE_scatter"+label, output_path=self.output_path)

    def plot_model_on_RPE_with_scatter_minima(self,plot_model=None, save=False, label=""):
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        if plot_model is None:
            plot_model = self.model
        self.plotter.committor_model_2d_RPE_data(plot_model,ax=ax[0])
        self.plotter.q_model_2d_RPE_data(plot_model,fig=fig, ax=ax[1])
        self.plotter.scatter_rc_minima_path(plot_model,ax=ax[1])
        fig.tight_layout()
        if save:
            save_fig_pdf_and_png(fig, "model_on_RPE_scatter"+label, output_path=self.output_path)


    def plot_model_on_RPE_with_scatter_deriv_rho(self,plot_model=None, save=False, label=""):
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        if plot_model is None:
            plot_model = self.model
        self.plotter.committor_model_2d_RPE_data(plot_model,ax=ax[0])
        self.plotter.q_model_2d_RPE_data(plot_model,fig=fig, ax=ax[1])
        self.plotter.scatter_rc_deriv_rho(plot_model,ax=ax[1])
        fig.tight_layout()
        if save:
            save_fig_pdf_and_png(fig, "model_on_RPE_scatter"+label, output_path=self.output_path)


    def plot_model_vs_theory(self, theoretical_committor_path,n_x,plot_model=None, save=False, label=""):
        fig, ax = plt.subplots(1,1)
        if plot_model is None:
            plot_model = self.model
        self.plotter.q_model_2d_RPE_data(plot_model,fig=fig, ax=ax)
        self.plotter.theoretical_q_contour(theoretical_committor_path,n_x,ax=ax)
        fig.tight_layout()
        if save:
            save_fig_pdf_and_png(fig, "q_model_vs_theory"+label , output_path=self.output_path)
    
    def plot_model_vs_theory_contours_data(self, theoretical_committor_path,n_x,plot_model=None, save=False, label=""):
        fig, ax = plt.subplots(1,1,figsize=(7,5))
        if plot_model is None:
            plot_model = self.model
        self.plotter.plot_potential_contour(ax=ax)
        self.plotter.plot_q_model_RPE_data_contours(plot_model,fig=fig, ax=ax)
        self.plotter.theoretical_q_contour(theoretical_committor_path,n_x,ax=ax,fig=fig)
        self.plotter.plot_states(ax=ax)

        fig.tight_layout()
        if save:
            save_fig_pdf_and_png(fig, "q_model_vs_theory"+label , output_path=self.output_path)

    def plot_RPE_error(self,vmax=3, save=False, label=""):
        fig,ax = plt.subplots(1,1,figsize=(6,5))
        self.plotter.plot_RPE_error(ax=ax,fig=fig,vmax=vmax)
        self.plotter.plot_potential_contour(ax=ax)
        fig.tight_layout()
        if save:
                save_fig_pdf_and_png(fig, "RPE_error"+label , output_path=self.output_path)
    
    def plot_model_vs_theory_committor(self, theoretical_committor_path,n_x,plot_model=None,save=False, label=""):
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        if plot_model is None:
            plot_model = self.model
        self.plotter.committor_model_2d_RPE_data(plot_model,fig=fig, ax=ax)
        self.plotter.theoretical_committor_contour(theoretical_committor_path,n_x,ax=ax)
        # fig.tight_layout()
        if save:
            save_fig_pdf_and_png(fig, "committor_model_vs_theory"+label, output_path=self.output_path)

    def plot_model_vs_theory_projection(self, theoretical_committor_path,n_x,plot_model=None,save=False, label=""):
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        if plot_model is None:
            plot_model = self.model
        self.plotter.plot_potential_contour(ax=ax)
        self.plotter.plot_q_model_projection_contour(plot_model,ax=ax)
        self.plotter.theoretical_q_contour(theoretical_committor_path,n_x,ax=ax)
        fig.tight_layout()
        if save:
            save_fig_pdf_and_png(fig, "q_model_vs_theory_projection"+label, output_path=self.output_path)

    def plot_model_data_contours_q(self, plot_model=None, save=False, label=""):
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        if plot_model is None:
            plot_model = self.model
        self.plotter.plot_potential_contour(ax=ax)
        self.plotter.q_2d_RPE_data(ax=ax)
        self.plotter.plot_q_model_RPE_data_contours(plot_model,ax=ax)
        fig.tight_layout()
        if save:
            save_fig_pdf_and_png(fig, "q_model_on_RPE"+label, output_path=self.output_path)

    def plot_loss_components(self,theoretical_committor_path=None,n_x=501,  plot_model=None, fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        if plot_model is None:
            plot_model = self.model
        self.plotter.plot_loss_allong_q_model(plot_model=plot_model, ax=ax)
        self.plotter.plot_plnp_allong_q_model(plot_model=plot_model, ax=ax)
        self.plotter.plot_weight_allong_q_model(plot_model=plot_model, ax=ax)
        self.plotter.plot_distribution_of_points_allong_q_model(plot_model=plot_model,ax=ax, density=True)
        self.plotter.plot_pAandpB_of_RPE_data_allong_q_model(plot_model=plot_model,ax=ax)
        self.plotter.plot_loss_normalized_q_allong_q_model(plot_model=plot_model, ax=ax,density=True)
        if theoretical_committor_path is not None:
            self.plotter.plot_loss_allong_theory_q(theoretical_committor_path=theoretical_committor_path,n_x=n_x,ax=ax)
        ax.grid()
        if fig is not None:
            fig.legend(bbox_to_anchor=(1.5, 0.9))
        ax.set_yscale("log")
    
    def plot_loss_model_vs_theory(self,theoretical_committor_path=None,n_x=501,  plot_model=None, fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        if plot_model is None:
            plot_model = self.model
        self.plotter.plot_loss_allong_q_model(plot_model=plot_model, ax=ax)
        # self.plotter.plot_loss_scaled_q_allong_q_model(plot_model=plot_model,ax=ax)
        # self.plotter.plot_loss_scaled_weight_sqrtrho(plot_model=plot_model,ax=ax)
        # self.plotter.plot_loss_normalized_q_allong_q_model(plot_model=plot_model, ax=ax, density=True)
        if theoretical_committor_path is not None:
            p_B_model_RPE, q_model_RPE = self.plotter.model_output_RPE(plot_model)
            q_min,q_max = np.nanmin(q_model_RPE),np.nanmax(q_model_RPE)
            self.plotter.plot_loss_allong_theory_q(theoretical_committor_path=theoretical_committor_path,n_x=n_x,ax=ax)
            ax.set_xlim([q_min-1,q_max+1])
        ax.grid()
        if fig is not None:
            ax.legend(bbox_to_anchor=(1.1, 0.9))
        ax.set_yscale("log")
    
    def plot_plnp_model_vs_theory(self,theoretical_committor_path=None,n_x=501,  plot_model=None, fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        if plot_model is None:
            plot_model = self.model

    
        self.plotter.plot_plnp_allong_q_model(plot_model=plot_model, ax=ax)
        if theoretical_committor_path is not None:
            p_B_model_RPE, q_model_RPE = self.plotter.model_output_RPE(plot_model)
            q_min,q_max = np.nanmin(q_model_RPE),np.nanmax(q_model_RPE)
            self.plotter.plot_plnp_allong_theory_q(theoretical_committor_path=theoretical_committor_path,n_x=n_x,ax=ax)
            ax.set_xlim([q_min-1,q_max+1])
        ax.grid()
        if fig is not None:
            ax.legend(bbox_to_anchor=(1.1, 0.9))
        ax.set_yscale("log")
    
    def plot_weight_model_vs_theory(self,theoretical_committor_path=None,n_x=501,  plot_model=None, fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        if plot_model is None:
            plot_model = self.model
        p_B_model_RPE, q_model_RPE = self.plotter.model_output_RPE(plot_model)
        q_min,q_max = np.nanmin(q_model_RPE),np.nanmax(q_model_RPE)
        self.plotter.plot_weight_allong_q_model(plot_model=plot_model, ax=ax, density=True)
        if theoretical_committor_path is not None:
            p_B_model_RPE, q_model_RPE = self.plotter.model_output_RPE(plot_model)
            q_min,q_max = np.nanmin(q_model_RPE),np.nanmax(q_model_RPE)
            self.plotter.plot_weight_allong_theory_q(theoretical_committor_path=theoretical_committor_path,n_x=n_x,ax=ax,density=True)
            ax.set_xlim([q_min-1,q_max+1])
        ax.grid()
        if fig is not None:
            ax.legend(bbox_to_anchor=(1.1, 0.9))
        ax.set_yscale("log")
    
    def plot_free_energy_allong_q(self,theoretical_committor_path=None,n_x=501, plot_model=None, fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        if plot_model is None:
            plot_model = self.model
        self.plotter.plot_free_energy_RPE_allong_q_model(plot_model, fig=fig, ax=ax)
        if theoretical_committor_path is not None:
            self.plotter.plot_free_energy_allong_theory_q(theoretical_committor_path=theoretical_committor_path, n_x=n_x, fig=fig, ax=ax, U_max=300)
        ax.legend()

    def plot_free_energy_allong_committor(self,theoretical_committor_path=None,n_x=501, plot_model=None, fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        if plot_model is None:
            plot_model = self.model
        self.plotter.plot_free_energy_RPE_allong_committor_model(plot_model, fig=fig, ax=ax)
        if theoretical_committor_path is not None:
            self.plotter.plot_free_energy_allong_theory_committor(theoretical_committor_path=theoretical_committor_path, n_x=n_x, fig=fig, ax=ax, U_max=300)
        ax.legend()
    
    def plot_gradient_q_model(self,plot_model=None, ax=None):
        if plot_model is None:
            plot_model = self.model
        
        # Call the function to plot gradient field
        if ax is None:
            fig,ax = plt.subplots(1,1)
        self.plotter.plot_q_model_projection(plot_model,ax=ax)
        self.plotter.plot_gradient_field_2d(plot_model, grid_size=30,ax=ax)
        self.plotter.plot_potential_contour(ax=ax)
        
    def all_analysis_plots(self,plot_model=None, theoretical_committor_path=None, n_x = None, save=False, label=""):
        self.initialize_plotter()
        self.plot_RPE_subplots(save=save)
        self.plot_model_on_RPE_with_scatter_minima(plot_model=plot_model,save=save,label=label)
        if theoretical_committor_path is not None and n_x is not None:
            self.plot_model_vs_theory_contours_data(theoretical_committor_path=theoretical_committor_path, n_x = n_x,plot_model=plot_model,save=save, label=label)
            self.plot_model_vs_theory_projection(theoretical_committor_path=theoretical_committor_path, n_x = n_x,plot_model=plot_model,save=save, label=label)

    def all_plots(self, plot_model=None,theoretical_committor_path=None, n_x = None, save=False, label=""):
        self.initialize_plotter()
        self.plot_RPE_subplots(save=save)
        self.plot_model_on_RPE(plot_model=plot_model, save=save,label=label)
        self.plot_model_on_RPE_with_scatter(plot_model=plot_model, save=save,label=label)
        self.plot_model_on_RPE_with_scatter_minima(plot_model=plot_model, save=save,label=label)
        self.plot_model_projection(plot_model=plot_model,save=save, label=label)
        self.plot_RPE_error(save=save, label=label)
        self.plot_gradient_q_model(plot_model=plot_model)
        if theoretical_committor_path is not None and n_x is not None:
            self.plot_model_vs_theory(theoretical_committor_path=theoretical_committor_path, n_x = n_x,plot_model=plot_model, save=save, label=label)
            self.plot_model_vs_theory_contours_data(theoretical_committor_path=theoretical_committor_path, n_x = n_x,plot_model=plot_model, save=save, label=label)
            self.plot_model_vs_theory_committor(theoretical_committor_path=theoretical_committor_path, n_x = n_x,plot_model=plot_model, save=save, label=label)
            self.plot_model_vs_theory_projection(theoretical_committor_path=theoretical_committor_path, n_x = n_x,plot_model=plot_model, save=save, label=label)