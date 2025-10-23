
""" 
Here the AIMMD analysis framework will be made.
It should contain the following:
- Load in a list of ops-storage files
- single direction analaysis of the TIS to compute the crossing probabilities A->B B->A etc.
- 


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpathsampling as paths
from pathlib import Path
from .Tools import interface_indicator, save_fig_pdf_and_png, count_sign_changes, find_first_crossings, count_forward_crossings

from openpathsampling.experimental.storage import Storage
from concurrent.futures import ProcessPoolExecutor

# this will be a analysis class which is seperate from the TIS_AIMMD class, 
# the AIMMD model and storage should be loaded into this analysis class 
# Dertermine the snapshots which are inside the defined stable statess



def Create_Stable_trainset(storage_list, descriptor_transform, descriptor_dim, states):
    n_dim = descriptor_dim
    weights_each_stable = [] 
    descriptors_each_stable= []
    shot_results_each_stable= []
    for index_state  in range(len(storage_list)):
        print("loading data for state {}".format(index_state))
        points= len(storage_list[0].snapshots)
        weights = np.ones(points)
        descriptors = np.zeros((points,n_dim))
        shot_results = np.zeros((points,len(states)))
        total_point_stored = 0
        for traj in storage_list[index_state].trajectories:
            start = total_point_stored
            traj_snapshots = len(traj)
            total_point_stored += traj_snapshots
            weights[start:total_point_stored]= [states[index_state](traj.get_as_proxy(i)) for i in range(len(traj))]
            shot_results[start:total_point_stored,0] =  states[0](traj.get_as_proxy(0))*np.ones(traj_snapshots)
            shot_results[start:total_point_stored,1] = states[1](traj.get_as_proxy(0))*np.ones(traj_snapshots)
            descriptors[start:total_point_stored,:] = descriptor_transform(traj)
        descriptors = descriptors[:total_point_stored,:]
        weights = weights[:total_point_stored]
        shot_results = shot_results[:total_point_stored]
        weights_each_stable.append(weights)
        shot_results_each_stable.append(shot_results)
        descriptors_each_stable.append(descriptors)

    return descriptors_each_stable, weights_each_stable, shot_results_each_stable


class RPE_toy:
    def __init__(self, CrPrW_Forward, CrPrW_Backward):
        
        """
        This is the analysis class of the TIS-AIMMD method, 
        here the crossing probabilities will be computed and from this the RPE 
        and Free energy allong selected collective variables.       
        
        """
        self.CrPrW_Forward = CrPrW_Forward
        self.CrPrW_Backward = CrPrW_Backward
        eps = 0.01
        maximum_forward = self.CrPrW_Backward.interfaces[0]
        maximum_backward = self.CrPrW_Forward.interfaces[0]
        print(maximum_forward)
        print(maximum_backward)
        print(CrPrW_Forward.interfaces)
        print(CrPrW_Backward.interfaces)
        index_q0 = int(np.where(abs(CrPrW_Forward.wham_cross_prob.index - maximum_forward)<eps)[0][0])
        index_q0_Backward = int(np.where(abs(-CrPrW_Backward.wham_cross_prob.index-maximum_backward)<eps)[0][0])
        index_Forward = CrPrW_Forward.wham_cross_prob.index[index_q0]
        index_Backward = CrPrW_Backward.wham_cross_prob.index[index_q0_Backward]
        p_Ai_0 = CrPrW_Forward.wham_cross_prob[index_Forward]
        p_Bi_0 = CrPrW_Backward.wham_cross_prob[index_Backward]
        self.c_A_0 = 1/p_Ai_0
        self.c_B_0 = 1/p_Bi_0
        self.data_Forward=[]
        self.data_Backward=[]
        self.interfaces_forward = CrPrW_Forward.interfaces
        self.interfaces_backward = CrPrW_Backward.interfaces
        self.data_Stable = None
        self.load_stable =False
        self.states=["A","B"]
        self.descriptors_total = None
        self.flux_compensation = [1,1]
        self.n_thermalization_forward = self.CrPrW_Forward.n_thermalization
        self.n_thermalization_backward = self.CrPrW_Backward.n_thermalization


    def load_RPE(self,states=None,descriptor_transform=None, descriptor_dims=None,number_traj_used=2000, n_jump=1, data_pickle=None, stable_storages=None, load_stable=False, per_interface=False):
        self.load_stable = load_stable
        if data_pickle==None:
            self.load_RPE_from_storage(states,descriptor_transform, descriptor_dims,number_traj_used=number_traj_used, n_jump=n_jump, stable_storages=stable_storages)
        else: 
            if per_interface:
                self.load_RPE_all_interfaces(data_pickle,load_stable=load_stable)
            else:
                self.load_RPE_pickle(data_pickle,load_stable=load_stable)

    def load_RPE_from_storage(self,states,descriptor_transform, descriptor_dims,number_traj_used=2000, n_jump=1, stable_storages=None):
        self.data_Forward = self.Create_RPE_trainset(self.CrPrW_Forward, descriptor_transform, 
                                        descriptor_dims, states,number_traj_used=number_traj_used, n_jump=n_jump, start_traj=self.n_thermalization_forward)
        self.data_Backward = self.Create_RPE_trainset(self.CrPrW_Backward, descriptor_transform, 
                                    descriptor_dims, states,number_traj_used=number_traj_used, n_jump=n_jump, start_traj=self.n_thermalization_backward)
        if stable_storages is not None:
            self.data_Stable = Create_Stable_trainset(stable_storages, descriptor_transform, descriptor_dims, states)
    
    def load_RPE_all_interfaces(self, data_pickle,load_stable=False):
        self.data_Forward = self.Create_RPE_trainset_from_interface_pickles(data_pickle,self.CrPrW_Forward.interfaces,"forward")
        self.data_Backward = self.Create_RPE_trainset_from_interface_pickles(data_pickle,self.CrPrW_Backward.interfaces,"backward")
        if load_stable:
            self.data_Stable = self.Create_RPE_trainset_from_interface_pickles(data_pickle,["A","B"],"stable")

    def load_RPE_pickle(self, data_pickle, load_stable=False):
        self.data_Forward = self.create_RPE_trainset_from_pickle(data_pickle,self.CrPrW_Forward.direction,self.CrPrW_Forward.interfaces)
        self.data_Backward = self.create_RPE_trainset_from_pickle(data_pickle,self.CrPrW_Backward.direction,self.CrPrW_Backward.interfaces)
        if load_stable:
            self.data_Stable = self.create_RPE_trainset_from_pickle(data_pickle,"stable",["A","B"])
    
    
    def add_stable(self, storage_stable, descriptor_transform, descriptor_dim, states):
        self.data_Stable = Create_Stable_trainset(storage_stable, descriptor_transform, descriptor_dim, states)

    def save_RPE(self,filename):
        # Example usage
        data_store = DataStore()

        # Add data for interfaces
        for i, interface in enumerate(self.CrPrW_Forward.interfaces):
            data_store.add_interface_data('forward', str(interface), self.data_Forward[0][i], self.data_Forward[1][i], self.data_Forward[2][i])

        for i, interface in enumerate(self.CrPrW_Backward.interfaces):
            data_store.add_interface_data('backward', str(interface), self.data_Backward[0][i], self.data_Backward[1][i], self.data_Backward[2][i])

        if self.data_Stable is not None:
            for i, state in enumerate(["A","B"]):
                data_store.add_interface_data('stable', str(state), self.data_Stable[0][i], self.data_Stable[1][i], self.data_Stable[2][i])

        # Save data to file
        data_store.save_to_file(filename)
    
    def Create_RPE_trainset_for_interface(self, CrPrW, descriptor_transform, descriptor_dim, states, 
                                        interface_index, number_traj_used=2000, start_traj=200, 
                                        n_jump=1, show_log=True):
        """Processes only a single interface instead of all interfaces."""
        n_dim = descriptor_dim
        n_steps = int(np.floor((number_traj_used - start_traj) / n_jump))
        
        interface = CrPrW.interfaces[interface_index]
        
        if show_log:
            print(f"Processing single interface {interface_index} with q: {interface}")
            interface_start_time = time.time()

        total_point_stored = 0
        points = len(CrPrW.storage_list[interface_index].snapshots) * 2
        weights = np.zeros(points)
        descriptors = np.zeros((points, n_dim))
        shot_results = np.zeros((points, 2))

        for i in range(n_steps):
            if i % 10 == 0 and show_log:
                elapsed_time = time.time() - interface_start_time
                steps_completed = i + 1
                avg_time_per_step = elapsed_time / steps_completed
                estimated_total_time = avg_time_per_step * n_steps
                estimated_remaining_time = estimated_total_time - elapsed_time
                print(f"Interface {interface_index}: Step {i}/{n_steps}. "
                    f"Elapsed time: {elapsed_time / 60:.2f} min, Estimated total: {estimated_total_time / 60:.2f} min, "
                    f"Remaining: {estimated_remaining_time / 60:.2f} min")

            traj_index = start_traj + i * n_jump
            step = CrPrW.storage_list[interface_index].steps[traj_index]
            traj = step.active[0].trajectory
            
            start = total_point_stored
            total_point_stored += len(traj)
            
            shot_results[start:total_point_stored, 0] = states[0](traj.get_as_proxy(-1)) + states[0](traj.get_as_proxy(0))
            shot_results[start:total_point_stored, 1] = states[1](traj.get_as_proxy(-1)) + states[1](traj.get_as_proxy(0))
            
            weights[start:total_point_stored] = CrPrW.path_weights_interface[interface_index][traj_index]
            weights = np.nan_to_num(weights, posinf=0, nan=0, neginf=0)
            descriptors[start:total_point_stored, :] = descriptor_transform(traj)

        descriptors = descriptors[:total_point_stored, :]
        weights = weights[:total_point_stored]
        shot_results = shot_results[:total_point_stored]

        if show_log:
            interface_elapsed_time = time.time() - interface_start_time
            print(f"Finished processing interface {interface_index}. Time taken: {interface_elapsed_time / 60:.2f} min.")

        return descriptors, weights, shot_results

    def save_RPE_for_interface(self, filepath, interface_index, descriptors, weights, shot_results, mode="forward"):
        """Save results for a single interface to a pickle file."""
        data_store = DataStore()

        if mode == "forward":
            interface = self.CrPrW_Forward.interfaces[interface_index]
        elif mode == "stable":
            # Expecting interface_index to be either "A" or "B"
            if interface_index not in ["A", "B"]:
                raise ValueError("In 'stable' mode, interface_index must be 'A' or 'B'")
            interface = interface_index
        else:  # Backward case
            interface = self.CrPrW_Backward.interfaces[interface_index]

        data_store.add_interface_data(mode, str(interface), descriptors, weights, shot_results)
        data_store.save_to_file(filepath)

        print(f"Saved RPE results for interface {interface} in {filepath} (mode: {mode})")


    def load_RPE_for_interface(self, filepath,interface, mode="forward"):
        """Load results for a single interface from a pickle file."""
        data_store = DataStore()
        data_store = data_store.load_from_file(filepath)
        weights_interface = data_store.data[mode][str(interface)].weights
        descriptors_interface = data_store.data[mode][str(interface)].descriptors
        shot_results_interface = data_store.data[mode][str(interface)].shot_results
        return descriptors_interface, weights_interface, shot_results_interface
    
    def Create_RPE_trainset_from_interface_pickles(self,files_folder, interfaces, mode="forward"):
        """Load RPE data from pickle files for each interface."""
        weights_interface = []
        descriptors_interface = []
        shot_results_interface = []
        for interface in interfaces:
            if mode=="stable":
                file_path = files_folder / f"RPE_{mode}_{interface}.pkl"
            else:
                file_path = files_folder / f"RPE_{mode}_interface_{interface_indicator(interface)}.pkl"
            print(f"Loading RPE data for interface {interface} from file: {file_path}")
            descriptors, weights, shot_results = self.load_RPE_for_interface(file_path, interface, mode)
            weights_interface.append(weights)   
            descriptors_interface.append(descriptors)
            shot_results_interface.append(shot_results)
        return descriptors_interface, weights_interface, shot_results_interface


    def Create_RPE_trainset(self,CrPrW, descriptor_transform, descriptor_dim, states, number_traj_used=2000,
                            start_traj=200, n_jump=1,show_log=True):
        n_steps = int(np.floor((number_traj_used - start_traj) / n_jump))
        if show_log:
            print("Starting analysis with {} trajectories per interface".format(n_steps))
        weights_interface = []
        descriptors_interface = []
        shot_results_interface = []
        for index_interface, interface in enumerate(CrPrW.interfaces):
            interface_start_time = time.time()
            descriptors, weights, shot_results = self.Create_RPE_trainset_for_interface(CrPrW, descriptor_transform, descriptor_dim, states,
                                                                                        index_interface, number_traj_used, start_traj, n_jump, show_log)
            weights_interface.append(weights)
            shot_results_interface.append(shot_results)
            descriptors_interface.append(descriptors)
            if show_log:
                # Calculate elapsed time for this interface
                interface_elapsed_time = time.time() - interface_start_time
                print("Finished processing interface {}. Time taken: {:.2f} minutes.".format(index_interface, interface_elapsed_time / 60))
        return descriptors_interface, weights_interface, shot_results_interface

    def create_RPE_trainset_from_pickle(self,data_pickle, mode,interfaces_states):
        data_store = DataStore()
        data_store = data_store.load_from_file(data_pickle)
        weights_interface = []
        descriptors_interface = []
        shot_results_interface = []
        for i, interface in enumerate(interfaces_states):
            weights_interface.append(data_store.data[mode][str(interface)].weights)
            descriptors_interface.append(data_store.data[mode][str(interface)].descriptors)
            shot_results_interface.append(data_store.data[mode][str(interface)].shot_results)
        return descriptors_interface, weights_interface, shot_results_interface


    def compute_flux_compensation_stable_states(self, interface_model,first_infterface_forward=None, first_interface_backward=None, data_Stable=None):
        self.flux_compensation = []
        self.first_crossings_lambda = []
        self.forward_crossings_lambda = []
        self.in_stable = []
        if data_Stable is None:
            data_Stable = self.data_Stable
        for state in range(2):
            q_stable_data = interface_model.log_prob(data_Stable[0][state], use_transform=False, batch_size=None)
            if state ==0:
                if first_infterface_forward is None:
                    q_interface = self.interfaces_forward[0] 
                else:
                    q_interface = first_infterface_forward
            else:
                if first_interface_backward is None:
                    q_interface = self.interfaces_backward[0]
                else:
                    q_interface = first_interface_backward
            my_array = np.array(q_stable_data)-q_interface
            my_array_state_def = np.array(data_Stable[1][state])-0.5
            in_stable = np.array(my_array <=0)[:,0] if state == 0 else np.array(my_array >=0)[:,0]
            self.in_stable.append(in_stable)
            self.first_crossings_lambda.append(len(find_first_crossings((-1)**state*q_stable_data, data_Stable[1][state], (-1)**state*q_interface)))
            print("Number of first crossings through lamdba {} is: {}".format(q_interface,self.first_crossings_lambda[state]))
            forward_crossing_state = count_forward_crossings(my_array_state_def)
            forward_crossing_interface = count_forward_crossings(my_array)
            self.forward_crossings_lambda.append(forward_crossing_interface)
            print("The forward crossings through interface {} is  {}".format(q_interface,forward_crossing_interface))
            print("The forward through interface stable_state is  {}".format(forward_crossing_state))

            self.flux_compensation.append(forward_crossing_interface/forward_crossing_state)
        print(self.flux_compensation)
        
    def create_total_trainset(self):
        # create trainset
        flux_scaling = []
        if self.descriptors_total is None:
            Total_snapshots = 0
            Total_forward = 0
            Total_backward = 0
            n_descriptor_dims = np.shape(self.data_Forward[0][0])[1]


            for i, interface in enumerate(self.interfaces_forward):
                size = np.shape(self.data_Forward[1][i])[0]
                Total_snapshots += size
                Total_forward += size

            for i, interface in enumerate(self.interfaces_backward):
                size = np.shape(self.data_Backward[1][i])[0]
                Total_snapshots += size 
                Total_backward += size
            if self.load_stable:
                Total_stable = 0
                for i, interface in enumerate(self.states):
                    size = np.shape(self.data_Stable[1][i])[0]
                    Total_snapshots += size 
                    Total_stable += size
                    flux_scaling.append(size)

            weights_total = np.ones(Total_snapshots,dtype=np.float32)
            descriptors_total = np.zeros((Total_snapshots,n_descriptor_dims),dtype=np.float32)
            shot_results_total = np.zeros((Total_snapshots,2),dtype=np.float32)
            start = 0
            if self.load_stable:
                equal_scaling_states = 1
                print("cA: ", self.c_A_0)
                print("cB: ", self.c_B_0)
                print("flux compenstation (Ni/Nstate): ", self.flux_compensation)
                print("forward flux first crossing through interfaces \lambda_1: {} [1/delta_t]".format(self.forward_crossings_lambda[0]/len(self.in_stable[0])))
                print("backward flux first crossing through interfaces \lambda_n: {} [1/delta_t]".format(self.forward_crossings_lambda[1]/len(self.in_stable[1])))
                print("forward flux forward crossing through interfaces \lambda_1: {} [1/delta_t]".format(self.forward_crossings_lambda[0]/len(self.in_stable[0])))
                print("backward flux forward crossing through interfaces \lambda_n: {} [1/delta_t]".format(self.forward_crossings_lambda[1]/len(self.in_stable[1])))

                kab = (self.first_crossings_lambda[0]/len(self.in_stable[0]))/self.c_A_0
                kba = (self.first_crossings_lambda[1]/len(self.in_stable[1]))/self.c_B_0
                print("Kab = ", kab)
                print("Kba = ", kba)
                Weight_factor = [self.c_A_0/self.flux_compensation[0], self.c_B_0/self.flux_compensation[1]]
                print(Weight_factor)
                Weight_factor = np.nan_to_num(Weight_factor, posinf=0) 
                print(Weight_factor)
                for i in range(len(self.states)):
                    
                    snapshots_interface = np.shape(self.data_Stable[0][i])[0]
                    weights_total[start:start+snapshots_interface] =Weight_factor[i]*self.in_stable[i]
                    # weights_total[start:start+snapshots_interface] =self.data_Stable[1][i]*Weight_factor[i] 

                    descriptors_total[start:start+snapshots_interface,:] = self.data_Stable[0][i]
                    shot_results_total[start:start+snapshots_interface,:] = self.data_Stable[2][i]
                    start += snapshots_interface
            

            for i, interface in enumerate(self.interfaces_forward):
                snapshots_interface = np.shape(self.data_Forward[0][i])[0]
                weights_total[start:start+snapshots_interface] = self.data_Forward[1][i]*self.c_A_0
                descriptors_total[start:start+snapshots_interface,:] = self.data_Forward[0][i]
                shot_results_total[start:start+snapshots_interface,:] = self.data_Forward[2][i]
                start += snapshots_interface


            for i, interface in enumerate(self.interfaces_backward):
                snapshots_interface = np.shape(self.data_Backward[0][i])[0]
                weights_total[start:start+snapshots_interface] = self.data_Backward[1][i]*self.c_B_0
                descriptors_total[start:start+snapshots_interface,:] = self.data_Backward[0][i]
                shot_results_total[start:start+snapshots_interface,:] = self.data_Backward[2][i]
                start += snapshots_interface

            print(start)
            print(Total_snapshots)


            self.weights_total = weights_total[:start]
            self.descriptors_total = descriptors_total[:start]
            self.shot_results_total = shot_results_total[:start]
            # Create a mask to filter out data points where weight is zero
            nonzero_mask = self.weights_total != 0
            # Scale the weights such that reactive paht contribute as 1
  
            # Apply the mask to filter descriptors_total, weights_total, and shot_results_total
            self.descriptors_total = self.descriptors_total[nonzero_mask]
            self.weights_total = self.weights_total[nonzero_mask]
            print("min weights_total: {}".format(np.min(self.weights_total)))
            self.weights_total = self.weights_total/np.min(self.weights_total)
            print("new min weight_total: {}".format(np.min(self.weights_total)))
            self.shot_results_total = self.shot_results_total[nonzero_mask]
        return self.descriptors_total, self.weights_total, self.shot_results_total
    

    #TODO: implement create trainset per mode to clean create trainset code.
    def create_trainset_mode(self, mode="forward"):
        pass
    
""" since the procedure of the crossing probabilities and weights is the same for both the forward and backward 
 direction the same class can be used for both instances.
 The class should do the following.
 Load the Storage files and max_min_cv values (and if max_min is not provided compute them)
 get the correct x binning for the cumulative distribution
 Compute the cumulative distributions of each interface in the cv space (in our case q-space)
Compute the resulting WHAM cumulative distribution function.

Obtain the unscaled RPE weight for each path in the TIS PE's
"""

#TODO sort the interfaces such that they are automatically correctly sorted
class Crossing_Probability_and_weights:
    def __init__(self, interfaces, direction, storage_folder: Path =None, storage_filename=None, storage_list=None, RPE_already_stored=False, type="db"):
        self.interfaces = interfaces
        self.direction = direction
        self.storage_folder = storage_folder
        self.storage_filename = storage_filename
        self.n_thermalization = None
        # If storage_list is provided, use it. Otherwise, load storage.
        if storage_list is not (None or False):
            self.storage_list = storage_list
        else:
            self.storage_path_list = self.import_storage_paths(type)
            self.storage_list = [] if RPE_already_stored else self.import_storage(type)       

    def import_storage_paths(self, type="db"):
        """Load storage paths based on interfaces."""
        storage_paths = []
        for interface_val in self.interfaces:
            load_path = Path(self.storage_folder) / f"{self.storage_filename}_{self.direction}_{interface_indicator(interface_val)}.{type}"
            storage_paths.append(load_path)
        return storage_paths

    def import_storage(self, type="db"):
        """Load actual storage objects from paths."""
        storage_list = []
        for storage_path in self.storage_path_list:
            print(f"Loading {self.direction} storage: {storage_path}")
            if type == "nc":
                storage_list.append(paths.AnalysisStorage(storage_path))
            elif type == "db":
                storage_list.append(Storage(storage_path))
            else:
                raise TypeError("Incorrect storage type given. Should be either 'db' or 'nc' storage")
        return storage_list

    # def Compute_crossing_prob_and_wham_path_weights(self, q_bins_WHAM, max_min_filename=None, n_thermalization=200, cutoff=0.01, ax=None):
        
    #     max_min_filename = "max_min_q_int" if max_min_filename is None else max_min_filename
    #     self.n_thermalization = n_thermalization
    #     print("import max min cv data:")
    #     self.import_max_min_cv(self.storage_folder,max_min_filename)
    #     print("Compute crossing probabilities.")
    #     if self.direction=="backward":
    #         self.max_cv = [-cv for cv in self.max_cv]
    #     self.TIS_crossing_probabilities(q_bins_WHAM, n_thermalization)
    #     print("Compute crossing probabilies using WHAM")
    #     self.create_wham_input(cutoff=cutoff, ax=ax)
    #     crossingProb = self.wham_crossing()
    #     wham_weights= self.wham_weights()
    #     path_weights = self.path_weights_all_TIS(wham_weights)
    #     return crossingProb, wham_weights, path_weights  

    def Compute_crossing_prob_and_wham_path_weights(
    self, 
    q_bins_WHAM, 
    max_min_filename=None, 
    n_thermalization=200, 
    cutoff=0.01, 
    tol=1e-10,
    ax=None, 
    bootstrap=False, 
    n_bootstrap=100,
    block_size=10
    ):
        max_min_filename = "max_min_q_int" if max_min_filename is None else max_min_filename
        self.n_thermalization = n_thermalization
        print("Import max-min CV data:")
        self.import_max_min_cv(self.storage_folder, max_min_filename)
        
        if self.direction == "backward":
            self.max_cv = [-cv for cv in self.max_cv]

        if bootstrap:
            print(f"Compute crossing probabilities with bootstrapping ({n_bootstrap} replicas)")
            self.bootstrap_crossing_probs(q_bins_WHAM, n_thermalization, n_bootstrap=n_bootstrap, cutoff=cutoff, tol=tol, block_size=block_size)
            crossingProb = self.bootstrap_mean
            wham_weights = None  # Not yet implemented with bootstrapping
            path_weights = None
        else:
            print("Compute crossing probabilities.")
            self.TIS_crossing_probabilities(q_bins_WHAM, n_thermalization)
            print("Compute WHAM input")
            self.create_wham_input(cutoff=cutoff, ax=ax)
            crossingProb = self.wham_crossing()
            wham_weights = self.wham_weights()
            path_weights = self.path_weights_all_TIS(wham_weights)

        return crossingProb, wham_weights, path_weights


    def full_wham(self, n_bins, n_thermalization, starting_interface=0, cutoff=0.01, plot=False):
        self.TIS_crossing_probabilities(n_bins, n_thermalization, plot=plot)
        self.create_wham_input(starting_interface=starting_interface, cutoff=cutoff, plot=plot)
        return self.wham_crossing()
  
    def import_max_min_cv(self, storage_folder, filename, show_log=False):
        self.max_cv = [] 
        if self.direction=="backward":
            max_cv_ind = 0
        else: 
            max_cv_ind = 1

        for interface_val in self.interfaces:
            interface_ind = interface_indicator(interface_val)
            if show_log:
                print(f"Loading max_cv for interface {interface_val}")
            fileStoreMax = "{}{}_{}.npy".format(filename, interface_ind,self.direction)
            fileStoreMax = Path(storage_folder/ fileStoreMax).with_suffix(".npy")
            max_cv_from_storage = np.load(fileStoreMax)
            if np.shape(max_cv_from_storage)[1]==2:
                max_cv_from_storage = max_cv_from_storage[:,max_cv_ind]
            self.max_cv.append(max_cv_from_storage)

    # def read_max_cv_storage(self, cv, interface_val, store_max):
    #     print("Collecting maximum per path for TIS with interface at cv={}".format(interface_val))
    #     interface_indicator = int(np.round(interface_val*100))
    #     fileStoreMax = "{}_{}_{}".format(store_max, interface_indicator)
    #     max_cv = np.zeros(len(self.storage_list[i].steps))
    #     for j, step in enumerate(self.storage_list.steps):
    #         traj = step.active[0].trajectory
    #         cv_values =np.array([cv(snapshot) for snapshot in traj ]).reshape(len(traj.snapshots))
    #         max_cv[j] = max(cv_values)
    #     np.save(fileStoreMax, max_cv)

    def TIS_crossing_probabilities(self,n_bins, n_thermalization, q_min=-20, q_max=20,ax=None, show_log=False):
        self.alldata = []
        self.allydata = []
        for i, interface_val in enumerate(self.interfaces):
            interface_indicator = int(np.round(interface_val*100))
            if show_log:
                print(f"Computing crossing probabilities for interface {interface_val}")
            x =  np.asarray(self.max_cv[i][n_thermalization:])
            n, bins = np.histogram(x, n_bins, density=True)
            dx = np.diff(bins)
            F1 = 1-np.cumsum(n)*dx
            xdata= bins[1:]
            ydata= F1
            self.alldata.append(xdata)
            self.allydata.append(ydata)
            if ax is not None:
                ax.grid(True)
                if self.direction=="forward":
                    ax.vlines(interface_val,ymax=1,ymin=0, label="q={}".format(interface_val))
                elif self.direction=="backward" : 
                    ax.vlines(-interface_val,ymax=1,ymin=0, label="q={}".format(interface_val))
                ax.plot(xdata,ydata, color="green", label="cum Hist")
                ax.legend(loc='right')
                ax.set_yscale('log')
                ax.set_xlim(q_min, q_max)
                ax.set_title('Crossing Probability for interface q={}'.format(interface_val))
                ax.set_xlabel('q_committor')
                ax.set_ylabel('Probability')

    def create_wham_input(self,starting_interface=0, cutoff=0.01, q_min=-20,q_max=20, ax=None, tol=1e-10):
        self.input_df = pd.DataFrame(data=np.array(self.allydata[starting_interface:]).T,
                        index=self.alldata[0])
        self.wham = paths.numerics.WHAM(cutoff=cutoff,tol=tol)
        self.prepared_tis = self.wham.prep_reverse_cumulative(self.input_df)
        if ax is not None:
            color = 'black'
            index= self.alldata[0]
            n_paths = sum([len(cycle) for cycle in self.max_cv[0]]) if isinstance(self.max_cv[0], list) else len(self.max_cv[0])
            ymin = 1 / max(n_paths, 1)  # Avoid divide by zero
            ymax= 1.5
            #plt.plot(index, exact, '-ok', lw=2)
            #for iface, color in zip(prepared.columns, ['r', 'g', 'b']):
            for iface in self.prepared_tis.columns:
                # NaN out the 0 data, to prevent it from plotting
                values = self.prepared_tis[iface].apply(lambda x: np.nan if x==0.0 else x)
                ax.plot(values, "-o", color=color)
                ax.plot(self.input_df[iface], "--",color=color)
                if self.direction=="forward":
                    ax.vlines(self.interfaces,ymin=ymin, ymax=ymax)
                    ax.set_xlim(self.interfaces[0]-5, self.interfaces[-1]+15)
                elif self.direction=="backward" : 
                    ax.vlines([-l for l in self.interfaces],ymin=ymin, ymax=ymax)
                    ax.set_xlim(-self.interfaces[0]-5, -self.interfaces[-1]+15)

                ax.set_ylim(ymin,ymax)
                ax.set_yscale('log')
                ax.set_xlabel(r"q",fontsize=20)
                ax.set_ylabel('Probability',fontsize=20)
            ax.grid()
    
    def wham_crossing(self):
        self.wham_cross_prob = self.wham.wham_bam_histogram(self.input_df)
        return self.wham_cross_prob
    
    def wham_weights(self):
        self.weights_bin = self.wham_cross_prob/np.sum(self.prepared_tis,axis=1)
        return self.weights_bin

    def path_weight_per_tis_ensemble(self, weights_bin, max_cv):
        #can only get in a 1d array of cv (thus seperate for each interface)
        path_weight = np.zeros(np.shape(max_cv)[0])
        for path_label, max_ in enumerate(max_cv):
            ind = np.where(weights_bin.index<max_)[0][-1] # this gives the last bin for which the maximum is higher (so the first edge of the bin the maximum is in)
            path_weight[path_label] = weights_bin[weights_bin.index[ind]]
            # path_weight = W_A[np.where(max>W_A[:,0])[0][0]]

        path_weight = np.nan_to_num(path_weight,posinf=0,nan=0, neginf=0)
        return path_weight

    def path_weights_all_TIS(self, weights_bin):
        self.path_weights_interface = []
        for i, interface_val in enumerate(self.interfaces):
            max_cv = self.max_cv[i]
            self.path_weights_interface.append(self.path_weight_per_tis_ensemble(weights_bin, max_cv))
        return self.path_weights_interface

    def bootstrap_crossing_probs(self, q_bins_WHAM, n_thermalization, n_bootstrap=100, cutoff=0.01, block_size=10,tol=1e-10):
        bootstrap_results = []

        for _ in range(n_bootstrap):
            self.alldata = []
            self.allydata = []

            for i, interface_val in enumerate(self.interfaces):
                raw_data = np.asarray(self.max_cv[i][n_thermalization:])
                if len(raw_data) < block_size:
                    continue

                total_length = len(raw_data)
                # Step 1: Split into blocks
                n_blocks = len(raw_data) // block_size
                blocks = np.array_split(raw_data[:n_blocks * block_size], n_blocks)
                resampled = [blocks[np.random.randint(len(blocks))] for _ in range(n_blocks)]
                resampled_data = np.concatenate(resampled)[:total_length]  # trim to match original size

                # Step 3: Histogram & cumulative probability
                n, bins = np.histogram(resampled_data, q_bins_WHAM, density=True)
                dx = np.diff(bins)
                F1 = 1 - np.cumsum(n) * dx
                xdata = bins[1:]
                ydata = F1

                self.alldata.append(xdata)
                self.allydata.append(ydata)

            if len(self.alldata) == 0:
                continue

            input_df = pd.DataFrame(data=np.array(self.allydata).T, index=self.alldata[0])
            wham = paths.numerics.WHAM(cutoff=cutoff, tol=tol)
            wham.prep_reverse_cumulative(input_df)
            result = wham.wham_bam_histogram(input_df)
            bootstrap_results.append(result)

        bootstrap_array = np.array(bootstrap_results)
        self.bootstrap_mean = np.mean(bootstrap_array, axis=0)
        self.bootstrap_std = np.std(bootstrap_array, axis=0)




import pickle

class InterfaceData:
    def __init__(self, descriptors, weights, shot_results):
        self.descriptors = descriptors
        self.weights = weights
        self.shot_results = shot_results

class DataStore:
    def __init__(self):
        self.data = {}

    def add_interface_data(self, mode, interface_name, descriptors, weights, shot_results):
        if mode not in self.data:
            self.data[mode] = {}
        self.data[mode][interface_name] = InterfaceData(descriptors, weights, shot_results)

    def save_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.data, file)

    @classmethod
    def load_from_file(cls, filename):
        instance = cls()
        with open(filename, 'rb') as file:
            instance.data = pickle.load(file)
        return instance
