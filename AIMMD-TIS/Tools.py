""" 
This script will for now include all tools which are usefull but do not have a specific place.
Such as:

Looking through the directory to find specific files 
Determining from a stbale state run what the q-values inside the state are such that the next interfaces can be choosen.
"""
import numpy as np 
import torch 
from torch.optim.lr_scheduler import LambdaLR
import json
import matplotlib.pyplot as plt
from pathlib import Path
from .Toy_potentials import potential_0, potential_1,potential_2,potential_3,potential_4_linear,potential_5_Z_pot, potential_linear_q, potential_WQ, potential_Face


def train_function(model, trainset, testset, n_epochs= 10, batch_size=10000, display_committor_every= 0, plotter=None, shuffle=True, plot_loss = False, stopping_criteria=0.01, theoretical_committor_path=None, N_epoch_lr=50):
    loss_train = []
    loss_test = []
    lr_train = []
    if plotter!=None:
        plotter.model_projections(model)
        plotter.model_loss_plnp_weight_allong_q(model)
    stopping_counter=0

        # Assuming optimizer has two groups.
    lambda_group1 = lambda epoch: 1 if epoch < N_epoch_lr else 0.95** (epoch - N_epoch_lr + 1)
    scheduler = LambdaLR(model.optimizer, lr_lambda=[lambda_group1])
    for epoch in range(n_epochs):
        train, new_lr, epochs, batch_size = model.train_decision(trainset)
        if train:
            lr_train.append(model.optimizer.param_groups[0]['lr'])
            loss_step = model.train_epoch(trainset,batch_size=batch_size, shuffle=True)
            loss_train.append(loss_step)
            print("epoch: {} with loss_step {}".format(epoch,loss_step))

        if display_committor_every !=0 and  plotter!=None:
            if epoch%display_committor_every==0:
                plotter.model_projections(model)
                plotter.model_loss_plnp_weight_allong_q(model)
                
        loss_test_step = model.test_loss(testset,batch_size=batch_size)
        print("epoch: {} with test loss {}".format(epoch,loss_test_step))
        loss_test.append(loss_test_step)
        if epoch>1:
            if abs(loss_test[epoch-1]-loss_test[epoch])<stopping_criteria:
                stopping_counter +=1
            else :
                stopping_counter = 0
            if stopping_counter==5:
                if plotter!=None:
                    plotter.model_projections(model)
                    plotter.model_loss_plnp_weight_allong_q(model)
                print("stopping criteria reached.")
                break
        scheduler.step()
    if plot_loss:
        plt.figure()
        plt.plot(loss_train, label="train loss")
        plt.plot(loss_test, label="test loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.show()
    return loss_train, loss_test, lr_train

# def train_function_(aimmd_store, model,trainset, testset, n_epochs= 10, batch_size=10000, display_committor_every= 0, plotter=None, shuffle=True, plot_loss = False, stopping_criteria=0.01, theoretical_committor_path=None, N_epoch_lr=1000):
#     # stop if we reach either:
#     max_epochs = 100000  # maximum number of training epochs overall
#     max_epochs_sans_improvement = 1000  # maximum number of training epochs without a decrease in test loss

#     batch_size = 128  # batch_size=None results in batches of the size of the trainset
#     #batch_size = 256

#     test_losses = []
#     test_losses.append(model.test_loss(testset, batch_size=batch_size))
#     min_loss = test_losses[0]
#     train_losses = []
#     train = True
#     no_decrease = []
#     i = 0

#     while train and i <= max_epochs:
#         train_losses.append(model.train_epoch(trainset, batch_size=batch_size))
#         test_losses.append(model.test_loss(testset, batch_size=batch_size))

#         if test_losses[-1] <= min_loss:
#             min_loss = test_losses[-1]
#             no_decrease = []
#             aimmd_store.rcmodels["best_model_pptrain"] = model  # always store the model with the best test loss
#         else:
#             no_decrease.append(1)
#             if sum(no_decrease) >= max_epochs_sans_improvement:
#                 train = False

#         i += 1

def combined_train_function_l1_regularized(aimmd_store, model, trainset, testset, max_epochs=10000, 
                            batch_size=10000, display_committor_every=0, plotter=None, 
                            plot_loss=False, stopping_criteria=None, N_epoch_lr=300, 
                            max_epochs_sans_improvement=300):

    # Initialize loss tracking lists
    loss_train = []
    loss_test = []
    lr_train = []
    
    # For early stopping
    min_loss = None
    no_decrease = []
    
    # Display initial model projections if plotter is provided
    if plotter is not None:
        plotter.model_projections(model)
        plotter.model_loss_plnp_weight_allong_q(model)
    
    stopping_counter = 0

    # Set up learning rate scheduler
    lambda_group1 = lambda epoch: 1 if epoch < N_epoch_lr else 0.95 ** (epoch - N_epoch_lr + 1)
    scheduler = LambdaLR(model.optimizer, lr_lambda=[lambda_group1])

    for epoch in range(max_epochs):
        # Train the model on the training set
        lr_train.append(model.optimizer.param_groups[0]['lr'])
        loss_step = model.train_epoch_smoothness(trainset, batch_size=batch_size, shuffle=True)

        loss_train.append(loss_step)
        print(f"Epoch {epoch}: Training loss = {loss_step}")

        # Display committor-related plots periodically if plotter is provided
        if display_committor_every != 0 and plotter is not None and epoch % display_committor_every == 0:
            plotter.model_projections(model)
            plotter.model_loss_plnp_weight_allong_q(model)

        # Evaluate test loss
        loss_test_step = model.test_loss_smoothness(testset, batch_size=batch_size)
        print(f"Epoch {epoch}: Test loss = {loss_test_step}")
        loss_test.append(loss_test_step)

        # Update the best model based on test loss
        if min_loss is None or loss_test_step <= min_loss:
            min_loss = loss_test_step
            no_decrease = []
            aimmd_store.rcmodels["best_model_RPEtrain"] = model  # store best model
        else:
            no_decrease.append(1)

        # Early stopping based on test loss improvement
        if sum(no_decrease) >= max_epochs_sans_improvement:
            print("Stopping early due to no improvement in test loss.")
            break

        # Check for stopping criteria based on loss difference
        if epoch > 1 and stopping_criteria is not None:
            if abs(loss_test[epoch - 1] - loss_test[epoch]) < stopping_criteria:
                stopping_counter += 1
            else:
                stopping_counter = 0
            if stopping_counter == max_epochs_sans_improvement:
                print("Stopping criteria reached based on test loss convergence.")
                break

        # Step the learning rate scheduler
        scheduler.step()

    # Plot the loss if plot_loss is True
    if plot_loss:
        plt.figure()
        plt.plot(loss_train, label="Training loss")
        plt.plot(loss_test, label="Test loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    return loss_train, loss_test, lr_train



def combined_train_function(aimmd_store, model, trainset, testset, max_epochs=10000, batch_size=10000, 
                            display_committor_every=0, plotter=None, plot_loss=False, 
                            stopping_criteria=0.01, N_epoch_lr=300, max_epochs_sans_improvement=300):

    # Initialize loss tracking lists
    loss_train = []
    loss_test = []
    lr_train = []
    
    # For early stopping
    min_loss = None
    no_decrease = []
    
    # Display initial model projections if plotter is provided
    if plotter is not None:
        plotter.model_projections(model)
        plotter.model_loss_plnp_weight_allong_q(model)
    
    stopping_counter = 0

    # Set up learning rate scheduler
    lambda_group1 = lambda epoch: 1 if epoch < N_epoch_lr else 0.95 ** (epoch - N_epoch_lr + 1)
    scheduler = LambdaLR(model.optimizer, lr_lambda=[lambda_group1])

    for epoch in range(max_epochs):
        # Train the model on the training set
        lr_train.append(model.optimizer.param_groups[0]['lr'])
        loss_step = model.train_epoch(trainset, batch_size=batch_size, shuffle=True)
        loss_train.append(loss_step)
        print(f"Epoch {epoch}: Training loss = {loss_step}")

        # Display committor-related plots periodically if plotter is provided
        if display_committor_every != 0 and plotter is not None and epoch % display_committor_every == 0:
            plotter.model_projections(model)
            plotter.model_loss_plnp_weight_allong_q(model)

        # Evaluate test loss
        loss_test_step = model.test_loss(testset, batch_size=batch_size)
        print(f"Epoch {epoch}: Test loss = {loss_test_step}")
        loss_test.append(loss_test_step)

        # Update the best model based on test loss
        if min_loss is None or loss_test_step <= min_loss:
            min_loss = loss_test_step
            no_decrease = []
            aimmd_store.rcmodels["best_model_RPEtrain"] = model  # store best model
        else:
            no_decrease.append(1)

        # Early stopping based on test loss improvement
        if sum(no_decrease) >= max_epochs_sans_improvement:
            print("Stopping early due to no improvement in test loss.")
            break

        # Check for stopping criteria based on loss difference
        if epoch > 1:
            if abs(loss_test[epoch - 1] - loss_test[epoch]) < stopping_criteria:
                stopping_counter += 1
            else:
                stopping_counter = 0
            if stopping_counter == max_epochs_sans_improvement:
                print("Stopping criteria reached based on test loss convergence.")
                break

        # Step the learning rate scheduler
        scheduler.step()

    # Plot the loss if plot_loss is True
    if plot_loss:
        plt.figure()
        plt.plot(loss_train, label="Training loss")
        plt.plot(loss_test, label="Test loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    return loss_train, loss_test, lr_train

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

def potential_switch(potential_name, settings):
    "Switching function to select one of the toy potentials"
    if potential_name == "potential_0":
        pes = potential_0(A=settings["A"], x0=settings["x0"], n_harmonics=settings["n_harmonics"])
    elif potential_name == "potential_1":
        pes = potential_1(n_harmonics=settings["n_harmonics"])
    elif potential_name == "potential_2":
        pes = potential_2(n_harmonics=settings["n_harmonics"])
    elif potential_name == "potential_3":
        pes = potential_3(n_harmonics=settings["n_harmonics"])
    elif potential_name  == "potential_4":
        pes = potential_4_linear(n_harmonics=settings["n_harmonics"])
    elif potential_name == "z-potential":
        pes = potential_5_Z_pot(n_harmonics=settings["n_harmonics"])
    elif potential_name == "linear_q":
        pes = potential_linear_q(n_harmonics=settings["n_harmonics"])
    elif potential_name == "wolfe-quapp":
        if "rotation_degrees" in settings and "scale" in settings:
            pes = potential_WQ(n_harmonics=settings["n_harmonics"], rotation_degrees=settings["rotation_degrees"], scale=settings["scale"])
        else :
            pes = potential_WQ(n_harmonics=settings["n_harmonics"])
    elif potential_name == "FacePotential":
        if "rotation_degrees" in settings and "scale" in settings:
            pes = potential_Face(n_harmonics=settings["n_harmonics"], rotation_degrees=settings["rotation_degrees"], scale=settings["scale"])
        else :
            pes = potential_Face(n_harmonics=settings["n_harmonics"])
    else :
        ValueError("unknown potential name given, choose from either: 'potential_0', 'potential_1', 'potential_2', 'potential_3', 'potential_4', 'linear_q', 'wolfe-quapp' or 'z-potential'. ")

    return pes


def train_test_split(data,train_size, test_size, shuffle_indexs):
    if len(data)==len(shuffle_indexs):
        data_shuffled = data[shuffle_indexs]
    else:
        ValueError("Incorrect shuffle_index array given")
    if len(data_shuffled)==(train_size+test_size):
        return data_shuffled[:train_size], data_shuffled[train_size:train_size+test_size]


def create_train_test_split( descriptors, weights, shot_results, split = [4,1]):
    total_dataset_size = len(weights)
    ratio_train = split[0]/sum(split)
    train_size = int(np.floor(ratio_train*total_dataset_size))
    test_size = total_dataset_size-train_size
    shuffle_rng_index = np.random.permutation(total_dataset_size)

    weights_train, weights_test = train_test_split(weights, train_size, test_size, shuffle_rng_index)
    descriptors_train, descriptors_test = train_test_split(descriptors, train_size, test_size, shuffle_rng_index)
    shot_results_train, shot_results_test = train_test_split(shot_results, train_size, test_size, shuffle_rng_index)
    trainset = descriptors_train, shot_results_train, weights_train,
    testset = descriptors_test, shot_results_test ,weights_test
    return trainset, testset

class SyntheticDataGenerator:
    def __init__(self, potential_grid, p_B, pes, beta):
        self.potential_grid = potential_grid
        self.p_B = p_B
        self.n_x, self.n_y = p_B.shape
        self.pes_dim = pes.n_dims_pot + pes.n_harmonics
        self.range_pes = [[pes.extent[0],pes.extent[1]],[pes.extent[2],pes.extent[3]]]
        self.beta = beta

    def boltzmann_distribution(self, potential):
        # Boltzmann distribution
        return np.exp(-self.beta * potential)

    def generate_data(self, num_points):
        synthetic_data = []
        labels = []
        positions = np.zeros((num_points,self.pes_dim))
        shot_result = np.zeros((num_points,2))

        # Generate random points
        random_x = np.random.uniform(0, self.n_x, num_points)
        random_y = np.random.uniform(0, self.n_y, num_points)
        x_int = np.floor(random_x).astype(int)
        y_int = np.floor(random_y).astype(int)
        random_u = np.random.uniform(0,1,num_points)
        i=0
        
        # Probability of going to B based on p_B grid
        p_b = self.p_B[x_int, y_int]

        # Assign label based on probability
        shot_result[:,0] =  (random_u>= p_b)
        shot_result[:,1] =  (random_u< p_b)

        potential = self.potential_grid[x_int, y_int]
        weights = self.boltzmann_distribution(potential)
        positions[:,0] = (random_x/self.n_x)*(self.range_pes[0][1]-self.range_pes[0][0])+self.range_pes[0][0]
        positions[:,1] = (random_y/self.n_y)*(self.range_pes[1][1]-self.range_pes[1][0])+self.range_pes[1][0]
        positions[:,2:] = np.random.normal((self.pes_dim-2))

        return positions, shot_result, weights

def interface_indicator(interface_value):
    return int(np.floor(interface_value*100))

def figure_storage_path(filename, suffix=".pdf",output_path=None):
    return Path(output_path / filename).with_suffix(suffix)


def save_fig_pdf_and_png(fig, filename, output_path=None):
    fig.savefig(figure_storage_path(filename, suffix=".pdf", output_path=output_path))
    fig.savefig(figure_storage_path(filename, suffix=".png", output_path=output_path))

def create_discrete_cmap(n_color_steps=20):
    cmap = plt.cm.Spectral  # define the colormap
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
    q_total = model.log_prob(descriptors,use_transform=False,batch_size=None)

    min_max_stable_q = np.zeros((len(stable_states),2))
    # Define `in_state` as a mask array if not provided
    in_state_arg = False
    if in_state is None:
        in_state = np.zeros_like(shot_results)
        in_state_arg = True
    # Calculate min and max values for each stable state based on `in_state`
    for state in range(len(stable_states)):
        if in_state_arg:
            if weights is not None and shot_results is not None:
                in_state[:,state] = (weights * shot_results[:,state]) > 0
            else:
                raise ValueError("Provide `in_state` or both `weights` and `shot_results` to calculate it.")
            
        state_mask = in_state[:, state].astype(bool)
        min_max_stable_q[state, 0] = np.min(q_total[state_mask])
        min_max_stable_q[state, 1] = np.max(q_total[state_mask])
        
        print(f"stable_state {stable_states[state]} data falls in q-range {min_max_stable_q[state]}") 


    interface_distance = 0.25
    if np.sum(np.isnan(min_max_stable_q)) == 0 :
        # Forward_interfaces = np.arange(ceil_decimal(min_max_stable_q[0,1],decimals=2),interface_distance,interface_distance)
        # Backward_interfaces = np.arange(0,floor_decimal(min_max_stable_q[1,0],decimals=2)+interface_distance,interface_distance)
        Forward_interfaces = interfaces_q_space(ceil_decimal(min_max_stable_q[0,1],decimals=2),overlap,direction="forward")
        Backward_interfaces = interfaces_q_space(floor_decimal(min_max_stable_q[1,0],decimals=2),overlap,direction="backward")

        print("Forward interfaces: {}".format(' '.join(map(str, Forward_interfaces.flatten()))))
        print("Backward interfaces: {}".format(' '.join(map(str, Backward_interfaces.flatten()))))
        return Forward_interfaces, Backward_interfaces
    return None,None

def model_to(model,mode="cuda"):
    model.nnet = model.nnet.to(mode)
    model._device = mode
    model.device= mode 
    return model
