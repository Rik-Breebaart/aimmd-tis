"""
This script will contain the relevant training functions and attributes for TIS-AIMMD training.
This includes the different loss functions 
train decision 

snapshot loss for visualization of loss distribution

"""
import torch
import numpy as np
import matplotlib.pyplot as plt



""" Loss functions

"""


def q_histogram_plot(model, trainset):
    q_output = model.log_prob(trainset.descriptors, use_transform=False, batch_size=None)
    loss_per_snapshot = snapshot_loss_normalized_q(torch.as_tensor(q_output), torch.as_tensor(trainset.weights), torch.as_tensor(trainset.shot_results))
    fig, ax = plt.subplots(figsize=(8, 4))
    n, bins, patches = ax.hist(q_output[:,0], bins=np.arange(-26,26,0.5), weights=loss_per_snapshot)
    plt.xlabel(r"$q$")
    plt.title(r"Histogram of $q$ model output weighted by the loss")
    plt.show()

# def q_histogram_plot(model, trainset):
#     q_output = model.log_prob(trainset.descriptors, use_transform=False, batch_size=None)
#     loss_per_snapshot = snapshot_loss_original(torch.as_tensor(q_output), torch.as_tensor(trainset.weights), torch.as_tensor(trainset.shot_results))
#     fig, ax = plt.subplots(figsize=(8, 4))
#     n, bins, patches = ax.hist(q_output[:,0], bins=np.arange(-26,26,0.5), weights=loss_per_snapshot)
#     plt.xlabel(r"$q$")
#     plt.title(r"Histogram of $q$ model output weighted by the loss")
#     plt.show()



def train_function(model, trainset, testset, n_epochs= 10, batch_size=4096, display_committor_every= 0, toy_aimmd_visualizer=None, shuffle=True, plot_loss = False, stopping_criteria=0.01):
    loss_train = []
    loss_test = []
    if toy_aimmd_visualizer!=None:
        toy_aimmd_visualizer.display_committor(model,n_epoch=0)
        toy_aimmd_visualizer.display_q_space(model, n_epoch=0)
        q_histogram_plot(model, trainset)
        q_histogram_plot(model, testset)
    stopping_counter=0
    # Assuming optimizer has two groups.
    N_epoch_lr = 30
    lambda_group1 = lambda epoch: 1 if epoch < N_epoch_lr else 0.95** (epoch - N_epoch_lr + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(model.optimizer, lr_lambda=[lambda_group1])
    for epoch in range(n_epochs):
        print("Starting epoch: {}".format(epoch))
        loss_step = model.train_epoch(trainset,batch_size=batch_size, shuffle=True)
        loss_train.append(loss_step)

        if display_committor_every !=0 and  toy_aimmd_visualizer!=None:
            if epoch%display_committor_every==0:
                toy_aimmd_visualizer.display_committor(model, n_epoch=epoch+1)
                toy_aimmd_visualizer.display_q_space(model, n_epoch=epoch+1)
                q_histogram_plot(model, trainset)
                q_histogram_plot(model, testset)
        
        loss_test_step = model.test_loss(testset, batch_size=4096)
        print("loss test step: {}".format(loss_test_step))
        loss_test.append(loss_test_step)
        if epoch>1:
            if abs(loss_test[epoch-1]-loss_test[epoch])<stopping_criteria:
                stopping_counter +=1
            else :
                stopping_counter = 0
            if stopping_counter==5:
                if toy_aimmd_visualizer!=None:
                    toy_aimmd_visualizer.display_committor(model, n_epoch=epoch+1)
                    toy_aimmd_visualizer.display_q_space(model, n_epoch=epoch+1)
                    q_histogram_plot(model, trainset)
                    q_histogram_plot(model, testset)
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
    return loss_train, loss_test





"""
Snapshots loss functions
"""

def snapshot_loss_original(q_output, weights_tensor, shot_results_tensor):
    """
    Loss for a binomial process.

    target - dictionary containing shooting point properties and NN output,
             keys are as in Properties

    NOTE: This is NOT normalized.
    """
    q = q_output
    shots = shot_results_tensor
    weights = weights_tensor
    t1 = shots[:, 0]*torch.log(1. + torch.exp(q[:, 0]))
    t2 = shots[:, 1]*torch.log(1. + torch.exp(-q[:, 0]))
    zeros = torch.zeros_like(t1)
    return weights*(torch.where(shots[:, 0] == 0, zeros, t1)
                       + torch.where(shots[:, 1] == 0, zeros, t2))

def snapshot_loss_normalized_q(q_output, weights_tensor, shot_results_tensor):
    """
    Loss for a binomial process.

    target - dictionary containing shooting point properties and NN output,
             keys are as in Properties

    NOTE: This is NOT normalized.
    """
    q = q_output
    # what if we project everything onto q (the reaction coordinate)
    q_detached = q.detach().numpy()
    H_q, q_bins_high_res = np.histogram(q_detached,
                            bins=100, density=False)

    q_bin_indices = np.digitize(q_detached, q_bins_high_res[:-1])-1

    # print("length of q in the loss function", len(q))
    weights = weights_tensor.float()
    shots = shot_results_tensor
    print(H_q)
    q_norm = torch.tensor(np.nan_to_num(1/H_q[q_bin_indices])).float()
    print(q_norm.size())
    weights =torch.where(torch.abs(q[:,0])<30, weights.dot(q_norm[:,0]),0.)
    # weights =target[Properties.weights]
    zeros = torch.zeros_like(q[:,0])
    q_limit = 4
    #using the limit cases
    exp_q = torch.exp(torch.clamp(q[:,0],-30,30))
    exp_minq = torch.exp(torch.clamp(-q[:,0],-30,30))

    t1 = torch.where(q[:,0]<-q_limit, exp_q, zeros) \
        + torch.where(q[:,0]>q_limit, q[:,0], zeros) \
        + torch.where(torch.abs(q[:,0])<=q_limit, torch.log(1. + exp_q), zeros) 
    
    t2 = torch.where(q[:,0]<-q_limit, -q[:,0], zeros) \
    + torch.where(q[:,0]>q_limit, exp_minq, zeros) \
    + torch.where(torch.abs(q[:,0])<=q_limit, torch.log(1. + exp_minq), zeros) 

    return weights*(torch.where(shots[:, 0] == 0, zeros, t1)
                       + torch.where(shots[:, 1] == 0, zeros, t2))


def snapshot_loss_normalized_q(q_output, weights_tensor, shot_results_tensor):
    """
    Loss for a binomial process.

    target - dictionary containing shooting point properties and NN output,
             keys are as in Properties

    NOTE: This is NOT normalized.
    """
    q = q_output
    # what if we project everything onto q (the reaction coordinate)
    q_detached = q.detach().numpy()
    H_q, q_bins_high_res = np.histogram(q_detached,
                            bins=100, density=False)

    q_bin_indices = np.digitize(q_detached, q_bins_high_res[:-1])-1

    # print("length of q in the loss function", len(q))
    weights = weights_tensor.float()
    shots = shot_results_tensor
    print(H_q)
    q_norm = torch.tensor(np.nan_to_num(1/H_q[q_bin_indices])).float()
    print(q_norm.size())
    print(q_norm)
    weights =weights.dot(q_norm[:,0])
    # weights =target[Properties.weights]
    shots = shot_results_tensor
    t1 = shots[:, 0]*torch.log(1. + torch.exp(q[:, 0]))
    t2 = shots[:, 1]*torch.log(1. + torch.exp(-q[:, 0]))
    zeros = torch.zeros_like(t1)
    return weights*(torch.where(shots[:, 0] == 0, zeros, t1)
                       + torch.where(shots[:, 1] == 0, zeros, t2))


def snapshot_lnP(q_output, shot_results_tensor):
    """
    Loss for a binomial process.

    target - dictionary containing shooting point properties and NN output,
             keys are as in Properties

    NOTE: This is NOT normalized.
    """
    q = q_output
    shots = shot_results_tensor
    t1 = shots[:, 0]*torch.log(1. + torch.exp(q[:, 0]))
    t2 = shots[:, 1]*torch.log(1. + torch.exp(-q[:, 0]))
    zeros = torch.zeros_like(t1)
    return (torch.where(shots[:, 0] == 0, zeros, t1)
                       + torch.where(shots[:, 1] == 0, zeros, t2))

def snapshot_loss_low_q_scaled(q_output, weights_tensor, shot_results_tensor):
    """
    Loss for a binomial process.

    target - dictionary containing shooting point properties and NN output,
             keys are as in Properties

    NOTE: This is NOT normalized.
    """
    
    q = q_output
    shots = shot_results_tensor
    a = 3
    with torch.no_grad():
        scaling = 1 + 9*torch.exp(-q[:, 0]**2 / a**2)
    weights = weights_tensor*scaling
    t1 = shots[:, 0]*torch.log(1. + torch.exp(q[:, 0]))
    t2 = shots[:, 1]*torch.log(1. + torch.exp(-q[:, 0]))
    zeros = torch.zeros_like(t1)
    return weights*(torch.where(shots[:, 0] == 0, zeros, t1)
                       + torch.where(shots[:, 1] == 0, zeros, t2))

def snapshot_loss_sqrt_rho_weight(q_output, weights_tensor, shot_results_tensor):
    """
    Loss for a binomial process.

    target - dictionary containing shooting point properties and NN output,
             keys are as in Properties

    NOTE: This is NOT normalized.
    """
    
    q = q_output
    shots = shot_results_tensor
    a = 3
    with torch.no_grad():
            H_q, q_bins = torch.histogram(q_output, bins=100, weight=weights_tensor)
            q_bin_indices = torch.bucketize(q_output, q_bins[:-1],right=True)-1
            scaling = torch.tensor(torch.nan_to_num(1/torch.sqrt(H_q[q_bin_indices]))[:,0]).float()
    weights = weights_tensor*scaling
    t1 = shots[:, 0]*torch.log(1. + torch.exp(q[:, 0]))
    t2 = shots[:, 1]*torch.log(1. + torch.exp(-q[:, 0]))
    zeros = torch.zeros_like(t1)
    return weights*(torch.where(shots[:, 0] == 0, zeros, t1)
                       + torch.where(shots[:, 1] == 0, zeros, t2))