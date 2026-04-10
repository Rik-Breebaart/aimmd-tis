import numpy as np
import torch
import matplotlib.pyplot as plt


def q_histogram_plot(model, trainset):
    q_output = model.log_prob(trainset.descriptors, use_transform=False, batch_size=None)
    loss_per_snapshot = snapshot_loss_normalized_q(
        torch.as_tensor(q_output),
        torch.as_tensor(trainset.weights),
        torch.as_tensor(trainset.shot_results),
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(q_output[:, 0], bins=np.arange(-26, 26, 0.5), weights=loss_per_snapshot)
    plt.xlabel(r"$q$")
    plt.title(r"Histogram of $q$ model output weighted by the loss")
    plt.show()


def snapshot_loss_original(q_output, weights_tensor, shot_results_tensor):
    q = q_output
    shots = shot_results_tensor
    weights = weights_tensor
    t1 = shots[:, 0] * torch.log(1.0 + torch.exp(q[:, 0]))
    t2 = shots[:, 1] * torch.log(1.0 + torch.exp(-q[:, 0]))
    zeros = torch.zeros_like(t1)
    return weights * (
        torch.where(shots[:, 0] == 0, zeros, t1)
        + torch.where(shots[:, 1] == 0, zeros, t2)
    )


def snapshot_loss_smoothness(
    q_pred: torch.Tensor,
    descriptors: torch.Tensor,
    reduction: str = "none",
):
    if not descriptors.requires_grad:
        descriptors.requires_grad_(True)

    q = q_pred.view(-1, 1)
    grad = torch.autograd.grad(
        outputs=q,
        inputs=descriptors,
        grad_outputs=torch.ones_like(q),
        create_graph=False,
        retain_graph=False,
    )[0]

    grad_sq = grad.abs() ** 2
    out = grad_sq.mean(dim=1)

    if reduction == "mean":
        out = out.mean()
    elif reduction == "sum":
        out = out.sum()
    elif reduction != "none":
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    return out.detach()


def snapshot_loss_normalized_q(q_output, weights_tensor, shot_results_tensor):
    q = q_output
    q_detached = q.detach().numpy()
    H_q, q_bins_high_res = np.histogram(q_detached, bins=100, density=False)
    q_bin_indices = np.digitize(q_detached, q_bins_high_res[:-1]) - 1

    weights = weights_tensor.float()
    shots = shot_results_tensor
    q_norm = torch.tensor(np.nan_to_num(1 / H_q[q_bin_indices])).float()
    weights = weights.dot(q_norm[:, 0])

    t1 = shots[:, 0] * torch.log(1.0 + torch.exp(q[:, 0]))
    t2 = shots[:, 1] * torch.log(1.0 + torch.exp(-q[:, 0]))
    zeros = torch.zeros_like(t1)
    return weights * (
        torch.where(shots[:, 0] == 0, zeros, t1)
        + torch.where(shots[:, 1] == 0, zeros, t2)
    )


def snapshot_lnP(q_output, shot_results_tensor):
    q = q_output
    shots = shot_results_tensor
    t1 = shots[:, 0] * torch.log(1.0 + torch.exp(q[:, 0]))
    t2 = shots[:, 1] * torch.log(1.0 + torch.exp(-q[:, 0]))
    zeros = torch.zeros_like(t1)
    return torch.where(shots[:, 0] == 0, zeros, t1) + torch.where(shots[:, 1] == 0, zeros, t2)


def snapshot_loss_low_q_scaled(q_output, weights_tensor, shot_results_tensor):
    q = q_output
    shots = shot_results_tensor
    a = 3
    with torch.no_grad():
        scaling = 1 + 9 * torch.exp(-(q[:, 0] ** 2) / a**2)
    weights = weights_tensor * scaling
    t1 = shots[:, 0] * torch.log(1.0 + torch.exp(q[:, 0]))
    t2 = shots[:, 1] * torch.log(1.0 + torch.exp(-q[:, 0]))
    zeros = torch.zeros_like(t1)
    return weights * (
        torch.where(shots[:, 0] == 0, zeros, t1)
        + torch.where(shots[:, 1] == 0, zeros, t2)
    )


def snapshot_loss_sqrt_rho_weight(q_output, weights_tensor, shot_results_tensor):
    q = q_output
    shots = shot_results_tensor
    with torch.no_grad():
        H_q, q_bins = torch.histogram(q_output, bins=100, weight=weights_tensor)
        q_bin_indices = torch.bucketize(q_output, q_bins[:-1], right=True) - 1
        scaling = torch.tensor(torch.nan_to_num(1 / torch.sqrt(H_q[q_bin_indices]))[:, 0]).float()
    weights = weights_tensor * scaling
    t1 = shots[:, 0] * torch.log(1.0 + torch.exp(q[:, 0]))
    t2 = shots[:, 1] * torch.log(1.0 + torch.exp(-q[:, 0]))
    zeros = torch.zeros_like(t1)
    return weights * (
        torch.where(shots[:, 0] == 0, zeros, t1)
        + torch.where(shots[:, 1] == 0, zeros, t2)
    )
