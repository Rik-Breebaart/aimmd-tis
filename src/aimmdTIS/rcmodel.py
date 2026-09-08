"""TIS-specific PyTorch reaction-coordinate model extensions."""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from aimmd.base import Properties
from aimmd.base.utils import get_batch_size_from_model_and_descriptors
from aimmd.pytorch import PytorchRCModel

from .stochastic_gates import StochasticGates
from .train_decision import (
    EE_SCALE_CUTOFF_DEFAULTS,
    EE_SCALE_CUTOFF_DOC,
    train_decision_ee_scale_cutoff,
)


def binomial_cutoff_loss(target):
    """
    Loss for a binomial process.
    With cut-off at q=30 and using taylor expansions of binomial loss for 
    high and low q (above q=4).

    target - dictionary containing shooting point properties and NN output,
             keys are as in Properties

    NOTE: This is NOT normalized.
    """
    q = target[Properties.q]
    # print("length of q in the loss function", len(q))
    shots = target[Properties.shot_results]
    # weights =torch.where(torch.abs(q[:,0])<45, target[Properties.weights],0.)
    weights =target[Properties.weights]

    # weights =target[Properties.weights]
    zeros = torch.zeros_like(q[:,0])
    q_limit = 10
    #using the limit cases
    exp_q = torch.exp(torch.clamp(q[:,0],-45,45))
    exp_minq = torch.exp(torch.clamp(-q[:,0],-45,45))

    t1 = torch.where(q[:,0]<-q_limit, exp_q, zeros) \
        + torch.where(q[:,0]>q_limit, q[:,0], zeros) \
        + torch.where(torch.abs(q[:,0])<=q_limit, torch.log(1. + exp_q), zeros) 
    
    t2 = torch.where(q[:,0]<-q_limit, -q[:,0], zeros) \
    + torch.where(q[:,0]>q_limit, exp_minq, zeros) \
    + torch.where(torch.abs(q[:,0])<=q_limit, torch.log(1. + exp_minq), zeros) 

    return weights.dot(torch.where(shots[:, 0] == 0, zeros, t1)
                       + torch.where(shots[:, 1] == 0, zeros, t2))

def binomial_limit_loss(target):
    """
    Loss for a binomial process.
    With cut-off at q=30 and using taylor expansions of binomial loss for 
    high and low q (above q=4).

    target - dictionary containing shooting point properties and NN output,
             keys are as in Properties
    """
    q = target[Properties.q]
    shots = target[Properties.shot_results]
    weights = target[Properties.weights]
    q_limit = 10
    exp_q = torch.exp(q[:,0])
    exp_minq = torch.exp(-q[:,0])
    zeros = torch.zeros_like(q[:,0])
    t1 = torch.where(q[:,0]<-q_limit, exp_q,zeros) \
        + torch.where(q[:,0]>q_limit, q[:,0], zeros) \
        + torch.where(torch.abs(q[:,0])<=q_limit, torch.log(1. + torch.exp(q[:,0])), zeros) 
    
    t2 = torch.where(q[:,0]<-q_limit, -q[:,0], zeros) \
    + torch.where(q[:,0]>q_limit, exp_minq, zeros) \
    + torch.where(torch.abs(q[:,0])<=q_limit, torch.log(1. + exp_minq), zeros) 
    return weights.dot(torch.where(shots[:, 0] == 0, zeros, t1)
                       + torch.where(shots[:, 1] == 0, zeros, t2))

def binomial_loss_softplus(target):
    q = target[Properties.q][:, 0]        # logits
    shots = target[Properties.shot_results]
    weights = target[Properties.weights]

    # softplus is numerically stable version of log(1+exp(.))
    t1 = shots[:, 0] *F.softplus(q)   # corresponds to log(1 + exp(q))
    t2 = shots[:, 1] *F.softplus(-q)  # corresponds to log(1 + exp(-q))

    loss_terms = torch.where(shots[:, 0] == 0, torch.zeros_like(q), t1) \
               + torch.where(shots[:, 1] == 0, torch.zeros_like(q), t2)
    if weights.device.type == "mps":
        weighted_loss = (weights * loss_terms).sum()
    else:
        weighted_loss = (weights.double() * loss_terms.double()).sum().float()

    return weighted_loss


def binomial_loss_softplus_downscaled(target):
    q = target[Properties.q][:, 0]        # logits
    shots = target[Properties.shot_results]
    weights = target[Properties.weights]

    # softplus is numerically stable version of log(1+exp(.))
    t1 = shots[:, 0] *F.softplus(q)   # corresponds to log(1 + exp(q))
    t2 = shots[:, 1] *F.softplus(-q)  # corresponds to log(1 + exp(-q))

    loss_terms = torch.where(shots[:, 0] == 0, torch.zeros_like(q), t1) \
               + torch.where(shots[:, 1] == 0, torch.zeros_like(q), t2)
    if weights.device.type == "mps":
        weighted_loss = (weights * loss_terms).sum()
    else:
        weighted_loss = (weights.double() * loss_terms.double()).sum().float()
    scaling = 1e-18
    return weighted_loss * scaling

# def binomial_loss_with_smoothness_penalty(target, smoothness_penalty_weight=None):
#     q = target[Properties.q]
#     shots = target[Properties.shot_results]
#     weights = target[Properties.weights]
#     t1 = shots[:, 0] * torch.log(1. + torch.exp(q[:, 0]))
#     t2 = shots[:, 1] * torch.log(1. + torch.exp(-q[:, 0]))
#     zeros = torch.zeros_like(t1)

#     # Compute the standard loss
#     loss = weights.dot(torch.where(shots[:, 0] == 0, zeros, t1)
#                        + torch.where(shots[:, 1] == 0, zeros, t2))
    

#     # Compute the smoothness penalty: Compute gradients with respect to q
#     if smoothness_penalty_weight is not None:
#         q_grad = torch.autograd.grad(outputs=q.sum(), inputs=q, create_graph=True)[0]
#         smoothness_loss = (q_grad ** 2).mean()  # Compute smoothness penalty
#         loss += smoothness_penalty_weight * smoothness_loss
                  
#     return loss



# ### TODO: Added by Rik Breebaart for (RE)TIS-AIMMD
# def binomial_loss_uniform_q(target):
#     """
#     Loss for a binomial process.
#     Normalized in q-space, an additional weight is added to compensate for high q density regions.
    
#     target - dictionary containing shooting point properties and NN output,
#              keys are as in Properties

#     NOTE: This is NOT normalized.
#     """
#     q = target[Properties.q]
#     weights = target[Properties.weights]
#     shots = target[Properties.shot_results]
#     H_q, q_bins = torch.histogram(q.detach().cpu(), bins=100, density=True)
#     q_bin_indices = torch.bucketize(q.detach().cpu(), q_bins[:-1],right=True)-1

#     weights = target[Properties.weights]
#     shots = target[Properties.shot_results]
#     q_norm = torch.tensor(torch.nan_to_num(1/H_q[q_bin_indices])[:,0]).float().to(weights.device)
#     weights_norm = weights* q_norm 

#     zeros = torch.zeros_like(q[:,0])
#     q_limit = 10
#     exp_q = torch.exp(q[:,0])
#     exp_minq = torch.exp(-q[:,0])

#     t1 = torch.where(q[:,0]<-q_limit, exp_q, zeros) \
#         + torch.where(q[:,0]>q_limit, q[:,0], zeros) \
#         + torch.where(torch.abs(q[:,0])<=q_limit, torch.log(1. + exp_q), zeros) 
    
#     t2 = torch.where(q[:,0]<-q_limit, -q[:,0], zeros) \
#     + torch.where(q[:,0]>q_limit, exp_minq, zeros) \
#     + torch.where(torch.abs(q[:,0])<=q_limit, torch.log(1. + exp_minq), zeros) 

#     return weights_norm.dot(torch.where(shots[:, 0] == 0, zeros, t1)
#                        + torch.where(shots[:, 1] == 0, zeros, t2))



def binomial_loss_uniform_q(target):
    """
    Stable version of binomial loss with q-space histogram reweighting.
    Uses softplus for numerical stability and float64 accumulation.

    Parameters
    ----------
    target : dict
        Dictionary with Properties.q, Properties.shot_results, Properties.weights
    bins : int
        Number of bins for q-histogram
    q_clip : float
        Clamp range for logits q to avoid exp overflow
    eps : float
        Small stabilizer for dividing by histogram counts

    Returns
    -------
    torch.Tensor
        Scalar loss (float32 for backprop)
    """
    bins=100
    q = target[Properties.q][:,0]
    shots = target[Properties.shot_results]
    weights = target[Properties.weights]

    # ---- Histogram reweighting (on CPU, detached) ----
    q_cpu = q.detach().cpu()
    H_q, q_bins = torch.histogram(q_cpu, bins=bins, density=True)
    q_bin_indices = torch.bucketize(q_cpu, q_bins[:-1], right=True) - 1
    # Safe inversion with eps
    q_norm = torch.tensor(torch.nan_to_num(1/H_q[q_bin_indices])).float().to(weights.device)
    # ---- Weight adjustment ----
    weights_norm = weights * q_norm

    # ---- Numerically stable t1/t2 using softplus ----
    t1 = F.softplus(q)   # log(1+exp(q))
    t2 = F.softplus(-q)  # log(1+exp(-q))

    # ---- Loss terms ----
    loss_terms = torch.where(shots[:, 0] == 0, torch.zeros_like(q), t1) \
               + torch.where(shots[:, 1] == 0, torch.zeros_like(q), t2)

    # ---- Weighted sum in float64 ----
    if weights.device.type == "mps":
        weighted_loss = (weights_norm * loss_terms).sum()
    else:
        weighted_loss = (weights_norm.double() * loss_terms.double()).sum().float()

    return weighted_loss

def new_loss_scaled_sqrtrhoq(target):
    """
    Loss for a binomial process.
    With cut-off at q=30 and using taylor expansions of binomial loss for 
    high and low q (above q=4).
    Normalized in q-space, an additional weight is added to compensate for high q density regions.
    
    target - dictionary containing shooting point properties and NN output,
             keys are as in Properties

    NOTE: This is NOT normalized.
    """
    q = target[Properties.q]
    weights = target[Properties.weights]
    shots = target[Properties.shot_results]
    # what if we project everything onto q (the reaction coordinate)
    # q_detached = q.todetach().numpy()
    # H_q, q_bins = np.histogram(q_detached,
    #                         bins=100, density=False)
    H_q, q_bins = torch.histogram(q.detach(), bins=100, weight=weights)
    q_bin_indices = torch.bucketize(q.detach(), q_bins[:-1],right=True)-1

    # print("length of q in the loss function", len(q))
    weights = target[Properties.weights]
    shots = target[Properties.shot_results]
    q_norm = torch.tensor(torch.nan_to_num(1/torch.sqrt(H_q[q_bin_indices]))[:,0]).float()
    # q_norm = 1+9*torch.exp(-q[:,0].detach()**2/5)
    # print(q_norm.shape)
    # print(weights.shape)
    # weights_norm=weights * q_norm
    weights_norm = weights* q_norm 
    # print(weights_norm.shape)
    # weights =target[Properties.weights]
    zeros = torch.zeros_like(q[:,0])
    q_limit = 6
    #using the limit cases
    exp_q = torch.exp(torch.clamp(q[:,0],-30,30))
    exp_minq = torch.exp(torch.clamp(-q[:,0],-30,30))

    t1 = torch.where(q[:,0]<-q_limit, exp_q, zeros) \
        + torch.where(q[:,0]>q_limit, q[:,0], zeros) \
        + torch.where(torch.abs(q[:,0])<=q_limit, torch.log(1. + exp_q), zeros) 
    
    t2 = torch.where(q[:,0]<-q_limit, -q[:,0], zeros) \
    + torch.where(q[:,0]>q_limit, exp_minq, zeros) \
    + torch.where(torch.abs(q[:,0])<=q_limit, torch.log(1. + exp_minq), zeros) 

    return weights_norm.dot(torch.where(shots[:, 0] == 0, zeros, t1)
                       + torch.where(shots[:, 1] == 0, zeros, t2))

### TODO: Added by Rik Breebaart for (RE)TIS-AIMMD
def new_loss_scaled_low_q(target):
    """
    Loss for a binomial process.
    With cut-off at q=30 and using taylor expansions of binomial loss for 
    high and low q (above q=4).
    Normalized in q-space, an additional weight is added to compensate for high q density regions.
    
    target - dictionary containing shooting point properties and NN output,
             keys are as in Properties

    NOTE: This is NOT normalized.
    """
    q = target[Properties.q]
    weights = target[Properties.weights]
    shots = target[Properties.shot_results]
    # what if we project everything onto q (the reaction coordinate)
    # q_detached = q.todetach().numpy()
    # H_q, q_bins = np.histogram(q_detached,
    #                         bins=100, density=False)
    # H_q, q_bins = torch.histogram(q.detach(), bins=100, weight=weights)
    # q_bin_indices = torch.bucketize(q.detach(), q_bins[:-1],right=True)-1

    # print("length of q in the loss function", len(q))
    weights = target[Properties.weights]
    shots = target[Properties.shot_results]
    # q_norm = torch.tensor(torch.nan_to_num(1/torch.sqrt(H_q[q_bin_indices]))[:,0]).float()
    q_norm = 1+9*torch.exp(-q[:,0].detach()**2/5)
    # print(q_norm.shape)
    # print(weights.shape)
    # weights_norm=weights * q_norm
    weights_norm = weights* q_norm 
    # print(weights_norm.shape)
    # weights =target[Properties.weights]
    zeros = torch.zeros_like(q[:,0])
    q_limit = 6
    #using the limit cases
    exp_q = torch.exp(torch.clamp(q[:,0],-30,30))
    exp_minq = torch.exp(torch.clamp(-q[:,0],-30,30))

    t1 = torch.where(q[:,0]<-q_limit, exp_q, zeros) \
        + torch.where(q[:,0]>q_limit, q[:,0], zeros) \
        + torch.where(torch.abs(q[:,0])<=q_limit, torch.log(1. + exp_q), zeros) 
    
    t2 = torch.where(q[:,0]<-q_limit, -q[:,0], zeros) \
    + torch.where(q[:,0]>q_limit, exp_minq, zeros) \
    + torch.where(torch.abs(q[:,0])<=q_limit, torch.log(1. + exp_minq), zeros) 

    return weights_norm.dot(torch.where(shots[:, 0] == 0, zeros, t1)
                       + torch.where(shots[:, 1] == 0, zeros, t2))



class TIS_EEScalePytorchRCModelMixin:
    """Expected-efficiency AIMMD model with TIS regularization helpers."""

    __doc__ += EE_SCALE_CUTOFF_DOC

    def __init__(self, nnet, optimizer, states,
                 ee_params=EE_SCALE_CUTOFF_DEFAULTS,
                 descriptor_transform=None, loss=None, cache_file=None,
                 n_out=None):
        super().__init__(nnet=nnet, optimizer=optimizer, states=states,
                         descriptor_transform=descriptor_transform, loss=loss,
                         cache_file=cache_file, n_out=n_out)
        defaults = copy.deepcopy(EE_SCALE_CUTOFF_DEFAULTS)
        defaults.update(ee_params)
        self.ee_params = defaults
        self.lr = self.ee_params["lr_0"]

    train_decision = train_decision_ee_scale_cutoff

    def _effective_mass_mean(self, trainset):
        shots = torch.as_tensor(trainset.shot_results, device=self._device, dtype=torch.float64)
        weights = torch.as_tensor(trainset.weights, device=self._device, dtype=torch.float64)
        return torch.mean(weights * torch.sum(shots, dim=-1))

    def _stochastic_gates(self):
        return [module for module in self.nnet.modules() if isinstance(module, StochasticGates)]

    def gate_saturation_stats(self, threshold=1e-6):
        """Count gate features whose erf gradient has numerically vanished."""
        total = saturated = 0
        for gates in self._stochastic_gates():
            grad_mag = gates.gate_probability_gradient_magnitude()
            total += grad_mag.numel()
            saturated += int((grad_mag < threshold).sum().item())
        return saturated, total

    def _gate_regularization(self, reference):
        """Normalized L0-style sparsity penalty on the stochastic gates.

        penalty = coefficient * (E[active features] / n_features)
        i.e. coefficient times the mean gate-open probability, so the
        coefficient is in O(1) units and independent of the descriptor count
        and of the RPE weight magnitude. mu is clamped to [mu_min, 1] elsewhere,
        so the per-gate gradient never vanishes and this stays a usable knob.
        """
        expected_active = reference.new_zeros(())
        n_features = 0
        for gates in self._stochastic_gates():
            expected_active = (
                expected_active + gates.expected_active_features()
            )
            n_features += int(gates.n_in)

        coefficient = self.ee_params.get(
            "stochastic_gate_regularization", 0.0
        )
        norm = float(n_features) if n_features > 0 else 1.0
        penalty = coefficient * (expected_active / norm)
        return penalty, expected_active
    def _network_parameters_without_gates(self):
        gate_ids = {
            id(p)
            for gate in self._stochastic_gates()
            for p in gate.parameters()
        }
        return [
            p for p in self.nnet.parameters()
            if id(p) not in gate_ids
        ]
    
    def train_epoch_smoothness(self, trainset, batch_size=None, shuffle=True,
                               normalization=True, weighted_smoothness=False):
        model_raw_sum = effective_mass_sum = smooth_sum = smooth_mass = 0.0
        l1_last = gate_last = active_features_last = 0.0
        full_mass_mean = self._effective_mass_mean(trainset)

        # Boosted gradient step for the gate logits, applied once per epoch on
        # the epoch-mean mu gradient (model fit + sparsity penalty). The shared
        # optimizer runs at the small net LR -- far too slow for mu to traverse
        # [mu_min, 1] during selection -- and aimmd storage cannot round-trip a
        # model whose optimizer has a second param group, so mu gets its extra
        # rate here instead. The step keeps the true gradient magnitude (bounded
        # by gate_cap), so a feature whose model-fit gradient beats the small
        # constant sparsity gradient stays open while unused features close.
        gate_modules = self._stochastic_gates()
        gate_nudge = float(self.ee_params.get("stochastic_gate_lr", 10.0)) if gate_modules else 0.0
        gate_cap = float(self.ee_params.get("stochastic_gate_step_cap", 0.15))
        gate_grad_accum = {id(g): torch.zeros_like(g.mu) for g in gate_modules}
        gate_grad_count = 0
        nudge_on = gate_nudge > 0.0 and float(
            self.ee_params.get("stochastic_gate_regularization", 0.0)
        ) > 0.0

        for target in trainset.iter_batch(batch_size, shuffle):
            def closure():
                nonlocal model_raw_sum, effective_mass_sum, smooth_sum, smooth_mass
                nonlocal l1_last, gate_last, active_features_last
                self.optimizer.zero_grad()
                target_t = {key: torch.as_tensor(value, device=self._device, dtype=self._dtype)
                            for key, value in target.items()}
                descriptors = target_t[Properties.descriptors].requires_grad_(True)
                q_pred = self.nnet(descriptors)
                target_t[Properties.q] = q_pred
                model_raw = self.loss(target_t)
                shot_counts = torch.sum(target_t[Properties.shot_results], dim=-1).to(torch.float64)
                weights = target_t[Properties.weights].to(torch.float64)
                batch_mass = torch.sum(shot_counts * weights)
                model_raw_sum += float(model_raw.detach().cpu())
                effective_mass_sum += float(batch_mass.detach().cpu())
                batch_norm = None
                model_loss = model_raw
                if normalization:
                    batch_norm = (batch_mass / full_mass_mean).clamp(min=torch.finfo(torch.float64).eps)
                    batch_norm = batch_norm.to(model_raw.dtype)
                    model_loss = model_raw / batch_norm

                smooth_term = model_raw.new_zeros(())
                smooth_weight = self.ee_params.get("smoothness_penalty_weight", 0.0)
                if smooth_weight:
                    q_grad = torch.autograd.grad(q_pred.sum(), descriptors, create_graph=True)[0]
                    grad_sq = torch.abs(q_grad).square().sum(-1)
                    if weighted_smoothness:
                        smooth_raw = torch.sum(grad_sq.to(weights.dtype) * weights * shot_counts).to(model_raw.dtype)
                        smooth_loss = smooth_raw if batch_norm is None else smooth_raw / batch_norm
                        smooth_sum += float(smooth_raw.detach().cpu())
                        smooth_mass += float(batch_mass.detach().cpu())
                    else:
                        # Carry the same full_mass_mean scale as the weighted
                        # branch and the model term so smoothness_penalty_weight
                        # has a mode-independent meaning.
                        mass_scale = full_mass_mean.to(grad_sq.dtype) if normalization else 1.0
                        smooth_loss = grad_sq.mean() * mass_scale
                        smooth_sum += float((smooth_weight * smooth_loss).detach().cpu()) * descriptors.shape[0]
                        smooth_mass += descriptors.shape[0]
                    smooth_term = smooth_weight * smooth_loss

                l1_term = model_raw.new_zeros(())
                if self.ee_params.get("l1_regularization", 0.0):
                    l1_term = self.ee_params["l1_regularization"] * sum(
                        parameter.abs().sum() 
                        for parameter in self._network_parameters_without_gates()
                        )
                    l1_last = float(l1_term.detach().cpu())
                gate_term, expected_active = self._gate_regularization(model_raw)
                gate_last = float(gate_term.detach().cpu())
                active_features_last = float(expected_active.detach().cpu())
                loss = model_loss + smooth_term + l1_term + gate_term
                loss.backward()
                clip_norm = self.ee_params.get("max_clipping_norm")
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), max_norm=clip_norm)
                return loss
            self.optimizer.step(closure)
            with torch.no_grad():
                if nudge_on:
                    for gate in gate_modules:
                        if gate.mu.grad is not None:
                            gate_grad_accum[id(gate)] += gate.mu.grad.detach()
                    gate_grad_count += 1
                for gate in gate_modules:
                    gate.clamp_mu()

        if nudge_on and gate_grad_count > 0:
            with torch.no_grad():
                for gate in gate_modules:
                    mean_grad = gate_grad_accum[id(gate)] / gate_grad_count
                    step = (mean_grad * (-gate_nudge)).clamp_(-gate_cap, gate_cap)
                    gate.mu.add_(step)
                    gate.clamp_mu()

        model_epoch = (float(full_mass_mean.detach().cpu()) * model_raw_sum /
                       max(effective_mass_sum, np.finfo(float).eps)
                       if normalization else model_raw_sum)
        if smooth_mass:
            smooth_epoch = smooth_sum / smooth_mass
            if weighted_smoothness:
                smooth_epoch *= self.ee_params.get("smoothness_penalty_weight", 0.0)
                if normalization:
                    smooth_epoch *= float(full_mass_mean.detach().cpu())
        else:
            smooth_epoch = 0.0
        return {"total_loss": model_epoch + smooth_epoch + l1_last + gate_last,
                "model_loss": model_epoch, "smoothness_loss": smooth_epoch,
            "l1_regularization": l1_last, "stochastic_gate_regularization": gate_last,
            "expected_active_features": active_features_last}

    def test_loss_smoothness(self, trainset, batch_size=None, normalization=True,
                             weighted_smoothness=False):
        if batch_size is None:
            batch_size = get_batch_size_from_model_and_descriptors(self, trainset.descriptors)
        was_training = self.nnet.training
        self.nnet.eval()
        model_raw_sum = effective_mass_sum = smooth_sum = smooth_mass = 0.0
        l1_last = gate_last = active_features_last = 0.0
        full_mass_mean = self._effective_mass_mean(trainset)
        try:
            for target in trainset.iter_batch(batch_size, False):
                target_t = {key: torch.as_tensor(value, device=self._device, dtype=self._dtype)
                            for key, value in target.items()}
                descriptors = target_t[Properties.descriptors].requires_grad_(True)
                q_pred = self.nnet(descriptors)
                target_t[Properties.q] = q_pred
                model_raw = self.loss(target_t)
                shot_counts = torch.sum(target_t[Properties.shot_results], dim=-1).to(torch.float64)
                weights = target_t[Properties.weights].to(torch.float64)
                batch_mass = torch.sum(shot_counts * weights)
                model_raw_sum += float(model_raw.detach().cpu())
                effective_mass_sum += float(batch_mass.detach().cpu())
                smooth_weight = self.ee_params.get("smoothness_penalty_weight", 0.0)
                if smooth_weight:
                    q_grad = torch.autograd.grad(q_pred.sum(), descriptors)[0]
                    grad_sq = torch.abs(q_grad).square().sum(-1)
                    if weighted_smoothness:
                        smooth_sum += float(torch.sum(grad_sq.to(weights.dtype) * weights * shot_counts).detach().cpu())
                        smooth_mass += float(batch_mass.detach().cpu())
                    else:
                        # Match the full_mass_mean scale used by the weighted
                        # branch and the model term (see train_epoch_smoothness).
                        mass_scale = float(full_mass_mean.detach().cpu()) if normalization else 1.0
                        smooth_sum += float((smooth_weight * grad_sq.mean()).detach().cpu()) * mass_scale * descriptors.shape[0]
                        smooth_mass += descriptors.shape[0]
                if self.ee_params.get("l1_regularization", 0.0):
                    l1_last = float((self.ee_params["l1_regularization"] * sum(
                        parameter.abs().sum() for parameter in self._network_parameters_without_gates())).detach().cpu())
                gate_term, expected_active = self._gate_regularization(model_raw)
                gate_last = float(gate_term.detach().cpu())
                active_features_last = float(expected_active.detach().cpu())
        finally:
            self.nnet.train(was_training)
        model_epoch = (float(full_mass_mean.detach().cpu()) * model_raw_sum /
                       max(effective_mass_sum, np.finfo(float).eps)
                       if normalization else model_raw_sum)
        smooth_epoch = smooth_sum / smooth_mass if smooth_mass else 0.0
        if weighted_smoothness:
            smooth_epoch *= self.ee_params.get("smoothness_penalty_weight", 0.0)
            if normalization:
                smooth_epoch *= float(full_mass_mean.detach().cpu())
        return {"total_loss": model_epoch + smooth_epoch + l1_last + gate_last,
                "model_loss": model_epoch,
                "model_per_effective_shot": model_raw_sum / max(effective_mass_sum, np.finfo(float).eps),
            "smoothness_loss": smooth_epoch, "l1_regularization": l1_last,
            "stochastic_gate_regularization": gate_last,
            "expected_active_features": active_features_last}


class TIS_EEScalePytorchRCModel(TIS_EEScalePytorchRCModelMixin, PytorchRCModel):
    pass


def register_legacy_aimmd_alias():
    """Make models saved before the package migration loadable with pickle."""
    import aimmd.pytorch
    import aimmd.pytorch.rcmodel

    aimmd.pytorch.rcmodel.TIS_EEScalePytorchRCModel = TIS_EEScalePytorchRCModel
    aimmd.pytorch.TIS_EEScalePytorchRCModel = TIS_EEScalePytorchRCModel