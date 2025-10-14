import torch
from torch import nn
import torch.nn.functional as F
import copy
import numpy as np


def dirac_GaussianApprox(x, epsilon=0.25):
    return torch.exp(-x**2 / (2 * epsilon**2)) / (epsilon * (2 * torch.pi)**0.5)

# Function to compute gradients per loss
def get_gradients(loss, model):
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
    flat_grad = torch.cat([g.view(-1) for g in grads])
    return flat_grad

# Solve MGDA QP for 2 losses
def solve_mgda_2(grads):  # grads: tensor of shape (2, D)
    g1, g2 = grads
    g1g1 = torch.dot(g1, g1).item()
    g2g2 = torch.dot(g2, g2).item()
    g1g2 = torch.dot(g1, g2).item()

    print ("MGDA", g1g1, g2g2, g1g2)

    # Solve: min α∈[0,1] || αg1 + (1-α)g2 ||²
    # That gives: α = clamp((g2g2 - g1g2) / (g1g1 + g2g2 - 2*g1g2), 0, 1)
    denom = g1g1 + g2g2 - 2 * g1g2
    if denom == 0:
        alpha = 0.5
    else:
        alpha = (g2g2 - g1g2) / denom
        alpha = max(0, min(1, alpha))

    return torch.tensor([alpha, 1 - alpha])
def ParetoOptimalAlpha (loss_ipl, loss_fpl, model):
    grad_ipl = get_gradients(loss_ipl, model)
    grad_ipl_norm = grad_ipl / (grad_ipl.norm() + 1e-8)
    grad_fpl = get_gradients(loss_fpl, model)
    grad_fpl_norm = grad_fpl / (grad_fpl.norm() + 1e-8)

    grads = torch.stack([ grad_fpl, grad_ipl ]) 
    # Switching, because in MGDA, they assign more weight to the loss which is already 0
    alphas = solve_mgda_2(grads)
    return alphas.float()

def weightedAlpha (loss_ipl, loss_fpl):
    with torch.no_grad():
        two_losses = torch.stack([loss_ipl, loss_fpl])
        alpha_new = torch.nn.functional.softmax(two_losses / 2., dim=0)
    return alpha_new.float()

def retrieve_gradients_forPC(loss, model):
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
    shape = []
    grad = []
    for g in grads:
        shape.append(g.shape)
        grad.append(g.clone())

    return shape, grad

def PCGrad (loss_ipl, loss_fpl, model, w_ipl_norm, w_fpl_norm, w_ipl_proj, w_fpl_proj, infeasibility_aversion_coeff):
    # Reference: https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/master/pcgrad.py
    shape, grad_ipl = retrieve_gradients_forPC(loss_ipl, model)
    shape, grad_fpl = retrieve_gradients_forPC(loss_fpl, model)
    # print("grad_ipl", grad_ipl)
    # print("grad_fpl", grad_fpl)

    grad_ipl_flatten = torch.cat([g.flatten() for g in grad_ipl])
    grad_fpl_flatten = torch.cat([g.flatten() for g in grad_fpl])
    # print ("grad_ipl_flatten", grad_ipl_flatten)
    # print ("grad_fpl_flatten", grad_fpl_flatten)
    ipl_norm = torch.norm(grad_ipl_flatten)  # L2 norm
    fpl_norm = torch.norm(grad_fpl_flatten)  # L2 norm
    # print ("grads norm", ipl_norm, fpl_norm)
    # grad_ipl_flatten = grad_ipl_flatten / (ipl_norm + 1e-8)
    # grad_fpl_flatten = grad_fpl_flatten / (fpl_norm + 1e-8)

    grads = [grad_ipl_flatten, grad_fpl_flatten]
    
    # print ("grads norm after", grad_ipl_flatten.norm()**2, grad_fpl_flatten.norm()**2)

    # g_i_g_j = torch.dot(grad_ipl_flatten, grad_fpl_flatten)
    # if (g_i_g_j < 0) and (ipl_norm > 1e-3):
    #     grad_fpl_flatten -= infeasibility_aversion_coeff * (g_i_g_j) * grad_ipl_flatten #/ (ipl_norm)

    g_i_g_j = torch.dot(grad_ipl_flatten, grad_fpl_flatten)
    if (g_i_g_j < 0) and (ipl_norm > 1e-3) and (fpl_norm > 1e-3):
        projection_of_fpl_on_ipl =  (g_i_g_j) * grad_ipl_flatten / (ipl_norm**2) # This is the direction goes against IPL
        normal_projection_of_fpl = grad_fpl_flatten - projection_of_fpl_on_ipl #Orthogonalto IPL

        projection_of_ipl_on_fpl =  (g_i_g_j) * grad_fpl_flatten / (fpl_norm**2)
        normal_projection_of_ipl = grad_ipl_flatten - projection_of_ipl_on_fpl
    
        
        pc_grad = [  w_ipl_norm * normal_projection_of_ipl.clone() + w_ipl_proj * projection_of_ipl_on_fpl.clone(),
                    w_fpl_norm * normal_projection_of_fpl.clone() + w_fpl_proj * projection_of_fpl_on_ipl.clone()]
        
        # print ("PCGrad", pc_grad)
    
    else:
        # print ("in else")
        pc_grad = [infeasibility_aversion_coeff * grad_ipl_flatten.clone(), 
                   (1-infeasibility_aversion_coeff) * grad_fpl_flatten.clone()]
        # print ("PCGrad", pc_grad)
    
    merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
    merged_grad = torch.stack([g  for g in pc_grad]).mean(dim=0)

    unflatten_grad, idx = [], 0
    for s in shape:
        length = np.prod(s)
        unflatten_grad.append(merged_grad[idx:idx + length].view(s).clone())
        idx += length
    return unflatten_grad

def _project_conflicting(self, grads, has_grads, shapes=None):
    shared = torch.stack(has_grads).prod(0).bool()
    pc_grad, num_task = copy.deepcopy(grads), len(grads)
    for g_i in pc_grad:
        random.shuffle(grads)
    for g_i in pc_grad:
        random.shuffle(grads)
        for g_j in grads:
            g_i_g_j = torch.dot(g_i, g_j)
            if g_i_g_j < 0:
                g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
    merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
    if self._reduction:
        merged_grad[shared] = torch.stack([g[shared]
                                        for g in pc_grad]).mean(dim=0)
    elif self._reduction == 'sum':
        merged_grad[shared] = torch.stack([g[shared]
                                        for g in pc_grad]).sum(dim=0)
    else: exit('invalid reduction method')

    merged_grad[~shared] = torch.stack([g[~shared]
                                        for g in pc_grad]).sum(dim=0)
    return merged_grad