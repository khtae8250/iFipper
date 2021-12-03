from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy

import torch
from torch.nn.functional import relu
from torch import nn
import torch.optim as optim
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from utils import measure_violations

def Gradient(label, m, w_sim, edge, lam_high):
    """         
        Solves an unconstrained optimization problem via gradient descent.
        If the output does not satisfy the violations limit m, it increases lambda, which controls the trade-off between fairness and accuracy, up to lam_high.

        Args: 
            label: Labels of the data
            m: The violations limit
            w_sim: Similarity matrix
            edge: Indices of similar pairs
            lam_high: Upper bound of lambda
            
        Return:
            flipped_label: Flipped labels for a given m
    """

    device = torch.device("cuda:0")
    n_optim_steps, learning_rate = 1000, 0.1
    lam_high, lam_low, temp = lam_high, 0, 0.1

    init_violations = measure_violations(label, edge)
    best_flips, best_violations = label.shape[0], init_violations
    best_flips_fail, best_violations_fail = label.shape[0], init_violations

    while (lam_high-lam_low) > 0.01:
        lam = (lam_high + lam_low) / 2
        losses = []
        c = torch.from_numpy(label).to(device)
        theta = torch.zeros_like(c).to(device)
        theta.requires_grad_()
        optimizer = optim.Adam([theta], lr=learning_rate)
        for ii in range(n_optim_steps):
            optimizer.zero_grad()
            loss = L_RB_Binary(c, theta, lam, temp, edge)
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
            if ii % 100 == 99:
                optimizer.param_groups[0]['lr'] *= 0.5

        raw_label = torch.sigmoid(theta).cpu().detach().numpy()
        rounded_label = np.round(raw_label)

        num_violations = measure_violations(rounded_label, edge)
        num_flips = np.sum(label != rounded_label)

        if num_violations < m: 
            if num_flips < best_flips:
                best_flips = num_flips
                best_violations = num_violations
                best_flipped_label = rounded_label
            lam_high = lam
        else:
            if num_violations < best_violations_fail:
                best_flips_fail = num_flips
                best_violations_fail = num_violations
                best_flipped_label_fail = rounded_label
            lam_low = lam

        if best_violations < m:
            flipped_label = best_flipped_label
        else:
            flipped_label = best_flipped_label_fail

    return flipped_label

def L_RB_Binary(c, theta, lam, temp, edge):
    """         
        Computes (sum (yi-yi')**2) + lambda * (sum Wij(yi-yj)**2).
        Here we use the RelaxedBernoulli function to enforce the output label close to either 0 or 1.

        Args: 
            theta: Raw labels from the network
            c: Original labels
            lam: Hyperparameter lambda that controls the trade-off bewtween fairness and accuracy
            temp: Temperature parameter for RelaxedBernoulli
            edge: Indices of similar pairs
            
        Return:
            flipped_label: Flipped labels for a given m
    """

    dist = RelaxedBernoulli(temperature=temp, logits = theta)
    y = dist.rsample()
    return torch.sum((y-c)**2) + lam*torch.sum((y[edge[:,0]]-y[edge[:,1]])**2) 
