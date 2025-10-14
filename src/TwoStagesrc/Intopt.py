import numpy as np
import torch
from torch import nn

class Intopt (nn.Module):
    def __init__(self, c, h, A, b, penalty, n_features, num_layers=5, 
    smoothing=False, thr=0.1, max_iter=None, method=1, mu0=None,
                 damping=0.5, target_size=1, **hyperparams):
        super(Intopt, self).__init__()
        self.c = c
        self.h = h
        self.A = A
        self.b = b
        self.penalty = penalty
        self.n_features = n_features
        self.num_layers = num_layers
        self.smoothing = smoothing
        self.thr = thr
        self.max_iter = max_iter
        self.method = method
        self.mu0 = mu0
        self.damping = damping
        self.target_size = target_size
        self.hyperparams = hyperparams
    