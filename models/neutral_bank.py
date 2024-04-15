import torch
import torch.nn as nn

import numpy as np

class NeutralBank(nn.Module):
    def __init__(self, count, shape, init_value, beta=0.9):
        super().__init__()

        self.count = count
        self.shape = shape

        self.beta = beta

        self.data = nn.Embedding(count, np.prod(shape))
        
        with torch.no_grad():
            self.data.weight.copy_(init_value.flatten()[None].expand(count, -1).to(self.data.weight.device))
    
    def forward(self, index):
        return self.data(index).unflatten(1, self.shape)

    def update(self, vec, pid):
        BETA = self.beta
        
        neu_new = vec.flatten(1) # B, V, 3
        neu_old = self.data(pid) # B, V, 3

        neu_new = torch.where(neu_new.sum(1, keepdims=True).isnan(), neu_old, neu_new) # mask out mesh with nan value

        unique_pid, inverse, counts = torch.unique(pid, return_inverse=True, return_counts=True)

        delta = ((BETA-1)*neu_old + (1-BETA)*neu_new) / counts[inverse].reshape(-1, 1)
        self.data.weight.scatter_add_(0, unique_pid[inverse].reshape(-1, 1).expand_as(delta), delta)

        # new_by_pid = torch.zeros_like(neu_new[unique_pid])
        # new_by_pid.scatter_reduce_(0, inverse.unsqueeze(-1), neu_new, reduce="mean")
        # old_by_pid = self.data(unique_pid)

        # delta = (BETA-1)*old_by_pid + (1-BETA)*new_by_pid
        # self.data.weight.scatter_add_(0, unique_pid.reshape(-1, 1), delta)

        # val <= BETA*old + (1-BETA)*new
        # val <= val + BETA*old - val + (1-BETA)*new
        # val <= val + (BETA-1)*old + (1-BETA)*new