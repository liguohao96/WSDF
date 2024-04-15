import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.reshape(self.shape)
    
    def jacobian(self, x, jac_out):
        # shape
        if jac_out is not None:
            return jac_out.reshape(x.shape) 
        else:
            return None

class GlobalAvgPool1d(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()

        self.dim = dim
    
    def forward(self, x):
        return x.mean(self.dim)

class InstanceNorm(nn.InstanceNorm1d):
    def __init__(self, channel):
        super().__init__(channel)
    
    def forward(self, x):
        x_t = x.transpose(1, 2)  # B, C, N
        out = super().forward(x_t)
        # out = nn.InstanceNorm1d.__call__(self, x_t)
        out = out.transpose(1, 2)
        return out

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()

        # self.register_buffer("mean", mean)
        # self.register_buffer("std",   std)
        self.register_buffer("mean", mean.clone().detach())
        self.register_buffer("std",   std.clone().detach())
    
    def forward(self, x):
        return (x - self.mean) / self.std

    @torch.jit.export
    def invert(self, x):
        return x * self.std + self.mean
    
    def __str__(self):
        return f"Normalize({tuple(self.mean.shape)}, {tuple(self.std.shape)})"