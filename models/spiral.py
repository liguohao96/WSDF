import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class MeshSamplingCOO(torch.nn.Module):
    def __init__(self, matmul_matrix):
        super().__init__()

        if matmul_matrix.layout == torch.sparse_coo:
            pass
        else:
            matmul_matrix = matmul_matrix.to_sparse_coo()
        
        matmul_matrix = matmul_matrix.coalesce()
        self.register_buffer("sampling_matrix_i", matmul_matrix.indices().clone().detach())
        self.register_buffer("sampling_matrix_v", matmul_matrix.values().clone().detach())

        self.sampling_matrix_s = matmul_matrix.shape
    
    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(1, 0, 2) # N, B, C
        x = x.flatten(1)

        with torch.cuda.amp.autocast(enabled=False):
            sampling_matrix = torch.sparse_coo_tensor(self.sampling_matrix_i, self.sampling_matrix_v, self.sampling_matrix_s)
            x = torch.matmul(sampling_matrix, x.float()) # N', BC
        x = x.reshape(sampling_matrix.size(0), B, C) # N', B, C
        x = x.permute(1, 0, 2)
        return x

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.sampling_matrix_s[1], self.sampling_matrix_s[0])

class MeshSamplingCSR(torch.nn.Module):
    def __init__(self, matmul_matrix):
        super().__init__()
        
        matmul_matrix = matmul_matrix.to_sparse_csr()

        self.register_buffer("sampling_matrix_crow", matmul_matrix.crow_indices().clone().detach())
        self.register_buffer("sampling_matrix_col",  matmul_matrix.col_indices().clone().detach())
        self.register_buffer("sampling_matrix_val",  matmul_matrix.values().clone().detach())

        self.sampling_matrix_s = matmul_matrix.shape

    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(1, 0, 2) # N, B, C
        x = x.flatten(1)

        with torch.cuda.amp.autocast(enabled=False):
            sampling_matrix = torch.sparse_csr_tensor(self.sampling_matrix_crow, 
                self.sampling_matrix_col, 
                self.sampling_matrix_val,
                self.sampling_matrix_s)
            x = torch.sparse.mm(sampling_matrix, x.float()) # N', BC
        x = x.reshape(-1, B, C) # N', B, C
        x = x.permute(1, 0, 2)
        return x

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.sampling_matrix_s[1], self.sampling_matrix_s[0])

class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.register_buffer("indices", indices.clone().detach())

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.layer.weight)
        # torch.nn.init.constant_(self.layer.bias, 0)
        torch.nn.init.kaiming_uniform_(self.layer.weight)
        # torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, nc = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
            # x = x.unflatten((0, 1), (n_nodes, -1))
        elif x.dim() == 3:
            bs,_,c = x.size()
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            # x = x.view(bs, n_nodes, -1)
            x = x.reshape(bs, n_nodes, nc*c)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))
        x = self.layer(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)

class SpiralBlock(nn.Module):
    def __init__(self, in_channels, out_channels, indices, norm=None, act=None):
        super().__init__()

        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.norm = norm(out_channels) if norm is not None else None
        self.act  = act() if act is not None else None

    def forward(self, x):
        out = self.conv(x)

        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out

class SpiralResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, indices, norm=None, act=None):
        super().__init__()
        int_channels = (in_channels + out_channels) // 2
        self.block_0 = SpiralBlock( in_channels, int_channels, indices, norm, act)
        self.block_1 = SpiralBlock(int_channels, out_channels, indices, norm, None)

    def forward(self, x):
        out = self.block_0(x)
        out = self.block_1(out)

        out += x
        return out