import numpy as np
import torch
import torch.nn as nn

from .mlp import MLP

class Recoupler(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, normalize=True):
        super().__init__()

        self.args_str = f"in_dim={in_dim}, out_dim={out_dim}, bias={bias},normalize={normalize}"

        assert len(in_dim) == 2

        num_k       = out_dim
        num_i,num_e = in_dim
        self.weight = nn.Parameter(torch.empty(num_k, num_i, num_e))
        self.bias   = nn.Parameter(torch.empty(num_k)) if bias is True else None
        
        self.normalize = normalize
        self.reset_parameter()
    
    def extra_repr(self):
        return self.args_str

    def reset_parameter(self):
        with torch.no_grad():
            nn.init.normal_(self.weight)

            if self.bias is not None:
                nn.init.normal_(self.bias,   0, 0.001)
    
    def forward(self, i_code, e_code):
        # return self.fusion

        uc = self.normal2uniform(i_code)
        us = self.normal2uniform(e_code)
        
        coef   = torch.einsum("bi,bj->bij", uc, us)
        u_coef = self.neglog2uniform(coef)
        n_coef = self.uniform2normal(u_coef)

        if self.normalize:
            weight = torch.nn.functional.normalize(self.weight.flatten(1), dim=1).reshape(self.weight.shape)
        else:
            weight = self.weight
        
        v = torch.einsum("kij,bij->bk", weight, n_coef)
        if self.bias is not None:
            v = v + self.bias
        
        return v

    @classmethod
    def cdf_normal(cls, x):
        sqrt2 = np.sqrt(2)
        return 0.5+0.5*torch.erf(x/sqrt2)        
    @classmethod
    def pdf_normal(cls, x):
        sqrt2pi = np.sqrt(2*np.pi)
        return torch.exp(-0.5*x.pow(2)) / sqrt2pi

    @classmethod
    def cdfinv_normal(cls, x):
        sqrt2 = np.sqrt(2)
        return sqrt2*torch.erfinv(2*x-1)

    @classmethod
    def normal2uniform(cls, x):
        return cls.cdf_normal(x)*0.9999 + 0.00005
    @classmethod
    def normal2uniform_grad(cls, x):
        return cls.pdf_normal(x)

    @classmethod
    def uniform2normal(cls, x):
        return cls.cdfinv_normal(x)
    @classmethod
    def uniform2normal_grad(cls, x):
        sqrt2pi = np.sqrt(2*np.pi)
        z = torch.erfinv(2*x-1)
        return sqrt2pi * torch.exp(z.pow(2))
    
    @classmethod
    def neglog2uniform(cls, x):
        # x - x*log(x)
        # x*(1-log(x))
        # x*log(e/x)
        eps = 1e-5
        # assert (x>0).all()
        return x * (1-torch.log(x+eps))
    @classmethod
    def neglog2uniform_grad(cls, x):
        return torch.log(x)

class RegisteredMeshDecoder(nn.Module):
    def __init__(self, 
        dim_I,                  # dim of Indentity code
        dim_E,                  # dim of Expression code
        dim_K,                  # dim of fused code
        num_vertex,             # num of predict vertex
        channel_per_vertex = 3, # predicted channel for each vertex
        mid_channels       = [256, 256]
        ):
        super().__init__()

        self.dim_I = dim_I
        self.dim_E = dim_E
        self.dim_K = dim_K

        self.num_vertex         = num_vertex
        self.channel_per_vertex = channel_per_vertex

        self.recoupler = Recoupler((dim_I, dim_E), dim_K)
        self.gen_model = MLP(dim_K, mid_channels, num_vertex*channel_per_vertex, 
            act_fn=lambda: nn.ELU(), 
            last_act_fn=None)

    def forward(self, i_code, e_code):
        f_code = self.recoupler(i_code, e_code)
        mesh_v = self.gen_model(f_code)

        mesh_v = mesh_v.unflatten(-1, (self.num_vertex, self.channel_per_vertex))

        return mesh_v
    
    def jacobian(self, i_code, e_code):
        device = i_code.device

        # forward
        f_code = self.recoupler(i_code, e_code)
        mesh_v = self.gen_model(f_code)

        # compute jacobian
        jac_v  = self.gen_model.jacobian(f_code, None).unflatten(1, (self.num_vertex, self.channel_per_vertex))  # (BS, NV, 3, K)

        g_out = torch.eye(f_code.size(1), device=device)
        ji,je = torch.autograd.grad(f_code.sum(0), [i_code, e_code], grad_outputs=g_out,
            is_grads_batched=True, create_graph=True)
        
        # print(ji.shape, je.shape, jn.shape)
        ji = torch.einsum("b...i,ibc->b...c", jac_v, ji)
        je = torch.einsum("b...i,ibc->b...c", jac_v, je)

        return ji, je