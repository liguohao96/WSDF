import os
import numpy as np
import torch

from typing import Tuple

torch_compile_fn = getattr(torch, "compile", lambda f:f)

@torch_compile_fn
def compute_normal(
    TH_v_bn3: Tuple[torch.Tensor] or torch.Tensor, 
    TH_f_bm3: Tuple[torch.Tensor] or torch.Tensor,
    TH_m_vf:  Tuple[torch.Tensor] or torch.Tensor or None=None) -> Tuple[torch.Tensor] or torch.Tensor:

    BS = len(TH_v_bn3)

    def nf2nv_matrix(TH_f_m3, device):
        i_n = TH_f_m3.flatten() # Nf*3
        i_m = torch.arange(NF, device=device).reshape(-1, 1).repeat(1, 3).flatten()

        i = torch.stack([i_n, i_m], axis=0)
        v = torch.ones(i.shape[1], device=device)
        TH_n_sel = torch.sparse_coo_tensor(i, v, (NV, NF))
        return TH_n_sel

    # shared
    S_TH_n_sel = TH_m_vf
    if isinstance(TH_f_bm3, torch.Tensor):
        # tensor
        # assert isinstance(TH_v_bn3, torch.Tensor)
        if TH_f_bm3.ndim == 2:
            TH_f_bm3 = TH_f_bm3.unsqueeze(0)
        NV = TH_v_bn3.shape[1]
        NF = TH_f_bm3.shape[1]

        if TH_f_bm3.shape[0] != BS:
            TH_f_bm3   = TH_f_bm3.expand(BS, NF, 3)
            S_TH_n_sel = nf2nv_matrix(TH_f_bm3[0], TH_v_bn3.device)

    TH_n_bn3 = []
    for bi in range(BS):
        TH_v_n3 = TH_v_bn3[bi]
        TH_f_m3 = TH_f_bm3[bi]
        TH_n_f = torch.cross(
            TH_v_n3[TH_f_m3[:, 1]] - TH_v_n3[TH_f_m3[:, 0]],
            TH_v_n3[TH_f_m3[:, 2]] - TH_v_n3[TH_f_m3[:, 0]],
            dim=1)
        TH_n_f  = torch.nn.functional.normalize(TH_n_f, dim=1)

        if S_TH_n_sel is None:
            TH_n_sel = nf2nv_matrix(TH_f_m3, TH_v_n3.device)
        else:
            TH_n_sel = S_TH_n_sel

        with torch.cuda.amp.autocast(enabled=False):
            TH_n_n3 = torch.mm(TH_n_sel.float(), TH_n_f.float())
        TH_n_bn3.append(TH_n_n3)

    if isinstance(TH_v_bn3, torch.Tensor):
        TH_n_bn3 = torch.stack(TH_n_bn3, dim=0)
        TH_n_bn3 = torch.nn.functional.normalize(TH_n_bn3, dim=2)
    elif isinstance(TH_v_bn3, (list, tuple)):
        TH_n_bn3 = [torch.nn.functional.normalize(n, dim=1) for n in TH_n_bn3]

    return TH_n_bn3
