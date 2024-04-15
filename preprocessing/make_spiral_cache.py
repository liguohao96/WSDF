import os
import sys
import torch
import scipy
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPIRAL_PLUS_ROOT = os.path.join(ROOT, "3rdparty", "spiralnet_plus")

sys.path.insert(0, SPIRAL_PLUS_ROOT)
import utils
import utils as spiral_utils
from utils import mesh_sampling as spiral_mesh_sampling
sys.path.pop(0)

def scipy_to_torch_sparse(scp_matrix):
    scp_matrix = scp_matrix.tocoo()
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, shape).coalesce()
    return sparse_tensor

def build_spiral_cache(vertex, triangle):
    from psbody.mesh import Mesh

    down_scales = [4, 4, 4, 4]

    template_mesh = Mesh(vertex.reshape(-1, 3), triangle.reshape(-1, 3))
    _, A, D, U, F, V = spiral_mesh_sampling.generate_transform_matrices(template_mesh, down_scales)

    # "A": A,  # adjacent
    # "D": D,  # down matrix
    # "U": U,  # up matrix
    # "F": F,  # face
    # "V": V   # vertices

    A_t = [scipy_to_torch_sparse(a) for a in A]
    D_t = [scipy_to_torch_sparse(d) for d in D]
    U_t = [scipy_to_torch_sparse(u) for u in U]
    F_t = [torch.as_tensor(f.astype(np.int32)) for f in F]
    V_t = [torch.as_tensor(v) for v in V]

    print("F.shape", [f.shape for f in F])
    seq_length = [9, 9, 9, 9, 9]
    dilation   = [1, 1, 1, 1, 1]
    S_i = [
        spiral_utils.preprocess_spiral(
            F[idx], seq_length[idx], V[idx], dilation[idx])
        for idx in range(len(F))
    ]

    return A_t, D_t, U_t, F_t, V_t, S_i

def main(args):

    if args.name == "FLAME2020":
        save_path = os.path.join(ROOT, "Data", "SpiralPlusCache_FLAME2020.pth")

        import pickle

        for t in ["bool", "int", "float", "complex", "object", "unicode", "str", "nan", "inf"]:
            if not hasattr(np, t):
                setattr(np, t, getattr(np, f"{t}_"))

        with open(os.path.join(ROOT, "Data", "FLAME2020", "generic_model.pkl"), "rb") as f:
            data = pickle.load(f, encoding="latin1")
        
        ver = data["v_template"].reshape(-1, 3)
        tri = data["f"].reshape(-1, 3)

    # vs, vt, vn, t_vs, t_vt, t_vn = load_mesh(args.mesh)
    As, Ds, Us, Fs, Vs, Ss = build_spiral_cache(ver, tri)

    save_obj = {
        "A": As,
        "D": Ds,
        "U": Us,
        "F": Fs,
        "V": Vs,
        "S": Ss,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"save @ {save_path}")
    torch.save(save_obj, save_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--mesh", type=str, required=True, help="mesh file")
    parser.add_argument("--name", type=str, required=True, choices=["FLAME2020"])

    args = parser.parse_args()

    main(args)
