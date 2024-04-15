import os
import sys

import torch
import numpy as np

import json
import pickle

from tqdm        import tqdm
from glob        import glob
from natsort     import natsorted
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, ROOT)
import datasets
from datasets.mesh_dataset import RegisteredMeshDataset
sys.path.pop(0)

for t in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, t):
        setattr(np, t, getattr(np, f"{t}_"))

def get_arap_sRT(P, Q, w=None):
    n, d  = P.shape

    P_avg = np.mean(P, axis=0) # 3
    Q_avg = np.mean(Q, axis=0) # 3

    x = P - P_avg[None, :]     # N, 3
    y = Q - Q_avg[None, :]     # N, 3

    s = np.mean(np.linalg.norm(y, axis=1)/np.linalg.norm(x, axis=1))

    if w is not None:
        W = np.diag(w)
        S = x.T.dot(W).dot(y)       # S = X W Yt
    else:
        S = x.T.dot(y)              # S = X W Yt
    U, E, Vt = np.linalg.svd(S)

    det = np.linalg.det(Vt.T.dot(U.T))
    M   = np.diag([1]*(d-1) + [np.sign(det)])
    R   = Vt.T.dot(M).dot(U.T)       # R = V () Ut

    R   = R.T
    T   = Q_avg - P_avg.dot(s*R)

    return s, R, T

def get_dataset_tag(dataset):
    import hashlib
    md5     = hashlib.md5((repr(dataset)).encode('utf-8'))
    tag     = f'{md5.hexdigest()}-{len(dataset)}'
    return tag

def main(args):

    if args.dataset.lower() in ["coma"]:
        # CoMA is in millimeter
        dataset = RegisteredMeshDataset(args.data_dir, args.label_file, 1000)

        with open(os.path.join(ROOT, "Data", "FLAME2020", "generic_model.pkl"), "rb") as f:
            data = pickle.load(f, encoding="latin1")
        
        v_template = 1000 * data["v_template"]  # convert flame model to millimeter

    v_template = v_template.astype(np.float32).reshape(-1, 3)

    print(f"build cache for {dataset}")

    tag = get_dataset_tag(dataset)
    cache_f = os.path.join(args.out_dir, f"DatasetCache_{tag}.pth")

    if os.path.exists(cache_f):
        print(f"find existing cache @ {cache_f}")
        ret = torch.load(cache_f)
    else:
        os.makedirs(os.path.dirname(cache_f), exist_ok=True)

        TH_ver, TH_pid, TH_eid, TH_vid, TH_fid = [], [], [], [], []
        for data in tqdm(dataset):
            vs = data["vertex"]

            s, r, t  = get_arap_sRT(vs, v_template) # estimate rigid pose with scale

            # template is always in mm
            vs = (vs.dot(r) + t/s)

            TH_ver.append(vs)
            TH_pid.append(data["label"][0])
            TH_eid.append(data["label"][1])
            TH_vid.append(data["label"][2])
            TH_fid.append(data["label"][3])
        TH_ver = torch.as_tensor(TH_ver).float()
        TH_pid = torch.as_tensor(TH_pid).long()
        TH_eid = torch.as_tensor(TH_eid).long()
        TH_vid = torch.as_tensor(TH_vid).long()
        TH_fid = torch.as_tensor(TH_fid).long()

        TH_svd_data = TH_ver.flatten(1)
        avg         = torch.mean(TH_svd_data, dim=0, keepdim=True)
        print("computing svd of masked data...")
        U, S, Vh    = torch.linalg.svd((TH_svd_data - avg), full_matrices=False)

        rate     = args.drop_ratio/100
        if rate > 0:
            std, vec = S[:100], Vh[:100]
            data     = (TH_svd_data - avg) # N, C
            proj     = torch.einsum("nc,kc->nk", data, vec) # N, K
            # log(prob) = SUM[-(x/var)^2]
            neg_prob = torch.sum((proj / std).square(), dim=-1)           # - log(prob)
            indx     = torch.argsort(neg_prob)[-int(rate*len(neg_prob)):] # mask out index, argsort output in increasing order
            mask     = torch.full(neg_prob.shape, True, dtype=torch.bool)
            mask[indx] = False

            print("recomputing svd of masked data...")
            TH_ver = TH_ver[mask]
            TH_pid = TH_pid[mask]
            TH_eid = TH_eid[mask]
            TH_vid = TH_vid[mask]
            TH_fid = TH_fid[mask]
            TH_svd_data = TH_ver.flatten(1)
            avg         = torch.mean(TH_svd_data, dim=0, keepdim=True)
            U, S, Vh    = torch.linalg.svd((TH_svd_data - avg), full_matrices=False)

        print(f"got {len(TH_ver)} samples")

        U  = U[:, :1000]
        S  = S[:1000]
        Vh = Vh[:1000]

        ret = {
            "U": U.cpu(), "S": S.cpu(), "Vh": Vh.cpu(), 
            "mean": avg.cpu(), 
            "ver": TH_ver.cpu(), 
            "pid": TH_pid.cpu(),
            "eid": TH_eid.cpu(),
            "vid": TH_vid.cpu(),
            "fid": TH_fid.cpu(),
            }

        if "neu_by_pid" in dataset.label_data:
            # these filename may not in current dataset split
            neu_labels = []
            for k, v in dataset.label_data["neu_by_pid"].items():
                for f in v:
                    neu_labels.append((f, [int(k), -1, -1, -1]))

            neu_dataset = dataset.make_subset_by_labels(neu_labels)

            neu_by_pid = defaultdict(list)
            for data in tqdm(neu_dataset):
                vs = data["vertex"]

                s, r, t  = get_arap_sRT(vs, v_template) # estimate rigid pose with scale

                vs = (vs.dot(r) + t/s)

                neu_by_pid[data["label"][0]].append(vs)

            neu_by_pid = {k: torch.as_tensor(v).mean(dim=0) for k, v in neu_by_pid.items()}
        
            ret["neu_by_pid"] = neu_by_pid
        
        if args.dataset_type == "train":
            del ret["eid"]
            del ret["vid"]
            del ret["fid"]
            if "neu_by_pid" in ret:
                del ret["neu_by_pid"]

        torch.save(ret, cache_f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",    type=str, choices=["coma"])
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--label_file", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, choices=["train", "valid", "test"])
    parser.add_argument("--drop_ratio", type=float, default=5)
    parser.add_argument("--out_dir",    type=str, required=True)

    parser.add_argument("--split_method", type=str, default=None)
    parser.add_argument("--neutral_from", type=str, default=None)
    parser.add_argument("--neuexp_names", type=str, nargs="+", default=[])

    args = parser.parse_args()

    if args.dataset == "coma":
        if args.split_method is None:
            args.split_method = "coma_interpolation"
        if args.neutral_from is None:
            args.neutral_from = "first_frame"

    print(args)
    main(args)