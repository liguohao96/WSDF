import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] =":4096:8"
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import cv2
import pickle
from tqdm        import tqdm
from typing      import Tuple, Iterable
from functools   import reduce
from contextlib  import nullcontext
from collections import defaultdict, OrderedDict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, ROOT)
import models
sys.path.pop(0)

for t in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, t):
        setattr(np, t, getattr(np, f"{t}_"))

def get_class(type_name):
    import importlib
    mod_name = ".".join(type_name.split(".")[:-1])
    obj_name = type_name.split(".")[-1]

    mod = importlib.import_module(mod_name)
    print(mod_name, mod)

    return getattr(mod, obj_name)

def load_wsdf(ckpt_path, config=None, device=None):
    '''
    ckpt_path: str
    config:   (str, dict) highest priority
    '''
    import yaml
    import zipfile

    device = device if device is not None else "cuda"

    # Config
    #---------------------------------------------#
    with nullcontext("Config"):
        cfg_file = None
        cfg_data = {}    # higher priority

        # detect from checkpoint
        if os.path.splitext(ckpt_path)[-1] == ".zip":
            try:
                cfg_file = zipfile.ZipFile(ckpt_path, "r").open("config.yaml")
            except FileNotFoundError:
                pass

        if isinstance(config, (str,)) and os.path.exists(config):
            cfg_data = yaml.load(open(config), yaml.FullLoader)
        elif isinstance(config, (dict,)):
            cfg_data = config

        if cfg_file is not None:
            config = yaml.load(cfg_file, yaml.FullLoader)
        else:
            config = {}
        
        config.update(cfg_data)

        EXP_NAME = config["name"]
        model_cfg = config["model"]

    # MeshTopology & Render setups
    #---------------------------------------------#
    with nullcontext("MeshTopology & Render"):
        TOPOLOGY_CONFIG = {}
        mesh_topology = config.get("mesh_topology", "FLAME2020")
        if mesh_topology == "FLAME2020":
            with open(os.path.join(ROOT, "Data", "FLAME2020", "generic_model.pkl"), "rb") as f:
                data = pickle.load(f, encoding="latin1")
            # v_template = 1000 * data["v_template"]  # convert flame model to millimeter
            texr = np.load(os.path.join(ROOT, "Data", "FLAME2020", "FLAME_texture.npz"))

            TOPOLOGY_CONFIG["ver"]     = torch.as_tensor(1000*data["v_template"].astype(np.float32)).reshape(-1, 3)
            TOPOLOGY_CONFIG["tri"]     = torch.as_tensor(data["f"].astype(np.int32)).reshape(-1, 3)
            TOPOLOGY_CONFIG["uv"]      = torch.as_tensor(texr["vt"].astype(np.float32)).reshape(-1, 2)
            TOPOLOGY_CONFIG["uv_tri"]  = torch.as_tensor(texr["ft"].astype(np.int32)).reshape(-1, 3)

        elif mesh_topology == "FaceScape":
            # read FaceScape model
            data = np.load(os.path.join(ROOT, "Data", "3DMM_FaceScape.npz"), allow_pickle=True)
            ver = (data["id_mean"] + data["exp_mean"]).reshape(-1, 3)
            tri = (data["tri"]).reshape(-1, 3)
            
            TOPOLOGY_CONFIG["ver"]     = torch.as_tensor(ver.astype(np.float32)).reshape(-1, 3)
            TOPOLOGY_CONFIG["tri"]     = torch.as_tensor(tri.astype(np.int32)).reshape(-1, 3)
            TOPOLOGY_CONFIG["uv"]      = torch.as_tensor(data["uv"].astype(np.float32)).reshape(-1, 2)
            TOPOLOGY_CONFIG["uv_tri"]  = torch.as_tensor(data["uv_tri"].astype(np.int32)).reshape(-1, 3)

        for k, v in TOPOLOGY_CONFIG.items():
            TOPOLOGY_CONFIG[k] = v.to(device)
        
        # render setup
        fov  = 15
        tan  = np.tan(np.deg2rad(fov/2))
        cot  = 1/tan
        zz   = -2500*(np.tan(np.deg2rad(12/2)) * cot)
        n, f = 500, 3000
        t    = n*tan
        r    = n*tan
        np_w2v = np.array([
            [ 1,  0,  0,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  1, zz],
            [ 0,  0,  0,  1]
            ], dtype=np.float32)
        np_v2n = np.array([
            [2*n/r,     0,  0, 0],
            [    0, 2*n/t,  0, 0],
            [    0,     0, -(f+n)/(f-n), -2*f*n/(f-n)],
            [    0,     0, -1, 0]
            # [  0,    0, (f+n)/(f-n), 2*f*n/(f-n)],
            # [  0,    0, -1, 0]
            ], dtype=np.float32)
        m2v = torch.as_tensor(np_w2v).unsqueeze(0).to(device)
        ndc = torch.as_tensor(np_v2n)
        reverse_y = torch.diag(torch.as_tensor([1,-1,1,1]).float())
        ndc = (ndc @ reverse_y).unsqueeze(0).to(device)

        def render_vertex(ver, tex=None):
            if tex is None:
                tex = torch.ones(ver.size(0), 16, 16, 3, device=ver.device)
            tri, uv, uv_tri = TOPOLOGY_CONFIG["tri"], TOPOLOGY_CONFIG["uv"], TOPOLOGY_CONFIG["tri"]
            img, alp, nrm, dep = render_nvdr(m2v, ndc, ver, tri, uv, uv_tri, tex, H=256, W=256)
            return 0.5+0.5*nrm

    # Networks
    #---------------------------------------------#
    with nullcontext("Networks"):
        NV = TOPOLOGY_CONFIG["ver"].numel()//3
        NUM_VEC = model_cfg.get("num_vec", 64)
        NUM_IND = model_cfg.get("num_ind", 16)
        NUM_EXP = model_cfg.get("num_exp", 16)

        Encoder_cfg = model_cfg["Encoder"]
        Decoder_cfg = model_cfg["Decoder"]
        NeuBank_cfg = model_cfg.get("NeuBank", {})

        ENC_CHANNELS  = Encoder_cfg.get("channels", [16, 32, 64, 64, 64, 128])
        DEC_CHANNELS  = Decoder_cfg.get("channels", [256]*3)

        from models.mesh_encoder import RegisteredMeshEncoder
        from models.mesh_decoder import RegisteredMeshDecoder
        from models.neutral_bank import NeutralBank
        from models.common       import Normalize
        from models.VAEs         import NaiveVAEs

        spiral_cache = torch.load(os.path.join(ROOT, "Data", f"SpiralPlusCache_{mesh_topology}.pth"), map_location="cpu")
        spiral_indices      = spiral_cache["S"]
        downsample_matrices = spiral_cache["D"]

        Encoder = RegisteredMeshEncoder(dim_I=2*NUM_IND, dim_E=2*NUM_EXP, 
            spiral_indices=spiral_indices, downsample_matrices=downsample_matrices, 
            mid_channels=ENC_CHANNELS[:-1], feature_channel=ENC_CHANNELS[-1]).to(device)
        VAEHead = NaiveVAEs()
        Decoder = RegisteredMeshDecoder(dim_I=NUM_IND, dim_E=NUM_EXP, dim_K=NUM_VEC, num_vertex=NV, mid_channels=DEC_CHANNELS).to(device)
        NeuBank = NeutralBank(1, (NV, 3), torch.zeros((NV, 3)))

        Norm    = Normalize(
            torch.full((1, NV, 3), 0.0), 
            torch.full((1, NV, 3), 1.0)).to(device)

        model_dict = {
            "Encoder": Encoder,
            "VAEHead": VAEHead,
            "Decoder": Decoder,
            "NeuBank": NeuBank,
            "Norm":    Norm,
        }

        Encoder = model_dict["Encoder"] 
        VAEHead = model_dict["VAEHead"] 
        Decoder = model_dict["Decoder"] 
        NeuBank = model_dict["NeuBank"] 

    # Save/Load
    #---------------------------------------------#
    with nullcontext("Save & Load"):
        def load_model(state_fpath):
            if os.path.splitext(state_fpath)[-1] == ".zip":
                from zipfile import ZipFile
                with ZipFile(state_fpath, "r") as f:
                    with f.open("param.pth", "r") as pth_f:
                        state = torch.load(pth_f)
            else:
                state = torch.load(state_fpath)

            model_state = state

            for mod, key in zip([Encoder, Decoder, NeuBank, Norm], ["Encoder", "Decoder", "NeuBank", "Norm"]):
                if key not in model_state:
                    continue

                if key in ["Encoder", "Decoder", "Norm"]:
                    tgt_state_dict = mod.state_dict()
                    src_state_dict, mismatch_state = OrderedDict(), OrderedDict()
                    for k, v in model_state[key].items():
                        if k in tgt_state_dict and v.shape == tgt_state_dict[k].shape:
                            src_state_dict[k] = v
                        else:
                            mismatch_state[k] = v
                            print(f"got shape mismatch for '{key}' curr:{tgt_state_dict[k].shape} load:{v.shape}")

                    missing_keys, unexpected_keys = mod.load_state_dict(src_state_dict, strict=False)
                    num_param = np.sum([p.numel() for p in src_state_dict.values()])
                    num_total = np.sum([p.numel() for p in tgt_state_dict.values()])
                    print(f"loaded {key} with {num_param/2**20:.3f}MB ({num_param/num_total*100:.2f}% of the model)")

                    if (len(mismatch_state) + len(missing_keys) + len(unexpected_keys)) != 0:
                        print(f"[WARNING] model [{key}]")
                        print(f"shape mismatch: {mismatch_state.keys()}")
                        print(f"missing keys:   {missing_keys}")
                        print(f"unexpected keys {unexpected_keys}")
                
                if key in ["NeuBank"]:
                    mod.data.weight = nn.Parameter(model_state[key]["data.weight"])

    # load weight for model in model_dict
    print(f"Loading from {ckpt_path}")
    load_model(ckpt_path)

    model_dict = {k: v.to(device) for k, v in model_dict.items()}
    model_dict["Topology"] = {k: v.to(device) for k,v in TOPOLOGY_CONFIG.items()}

    return model_dict
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str, help='', required=False, default=None)
    parser.add_argument("--config",      type=str, help='', required=False, default=None)

    args = parser.parse_args()
    load_wsdf(args.checkpoint, args.config)