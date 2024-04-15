import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] =":4096:8"
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed   as dist
import numpy as np

import cv2
import pickle
from tqdm        import tqdm
from typing      import Tuple, Iterable
from functools   import reduce
from contextlib  import nullcontext
from collections import defaultdict

record_function = torch.profiler.record_function

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

sys.path.insert(0, ROOT)
from preprocessing.generate_dataset_cache import get_dataset_tag
from utils import device_control
from utils.mesh_tool import compute_normal
sys.path.pop(0)

for t in ["bool", "int", "float", "complex", "object", "unicode", "str"]:
    if not hasattr(np, t):
        setattr(np, t, getattr(np, f"{t}_"))

class TrainSamplerByID(object):
    RATE_D_S = 1
    def __init__(self, id_list, pid_list, batch_size=1, keep_last=True, seed=110) -> None:
        super().__init__()

        pids = np.unique(pid_list)

        self.pid_set = pids

        self.rng       = np.random.RandomState(seed=seed)
        self.extra_rng = np.random.RandomState(seed=seed)

        vid_2_i = dict()
        pid_2_i = dict()
        i_2_fid = list()
        
        assert len(id_list)  == len(pid_list)
        fid = 0
        for i, (id, pid) in enumerate(zip(id_list, pid_list)):
            if pid not in pid_2_i:
                pid_2_i[pid] = list()
            pid_2_i[pid].append(i)

        self.id_list  = id_list
        self.pid_list = pid_list

        self.pid_2_i = pid_2_i

        self._batch_size  = batch_size
        self._keep_last   = keep_last

        # keep_last = True
        self._sections = (len(self.pid_list) // self._batch_size) + (1 if len(self.id_list) % self._batch_size > 0 else 0)
        if self._keep_last:
            self._length      = self._sections
        else:
            self._length      = (len(self.pid_list) // self._batch_size)

    def random(self, choice, size=1, weight=None, rng=None):
        choice  = choice
        replace = len(choice) < size
        prob    = None
        if weight is not None:
            prob = weight.flatten() / np.sum(weight)
        rng = self.rng if rng is None else rng
        return rng.choice(choice, size=size, replace=replace, p=prob)

    def random_exclude(self, choice, exclude, size=1, weight=None, rng=None):
        exclude = set(exclude) if isinstance(exclude, Iterable) else set([exclude])
        choice  = [ c for c in choice if c not in exclude ]
        replace = len(choice) < size
        prob    = None
        if weight is not None:
            prob = weight.flatten() / np.sum(weight)
        rng = self.rng if rng is None else rng
        return rng.choice(choice, size=size, replace=replace, p=prob)
    

    def __len__(self):
        if dist.is_initialized():
            return (self._length // dist.get_world_size()) + (1 if self._length % dist.get_world_size() > 0 else 0)
        else:
            return self._length


    def __iter__(self):
        total_random_id = list(range(len(self.id_list)))         # N
        self.rng.shuffle(total_random_id)

        # total_random_id = total_random_id[:self._length * self._batch_size] # L*B

        # total_random_id = np.array(total_random_id).reshape(self._length, self._batch_size) # L, B

        # if dist.get_rank() >= 0:
        #     total_random_id = total_random_id[dist.get_rank()::dist.get_world_size()]
        if dist.is_initialized():
            # N*L'*B
            # dist.barrier()
            total_size = len(self) * self._batch_size # L'*B
            total_size *= dist.get_world_size()       # N*L'*B
            total_random_id += total_random_id[:total_size - len(total_random_id)] # N*L'*B

            total_random_id = np.array(total_random_id).reshape(dist.get_world_size(), -1, self._batch_size)[dist.get_rank()] # L'*B
            assert len(total_random_id) == len(self)
        else:
            # L*B
            if self._keep_last is True:
                total_random_id = np.array_split(total_random_id, self._sections)
            else:
                total_random_id = np.array(total_random_id[:self._length*self._batch_size]).reshape(self._length, self._batch_size)

        for i in range(len(self)):
            batch = []

            random_id = total_random_id[i] # B
            
            dynamic_id = []
            for cur_i in random_id:
                pid    = self.pid_list[cur_i]    # 0, len(unique(self.vid_list))
                i_in_same_pid = self.pid_2_i[pid]      # 0, len(self.id_list)
                weight = None
                try:
                    rand_i = self.random_exclude(i_in_same_pid, cur_i, self.RATE_D_S, weight, rng=self.extra_rng)
                except Exception as ex:
                    print(f"unable to choice from {i_in_same_pid} while excluding {cur_i}")
                    rand_i = [cur_i] * self.RATE_D_S
                dynamic_id.extend(rand_i)
                
            batch.extend([ self.id_list[j] for j in random_id  ])
            batch.extend([ self.id_list[j] for j in dynamic_id ])

            yield batch

def render_nvdr(m2v, prj, ver, tri, uv, uv_tri, tex, H=512, W=512, FOV=90, locals={}):
    import nvdiffrast.torch as dr
    BS     = ver.size(0)
    device = ver.device
    drctx = locals.get("drctx", dr.RasterizeCudaContext(device=device))

    tri_l = tri.long()
    face_nrm = torch.cross(ver[:,tri_l[:,1]]-ver[:,tri_l[:,0]], ver[:,tri_l[:,2]]-ver[:,tri_l[:,0]], dim=-1)

    v_nrm = torch.zeros_like(ver) # B, N, 3
    v_nrm.scatter_add_(1, tri_l[None, :, 0, None].expand(BS, -1, 3), face_nrm)
    v_nrm.scatter_add_(1, tri_l[None, :, 1, None].expand(BS, -1, 3), face_nrm)
    v_nrm.scatter_add_(1, tri_l[None, :, 2, None].expand(BS, -1, 3), face_nrm)
    v_nrm  = F.normalize(v_nrm, dim=-1)

    v_nrm  = compute_normal(ver, tri_l)

    tri_   = tri.int().contiguous()

    m2v_matx = m2v
    ndc_proj = prj

    ver_  = F.pad(ver, (0, 1), "constant", 1)
    ver_v = ver_  @ m2v_matx.transpose(1, 2)
    ver_n = ver_v @ ndc_proj.transpose(1, 2)
    tex_  = tex.contiguous()
    pos   = ver_n.contiguous()
    rast, rast_db = dr.rasterize(drctx, pos, tri_, resolution=[H, W])
    if tex_.ndim == 4:
        uv     = uv.float().contiguous()
        uv_tri = uv_tri.int().contiguous()
        im_uv,uv_da = dr.interpolate(uv, rast, uv_tri, rast_db, diff_attrs="all")
        # im_uv,uv_da = dr.interpolate(uv, rast, uv_tri)
        im_ndr      = dr.texture(tex_, im_uv, uv_da=uv_da)
    else:
        im_ndr, _   = dr.interpolate(tex_, rast, tri_, rast_db)
    im_ndr = dr.antialias(im_ndr, rast, pos, tri_)

    v_nrm  = v_nrm @ m2v_matx[:,:3,:3].transpose(1, 2) # B,Nv,3
    z_view = ver_v[...,2:3] - ver_v[...,2:3].min(-2, keepdim=True).values.detach()
    v_nd   = torch.cat([v_nrm, z_view], dim=-1)

    nrm_dpt, _ = dr.interpolate(v_nd, rast, tri, rast_db)

    im_nrm = nrm_dpt[..., :3]
    im_dep = nrm_dpt[..., 3:4]

    im_alp = (rast[:,:,:,-1].reshape(BS,H,W,1)>=1).float()
    im_dep = im_alp * (im_dep.reshape(BS,H,W,1))

    return im_ndr, im_alp, im_nrm, im_dep

def get_class(type_name):
    import importlib
    mod_name = ".".join(type_name.split(".")[:-1])
    obj_name = type_name.split(".")[-1]

    mod = importlib.import_module(mod_name)
    print(mod_name, mod)

    return getattr(mod, obj_name)

def train(args):

    device_control.make_deterministic(seed=172592)

    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Config
    #---------------------------------------------#
    with nullcontext("Config"):
        import yaml
        with open(args.conf) as f:
            config = yaml.load(f, yaml.FullLoader)

        SAVE_ROOT = os.path.join(ROOT, "temp", config["name"])
        os.makedirs(SAVE_ROOT, exist_ok=True)
        
        EXP_NAME = config["name"]

        model_cfg = config["model"]
        train_cfg = config["train"]

        BLF_MIN_COUNT = train_cfg.get("belief_min_count", 5)

    # Device/DDP
    #---------------------------------------------#
    with nullcontext("Device & DDP"):
        device = device_control.init_device(args.enable_ddp)
        device_control.ddp_enabled = args.enable_ddp
        vram_t = torch.cuda.mem_get_info(device)[1]

        log_dir = os.path.join(ROOT, "Data", "log", config["name"])
        if device_control.get_rank() == 0:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)
        else:
            writer = None

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

    # Datasets
    #---------------------------------------------#
    with nullcontext("Dataset"):
        CACHED_TRAINDATA = {"ver": [], "pid": []}
        CACHED_TESTDATA  = {}
        test_dataset_d   = {}

        dataset_cfg = config.get("dataset")
        pid_count = 0
        for cfg in dataset_cfg.get("train"):
            train_dataset = get_class(cfg.get("type"))(*cfg.get("args"), **cfg.get("kwargs"))
            cached_data   = torch.load(os.path.join(ROOT, "Data", f"DatasetCache_{get_dataset_tag(train_dataset)}.pth"))

            # pid may be [1, ... 1, 3, ...]
            # convert to [0, ...,0, 1, ...]
            unique_pid, inverse, count = torch.unique(cached_data["pid"], return_inverse=True, return_counts=True) # unique_pid: [1, 3, ...]
            pid = torch.arange(len(unique_pid), device=inverse.device)[inverse] + pid_count                        # arange(len(unique_pid)): [0, 1, ...]

            CACHED_TRAINDATA["ver"].append(cached_data["ver"])
            CACHED_TRAINDATA["pid"].append(pid)

            pid_count += len(unique_pid)
        
        CACHED_TRAINDATA = {k: torch.cat(v, dim=0) for k, v in CACHED_TRAINDATA.items()}

        print(f"#sample={CACHED_TRAINDATA['ver'].size(0)} #pid={pid_count}")
        print(f"vertex: {(CACHED_TRAINDATA['ver'].numel()*4)//2**20}MB, total VRAM: {vram_t//2**20}MB")
        if CACHED_TRAINDATA["ver"].numel()*4 < vram_t*0.4:
            CACHED_TRAINDATA = {k: v.to(device) for k, v in CACHED_TRAINDATA.items()}

        unique_pid, inverse, count = torch.unique(CACHED_TRAINDATA["pid"], return_inverse=True, return_counts=True)
        NeugtBelief = (1 - torch.exp(BLF_MIN_COUNT - count.float())).clamp_min(0).to(device)

        for k, cfg in dataset_cfg.get("test").items():
            test_dataset = get_class(cfg.get("type"))(*cfg.get("args"), **cfg.get("kwargs"))
            test_dataset_d[k]  = test_dataset
            CACHED_TESTDATA[k] = torch.load(os.path.join(ROOT, "Data", f"DatasetCache_{get_dataset_tag(test_dataset)}.pth"))

    # Networks
    #---------------------------------------------#
    with nullcontext("Networks"):
        NV = TOPOLOGY_CONFIG["ver"].numel()//3
        NUM_VEC = model_cfg.get("num_vec", 64)
        NUM_IND = model_cfg.get("num_ind", 16)
        NUM_EXP = model_cfg.get("num_exp", 16)

        NUM_PID = len(torch.unique(CACHED_TRAINDATA["pid"]))

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

        spiral_cache = torch.load(os.path.join(ROOT, "Data", "SpiralPlusCache_FLAME2020.pth"), map_location="cpu")
        spiral_indices      = spiral_cache["S"]
        downsample_matrices = spiral_cache["D"]

        Encoder = RegisteredMeshEncoder(dim_I=2*NUM_IND, dim_E=2*NUM_EXP, 
            spiral_indices=spiral_indices, downsample_matrices=downsample_matrices, 
            mid_channels=ENC_CHANNELS[:-1], feature_channel=ENC_CHANNELS[-1]).to(device)
        VAEHead = NaiveVAEs()
        Decoder = RegisteredMeshDecoder(dim_I=NUM_IND, dim_E=NUM_EXP, dim_K=NUM_VEC, num_vertex=NV, mid_channels=DEC_CHANNELS).to(device)
        NeuBank = NeutralBank(NUM_PID, (NV, 3), torch.mean(CACHED_TRAINDATA["ver"], dim=0), beta=NeuBank_cfg.get("beta", 0.99))

        Norm    = Normalize(
            torch.mean(CACHED_TRAINDATA["ver"], dim=0, keepdims=True), 
            torch.std( CACHED_TRAINDATA["ver"], dim=0, keepdims=True)).to(device)

        model_dict = {
            "Encoder": Encoder,
            "VAEHead": VAEHead,
            "Decoder": Decoder,
            "NeuBank": NeuBank,
            "Norm":    Norm
        }
        multi_device_ctx, main_process_dcr, model_dict = device_control.send_model_to_device(model_dict, device)

        Encoder = model_dict["Encoder"] 
        VAEHead = model_dict["VAEHead"] 
        Decoder = model_dict["Decoder"] 
        NeuBank = model_dict["NeuBank"] 

        print(Encoder)
        print(Decoder)
    
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
                if key in model_state:
                    mod.load_state_dict(model_state[key])
                    num_param = np.sum([p.numel() for p in model_state[key].values()])
                    num_total = np.sum([p.numel() for p in mod.state_dict().values()])
                    print(f"loaded {key} with {num_param/2**20:.3f}MB ({num_param/num_total*100:.2f}% of the model)")

        def save_model(state_fname):
            import io
            from zipfile import ZipFile
            pth_bits = io.BytesIO()

            with ZipFile(state_fname, "w") as zf:
                state_dict = {}
                for k, m in model_dict.items():
                    state_dict[k] = m.state_dict()
                torch.save(state_dict, pth_bits)

                zf.writestr("param.pth", pth_bits.getbuffer())

                import yaml
                zf.writestr("config.yaml", yaml.dump(config, indent=2, sort_keys=True))
        
        if args.checkpoint is not None:
            print(f"Loading from {args.checkpoint}")
            load_model(args.checkpoint)
        
    # Train Setups
    #---------------------------------------------#
    with nullcontext("Train Setups"):
        loss_cfg  = config.get("loss",  {})
        train_cfg = config.get("train", {})

        optimizer_cfg = config.get("optimizer", {"type": "torch.optim.AdamW"})
        scheduler_cfg = config.get("scheduler", {"type": "torch.optim.lr_scheduler.StepLR"})

        amp_scaler = torch.cuda.amp.GradScaler(init_scale=1, enabled=args.enable_amp)

        E_BEG, E_END = 0, train_cfg.get("training_epochs", 100)

        ACCUMULATE_ITER = train_cfg.get("accumulate_iter", 1)
        LOG_STATE_STEPS = train_cfg.get("log_state_steps", 1000)
        EVALUATE_EPOCHS = train_cfg.get("evaluate_epochs", list(range(0, E_END, 10)) + [E_END-1])
        SAVING_N_EPOCHS = train_cfg.get("saving_n_epochs", 10)

        BATCH_SIZE = args.batch_size
        LOCAL_BATCH_SIZE = BATCH_SIZE // device_control.get_world_size()

        print(f"train with {len(CACHED_TRAINDATA['ver'])} samples")

        id_list  = np.arange(len(CACHED_TRAINDATA["ver"]))
        pid_list = CACHED_TRAINDATA["pid"].detach().cpu().numpy()

        # train_loader = torch.utils.data.DataLoader(range(len(id_list)), 
        #     batch_size=LOCAL_BATCH_SIZE,
        #     num_workers=0, shuffle=True)

        train_sampler= TrainSamplerByID(id_list, pid_list, batch_size=LOCAL_BATCH_SIZE, seed=110)
        train_sampler.RATE_D_S = 3
        train_loader = torch.utils.data.DataLoader(id_list, 
            batch_sampler=train_sampler, pin_memory=True)

        test_loader_d = {}

        for k, v in test_dataset_d.items():
            test_loader_d[k] = torch.utils.data.DataLoader(np.arange(len(v)), batch_size=LOCAL_BATCH_SIZE, drop_last=False)
        
        pgrp = [
            {
                "name": "encoder",
                "params": Encoder.parameters(),
            },
            {
                "name": "decoder",
                "params": Decoder.parameters(),
            }
        ]
        
        optm = get_class(optimizer_cfg.get("type"))(pgrp, *optimizer_cfg.get("args", []), **optimizer_cfg.get("kwargs", {}))
        schd = get_class(scheduler_cfg.get("type"))(optm, *scheduler_cfg.get("args", []), **scheduler_cfg.get("kwargs", {}))

        weight_d = loss_cfg.get("weight", {})
        def compute_loss(data, step=None):
            BS = data["ver"].size(0)

            output_dict = {}

            loss_d = {}

            ver_norm = Norm(data["ver"])

            ind_prm, exp_prm = Encoder(ver_norm)

            ind_vec, (ind_mu, ind_lv) = VAEHead(ind_prm)
            exp_vec, (exp_mu, exp_lv) = VAEHead(exp_prm)

            # recon
            rec_norm = Decoder(ind_vec, exp_vec)
            output_dict["ver"] = Norm.invert(rec_norm)

            # neutrialize
            neu_norm = Decoder(ind_vec, torch.zeros_like(exp_vec))
            neugt    = Norm(NeuBank(data["pid"])).detach()
            belief   = NeugtBelief[data["pid"]].detach()

            l_rec = (rec_norm - ver_norm).square().flatten(1).sum(1).mean()
            l_neu = ((neu_norm - neugt).square()*belief[:,None,None]).flatten(1).sum(1).mean()

            loss_d["rec"] = l_rec
            loss_d["neu"] = l_neu

            # KL loss 
            ind_kl = 0.5 * (ind_mu.square() + ind_lv.exp() - ind_lv - 1 ).sum(1).mean()
            exp_kl = 0.5 * (exp_mu.square() + exp_lv.exp() - exp_lv - 1 ).sum(1).mean()
            l_kl  = ind_kl + exp_kl
            loss_d["kl"] = l_kl

            # MI loss
            i_mu, i_lv = ind_mu, ind_lv

            # match with original code
            i_mu, i_lv = i_mu[:BS//4], i_lv[:BS//4]

            std    = torch.exp(i_lv/2)
            z      = VAEHead.draw_from_gaussian(i_mu, i_lv)  # M, C
            diff   = i_mu[None, :, :] - z[:, None, :]                       # i, j, c
            log_qz = -0.5*( (diff).square()*(-i_lv[None]).exp() + i_lv[None] + np.log(2*np.pi) )
            l_mi   = torch.logsumexp(log_qz.sum(-1), dim=1).sum()
            loss_d["mi"] = l_mi

            # Jac loss
            if weight_d.get("jac", 0) > 0:
                NUM_JAC = BS

                i_vec, e_vec = ind_vec, exp_vec

                # match with original code
                i_vec = i_vec[:BS//4].unsqueeze(1).expand(-1, 4, -1).flatten(0, 1)

                vecs   = tuple(map(lambda x:x[:NUM_JAC].clone().detach().requires_grad_(True), [i_vec, e_vec]))
                fx     = Norm.invert(Decoder(*vecs)).flatten(1)
                fb     = Norm.invert(Decoder(vecs[0], torch.zeros_like(vecs[1]))).flatten(1)

                # autograd
                # g_out = torch.eye(fx.size(1), device=fx.device)
                # grads = torch.autograd.grad(fx.sum(0), vecs, grad_outputs=g_out,
                #     is_grads_batched=True, create_graph=True)
                # Jl = [grad.transpose(0, 1) for grad in grads]

                # jacobian
                # f  = lambda i,e : Norm.invert(Decoder(i, e)).flatten(1)
                # Jl = torch.autograd.functional.jacobian(f, vecs, create_graph=True)
                # Jl = [J.sum(dim=-2) for J in Jl]

                # dirty
                J  = Decoder.jacobian(*vecs) # B, V, 3, C
                J  = [ torch.einsum("bvkc,vk->bvkc", j, Norm.std.squeeze(0)).flatten(1, 2) for j in J ]
                Jl = J[:2]

                _, x = vecs     # B, C
                _,Jx = Jl       # B, V*3, C

                GAMMA, GAMMA_t = loss_cfg.get("gamma", [0, 5000])

                v = GAMMA * torch.einsum("bc,bc->b", x, x).detach() - torch.einsum("bf,bfc,bc->b", fx-fb, Jx, x.detach())

                v_t = GAMMA_t * torch.einsum("bc,bc->b", x, x).detach() - torch.einsum("bf,bfc,bc->b", fx-fb, Jx, x.detach())

                l   = v.relu().mean()
                l_t = (-v_t).relu().mean()

                l_jac = l + l_t

                loss_d["jac"] = l_jac
            
            loss_d["all"] = reduce(lambda a,b:a+b, [w*loss_d[k] for k, w in weight_d.items() if w > 0], 0)
            return loss_d, output_dict
        
        def log_state_function(epoch, step):
            save_image_dict = {}

            BS = 4

            # recon
            index = torch.randint(len(CACHED_TRAINDATA["ver"]), (BS, ), device=CACHED_TRAINDATA["ver"].device)
            ver   = CACHED_TRAINDATA["ver"][index].to(device)

            ind_prm, exp_prm = Encoder(Norm(ver))
            ind_vec, (ind_mu, ind_lv) = VAEHead(ind_prm)
            exp_vec, (exp_mu, exp_lv) = VAEHead(exp_prm)

            rec = Norm.invert(Decoder(ind_vec, exp_vec))
            neu = Norm.invert(Decoder(ind_vec, 0*exp_vec))
            ver, rec, neu = map(render_vertex, (ver, rec, neu)) # b,h,w,3
            log_im = torch.stack((ver, rec, neu), dim=0).transpose(1, 2).flatten(0, 1).flatten(1, 2).detach().cpu().numpy()

            H = ver.size(1)
            cv2.putText(log_im, f"input", (10, 0*H+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0.8, 0.8, 0.1), 2)
            cv2.putText(log_im, f"recon", (10, 1*H+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0.8, 0.8, 0.1), 2)
            cv2.putText(log_im, f"neu",   (10, 2*H+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0.8, 0.8, 0.1), 2)
            save_image_dict["recon"] = log_im

            # random
            ind_vec = torch.randn(BS, NUM_IND, device=device)
            exp_vec = torch.randn(BS, NUM_EXP, device=device)

            gen = Decoder(ind_vec, exp_vec)
            neu = Decoder(ind_vec, 0*exp_vec)
            inv = Decoder(ind_vec, -exp_vec)

            gen, neu, inv = map(Norm.invert,   (gen, neu, inv))
            gen, neu, inv = map(render_vertex, (gen, neu, inv))

            log_im = torch.stack((gen, neu, inv), dim=0).transpose(1, 2).flatten(0, 1).flatten(1, 2).detach().cpu().numpy()

            H = ver.size(1)
            cv2.putText(log_im, f"rand_exp", (10, 0*H+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0.8, 0.8, 0.1), 2)
            cv2.putText(log_im, f"neu",      (10, 1*H+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0.8, 0.8, 0.1), 2)
            cv2.putText(log_im, f"inv_exp",  (10, 2*H+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0.8, 0.8, 0.1), 2)
            save_image_dict["random"] = log_im

            if writer is not None:

                for k, v in save_image_dict.items():
                    img = (255*v).astype(np.uint8).copy()
                    writer.add_image(k, img, step, dataformats="HWC")
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(SAVE_ROOT, f"{k}-{step}.png"), img)

        def eval_function():
            pass

            metric_d = {}

            def evaluation_avd(recon, target):
                if isinstance(recon, np.ndarray):
                    avd = np.linalg.norm(recon - target, ord=2, axis=-1) # N, V
                    avd = np.mean(avd, axis=-1)                        # N
                elif isinstance(recon, torch.Tensor):
                    avd  = torch.linalg.norm(recon - target, ord=2, dim=-1).mean(-1)
                return avd

            def evaluation_std(recon):
                # recon: N, V, 3

                ###### Variation 1 ######
                # paper: Disentangled Representation Learning for 3D Face Shape
                # original code: https://github.com/zihangJiang/DR-Learning-for-3D-Face/blob/9a6dbc97207d5d1b81025d843019f1e440a53697/src/measurement.py#L123
                dis = np.sqrt(np.sum(np.square(np.std(recon, axis = 0)),axis = -1)) # data_array.reshape((47,11510,3)) -> recon
                dis = np.mean(dis)  # line 125

                # our implement:
                s = np.std(recon, axis=0)      # V, 3
                d = np.linalg.norm(s, axis=-1) # V
                r1 = np.mean(d)

                # check
                assert np.allclose(r1, dis)
                ###### Variation 1 ######

                ###### Variation 2 ######
                # paper: Disentangled Representation Learning for 3D Face Shape
                # original code: https://github.com/rmraaron/FaceExpDisentanglement/blob/552f4129283fcc169946884e91ea2f5c7803170f/Model/evaluation_vae.py#L263
                def all_euc_std(ne_arr):
                    return np.std(np.sqrt(np.sum((ne_arr - ne_arr.mean(0)) ** 2, axis=2)))
                euc_3dv = all_euc_std(recon)

                # our implement:
                d = np.linalg.norm(recon-np.mean(recon, axis=0, keepdims=True), axis=-1) # N, V
                r2 = np.std(d)

                # check
                assert np.allclose(r2, euc_3dv), f"{r2}, {euc_3dv}"
                ###### Variation 2 ######

                return r1, r2

            test_ds_name = "coma"
            with torch.no_grad():
                group_by_id = defaultdict(list)

                recon_error_list = []
                id_decompos_list = []
                id_decompos_pids = []
                id_decompos_list2= []
                neu_error_pids   = []
                neu_error_list   = []

                cached_datast = CACHED_TESTDATA[test_ds_name]
                neugt_by_id   = cached_datast.get("neu_by_pid", {})

                for data in test_loader_d[test_ds_name]:
                    index = data

                    ver = cached_datast["ver"][index].to(device, non_blocking=True)
                    pid = cached_datast["pid"][index].to(device, non_blocking=True)

                    ind_prm, exp_prm = Encoder(Norm(ver))
                    ind_vec, (ind_mu, ind_lv) = VAEHead(ind_prm)
                    exp_vec, (exp_mu, exp_lv) = VAEHead(exp_prm)

                    rec = Norm.invert(Decoder(ind_vec, exp_vec))
                    neu = Norm.invert(Decoder(ind_vec, 0*exp_vec))

                    # if device_control.ddp_enabled:
                    #     ver = device_control.gather(ver)
                    #     pid = device_control.gather(pid)
                    #     rec = device_control.gather(rec)
                    #     neu = device_control.gather(neu)
                    
                    ver, pid, rec, neu = map(lambda x:x.detach().cpu().numpy(), [ver, pid, rec, neu])
                    
                    n_avd  = evaluation_avd(rec, ver)
                    recon_error_list.extend(n_avd)
                    
                    for pi, v in zip(pid, neu):
                        group_by_id[pi].append(v)

                for k, v in group_by_id.items():
                    v = np.stack(v, axis=0)        # N, V, 3
                    r1, r2 = evaluation_std(v)
                    id_decompos_pids.append(k)
                    # id_decompos_list.append(r1)
                    id_decompos_list2.append(r2)

                    if k in neugt_by_id:
                        # neutralization
                        neu = neugt_by_id[k].cpu().numpy()
                        avd = evaluation_avd(v, neu[None])
                        neu_error_pids.extend([k]*len(avd))
                        neu_error_list.extend(avd)
                    else:
                        print(f"{k} is not evaluated with NEU because of missing GT (got {neugt_by_id.keys()})")

                metric_d["reconstruction.mean"]   = np.mean(  recon_error_list)
                metric_d["reconstruction.median"] = np.median(recon_error_list)
                # metric_d["ind_decompose.mean"]    = np.mean(  id_decompos_list)
                # metric_d["ind_decompose.median"]  = np.median(id_decompos_list)
                metric_d["ind_decompose2.mean"]   = np.mean( id_decompos_list2)
                metric_d["ind_decompose2.median"] = np.median(id_decompos_list2)
                metric_d["neutralization.mean"]    = np.mean(  neu_error_list)
                metric_d["neutralization.median"]  = np.median(neu_error_list)
            return metric_d

        def save_function(epoch, step, tag=None):
            import io
            import shutil
            from zipfile import ZipFile

            if tag is not None:
                filename = f"{epoch}-{step}-{tag}"
            else:
                filename = f"{epoch}-{step}"
            
            save_model(os.path.join(SAVE_ROOT, filename+".zip"))

    # Train Loop
    #---------------------------------------------#
    with nullcontext("Train Loop"):
        epoch = step = 0

        with record_function("evaluate"):
            # eval
            print(epoch, E_END)
            if epoch in EVALUATE_EPOCHS:
                Encoder.eval()
                VAEHead.eval()
                Decoder.eval()
                eval_result = eval_function()
                for k, v in eval_result.items():
                    print(f"[{k:^30}] {v:.4f}")
                Encoder.train()
                VAEHead.train()
                Decoder.eval()

        Encoder.train()
        VAEHead.train()
        Decoder.train()

        for epoch in range(E_BEG, E_END):
            progress = tqdm(desc=f"{EXP_NAME}-[{epoch}]", total=len(train_loader), file=sys.stdout)
            for i, data in enumerate(train_loader):

                with record_function("load data"):
                    index = data
                    data  = {"index": index}
                    for k, v in CACHED_TRAINDATA.items():
                        data[k] = v[index]
                    for k, v in data.items():
                        data[k] = data[k].to(device, non_blocking=True)
                    
                with record_function("forward"), torch.cuda.amp.autocast(enabled=amp_scaler.is_enabled()):
                    loss_d, output_d = compute_loss(data, step)
                
                with record_function("update EMA"), torch.no_grad():
                    if device_control.ddp_enabled:
                        all_ver = device_control.gather(output_d["ver"])
                        all_pid = device_control.gather(data["pid"])

                        NeuBank.update(all_ver, all_pid)

                        device_control.broadcast_to_all(NeuBank.data.weight)
                    else:
                        NeuBank.update(output_d["ver"], data["pid"])

                with record_function("backward"):
                    amp_scaler.scale(loss_d["all"]).backward()

                with record_function("update_weight"):
                    if (step + 1) % ACCUMULATE_ITER == 0:
                        amp_scaler.unscale_(optm)
                        for pg in optm.param_groups:
                            params = pg['params']
                            # torch.nn.utils.clip_grad_value_(params, 5)
                            for p in filter(lambda p: p.grad is not None, params):
                                pass
                                p.grad.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                        device_control.broadcast_optimizer(optm)
                        amp_scaler.step(optm)
                        amp_scaler.update()
                        optm.zero_grad(set_to_none=True)
                
                with record_function("log_state"):
                    if step % LOG_STATE_STEPS == 0:
                        for k, v in loss_d.items():
                            print(f"[{k:^10}] {v.item()}")
                        log_state_function(epoch, step)

                progress.update()
                step += 1
            progress.close()

            with record_function("evaluate"):
                # eval
                print(epoch, E_END)
                if epoch in EVALUATE_EPOCHS:
                    Encoder.eval()
                    VAEHead.eval()
                    Decoder.eval()
                    eval_result = eval_function()
                    for k, v in eval_result.items():
                        print(f"[{k:^30}] {v:.4f}")
                    Encoder.train()
                    VAEHead.train()
                    Decoder.train()

            if (epoch+1) % SAVING_N_EPOCHS == 0 and (epoch+1 == E_END):
                save_function(epoch, step)
            schd.step()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("conf", type=str)
    # parser.add_argument("-bs", "--batch_size", default=64, type=int)
    parser.add_argument("-bs", "--batch_size", default=None, type=int)
    parser.add_argument("-nw", "--num_worker", default=None, type=int)
    parser.add_argument("-ec", "--extra_conf", type=str, default="")
    parser.add_argument("--checkpoint",  type=str, help='', required=False, default=None)
    parser.add_argument("--profile", dest="is_profile", action="store_true", help='enable profile (default: False)')
    parser.add_argument("--amp",     dest="enable_amp", action="store_true", help='enable amp (default: False)')
    parser.add_argument("--ddp",     dest="enable_ddp", action="store_true", help='enable ddp (default: False)')

    args = parser.parse_args()
    train(args)
