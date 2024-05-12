import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] =":4096:8"
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

import numpy as np
import cv2
import imageio
import json

from tqdm        import tqdm
from typing      import Tuple, Iterable
from functools   import reduce, lru_cache
from contextlib  import nullcontext
from collections import defaultdict, OrderedDict

from diffusers import DDPMScheduler, DDIMScheduler

record_function = torch.profiler.record_function

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, ROOT)
from utils import device_control
from utils.io.mesh_io import save_mesh
from utils.load import load_wsdf
from utils.mesh_tool import compute_normal
sys.path.pop(0)

with nullcontext("utils"):

    def depth_norm_0_1(depth):
        df = depth.flatten(1)
        min, max = df.min(1, keepdim=True).values, df.max(1, keepdim=True).values

        df = (df - min) / (max - min)
        return df.reshape(depth.shape)

    class Camera(nn.Module):
        def __init__(self, 
            obj_rot=[0.,0.,0.], obj_t3d=[0.,0.,0.], 
            cam_rot=[0.,0.,0.], cam_t3d=[0.,0.,1.],
            near = 0.1, far = 100, fov_degree = 45
            ):
            super().__init__()

            self.register_parameter("obj_rot", nn.Parameter(torch.as_tensor(obj_rot, dtype=torch.float32)))
            self.register_parameter("obj_t3d", nn.Parameter(torch.as_tensor(obj_t3d, dtype=torch.float32).reshape(3, 1)))

            self.register_parameter("cam_rot", nn.Parameter(torch.as_tensor(cam_rot, dtype=torch.float32)))
            self.register_parameter("cam_t3d", nn.Parameter(torch.as_tensor(cam_t3d, dtype=torch.float32).reshape(3, 1)))

            self.register_parameter("near",       nn.Parameter(torch.as_tensor([near], dtype=torch.float32)))
            self.register_parameter("far_offset", nn.Parameter(torch.as_tensor([far-near], dtype=torch.float32)))
            self.register_parameter("fov_r",      nn.Parameter(torch.as_tensor([np.deg2rad(fov_degree)], dtype=torch.float32)))

            self.register_buffer("padding", torch.as_tensor([0., 0, 0, 1]).reshape(1, 4))

            self.register_buffer("axis_flip", torch.diag(torch.as_tensor([1., 1, -1, 1])))

        @staticmethod
        def rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
            batch_size = rot_vecs.shape[0]
            device = rot_vecs.device

            angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
            rot_dir = rot_vecs / angle

            cos = torch.unsqueeze(torch.cos(angle), dim=1)
            sin = torch.unsqueeze(torch.sin(angle), dim=1)

            # Bx1 arrays
            rx, ry, rz = torch.split(rot_dir, 1, dim=1)
            K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

            zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))

            ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
            rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
            return rot_mat
        
        def forward(self):
            rot = self.rodrigues(self.obj_rot.unsqueeze(0)).squeeze(0)
            m34 = torch.cat([rot, self.obj_t3d], dim=1)
            m44 = torch.cat([m34, self.padding], dim=0)
            obj_matx = m44

            rot = self.rodrigues(self.cam_rot.unsqueeze(0)).squeeze(0)
            m34 = torch.cat([rot, self.cam_t3d], dim=1)
            m44 = torch.cat([m34, self.padding], dim=0)
            cam_matx = m44

            # https://www.songho.ca/opengl/gl_projectionmatrix.html
            # fx,  0, 0, 0
            #  0, fy, 0, 0
            #  0,  0, s*(f+n)/(f-n), -2*f*n/(f-n)
            #  0,  0, s, 0

            fx = fy = 1 / torch.tan(self.fov_r.relu()/2)
            n,  f_n = self.near.relu(), self.far_offset.relu()
            s = -1

            v33, v34 = s*(f_n+2*n)/f_n, -2*(f_n+n)*n/f_n

            ndc_matx = torch.zeros((4, 4), dtype=torch.float32, device=obj_matx.device)

            ndc_matx[0, 0] = fx
            ndc_matx[1, 1] = -fy
            ndc_matx[2, 2] = v33
            ndc_matx[2, 3] = v34
            ndc_matx[3, 2] = s
            ndc_matx[3, 3] = 0

            return obj_matx, cam_matx, ndc_matx

with nullcontext("render"):

    with nullcontext("nvdiff"):
        import nvdiffrast.torch as dr
        glctx = None

        @torch.autocast(device_type="cuda", dtype=torch.float32)
        def nvdiff_render(campose, ndcproj, objpose, ver, tri, uv, uv_tri, material, environment, H=512, W=512, SPP=1, background=None):
            global glctx
            if glctx is None:
                device = campose.device
                try:
                    glctx  = dr.RasterizeGLContext(device=device)
                except:
                    glctx  = dr.RasterizeCudaContext(device=device)
            
            BS = ver.size(0)

            tri_l = tri.long()
            v_tri = tri.int()
            face_nrm = torch.cross(ver[:,tri_l[:,1]]-ver[:,tri_l[:,0]], ver[:,tri_l[:,2]]-ver[:,tri_l[:,0]], dim=-1)

            v_nrm  = compute_normal(ver, tri_l)

            v_homo = F.pad(ver, (0, 1), "constant", 1).contiguous()

            # o2w, w2v = objpose, torch.linalg.inv(campose)
            o2w, w2v = objpose, campose # different from text_3d.py

            v_world = v_homo @ o2w.transpose(1, 2)
            v_view  = v_world@ w2v.transpose(1, 2)
            v_ndc   = v_view @ ndcproj.transpose(1, 2)

            nrm = v_nrm @ o2w[:, :3, :3].transpose(1, 2)

            rast, rast_db = dr.rasterize(glctx, v_ndc, v_tri, (H*SPP, W*SPP))    

            mask = (rast[..., 3] > 0).float().unsqueeze(-1)

            # geometry buffer
            # nrm_d   = torch.cat([nrm, v_view[...,3:]], dim=-1)
            nrm_d   = torch.cat([nrm, v_view[...,2:3]-v_view[...,2:3].min(-2, keepdim=True).values.detach()], dim=-1)
            nrm_d,_ = dr.interpolate(nrm_d, rast, v_tri, rast_db=rast_db)

            normal = nrm_d[..., :3]
            depth  = nrm_d[..., 3:]

            # material
            if hasattr(material, "sample2d"):
                texc, texd = dr.interpolate(uv, rast, uv_tri, rast_db=rast_db)
                mat = material.sample2d(texc)
            elif hasattr(material, "sample3d"):
                texc, texd = dr.interpolate(uv, rast, uv_tri, rast_db=rast_db)
                mat = material.sample3d(pos)
            elif isinstance(material, torch.Tensor):
                texc, texd = dr.interpolate(uv, rast, uv_tri, rast_db=rast_db, diff_attrs="all")
                mat = dr.texture(material, texc, texd, filter_mode='linear-mipmap-linear')

            albedo = mat[...,:3]
            if background is not None:
                image = mask*albedo + (1-mask)*background

            render_dict = {
                "image":  image,
                "normal": normal,
                "depth":  depth
            }

            return render_dict

with nullcontext("geometry"):

    class ParameterGeometry(torch.nn.Module):
        def __init__(self, checkpoint_path, mesh_scale, mesh_shift):
            super().__init__()
            self.mesh_scale = mesh_scale
            self.register_buffer("mesh_shift", torch.as_tensor(mesh_shift).reshape(1, 1, 3))

            ############### load wsdf ################
            model_dict = load_wsdf(checkpoint_path, device="cpu")
            Decoder = model_dict["Decoder"]
            Norm    = model_dict["Norm"]
            Topology= model_dict["Topology"]


            uv     = Topology["uv"].reshape(-1, 2).clone()
            uv_tri = Topology["uv_tri"].reshape(-1, 3)

            uv[:, 1] = 1 - uv[:, 1]
            uv       = 2 * uv - 1

            self.topology = Topology
            self.decoder = Decoder.requires_grad_(False).eval()
            self.norm    = Norm.requires_grad_(False).eval()

            self.register_buffer("tri",    Topology["tri"].reshape(-1, 3))
            self.register_buffer("tri_long", Topology["tri"].reshape(-1, 3).long())
            self.register_buffer("uv",     uv)
            self.register_buffer("uv_tri", uv_tri)

            self.g_param_0 = nn.Parameter(torch.zeros(1, Decoder.dim_I))
            self.g_param_1 = nn.Parameter(torch.zeros(1, Decoder.dim_E))

            v = self.decoder(self.g_param_0, self.g_param_1)
            self.deform = nn.Parameter(0.1*torch.randn_like(v))

        def get_ver(self):
            pred = self.norm.invert(self.decoder(self.g_param_0, self.g_param_1))
            
            # verts = (pred + self.deform).squeeze(0)  # Nv, 3
            verts = (self.mesh_shift + self.mesh_scale*pred)  # 1, Nv, 3
            return verts
        
        def render(self, target, material, environment=None):
            opt_ver = self.get_ver() 
            
            (H, W), SPP, background   = target["resolution"], target["spp"], target["background"]
            objpose, campose, ndcproj = target["objpose"], target["campose"], target["ndcproj"]

            ver = opt_ver.expand(objpose.size(0), -1, -1)

            render_ret = nvdiff_render(campose, ndcproj, objpose, ver, self.tri, self.uv, self.uv_tri, material, environment, H=H, W=W, SPP=SPP,
                                    background=background)

            render_ret["ver"]    = ver
            render_ret["tri"]    = self.tri_long
            render_ret["uv"]     = self.uv
            render_ret["uv_tri"] = self.uv_tri
            return render_ret

        def export_mesh(self):
            ver = self.get_ver().squeeze(0)
            return {"ver": ver, "tri": self.tri, "uv": self.topology["uv"].reshape(-1, 2), "uv_tri": self.uv_tri}

with nullcontext("material"):
    import tinycudann as tcnn

    class _MLP(torch.nn.Module):
        def __init__(self, cfg, loss_scale=1.0):
            super(_MLP, self).__init__()
            self.loss_scale = loss_scale
            net = (torch.nn.Linear(cfg['n_input_dims'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
            for i in range(cfg['n_hidden_layers']-1):
                net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
            net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_output_dims'], bias=False),)
            self.net = torch.nn.Sequential(*net).cuda()
            
            self.net.apply(self._init_weights)
            
            # if self.loss_scale != 1.0:   x.to(torch.float32)
            #     self.net.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.loss_scale, ))

        def forward(self, x):
            return self.net(x.to(torch.float32))

        @staticmethod
        def _init_weights(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0.0)

    class MLPTexture2D(torch.nn.Module):
        def __init__(self, AABB, channels = 3, internal_dims = 32, hidden = 1, min_max = None):
            super().__init__()

            self.channels = channels
            self.internal_dims = internal_dims
            self.AABB = AABB
            self.min_max = min_max

            # Setup positional encoding, see https://github.com/NVlabs/tiny-cuda-nn for details
            desired_resolution = 4096
            base_grid_resolution = 16
            num_levels = 16
            per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))
            enc_cfg =  {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": base_grid_resolution,
                "per_level_scale" : per_level_scale
            }
            gradient_scaling = 1 #128
            self.encoder = tcnn.Encoding(2, enc_cfg)
            mlp_cfg = {
                "n_input_dims" : self.encoder.n_output_dims,
                "n_output_dims" : self.channels,
                "n_hidden_layers" : hidden,
                "n_neurons" : self.internal_dims
            }
            self.net = _MLP(mlp_cfg, gradient_scaling)

        # Sample texture at a given location
        def sample2d(self, texc):
            # _texc = (texc.view(-1, 2) - self.AABB[0][None, ...]) / (self.AABB[1][None, ...] - self.AABB[0][None, ...])
            _texc = texc.view(-1, 2)
            _texc = 0.5 + 0.49 * _texc
            _texc = torch.clamp(_texc, min=0, max=1)
            
            p_enc = self.encoder(_texc.contiguous())
            out = self.net.forward(p_enc)
            # Sigmoid limit and scale to the allowed range
            out = torch.sigmoid(out) * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]
            return out.view(*texc.shape[:-1], self.channels) # Remap to [n, h, w, c]

        def cleanup(self):
            tcnn.free_temporary_memory()            
        
        def export_image(self, h=512, w=512):
            device = next(self.parameters()).device
            query_point = torch.stack(torch.meshgrid(torch.linspace(-1, 1, w), torch.linspace(-1, 1, h)), dim=-1).reshape(-1, 2)[..., [1, 0]].to(device)
            return self.sample2d(query_point).reshape(h, w, -1)

def get_modules(FLAGS):
    pass
    device = torch.device("cuda")

    ################# Camera Model ######################
    cam_pose = Camera( cam_t3d = [0, 0, -10], fov_degree = 15).to(device)
    cam_pose.requires_grad_(False)
    cam_pose.obj_rot.requires_grad_(True)
    cam_pose.obj_t3d.requires_grad_(True)

    ################# Geometry Model ######################
    geometry = ParameterGeometry(FLAGS.checkpoint, FLAGS.mesh_scale, FLAGS.mesh_shift).to(device)
    print(geometry)

    ################# Texture Model ######################
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    mlp_min = torch.cat((kd_min[0:3], ks_min), dim=0)
    mlp_max = torch.cat((kd_max[0:3], ks_max), dim=0)

    material = MLPTexture2D(None, channels=6, min_max=[mlp_min, mlp_max]).to(device)

    h, w = 512, 512
    optimizer = torch.optim.AdamW(material.parameters(), lr=1e-3, betas=(0.9, 0.99))
    query_point = torch.stack(torch.meshgrid(torch.linspace(-1, 1, w), torch.linspace(-1, 1, h)), dim=-1).reshape(-1, 2)[..., [1, 0]].to(device)
    if FLAGS.tex_file is not None:
        initial_material = imageio.imread(FLAGS.tex_file)/255.
    else:
        initial_material = np.full((h, w, 6), 0.5)

    if initial_material.shape[:2] != (h, w):
        initial_material = cv2.resize(initial_material, (h, w), interpolation=cv2.INTER_AREA)
    initial_material = torch.as_tensor(initial_material).float().to(device)
    for i in range(100):
        queryed_material = material.sample2d(query_point)[..., :3].reshape(h, w, -1)
        F.mse_loss(queryed_material, initial_material).backward()
        optimizer.step()
        optimizer.zero_grad()

    ################# Environment Model ######################
    environment = None

    return {
        "cam_pose": cam_pose,
        "geometry": geometry,
        "material": material,
        "envlight": environment,
    }

def main(args):
    device = torch.device("cuda")

    device_control.make_deterministic(seed=args.seed)

    deterministic = True
    torch.backends.cudnn.benchmark     = deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

    low_precision = True
    torch.backends.cuda.matmul.allow_tf32 = low_precision
    torch.backends.cudnn.allow_tf32       = low_precision
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = low_precision

    module_dict = get_modules(args)

    cam_pose, geometry = module_dict["cam_pose"], module_dict["geometry"]
    material, envlight = module_dict["material"], module_dict["envlight"]

    BACKGROUND = "white"

    @lru_cache
    def get_background(background, batch_size, resolution):
        if background == 'white':
            # return torch.ones(batch_size, resolution[0], resolution[1], 3, dtype=torch.float32, device=device) 
            return torch.ones((1,1,1,3), dtype=torch.float32, device=device) 
        if background == 'black':
            # return torch.zeros(batch_size, resolution[0], resolution[1], 3, dtype=torch.float32, device=device) 
            return torch.zeros((1,1,1,3), dtype=torch.float32, device=device) 

    def prepare_batch(target, background='black'):
        # prompt

        objmatx, cammatx, ndcproj = cam_pose()

        target['objpose'] = objmatx.unsqueeze(0)
        target['campose'] = cammatx.unsqueeze(0)
        target['ndcproj'] = ndcproj.unsqueeze(0)

        batch_size = 1
        resolution = (512, 512)
        target["background"] = get_background(background, batch_size, resolution)
        target["resolution"] = resolution
        target["spp"]        = 1
        return target
    
    image = imageio.imread(args.fitting_image)

    # mask & lmk
    with nullcontext("initial"):
        import mediapipe as mp

        face_tracker = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        results = face_tracker.process(image)
        lmk_list = []
        for idx, res in enumerate(results.multi_face_landmarks):
            # print(res, dir(res))
            lmk  = [[p.x, p.y] for p in res.landmark]
            lmk_list.append(lmk)
        
        lmk_mp = np.array(lmk_list, dtype=np.float32)

        gt_lmk = torch.as_tensor(lmk_mp, dtype=torch.float32).to(device)  # 1, N, 2

        gt_img = torch.as_tensor([image/255.], dtype=torch.float32).permute(0, 3, 1, 2)
        gt_img = F.interpolate(gt_img, (512, 512), mode="bilinear", align_corners=False)
        gt_img = gt_img.permute(0, 2, 3, 1).clone().detach().to(device)

        print(gt_lmk.shape, gt_lmk.min(), gt_lmk.max())

    with nullcontext("landmark"):
        from easydict import EasyDict
        config = EasyDict({
                "flame_model_path":        os.path.join(ROOT, "Data", "FLAME2020", "generic_model.pkl"),
                "flame_lmk_embedding_path":os.path.join(ROOT, "Data", "FLAME2020", "landmark_embedding.npy"),
                "flame_tex_path":          os.path.join(ROOT, "Data", "FLAME2020", "FLAME_texture.npz"),
                "tex_path":                os.path.join(ROOT, "Data", "FLAME2020", "FLAME_albedo_from_BFM.npz"),
                "flame_mediapipe_lmk_embedding_path": os.path.join(ROOT, "Data", "FLAME2020", "mediapipe_landmark_embedding.npz"),
                # "flame_gaze_lmk_embedding_path":      os.path.join(ROOT, "Data", "FLAME2020", "gaze_landmark_embedding2.npz"),
                "n_shape": 100,
                "n_exp":   50,
                "n_tex":   50,
                "tex_type": 'BFM',
            })
        lmk_embeddings_mediapipe  = np.load(config.flame_mediapipe_lmk_embedding_path, allow_pickle=True, encoding='latin1')
        lmk_faces_idx_mediapipe   = torch.tensor(lmk_embeddings_mediapipe['lmk_face_idx'].astype(np.int64), dtype=torch.long).to(device)
        lmk_bary_coords_mediapipe = torch.tensor(lmk_embeddings_mediapipe['lmk_b_coords'], dtype=torch.float32).to(device)

        # lmk_embeddings_gaze  = np.load(config.flame_gaze_lmk_embedding_path, allow_pickle=True, encoding='latin1')
        # lmk_faces_idx_gaze   = torch.tensor(lmk_embeddings_gaze['lmk_face_idx'].astype(np.int64), dtype=torch.long).to(device)
        # lmk_bary_coords_gaze = torch.tensor(lmk_embeddings_gaze['lmk_b_coords'], dtype=torch.float32).to(device)

        # emoca/gdl/layers/losses/MediaPipeLandmarkLosses.py
        used_mediapipe_indices = torch.as_tensor([276, 282, 283, 285, 293, 295, 296, 300, 334, 336,  46,  52,  53,
            55,  63,  65,  66,  70, 105, 107, 249, 263, 362, 373, 374, 380,
            381, 382, 384, 385, 386, 387, 388, 390, 398, 466,   7,  33, 133,
            144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
            168,   6, 197, 195,   5,   4, 129,  98,  97,   2, 326, 327, 358,
            0,  13,  14,  17,  37,  39,  40,  61,  78,  80,  81,  82,  84,
            87,  88,  91,  95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
            308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
            415], dtype=torch.long).to(device)

        # https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
        # rightEyeIris: [473, 474, 475, 476, 477],
        # leftEyeIris: [468, 469, 470, 471, 472],
        used_mediapipe_gz_indices = torch.as_tensor([468, 473], dtype=torch.long).to(device)

        # self.register_buffer("lmk_mp_face_idx",   lmk_faces_idx_mediapipe)
        # self.register_buffer("lmk_mp_bary_wgt",   lmk_bary_coords_mediapipe)
        # self.register_buffer("lmk_gaze_face_idx", lmk_faces_idx_gaze)
        # self.register_buffer("lmk_gaze_bary_wgt", lmk_bary_coords_gaze)

        # self.register_buffer("used_mediapipe_indices",    used_mediapipe_indices)
        # self.register_buffer("used_mediapipe_gz_indices", used_mediapipe_gz_indices)

        # lmk_wgt[:, used_mediapipe_indices, :] = 1

        def surface_sample_and_project(ver, tri, lmk_face_idx, lmk_bary_wgt, m2v, ndc):
            BS = ver.size(0)

            # surface sample
            face_vi3 = tri[None, lmk_face_idx, :] # 1, N, 3
            face_v33 = torch.gather(ver, 1, face_vi3.flatten(1).unsqueeze(-1).expand(BS, -1, 3)).unflatten(1, (-1, 3)) # B, N, 3, 3
            surf_pnt = torch.einsum('bnkc,nk->bnc', [face_v33, lmk_bary_wgt])

            # project 2d
            homo_v = F.pad(surf_pnt, (0, 1), "constant", 1)
            proj_v = (homo_v @ (m2v.transpose(-1, -2) @ ndc.transpose(-1, -2)))
            proj_d = proj_v / proj_v[..., 3:]

            return surf_pnt, proj_d

    optim  = torch.optim.AdamW([
        {"name": "position", "params": [p for p in cam_pose.parameters() if p.requires_grad is True], "lr": 1e-2, "betas": (0.9, 0.99)},
        {"name": "geometry", "params": [p for p in geometry.parameters() if p.requires_grad is True], "lr": 1e-2, "betas": (0.9, 0.99)},
        {"name": "material", "params": material.parameters(), "lr": 3e-3, "betas": (0.9, 0.99)},
    ])
    for pg in optim.param_groups:
        num_p = np.sum([p.numel() for p in pg["params"]])
        print(f"[{pg['name']}] #params:{num_p/2**20:.4}MB")
    scaler = torch.cuda.amp.GradScaler(enabled=True)  
    
    SAVE_ROOT = os.path.join(ROOT, "temp", f"APP-fit_2d", f"{args.name}_seed{args.seed}")
    os.makedirs(SAVE_ROOT, exist_ok=True)

    NUM_STEPS = args.iter
    for it in tqdm(range(NUM_STEPS), desc=f"{args.name}_seed{args.seed}"):

        with record_function("data"):
            target = prepare_batch({}, BACKGROUND)

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            with record_function("render"):
                result_dict = geometry.render(target, material, envlight)

            rd_img = result_dict["image"]
            # gen_rgb_512 = result_dict["image"].permute(0, 3, 1, 2) # [B, 3, 512, 512]

            kpt3d, kpt2d = surface_sample_and_project(
                result_dict["ver"], result_dict["tri"], 
                lmk_faces_idx_mediapipe, lmk_bary_coords_mediapipe,
                target["campose"]@target["objpose"], target["ndcproj"])
                # torch.linalg.inv(target["campose"])@target["objpose"], target["ndcproj"])
            
            rd_lmk = kpt2d[..., :2]*0.5 + 0.5

            kpt_loss = ((gt_lmk[:, used_mediapipe_indices] - rd_lmk).square()).mean()
            img_loss = F.l1_loss(gt_img, rd_img)
            reg_loss = geometry.g_param_0.flatten().norm() + geometry.g_param_1.flatten().norm() + geometry.deform.flatten().norm()

            # loss = kpt_loss + img_loss + 0.1*reg_loss
            if it % 10 < 6:
                loss = 100*kpt_loss + 0.0001*reg_loss
            else:
                loss = 100*kpt_loss + img_loss + 0.0001*reg_loss
        
        with record_function("backward"):
            scaler.scale(loss).backward()

        with record_function("optimize"):
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            
        # visualization
        if it % 100 == 0 or it in [NUM_STEPS-1]:
            with torch.no_grad():
                target = prepare_batch({}, BACKGROUND)

                batch_size = target["campose"].size(0)

                print("loss", kpt_loss, img_loss, reg_loss)
                print("g param 0", geometry.g_param_0.flatten())
                print("g param 1", geometry.g_param_1.flatten())
                print("deform norm", geometry.deform.flatten().norm())

                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    result_dict = geometry.render(target, material, envlight)
                    img  = result_dict["image"][..., 0:3]
                    nrm  = result_dict["normal"][..., 0:3]
                    dpt  = result_dict["depth"]

                    pred_rgb_512    = img.permute(0, 3, 1, 2).contiguous() # [B, 3, 512, 512]
                    normal_512      = 0.5+0.5*nrm.permute(0, 3, 1, 2)
                    depth_512       = depth_norm_0_1(dpt).permute(0, 3, 1, 2).expand(-1, 3, -1, -1)

                    kpt3d, kpt2d = surface_sample_and_project(
                        result_dict["ver"], result_dict["tri"], 
                        lmk_faces_idx_mediapipe, lmk_bary_coords_mediapipe,
                        target["campose"]@target["objpose"], target["ndcproj"])
                        # torch.linalg.inv(target["campose"])@target["objpose"], target["ndcproj"])
                    
                    rd_lmk = kpt2d[..., :2]*0.5 + 0.5

                imgs = torch.cat([gt_img.permute(0, 3, 1, 2), pred_rgb_512, normal_512, depth_512], dim=0) # 3*B, 3, H, W
                (H,W),N= imgs.shape[2:], imgs.size(0) // batch_size

                # time_s = valid_scheduler.timesteps[t_index].expand(batch_size).detach().cpu().numpy()
                
                imgs = imgs.unflatten(0, (-1, batch_size)).permute(1, 3, 0, 4, 2).reshape(batch_size*H, N*W, -1) # N*B, 3, H, W -> B*H, N*W, 3
                imgs = (255*imgs.clamp(0, 1)).detach().cpu().numpy().astype(np.uint8)

                rd_lmk = rd_lmk.detach().cpu().numpy()
                for bi in range(batch_size):
                    for ki in range(rd_lmk.shape[1]):
                        x, y = int(rd_lmk[bi, ki, 0]*W), int(rd_lmk[bi, ki, 1]*H)
                        cv2.circle(imgs, (x+W, y), 3, (220, 220, 20), -1)

                print(imgs.shape)
                os.makedirs(SAVE_ROOT, exist_ok=True)
                cv2.imwrite(os.path.join(SAVE_ROOT, f"valid-{it}.png"), cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR))

                # writer.add_image("valid", imgs, it, dataformats="HWC")

    # generate mesh
    if args.save_ply is True:
        tex_image = material.export_image().detach().cpu().numpy()
        mesh_dict = { k:v.detach().cpu().numpy() for k, v in geometry.export_mesh().items() }

        print(tex_image.shape)
        imageio.imwrite(os.path.join(SAVE_ROOT, "texture.png"), tex_image[..., :3])

        save_mesh(os.path.join(SAVE_ROOT, "mesh.ply"), 
            (mesh_dict["ver"].astype(np.float), mesh_dict["uv"].astype(np.float),    None,
            mesh_dict["tri"].astype(np.uint32), mesh_dict["uv_tri"].astype(np.uint32), None),
            [f"TextureFile texture.png"]
            )

    # generate animation
    if args.save_gif is True:
        with torch.no_grad():
            frame_list = []

            target  = prepare_batch({}, BACKGROUND)
            objpose = target["objpose"]
            campose = target["campose"]

            for deg in np.linspace(0, 720, 100):  # same as tex_2d.py
                r = np.deg2rad(deg)
                rot = np.array([
                    [ np.cos(r), 0, np.sin(r),    0],
                    [         0, 1,         0,    0],
                    [-np.sin(r), 0, np.cos(r),    0],
                    [         0, 0,         0,    1],
                ])

                target["campose"][:,:3,:3] = torch.as_tensor([rot[:3,:3]], dtype=torch.float32, device=device)

                batch_size = target["campose"].size(0)

                result_dict = geometry.render(target, material, envlight)
                image = result_dict["image"][..., 0:3]
                normal= 0.5+0.5*result_dict["normal"]

                image = torch.cat([image, normal], dim=2).clamp(0, 1)
                
                frame_list.extend((255*image.detach().cpu().numpy()).astype(np.uint8))
        
            imageio.mimsave(os.path.join(SAVE_ROOT, f"rot-{it}.mp4"), frame_list, fps=25)
            imageio.mimsave(os.path.join(SAVE_ROOT, f"rot-{it}.gif"), frame_list, fps=25)

            # change expression
            frame_list = []
            origin_g_param_1 = geometry.g_param_1.clone()

            target = prepare_batch({}, BACKGROUND)
            batch_size = target["campose"].size(0)

            for s in tqdm( (1+np.sin(np.linspace(0, 2*np.pi, 200))).tolist() ):  # change expression intensity from 1 -> 2 -> 0 -> 1
                geometry.g_param_1.copy_(s*origin_g_param_1)
                result_dict = geometry.render(target, material, envlight)
                image = result_dict["image"][..., 0:3]
                normal= 0.5+0.5*result_dict["normal"]

                image = torch.cat([image, normal], dim=2).clamp(0, 1) # B,H,2*W,3
                image = image.flatten(0, 1)                           # B*H,2*W,3
                
                frame_list.append((255*image.detach().cpu().numpy()).astype(np.uint8))
            
            geometry.g_param_1.copy_(origin_g_param_1)
            imageio.mimsave(os.path.join(SAVE_ROOT, f"exp-{it}.mp4"), frame_list, fps=25)
            imageio.mimsave(os.path.join(SAVE_ROOT, f"exp-{it}.gif"), frame_list, fps=25)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--config',      type=str, default=None, help="config file")
    parser.add_argument("--checkpoint",  type=str, required=True, default=None, help="checkpoint for WSDF")

    parser.add_argument('--seed',   type=int, default=0, help="seed for random")
    parser.add_argument('--name',   type=str, default="default")

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--iter',       type=int, default=1000)

    parser.add_argument('--cfg_weight', type=float, default=100)

    # Texture/Geometry
    parser.add_argument('--tex_file',   type=str, default=None)                     # default for FLAME2020
    parser.add_argument('--mesh_scale', type=float, default=0.01)                   
    parser.add_argument('--mesh_shift', nargs=3, type=float, default=[0, 0, 0.5])

    # Fitting Target
    parser.add_argument('--fitting_image', type=str, default=None)

    # Saving
    parser.add_argument("--save_gif", type=lambda x:eval(x), default=True)
    parser.add_argument("--save_ply", type=lambda x:eval(x), default=True)

    args = parser.parse_args()

    args.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    args.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    args.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    args.ks_max              = [ 1.0,  1.0,  1.0]

    if args.config is not None:
        import yaml
        data = yaml.load(open(args.config, 'r'), yaml.FullLoader)
        for key in data:
            args.__dict__[key] = data[key]

    main(args)