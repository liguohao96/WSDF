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

# ./
from sd_guidance import StableDiffusion, PerpNegUtil

record_function = torch.profiler.record_function

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, ROOT)
from utils import device_control
from utils.io.mesh_io import save_mesh
from utils.load import load_wsdf
from utils.mesh_tool import compute_normal
sys.path.pop(0)

with nullcontext("utils"):
    # code from https://github.com/Gorilla-Lab-SCUT/Fantasia3D

    class Fantasia3DUtil:
        #----------------------------------------------------------------------------
        # Matrix helpers.
        #----------------------------------------------------------------------------

        def focal_length_to_fovy(focal_length, sensor_height):
            return 2 * np.arctan(0.5 * sensor_height / focal_length)

        # Reworked so this matches gluPerspective / glm::perspective, using fovy
        def perspective(fovy=0.7854, aspect=1.0, n=0.1, f= 1000.0, device=None):
            y = np.tan(fovy / 2)
            return torch.tensor([[1/(y*aspect),    0,            0,              0], 
                                 [           0, 1/-y,            0,              0], 
                                 [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                                 [           0,    0,           -1,              0]], dtype=torch.float32, device=device)

        # Reworked so this matches gluPerspective / glm::perspective, using fovy
        def perspective_offcenter(fovy, fraction, rx, ry, aspect=1.0, n=0.1, f=1000.0, device=None):
            y = np.tan(fovy / 2)

            # Full frustum
            R, L = aspect*y, -aspect*y
            T, B = y, -y

            # Create a randomized sub-frustum
            width  = (R-L)*fraction
            height = (T-B)*fraction
            xstart = (R-L)*rx
            ystart = (T-B)*ry

            l = L + xstart
            r = l + width
            b = B + ystart
            t = b + height

            # https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix
            return torch.tensor([[2/(r-l),        0,  (r+l)/(r-l),              0], 
                                 [      0, -2/(t-b),  (t+b)/(t-b),              0], 
                                 [      0,        0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                                 [      0,        0,           -1,              0]], dtype=torch.float32, device=device)

        def translate(x, y, z, device=None):
            return torch.tensor([[1, 0, 0, x], 
                                 [0, 1, 0, y], 
                                 [0, 0, 1, z], 
                                 [0, 0, 0, 1]], dtype=torch.float32, device=device)

        def rotate_x(a, device=None):
            s, c = np.sin(a), np.cos(a)
            return torch.tensor([[1,  0, 0, 0], 
                                 [0,  c, s, 0], 
                                 [0, -s, c, 0], 
                                 [0,  0, 0, 1]], dtype=torch.float32, device=device)

        def rotate_x_1(a, device=None):
            s, c = np.sin(a), np.cos(a)
            return torch.tensor([[1,  0, 0], 
                                 [0,  c, s], 
                                 [0, -s, c]], dtype=torch.float32, device=device)


        def rotate_y(a, device=None):
            s, c = np.sin(a), np.cos(a)
            return torch.tensor([[ c, 0, s, 0], 
                                 [ 0, 1, 0, 0], 
                                 [-s, 0, c, 0], 
                                 [ 0, 0, 0, 1]], dtype=torch.float32, device=device)

        def rotate_y_1(a, device=None):
            s, c = np.sin(a), np.cos(a)
            return torch.tensor([[ c, 0, s], 
                                 [ 0, 1, 0], 
                                 [-s, 0, c]], dtype=torch.float32, device=device)

        def rotate_y_2(a, device=None):
            s, c = np.sin(a), np.cos(a)
            return np.array([[ c, 0, s], 
                                 [ 0, 1, 0], 
                                 [-s, 0, c]])

        def rotate_x_2(a, device=None):
            s, c = np.sin(a), np.cos(a)
            return np.array([[1,  0, 0], 
                                 [0,  c, s], 
                                 [0, -s, c]])

    from torch.cuda.amp import custom_bwd, custom_fwd 
    class SpecifyGradient(torch.autograd.Function):
        @staticmethod
        @custom_fwd
        def forward(ctx, input_tensor, gt_grad):
            ctx.save_for_backward(gt_grad)
            # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
            return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_scale):
            gt_grad, = ctx.saved_tensors
            gt_grad = gt_grad * grad_scale
            return gt_grad, None

    def depth_norm_0_1(depth):
        df = depth.flatten(1)
        min, max = df.min(1, keepdim=True).values, df.max(1, keepdim=True).values

        df = (df - min) / (max - min)
        return df.reshape(depth.shape)

with nullcontext("dataset"):
    # code from https://github.com/Gorilla-Lab-SCUT/Fantasia3D

    class Fantasia3DDatasetMesh(torch.utils.data.Dataset):

        def __init__(self, FLAGS, validate=False, gif=False):
            # Init 
            self.FLAGS              = FLAGS
            self.validate           = validate
            self.gif                = gif
            self.aspect             = FLAGS.train_res[1] / FLAGS.train_res[0]
            self.fovy_range_min     = np.deg2rad(FLAGS.fovy_range[0])
            self.fovy_range_max     = np.deg2rad(FLAGS.fovy_range[1])
            self.elevation_range_min= np.deg2rad(FLAGS.elevation_range[0])
            self.elevation_range_max= np.deg2rad(FLAGS.elevation_range[1])
            self.angle_front        = np.deg2rad(FLAGS.front_threshold)
            self.cam_radius         = FLAGS.cam_radius

            self.pos_prompt_s = []
            self.neg_prompt_s = []

            if FLAGS.add_directional_text:
                for d in ['front', 'side', 'back', 'side']:
                    text = f"{FLAGS.text}, {d} view"
                    negative_text = f"{FLAGS.negative_text}"
                    self.pos_prompt_s.append(text)
                    self.neg_prompt_s.append(negative_text)
            else: 
                self.pos_prompt_s.append(FLAGS.text)
                self.neg_prompt_s.append(FLAGS.negative_text)
            
            print(self.pos_prompt_s)

        def _gif_scene(self, itr):
            fovy = np.deg2rad(45)
            proj_mtx = Fantasia3DUtil.perspective(fovy, self.FLAGS.display_res[1] / self.FLAGS.display_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

            rotate_y = (itr / 50) * np.pi * 2
            rotate_x =  np.deg2rad(0)
            trans_3d = [0, 0, -self.cam_radius]
            center_p = [0, 0, 0]
            prompt_base = 0

            mv = Fantasia3DUtil.translate(*trans_3d) @ (Fantasia3DUtil.rotate_x(rotate_x) @ Fantasia3DUtil.rotate_y(rotate_y)) \
                  @ Fantasia3DUtil.translate(-1*center_p[0], -1*center_p[1], -1*center_p[0])

            # angle_front = np.deg2rad(45)
            if self.FLAGS.add_directional_text:
                prompt_index = get_view_direction(thetas= rotate_x, phis = rotate_y, front= self.angle_front)
            else:
                prompt_index = 0
            prompt_index = prompt_base + prompt_index

            pos_prompt, unc_prompt = self.pos_prompt_s[prompt_index], self.neg_prompt_s[prompt_index]
            all_prompt = {
                "default": pos_prompt,
                "uncond":  unc_prompt,
                "front": self.pos_prompt_s[prompt_base + 0],
                "side":  self.pos_prompt_s[prompt_base + 1],
                "back":  self.pos_prompt_s[prompt_base + 2],
            }

            pose_obj = {"fovy": fovy, "rotate_xy": [rotate_x, rotate_y], "trans_3d": trans_3d}
            pose_str = json.dumps(pose_obj)
            # normal_rotate =  Fantasia3DUtil.rotate_y_1(0)
            # mvp    = proj_mtx @ mv
            campos = torch.linalg.inv(mv)
            objpos = torch.eye(4)
            return objpos[None, ...], campos[None, ...], proj_mtx[None, ...], self.FLAGS.display_res, self.FLAGS.spp, pos_prompt, unc_prompt, all_prompt, pose_str

        def _validate_scene(self, itr):
            # face region
            fovy = np.deg2rad(45)
            proj_mtx = Fantasia3DUtil.perspective(fovy, self.FLAGS.train_res[1] / self.FLAGS.train_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
            # ang    = (itr / 4) * np.pi * 2
            rotate_y = (itr / 8) * np.pi * 2
            # rotate_x = np.random.uniform(-np.pi/4,np.pi/18)
            # rotate_x = np.random.uniform(np.pi/18,np.pi/6)

            rotate_x = np.random.uniform(-np.pi/18, np.pi/18)

            trans_3d = [0, 0, -self.cam_radius]
            center_p = [0, 0, 0]
            prompt_base = 0

            mv = Fantasia3DUtil.translate(*trans_3d) @ (Fantasia3DUtil.rotate_x(rotate_x) @ Fantasia3DUtil.rotate_y(rotate_y)) \
                  @ Fantasia3DUtil.translate(-1*center_p[0], -1*center_p[1], -1*center_p[0])
            
            # angle_front = np.deg2rad(45)
            if self.FLAGS.add_directional_text:
                prompt_index = get_view_direction(thetas= rotate_x, phis = rotate_y, front= self.angle_front)
            else:
                prompt_index = 0
            prompt_index = prompt_base + prompt_index

            pos_prompt, unc_prompt = self.pos_prompt_s[prompt_index], self.neg_prompt_s[prompt_index]
            all_prompt = {
                "default": pos_prompt,
                "uncond":  unc_prompt,
                "front": self.pos_prompt_s[prompt_base + 0],
                "side":  self.pos_prompt_s[prompt_base + 1],
                "back":  self.pos_prompt_s[prompt_base + 2],
            }

            pose_obj = {"fovy": fovy, "rotate_xy": [rotate_x, rotate_y], "trans_3d": trans_3d}
            pose_str = json.dumps(pose_obj)
            # normal_rotate =  Fantasia3DUtil.rotate_y_1(0)
            # mvp    = proj_mtx @ mv
            campos = torch.linalg.inv(mv)
            objpos = torch.eye(4)
            return objpos[None, ...], campos[None, ...], proj_mtx[None, ...], self.FLAGS.display_res, self.FLAGS.spp, pos_prompt, unc_prompt, all_prompt, pose_str

        def _train_scene(self, itr):
            fovy =  np.random.uniform(self.fovy_range_min, self.fovy_range_max)
            proj_mtx = Fantasia3DUtil.perspective(fovy, self.FLAGS.train_res[1] / self.FLAGS.train_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

            rotate_y = np.random.uniform(np.deg2rad(-45), np.deg2rad(45))

            rotate_x = -np.random.uniform(self.elevation_range_min, self.elevation_range_max)

            x = np.random.uniform(-self.FLAGS.camera_random_jitter, self.FLAGS.camera_random_jitter)
            y = np.random.uniform(-self.FLAGS.camera_random_jitter, self.FLAGS.camera_random_jitter)

            trans_3d = [x, y, -self.cam_radius]
            center_p = [0, 0, 0]
            prompt_base = 0

            mv = Fantasia3DUtil.translate(*trans_3d) @ (Fantasia3DUtil.rotate_x(rotate_x) @ Fantasia3DUtil.rotate_y(rotate_y)) \
                  @ Fantasia3DUtil.translate(-1*center_p[0], -1*center_p[1], -1*center_p[0])
            
            # angle_front = np.deg2rad(45)
            if self.FLAGS.add_directional_text:
                prompt_index = get_view_direction(thetas= rotate_x, phis = rotate_y, front= self.angle_front)
            else:
                prompt_index = 0
            prompt_index = prompt_base + prompt_index

            pos_prompt, unc_prompt = self.pos_prompt_s[prompt_index], self.neg_prompt_s[prompt_index]
            all_prompt = {
                "default": pos_prompt,
                "uncond":  unc_prompt,
                "front": self.pos_prompt_s[prompt_base + 0],
                "side":  self.pos_prompt_s[prompt_base + 1],
                "back":  self.pos_prompt_s[prompt_base + 2],
            }

            pose_obj = {"fovy": fovy, "rotate_xy": [rotate_x, rotate_y], "trans_3d": trans_3d}
            pose_str = json.dumps(pose_obj)
            # normal_rotate =  Fantasia3DUtil.rotate_y_1(0)
            # mvp    = proj_mtx @ mv
            campos = torch.linalg.inv(mv)
            objpos = torch.eye(4)
            return objpos[None, ...], campos[None, ...], proj_mtx[None, ...], self.FLAGS.display_res, self.FLAGS.spp, pos_prompt, unc_prompt, all_prompt, pose_str

        def __len__(self):
            if self.gif == True:
                return 100
            else:
                # return 4 if self.validate else (self.FLAGS.iter + 1) * self.FLAGS.batch
                return 8 if self.validate else 1000*self.FLAGS.batch_size

        def __getitem__(self, itr):
            if self.gif:
                objpose, campose, ndcproj, iter_res, iter_spp, pos_prompt, unc_prompt, all_prompt, pose_str = self._gif_scene(itr)
            elif self.validate:
                objpose, campose, ndcproj, iter_res, iter_spp, pos_prompt, unc_prompt, all_prompt, pose_str = self._validate_scene(itr)
            else:
                objpose, campose, ndcproj, iter_res, iter_spp, pos_prompt, unc_prompt, all_prompt, pose_str = self._train_scene(itr)

            return {
                'objpose' : objpose,
                'campose' : campose,
                'ndcproj' : ndcproj,
                'resolution' : iter_res,
                'spp' :        iter_spp,
                'pos_prompt': pos_prompt,
                'unc_prompt': unc_prompt,
                'all_prompt': all_prompt,
                'pose_str': pose_str
            }
        def collate(self, batch):
            iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']
            return {
                'objpose' : torch.cat(list([item['objpose'] for item in batch]), dim=0),
                'campose' : torch.cat(list([item['campose'] for item in batch]), dim=0),
                'ndcproj' : torch.cat(list([item['ndcproj'] for item in batch]), dim=0),
                'resolution' : iter_res,
                'spp' :        iter_spp,
                'pos_prompt' :   [item['pos_prompt'] for item in batch],
                'unc_prompt' :   [item['unc_prompt'] for item in batch],
                'all_prompt' :   [item['all_prompt'] for item in batch],
                'pose_str' :     [item['pose_str']   for item in batch]
            }

    @torch.no_grad()
    def get_view_direction(thetas, phis, front):
        #                   phis [B,];  -pi~pi        thetas: [B,] -pi/2~pi/2 
        # front = 0         [-front, front) 
        # side (left) = 1   [front, pi - front)
        # back = 2          [pi - front, pi) or [-pi, -pi+front)
        # side (right) = 3  [-pi+front, - front)

        if (phis >= -front) and (phis < front) :
            prompt_index = 0
        elif  (phis >= front ) and (phis < np.pi - front ):
            prompt_index = 1
        elif (phis >= np.pi - front) or  (phis < -np.pi + front):
            prompt_index = 2
        elif (phis >= -np.pi + front) and (phis < -front):
            prompt_index = 3

        return prompt_index

    class Fantasia3DDatasetMeshCfg(object):
        batch_size      = 1
        add_directional_text = False
        text            = ""
        negative_text   = ""
        fovy_range      = [30, 45]
        elevation_range = [-5, 5]
        train_res       = [512, 512]
        cam_radius      = 4
        front_threshold = [45]
        # cam_near_far    = [1, 50]
        cam_near_far    = [0.01, 10]

        display_res     = [512, 512]
        spp             = 1
        camera_random_jitter = 0.4

        def __init__(self, args) -> None:
            super().__init__()

            for k in vars(args):
                if hasattr(self, k):
                    setattr(self, k, getattr(args, k))

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

            o2w, w2v = objpose, torch.inverse(campose)

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

            return nvdiff_render(campose, ndcproj, objpose, ver, self.tri, self.uv, self.uv_tri, material, environment, H=H, W=W, SPP=SPP,
                                    background=background)

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

    ################# StableDiffusion ######################
    guidance = StableDiffusion(device = 'cuda',
                               sd_version = FLAGS.sd_version,
                               hf_key     = FLAGS.hf_key,
                               time_step_range     = FLAGS.time_step_range,
                               guidance_weight     = FLAGS.cfg_weight,
                               release_decoder     = False,
                               release_textencoder = False,
                               gradient_checkpoint = FLAGS.gradient_checkpoint,
                               )
    guidance.eval()
    for p in guidance.parameters():
        p.requires_grad_(False)

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
        "guidance": guidance,
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

    guidance, geometry = module_dict["guidance"], module_dict["geometry"]
    material, envlight = module_dict["material"], module_dict["envlight"]

    @lru_cache
    def get_pos_prompt(s):
        print(f"compute pos prompt '{s}'")
        return guidance.get_text_embeds([s], batch=1)

    @lru_cache
    def get_neg_prompt(s):
        print(f"compute neg prompt '{s}'")
        return guidance.get_uncond_embeds([s], batch=1)

    @lru_cache
    def get_background(background, batch_size, resolution):
        if background == 'white':
            # return torch.ones(batch_size, resolution[0], resolution[1], 3, dtype=torch.float32, device=device) 
            return torch.ones((1,1,1,3), dtype=torch.float32, device=device) 
        if background == 'black':
            # return torch.zeros(batch_size, resolution[0], resolution[1], 3, dtype=torch.float32, device=device) 
            return torch.zeros((1,1,1,3), dtype=torch.float32, device=device) 

    @torch.no_grad()
    def prepare_batch(target, background= 'black',it = 0,coarse_iter=0):
        # prompt

        text_z   = torch.cat([get_pos_prompt(t) for t in target["pos_prompt"]], dim=0)
        uncond_z = torch.cat([get_neg_prompt(t) for t in target["unc_prompt"]], dim=0)

        if args.is_perpneg:
            text_z_batch = []
            weight_batch = []
            for i in range(len(target["all_prompt"])):
                prompt = target["all_prompt"][i]
                embeddings  = { k:get_pos_prompt(v) for k, v in prompt.items()}
                azimuth_val = np.rad2deg(json.loads(target["pose_str"][i])["rotate_xy"][1])
                azimuth_val = azimuth_val if azimuth_val < 180 else azimuth_val - 360
                text_z, weight = PerpNegUtil.get_pos_neg_text_embeddings(embeddings, azimuth_val, 2, 10, -2)

                text_z_batch.append(text_z)
                weight_batch.append(weight)
            # target["text_z"]   = text_z
            target["text_z"]   = torch.stack(text_z_batch, dim=1).to(device, non_blocking=True) # K, B, ...
            target["weight"]   = torch.stack(weight_batch, dim=1).to(device, non_blocking=True) # K, B
            target["uncond_z"] = uncond_z  # B, ...
        else:
            text_z   = torch.cat([get_pos_prompt(t) for t in target["pos_prompt"]], dim=0)
            target["text_z"]   = text_z.unsqueeze(0) # K, B, ...
            target["uncond_z"] = uncond_z  # B, ...

        target['objpose'] = target['objpose'].to(device, non_blocking=True)
        target['campose'] = target['campose'].to(device, non_blocking=True)
        target['ndcproj'] = target['ndcproj'].to(device, non_blocking=True)

        batch_size = target['campose'].shape[0]
        resolution = tuple(target['resolution'])
        target["background"] = get_background(background, batch_size, resolution)
        return target

    CFG_WEIGHT = args.cfg_weight
    BACKGROUND = "white"

    FLAGS      = Fantasia3DDatasetMeshCfg(args)
    dataset_valid = Fantasia3DDatasetMesh(FLAGS, validate=True)
    dataset_train = Fantasia3DDatasetMesh(FLAGS, validate=False)
    dataset_gif   = Fantasia3DDatasetMesh(FLAGS, gif=True)

    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=FLAGS.batch_size, collate_fn=dataset_valid.collate)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch_size, collate_fn=dataset_train.collate)
    dataloader_gif   = torch.utils.data.DataLoader(dataset_gif,   batch_size=FLAGS.batch_size, collate_fn=dataset_train.collate, 
                                                    shuffle=False, drop_last=False)

    valid_scheduler = DDIMScheduler.from_config(guidance.model_key, subfolder="scheduler")
    valid_scheduler.set_timesteps(20)

    optim  = torch.optim.AdamW([
        {"name": "geometry", "params": [p for p in geometry.parameters() if p.requires_grad is True], "lr": 5e-2, "betas": (0.9, 0.99)},
        {"name": "material", "params": material.parameters(), "lr": 1e-3, "betas": (0.9, 0.99)},
    ])
    for pg in optim.param_groups:
        num_p = np.sum([p.numel() for p in pg["params"]])
        print(f"[{pg['name']}] #params:{num_p/2**20:.4}MB")
    scaler = torch.cuda.amp.GradScaler(enabled=True)  
    
    SAVE_ROOT = os.path.join(ROOT, "temp", f"APP-tex_3d", f"{args.name}_seed{args.seed}")
    os.makedirs(SAVE_ROOT, exist_ok=True)

    NUM_STEPS = args.iter
    for it in tqdm(range(NUM_STEPS), desc=f"{args.name}_seed{args.seed}"):

        with record_function("data"):
            try:
                target = next(train_iter)
            except:
                train_iter = iter(dataloader_train)
                target = next(train_iter)

            target = prepare_batch(target, BACKGROUND)

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            with record_function("render"):
                result_dict = geometry.render(target, material, envlight)
            gen_rgb_512 = result_dict["image"].permute(0, 3, 1, 2) # [B, 3, 512, 512]

            latents = guidance.encode_imgs(gen_rgb_512)
            assert latents.grad_fn is not None

            BS = latents.size(0)

            with record_function("guidance"):
                t = torch.randint(guidance.min_step, guidance.max_step + 1, [BS], dtype=torch.long)
                noise_pred, noise, w = guidance.guidance(latents, t, uncond_text_z=target["uncond_z"], cond_text_z=target["text_z"], 
                                cond_text_weight=target.get("weight", None))
                grad = noise_pred - noise
                w    = w.to(grad.device)

                grad     = torch.nan_to_num(w*grad)
                gd_loss  = SpecifyGradient.apply(latents, grad)     

            img_loss = 0 # torch.tensor([0], dtype=torch.float32, device="cuda")
            reg_loss = geometry.g_param_0.flatten().norm() + geometry.g_param_1.flatten().norm() + geometry.deform.flatten().norm()

            loss = gd_loss + reg_loss + img_loss
        
        with record_function("backward"):
            scaler.scale(loss).backward()

        with record_function("optimize"):
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            
        # visualization
        if it % 100 == 0 or it in [NUM_STEPS-1]:
            with torch.no_grad():
                try:
                    target = next(valid_iter)
                except:
                    valid_iter = iter(dataloader_valid)
                    target = next(valid_iter)

                target = prepare_batch(target, BACKGROUND)

                prompt = target["pos_prompt"]
                unc_pt = target["unc_prompt"]
                pose_s = target["pose_str"]

                batch_size = len(prompt)

                print("loss", gd_loss, img_loss, reg_loss)
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
                    latents         = guidance.encode_imgs(pred_rgb_512)
                    # text_embeddings = torch.cat([target["uncond_z"], target["text_z"]], dim=0)

                    # t_index = np.random.randint(10, len(valid_scheduler.timesteps), (1,)).item()
                    t_index = 10
                    t = valid_scheduler.timesteps[t_index].expand(batch_size)
                    time_s = t.detach().cpu().numpy().tolist()

                    # add noise
                    noise = torch.randn_like(latents)
                    codes = valid_scheduler.add_noise(latents, noise, t)

                    for ti, t in enumerate(valid_scheduler.timesteps[t_index:]):
                    # for ti, t in enumerate(valid_scheduler.timesteps[t_index:]):
                        with torch.no_grad():
                            noise_pred, noise, w = guidance.guidance(codes, t.expand(batch_size), target["uncond_z"], target["text_z"],
                                                                    cond_text_weight=target.get("weight", None), add_noise=False, scheduler=valid_scheduler, cfg_weight=7.5)
                            diffusion_pred = valid_scheduler.step(noise_pred, t, codes)
                            previous_noisy_sample = diffusion_pred.prev_sample
                            if ti == 0:
                                predict_code = diffusion_pred.pred_original_sample
                        codes = previous_noisy_sample
                
                with torch.no_grad():
                    imgs = guidance.decode_latents(torch.cat([latents, predict_code, codes], dim=0)) # k*B, 3, H, W

                imgs = torch.cat([pred_rgb_512, normal_512, depth_512, imgs], dim=0) # 3*B, 3, H, W
                (H,W),N= imgs.shape[2:], imgs.size(0) // batch_size

                # time_s = valid_scheduler.timesteps[t_index].expand(batch_size).detach().cpu().numpy()
                
                imgs = imgs.unflatten(0, (-1, batch_size)).permute(1, 3, 0, 4, 2).reshape(batch_size*H, N*W, -1) # N*B, 3, H, W -> B*H, N*W, 3
                imgs = (255*imgs.clamp(0, 1)).detach().cpu().numpy().astype(np.uint8)
                for bi in range(batch_size):
                    cv2.putText(imgs, f"prompt:{prompt[bi]}", (10, bi*H+20),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 20), 2)
                    cv2.putText(imgs, f"unc_pt:{unc_pt[bi]}", (10, bi*H+40),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 20), 2)
                    cv2.putText(imgs, f"pose_s:{pose_s[bi]}", (10, bi*H+60),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 20), 2)
                    cv2.putText(imgs, f"time_s:{time_s[bi]}", (10, bi*H+80),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 20), 2)
                print("t_index", t_index)
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
            for target in tqdm(dataloader_gif):
                target = prepare_batch(target, BACKGROUND)
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
            for target in dataloader_valid:
                target = prepare_batch(target, BACKGROUND)
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
                break
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

    # Dataset
    parser.add_argument('--cam_radius', type=float, default=4.0)

    parser.add_argument('--camera_random_jitter', type=float, default=0.4, help="A large value is advantageous for the extension of objects such as ears or sharp corners to grow.")
    parser.add_argument('--fovy_range',      nargs=2, type=float, default=[30, 45])
    parser.add_argument('--elevation_range', nargs=2, type=int, default=[-5, 5], help="The elevatioin range must in [-90, 90].")

    parser.add_argument('--time_step_range',  nargs=2, type=float, default=[0.02, 0.5], help="The time step range")

    parser.add_argument('--text',          type=str, default="", help="text prompt")
    parser.add_argument("--negative_text", type=str, default="", help="adding negative text can improve the visual quality in appearance modeling")
    parser.add_argument("--add_directional_text", type=lambda x:eval(x), default=True)

    # Guidance
    parser.add_argument('--is_perpneg', type=lambda x:eval(x), default=True)
    parser.add_argument("--sd_version", type=str, default="2.1")
    parser.add_argument("--hf_key",     type=str, default=None)
    parser.add_argument("--gradient_checkpoint", type=lambda x:eval(x), default=False)

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