import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

def make_deterministic(seed):
    import numpy as np
    import random

    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

ddp_enabled = False

def init_device(enable_ddp=True):

    if enable_ddp:
        dist.init_process_group("nccl")
        # dist.init_process_group("gloo")
        rank       = dist.get_rank()
        device_id  = rank % torch.cuda.device_count()

        print(f"WORLD: {dist.get_world_size()} RANK: {rank} DEV: {device_id}")
        if rank != 0:
            print(f"RANK[{rank}] set stdout to NULL")
            sys.stdout.flush()
            sys.stdout = open(os.devnull, "w")
        else:
            sys.stdout.flush()
    else:
        device_id = 0

    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    # torch.set_default_device(device)

    return device

def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1

def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0

class DDPWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        def convert_nograd_params_to_buffers(module):
            ret = [[pn, prm] for pn, prm in module.named_parameters(recurse=False) if prm.requires_grad is False]
            for pn, prm in ret:
                delattr(module, pn)
                module.register_buffer(pn, prm.data.clone())
            for m in module.children():
                convert_nograd_params_to_buffers(m)

        for k, v in kwargs.items():
            if dist.is_initialized():
                v = torch.nn.SyncBatchNorm.convert_sync_batchnorm(v)
            convert_nograd_params_to_buffers(v)
            setattr(self, k, v)

        t_size = 0
        ddp_ignore = list()
        for n,ps in self.named_parameters():
            p_size = ps.numel()
            t_size += p_size
            if p_size == 0:
                ddp_ignore.append(n)
        self._ddp_params_and_buffers_to_ignore = ddp_ignore

        self.num_iterations = 0

    def forward(self, params):
        # self.num_iterations += 1
        # return next(self.parameters()) + x
        return torch.stack([p.mean() for p in params])

def send_model_to_device(model_or_dict, device):
    from contextlib import nullcontext, contextmanager

    mdc = None  # multi device context
    mpd = None  # main process decorator
    ret = model_or_dict

    if dist.is_initialized():

        if isinstance(model_or_dict, (dict,)):
            whole_model = DDPWrapper(**model_or_dict)

            whole_model.to(device)

            t_size = 0
            ddp_ignore = list()
            for n, ps in whole_model.named_parameters():
                p_size = ps.numel()
                t_size += p_size
                if p_size == 0:
                    ddp_ignore.append(n)

            whole_model._ddp_params_and_buffers_to_ignore = ddp_ignore

            # whole_model_ddp = DDP(whole_model, device_ids=[device.index], bucket_cap_mb=128, find_unused_parameters=True)
            # whole_model = whole_model_ddp.module

            sync_ddp()

            for n, p in whole_model.named_parameters():
                if p.is_contiguous() is False:
                    print(f"{n} is not contiguous")
            for n, p in whole_model.named_buffers():
                if p.is_contiguous() is False:
                    print(f"{n} is not contiguous")

            for n, p in whole_model.named_parameters():
                if p not in ddp_ignore:
                    broadcast_to_all(p, source=0)
            for n, b in whole_model.named_buffers():
                if b not in ddp_ignore:
                    broadcast_to_all(b, source=0)
            
            sync_ddp()

            num_iterations = 0
            @contextmanager
            def ddp_forward(params):
                # whole_model_ddp(params)
                nonlocal num_iterations
                if (num_iterations ) % 1000 == 0:
                    sync_check(whole_model)
                num_iterations += 1
                yield
                return
                # try:
                    # # whole_model_ddp(0)
                    # print("fake forward")
                    # # yield whole_model_ddp
                    # yield None
                    # print("after yield")
                    # nonlocal num_iterations
                    # if (num_iterations ) % 10 == 0:
                    #     sync_check(whole_model)
                    # num_iterations += 1
                    # return
                # except Exception as ex:
                    # import traceback
                    # print(traceback.format_exc())
            
            rank = dist.get_rank()

            if rank in [-1, 0]:
                def ddp_zeroonly(func):
                    return func
            else:
                def ddp_zeroonly(func):
                    def noop(*args, **kwargs):
                        return None
                    return noop

            mdc = ddp_forward
            mpd = ddp_zeroonly
            ret = {k:getattr(whole_model, k) for k in model_or_dict}
    else:
        mdc = nullcontext
        mpd = (lambda x:x)
        if isinstance(model_or_dict, (dict,)):
            ret = {k:v.to(device) for k,v in model_or_dict.items()}
    return mdc, mpd, ret

def sync_ddp():
    if dist.is_initialized():
        dist.barrier()

def broadcast_to_all(value, source=0):
    dist.broadcast(value, src=source)
    return value

# name2op = {
#     "mean": dist.ReduceOp.AVG,
#     "sum":  dist.ReduceOp.SUM
# }
def reduce_to_one(value, reduction="mean", 
    name2op={"mean": dist.ReduceOp.AVG, "sum": dist.ReduceOp.SUM}, target="all"):
    reduce_op = name2op[reduction]
    if target == "all":
        async_worker = dist.all_reduce(value, op=reduce_op)
    elif isinstance(target, (int,)):
        async_worker = dist.reduce(value, target, op=reduce_op)
    return value

def broadcast_optimizer(optimizer):
    if dist.is_initialized():
        workers = []
        params  = []
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                if p.grad is not None:
                    params.append(p)
        
        with torch.no_grad():
            total = torch.cat([p.grad.to(memory_format=torch.contiguous_format).flatten() for p in params])
            # dist.all_reduce(total, op=dist.ReduceOp.AVG, async_op=False)
            workers.append(dist.all_reduce(total, op=dist.ReduceOp.AVG, async_op=True))

            for p, grad in zip(params, torch.split(total, [np.prod(p.shape, dtype=np.int64, initial=1) for p in params])):
                p.grad = grad.reshape(p.grad.shape)
                # p.grad.copy_(grad.reshape(p.grad.shape))

            # for p in params:
            #     workers.append(dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True))
        
        for w in workers:
            w.wait()
        # dist.barrier()

def gather(value, target="all"):
    if target == "all":
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                value_o = torch.cat([value]*dist.get_world_size())
                dist.all_gather_into_tensor(value_o, value)
                ret = value_o
            else:
                tensor_list = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list, value)
                return torch.cat(tensor_list)
    elif isinstance(target, int):
        gather_list = [torch.empty_like(value) for _ in dist.get_world_size()]
        dist.gather(value, gather_list, target)
        ret = torch.cat(gather_list)
    dist.barrier() 
    return ret

def sync_check(model):
    # print("ddp sync checking")
    with torch.no_grad():
        for name, param in model.named_parameters():
            w = param.data.to(memory_format=torch.contiguous_format)

            tensor_list = [torch.zeros_like(w) for i in range(dist.get_world_size())]
            dist.all_gather(tensor_list, w)

            nl, nr = 0, 1
            tl, tr = tensor_list[nl], tensor_list[nr]
            close = torch.allclose(tl, tr)
            if not close:
                diff = (tl - tr).abs().mean()
                print(f"!!! Parameter Diff between DDP nodes {nl}/{nr} !!! Name:{name} Diff:{diff.item()}")