import torch

def chunk_fn(fn, chunk_size, input_tensor_list, *args, **kwargs):
    input_chunk_list = [torch.split(t, chunk_size, dim=0) for t in input_tensor_list]

    all_ret = []
    for i, input_tuple in enumerate(zip(*input_chunk_list)):
        r = fn(*input_tuple, *args, **kwargs)
        all_ret.append(r)
    
    r = all_ret[0]
    if len(all_ret) == 1:
        return r

    if isinstance(r, (list, tuple)):
        # ret = list(map(lambda i: torch.cat([ one[i] for one in all_ret], dim=0), range(len(r)) ))
        ret = [ torch.cat([one[i] for one in all_ret], dim=0) for i in range(len(r)) ]
    elif isinstance(r, (dict,)):
        # ret = dict(map(lambda k: (k,torch.cat([ one[k] for one in all_ret], dim=0)), r.keys() ))
        ret = { k: torch.cat([one[k] for one in all_ret], dim=0) for k in r.keys() }
    else:
        ret = torch.cat(all_ret, dim=0)
    return ret