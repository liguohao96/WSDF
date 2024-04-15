import os
import sys

import json

from tqdm import tqdm
from glob import glob
from natsort import natsorted

from collections import defaultdict

SEED = 210

def traverse_coma(rootdir, collect_fn):
    rootdir  = os.path.abspath(rootdir)
    pid_dirs = [f for f in natsorted(glob(os.path.join(rootdir, "*"))) if os.path.isdir(f)]

    total = 0
    total += len(glob(os.path.join(rootdir, "*", "*", "*.ply")))

    pbar = tqdm(total=total)

    current_si = 0
    current_vi = 0
    for pi, pid_dir in enumerate(pid_dirs):
        pid_name = os.path.split(pid_dir)[-1]
        vid_dirs = [f for f in natsorted(glob(os.path.join(pid_dir, "*"))) if os.path.isdir(f)]

        for vi, vid_dir in enumerate(vid_dirs):
            vid_name = os.path.split(vid_dir)[-1]
            obj_list = natsorted(glob(os.path.join(vid_dir, "*.ply")))

            for fi, obj_f in enumerate(obj_list):
                pbar.update()
                try:
                    # name = os.path.splitext(os.path.split(obj_f)[-1])[0]
                    name = os.path.relpath(os.path.abspath(obj_f), rootdir)

                    # vs, vt, vn, t_vs, t_vt, t_vn = mesh_io.load_mesh(obj_f)
                    # vs = vs.astype(np.float32)

                    # # save_ply(f"{pi}-{current_vi}-{fi}.org.ply", vs)
                    # # s_kpt = ver2kpt@vs
                    # s, r, t  = get_arap_sRT(vs, v_template) # estimate rigid pose with scale

                    # if do_scale:
                    #     # template is always in mm
                    #     if do_pose:
                    #         vs = vs.dot(s*r) + t
                    #     else:
                    #         vs = vs*s
                    # else:
                    #     # CoMA data in meter
                    #     if do_pose:
                    #         vs = (vs.dot(r) + t/s) * 1000
                    #     else:
                    #         vs = vs * 1000
                    # vs = vs.astype(np.float32)

                    collect_fn(current_si, name, pid_name, vid_name, current_vi, fi)

                    current_si += 1

                except Exception as ex:
                    import traceback
                    print(traceback.format_exc())
                    os._exit(1)
            current_vi += 1

def main(args):

    os.makedirs(args.out_dir, exist_ok=True)
    train_json = os.path.join(args.out_dir, f"{args.dataset}-{args.split_method}_train.json")
    valid_json = os.path.join(args.out_dir, f"{args.dataset}-{args.split_method}_valid.json")
    test_json  = os.path.join(args.out_dir, f"{args.dataset}-{args.split_method}_test.json")

    dataset_meta = {
        "pid_names":  [],
        "eid_names":  [],
        "total":      [],                # (name, (pid, eid, vid, fid))
        "neu_by_pid": defaultdict(list), # pid -> List[name]
    }

    def collect_fn(index, name, pid_name, eid_name, vid, fid):
        if pid_name in dataset_meta["pid_names"]:
            pid = dataset_meta["pid_names"].index(pid_name)
        else:
            pid = len(dataset_meta["pid_names"])
            dataset_meta["pid_names"].append(pid_name)

        if eid_name in dataset_meta["eid_names"]:
            eid = dataset_meta["eid_names"].index(eid_name)
        else:
            eid = len(dataset_meta["eid_names"])
            dataset_meta["eid_names"].append(eid_name)
        
        dataset_meta["total"].append((name, (pid, eid, vid, fid)))

        if eid_name in args.neuexp_names:
            dataset_meta["neu_by_pid"][pid].append(name)

    if args.dataset == "coma":
        traverse_coma(args.data_dir, collect_fn)

    dataset_meta["train"] = []
    dataset_meta["valid"] = []
    dataset_meta["test"]  = []

    if args.split_method == "coma_interpolation":
        for i, data in enumerate(dataset_meta["total"]):
            if (i % 100) < 10:
                dataset_meta["test"].append(data)
            else:
                dataset_meta["train"].append(data)

    if args.neutral_from == "first_frame":
        first_frame_by_pid_vid = defaultdict(lambda:(float('inf'), None)) # (pid, vid) -> (fid, idx)

        data_list = dataset_meta["total"] # if args.neutral_from_total else dataset_meta["train"]
        for i, data in enumerate(data_list):
            pid, eid, vid, fid = data[1]

            if fid < first_frame_by_pid_vid[(pid, vid)][0]:
                first_frame_by_pid_vid[(pid, vid)] = (fid, i)
        
        dataset_meta["neu_by_pid"] = defaultdict(list)

        for k, v in first_frame_by_pid_vid.items():
            pid, _ = k
            _, idx = v
            if idx is not None:
                fname, _ = data_list[idx]
                dataset_meta["neu_by_pid"][pid].append(fname)
    
    filterd_train = [[name, pevf] for name, pevf in dataset_meta["train"]]

    train_meta = {
        "pid_names":  dataset_meta["pid_names"],
        "labels":     filterd_train,             # (name, (pid, eid, vid, fid))
    }

    valid_meta = {
        "pid_names":  dataset_meta["pid_names"],
        "eid_names":  dataset_meta["eid_names"],
        "labels":     dataset_meta["valid"],     # (name, (pid, eid, vid, fid))
        "neu_by_pid": dataset_meta["neu_by_pid"]
    }

    test_meta = {
        "pid_names":  dataset_meta["pid_names"],
        "eid_names":  dataset_meta["eid_names"],
        "labels":     dataset_meta["test"],      # (name, (pid, eid, vid, fid))
        "neu_by_pid": dataset_meta["neu_by_pid"]
    }

    def save_if_not_exist(obj, out_json):
        if os.path.exists(out_json):
            print(f"find existing cache @ {out_json}, skipping...")
        else:
            with open(out_json, "w") as f:
                json.dump(obj, f)
    
    save_if_not_exist(train_meta, train_json)
    save_if_not_exist(valid_meta, valid_json)
    save_if_not_exist(test_meta, test_json)

    print("="*20)
    print(f"dataset name: {args.dataset}")
    print(f"dataset pids: [{len(dataset_meta['pid_names'])}] {dataset_meta['pid_names']}")
    print(f"dataset eids: [{len(dataset_meta['eid_names'])}] {dataset_meta['eid_names']}")
    print(f"train has {len(train_meta['labels'])} | saved @: {train_json}")
    print(f"valid has {len(valid_meta['labels'])} | saved @: {train_json}")
    print(f"test  has {len(test_meta['labels'])}  | saved @: {train_json}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",  type=str, choices=["coma"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir",  type=str, required=True)

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