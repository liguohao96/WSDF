import os
import io
import json
import numpy as np

from utils.io.mesh_io import load_mesh

class RegisteredMeshDataset(object):
    def __init__(self, data_root, label_file, scale_to_millimeter=1):
        super().__init__()

        self.args_str = f"{data_root}, {label_file}, {scale_to_millimeter}"

        self.data_root = data_root
        self.xyz_scale = scale_to_millimeter

        if isinstance(label_file, (str,)):
            with open(label_file, "r") as f:
                label_data = json.load(f)
        elif isinstance(label_file, (io.BytesIO, io.TextIOWrapper)):
            label_data = json.load(label_file)
        
        self.label_data = label_data
        self.labels     = label_data["labels"]
    
    def make_subset_by_fnames(self, fnames):
        fnames = set(fnames)

        subset_labels = []
        for filename, label in self.labels:
            if filename in fnames:
                subset_labels.append((filename, label))
        
        return self.make_subset_by_labels(subset_labels)

    def make_subset_by_labels(self, labels):
        subset_jsonobj = {}
        subset_jsonobj.update(self.label_data)
        subset_jsonobj["labels"] = labels

        buffer = io.BytesIO(json.dumps(subset_jsonobj).encode())

        return RegisteredMeshDataset(self.data_root, buffer, self.xyz_scale)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # read mesh
        filename, label = self.labels[index]
        vs, vt, vn, t_vs, t_vt, t_vn = load_mesh(os.path.join(self.data_root, filename))
        vs = self.xyz_scale * vs.astype(np.float32)

        return {
            "vertex": vs,
            "label": label
            # "pid":    pid,
            # "eid":    pid,
            # "fid":    fid
        }

    def __repr__(self):
        return f"RegisteredMeshDataset({self.args_str})"