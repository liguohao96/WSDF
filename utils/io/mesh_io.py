import os
import numpy as np
from glob import glob

class OBJReader(object):
    def __init__(self):
        super().__init__()
    
    @classmethod
    def load_obj(cls, fp):
        vs, t_vs = [], []
        vt, t_vt = [], []
        vn, t_vn = [], []

        for ln, lc in enumerate(fp):
            line            = lc.strip()
            splitted_line   = line.split()
            line_type       = splitted_line[0] if len(splitted_line) != 0 else None
            
            splited_num     = 4
            if not splitted_line:
                continue
            elif line_type == 'v':
                vs.append([float(v) for v in splitted_line[1:4]])
            elif line_type == 'vt':
                splited_num = 3
                vt.append([float(v) for v in splitted_line[1:3]])
            elif line_type == 'vn':
                vn.append([float(v) for v in splitted_line[1:4]])
            elif line_type == 'f':
                f_len      = len(splitted_line[1].split('/'))
                
                f_arr      = [list() for i in range(3)]
                
                for c in splitted_line[1:]:
                    fs     = c.split('/')
                    for i, fi in enumerate(fs):
                        if len(fi) > 0:
                            fi = int(fi)
                            f_arr[i].append(fi)
                
                s_ids      = f_arr[0]
                t_ids      = f_arr[1]
                n_ids      = f_arr[2]

                
                tri_ids    = [(ind - 1) if (ind > 0) else (len(vs) + ind)
                                   for ind in s_ids]
                tex_ids    = [(ind - 1) if (ind > 0) else (len(vt) + ind)
                                   for ind in t_ids]
                nrm_ids    = [(ind - 1) if (ind > 0) else (len(vn) + ind)
                                   for ind in n_ids]
                assert len(tex_ids) in [0, 3], f"line: {ln}, got {len(tex_ids)} tex"
                t_vs.append(tri_ids)
                t_vt.append(tex_ids)
                if len(nrm_ids) > 0:
                    t_vn.append(nrm_ids)
            else:
                continue
            assert len(splitted_line) == splited_num, f"Error with line {ln+1}: '{lc}'"
            
        vs    = np.asarray(vs)
        vt    = np.asarray(vt)
        vn    = np.asarray(vn)
        t_vs  = np.asarray(t_vs, dtype=int)
        t_vt  = np.asarray(t_vt, dtype=int)
        t_vn  = np.asarray(t_vn, dtype=int)
        return vs, vt, vn, t_vs, t_vt, t_vn

def load_mesh(mesh_f, filters=None):
    if os.path.splitext(mesh_f)[-1] == ".obj":
        with open(mesh_f) as f:
            ret = OBJReader.load_obj(f)
        return ret
    elif os.path.splitext(mesh_f)[-1] == ".ply":
        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(mesh_f)

        vs   = np.stack([plydata["vertex"][k] for k in "xyz"], axis=-1).astype(np.float32)
        t_vs = np.vstack(plydata["face"].data["vertex_indices"]).astype(np.int32)

        vt   = vn   = None
        t_vt = t_vn = None

        return vs, vt, vn, t_vs, t_vt, t_vn

def save_mesh(mesh_f, mesh_tuple, comments=None):
    vs, vt, vn, t_vs, t_vt, t_vn = mesh_tuple

    if os.path.splitext(mesh_f)[-1] == ".ply":
        from plyfile import PlyData, PlyElement
        from numpy.lib import recfunctions as rfn

        v_type  = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        vi_type = [("vertex_indices", "i4", (3,))]
        tc_type = [("texcoord", "f4", (6,))]

        assert np.max(t_vs) < len(vs), f"got index {np.max(t_vs)}, #ver={len(vs)}"

        ver = rfn.unstructured_to_structured(vs, np.dtype(v_type))
        tri = rfn.unstructured_to_structured(t_vs, np.dtype(vi_type))

        if vt is not None and t_vt is not None:
            tex_coord = vt[t_vt.flatten()].reshape(-1, 6)

            tri = np.zeros((len(t_vs)), dtype=np.dtype(vi_type + tc_type))

            tri["vertex_indices"][:] = t_vs
            tri["texcoord"][:] = tex_coord

            assert np.max(tri["vertex_indices"]) < len(vs), f"got index {np.max(tri['vertex_indices'])}, #ver={len(vs)}"
        
        ver_el = PlyElement.describe(ver, 'vertex')
        tri_el = PlyElement.describe(tri, 'face')

        el_list = [ver_el, tri_el]

        comments = comments if comments is not None else []

        PlyData([ver_el, tri_el], comments=comments, text=True).write(mesh_f)