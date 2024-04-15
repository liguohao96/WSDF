import os
import sys
import torch
import numpy as np
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main(args):

    if args.name == "FLAME2020":
        import pickle

        for t in ["bool", "int", "float", "complex", "object", "unicode", "str", "nan", "inf"]:
            if not hasattr(np, t):
                setattr(np, t, getattr(np, f"{t}_"))

        with open(os.path.join(ROOT, "Data", "FLAME2020", "generic_model.pkl"), "rb") as f:
            data = pickle.load(f, encoding="latin1")
        
        ver = data["v_template"].reshape(-1, 3)
        tri = data["f"].reshape(-1, 3)

        texr = np.load(os.path.join(ROOT, "Data", "FLAME2020", "FLAME_texture.npz"))

        # texr["mean"] is BGR 512x512x3
        cv2.imwrite(os.path.join(ROOT, "Data", "FLAME2020", "tex_mean.png"), texr["mean"])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--mesh", type=str, required=True, help="mesh file")
    parser.add_argument("--name", type=str, required=True, choices=["FLAME2020"])

    args = parser.parse_args()

    main(args)
