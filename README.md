# 3D Face Modeling via Weakly-supervised Disentanglement Network joint Identity-consistency Prior (WSDF)


[[project page](https://liguohao96.github.io/WSDF)]
[[arXiv Paper](https://arxiv.org/abs/2404.16536)]

This is the official code for FG 2024 paper.

WSDF learns a 3D Face Model from registered meshes through disentangling identity and expression without expression labels.

|Application|Demo|run time|
|:-:|:-:|:-:|
|text-to-3d|"a DSLR photo of a man with tiger makeup, ..." <video width="100" height="50" src="https://github.com/liguohao96/WSDF/assets/16358157/4a0190e9-46f4-41b2-adc0-e0bd41adb263"></video>|~5min on single RTX 3090|
|image-fitting|fit & interpolating expression <video width="100" height="50" src="https://github.com/liguohao96/WSDF/assets/16358157/9a13c8ae-73ab-47f0-a862-4788cff4db3f"></video>|~15sec|

## Change Logs

- 2024/04/15: initial commit

## Setups

- **clone**

    ```git clone https://github.com/liguohao96/WSDF.git --recursive```

- **environment**

    instal python packages through
    `pip install -r requirements.txt`.
    or manually install `torch, numpy, opencv-python, pillow, imageio, plyfile, nvdiffrast`

- **face model**

    **FLAME**

    please first down `FLAME 2020` and `FLAME texture space` from [here](https://flame.is.tue.mpg.de/download.php) and put extracted files under `Data/FLAME2020`, the directory should be like: 
    ```
    Data/FLAME2020
    ├── female_model.pkl
    ├── FLAME_texture.npz
    ├── generic_model.pkl
    ├── male_model.pkl
    ├── tex_mean.png 
    └── ...
    ```

- **datasets**

    **CoMA**
    please download **Registered Data** from [original website](https://coma.is.tue.mpg.de/download.php) and unpack into `Data/Datasets/CoMA/COMA_data`

    the directory should be like:
    ```
    Data/Datasets
    ├── CoMA
    │   ├── COMA_data
    │   │   ├── FaceTalk_xxx
    │   │   └── ...
    ```


## Preprocessing

- **build cache for SpiralConv**

    First, make sure `3rdparty/spiralnet_plus` is downloaded (if not please run `git submodules sync --recursive`).
    Then, run
    ```shell
    python preprocessing/make_spiral_cache.py --name FLAME2020
    ```
    this will build sampling matrix and indices for SpiralConv and save at `Data/SpiralCache_FLAME2020.pth`.

- **split datasets into train/valid/test**

    **CoMA**
    ```shell
    bash preprocessing/coma/run.sh [file path to COMA_data]
    ```
    this will generate label files in `Data/dataset_split/CoMA`.

## Training VAEs

Train VAEs with specified config:
```shell
python train.py [config file] --batch_size 8
```
It will save intermediate result and final checkpoints at `./temp/[name]`

## Trained Checkpoints

|Link|Config|Training Datasets|
|:-:|:-:|:-:|
|[google drive](https://drive.google.com/file/d/1qQ_TCkLlsXj6_QTvRLwwFQIYa_b1Bzoe/view?usp=sharing)|coma_full.yaml|CoMA|

- **using trained checkpoint**

    assuming checkpoints are downloaded at `Data/checkpoints/coma_ep100.zip`
    ```python
    from utils.load import load_wsdf
    model_dict = load_wsdf("Data/checkpoints/coma_ep100.zip")

    print(model_dict.keys())
    # dict_keys(['Encoder', 'VAEHead', 'Decoder', 'NeuBank', 'Norm', 'Topology'])

    print(model_dict["Topology"].keys())
    # dict_keys(['ver', 'tri', 'uv', 'uv_tri'])

    norm, decoder  = model_dict["Norm"].cuda(), model_dict["Decoder"].cuda()
    i_code, e_code = torch.randn(1, decoder.dim_I, device="cuda"), torch.randn(1, decoder.dim_E, device="cuda")
    ver = norm.invert(decoder(i_code, e_code))  # 1, NV, 3

    # export
    from utils.io import mesh_io
    mesh_io.save_mesh("out.ply", (ver.squeeze(0).detach().cpu().numpy(), None, None, model_dict["Topology"]["tri"].cpu().numpy(), None, None))

    ```

## Application: Text-to-3D

First, make sure `tiny-cuda-nn` pytorch extension is installed through pip (`pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`) or manually following [offical guide](https://github.com/NVlabs/tiny-cuda-nn/tree/master?tab=readme-ov-file#pytorch-extension).

Then, generate initial texture through `python preprocessing/make_tex_mean.py --name FLAME2020`

Finally, generate 3D mesh and texture through
```shell
python applications/text_3d.py --checkpoint [checkpoint file] --save_gif True --save_ply True --config [config file] 
```
or
```shell
python applications/text_3d.py --checkpoint [checkpoint file] --save_gif True --save_ply True \
--text "a DSLR photo of [], ..." --negative_text "blur, ..." --tex_file Data/FLAME2020/tex_mean.png --name [name]
```
It will save intermediate result and final mesh at `./temp/APP-tex_3d/[name]`

## Application: Image-Fitting

First, follow the first two steps in `Text-to-3D` (install `tiny-cuda-nn` and generate initial texture).

Then, download `FLAME Mediapipe Landmark Embedding` from [here](https://flame.is.tue.mpg.de/download.php) and put extracted files under `Data/FLAME2020`.

Finally, fitting image through
```shell
python applications/fit_2d.py --checkpoint [checkpoint file] --save_gif True --save_ply True --config [config file] 
```
or
```shell
python applications/fit_2d.py --checkpoint [checkpoint file] --save_gif True --save_ply True \
--fitting_image [image file] --tex_mean Data/FLAME2020/tex_mean.png --name [name]
```
It will save intermediate result and final mesh at `./temp/APP-fid_2d/[name]`

## Acknowledgments

Our network and evaluation codes are based on [spiralnet_plus](https://github.com/sw-gong/spiralnet_plus) and [FaceExpDisentanglement](https://github.com/rmraaron/FaceExpDisentanglement), thanks for their great works.

Our `text-to-3d` codes are based on [Fantasia3D](https://github.com/Gorilla-Lab-SCUT/Fantasia3D) and [Perp-Neg](https://github.com/Perp-Neg/Perp-Neg-stablediffusion), thantks for their great works.


## Citation

```
@inproceedings{li2024wsdf,
  title  = {3D Face Modeling via Weakly-supervised Disentanglement Network joint Identity-consistency Prior}, 
  author = {Guohao Li and Hongyu Yang and Di Huang and Yunhong Wang},
  booktitle = {18th {IEEE} International Conference on Automatic Face and Gesture
                  Recognition, {FG} 2024, Istanbul, Turkey, May 27-31, 2024},
  year   = {2024},
}
```
