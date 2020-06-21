# Steet View Panoptic Segmentation with PyTorch and Detectron2

==============================

## Installation

- Install [Anaconda Python](https://docs.anaconda.com/anaconda/install/mac-os/) for MacOS.
- Use the `environment.yml` to create the virtual environment and install packages by the following command line:

```bash
conda env create -f environment.yml
conda activate detectron2
```

Note: This `.yml` file is created on my mac-mini by this command: `conda env export | grep -v "^prefix: " > environment.yml`. It will install pytorch, torchvision, opencv and other needed packages for this practice.

If you want to specify a different install path than the default for your system, use the `-p` configuration option:

```bash
conda env create -f environment.yml -p /home/user/anaconda3/envs/env_name
```

or you can choose to use the following command to install it

## Install the Detectron2 for MacOS (CPU)

```bash
git clone https://github.com/facebookresearch/detectron2.git
```

Open `detectron2/detectron2/config/defaults.py` change the line 28:

```python
# _C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE = "cpu"
```

Then in the root foler run the following command to install the pacakge:

```
CC=clang CXX=clang++ python -m pip install -e detectron2
```

## Notebook for Image/Video Segmentation.

1. [CPU macOS (image, video)](<./Detectron2\ Test(CPU-MacOS).ipynb>)
2. [GPU Colab: Detectron2 Tutorial](./Detectron2_Tutorial_Colab_GPU.ipynb) This is original tutorial notebook open this with Colab to enable the GPU.

Reference:
https://github.com/facebookresearch/detectron2
