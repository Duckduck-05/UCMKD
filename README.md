# Cross-Modal Knowledge Distillation without Paired Data: Theoretical Foundations and Algorithms

[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/virtual/2026/poster/60546)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

Official implementation of the paper:

> **Cross-Modal Knowledge Distillation without Paired Data: Theoretical Foundations and Algorithms**  
> *Accepted at ICML 2026*

## Overview

We study cross-modal knowledge distillation in the **unpaired** setting, where audio and visual samples from the same class are not aligned at the instance level. We provide theoretical foundations for why distillation remains effective under marginal distribution mismatch, and propose algorithms based on optimal transport and bilevel optimization.

## Datasets

| Dataset | Modalities | Classes | Split |
|---------|-----------|---------|-------|
| [RAVDESS](https://zenodo.org/record/1188976) | Audio + Video | 8 (emotion) | train / val / test |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | Audio + Video | 6 (emotion) | train / test |
| [AVE](https://github.com/YapengTian/AVE-ECCV18) | Audio + Video | 28 (event) | train / val / test |
| [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) | Audio + Video | 50 (sound) | train / test |

## Repository Structure

```
UCMKD/
├── kd_losses.py          # Shared KD losses: ReviewKD, NORM
├── ravdess/              # RAVDESS experiments
│   ├── main_overlap_tag.py
│   ├── run.sh
│   └── utils/
│       ├── RavvdessDataset.py
│       ├── helper.py
│       ├── model_res.py
│       └── module.py
├── cremad/               # CREMA-D experiments
│   ├── main_overlap_tag.py
│   ├── run.sh
│   └── utils/
│       ├── CremadDataset.py
│       ├── helper.py
│       ├── model_res.py
│       └── module.py
├── ave/                  # AVE experiments
│   ├── main_overlap_tag.py
│   ├── run.sh
│   └── utils/
│       ├── AVEDataset.py
│       ├── helper.py
│       ├── model_res.py
│       └── module.py
└── vggsound/             # VGGSound experiments
    ├── main_overlap_tag.py
    ├── run.sh
    └── utils/
        ├── VGGSoundDataset.py
        ├── helper.py
        ├── model_res.py
        └── module.py
```

## Installation

```bash
pip install torch torchvision torchaudio timm wandb pot geomloss scipy librosa pillow pandas
```

Or use the provided environment file:

```bash
conda env create -f environment.yml
conda activate crkd
```

## Data Preparation

Each dataset directory should follow this layout (see `utils/data/` for preprocessing scripts):

**RAVDESS:**
```
$DATA_ROOT/
├── aud_features/     # .npy mel spectrogram features
├── vid_features/     # image frames
└── data_file/
    ├── spa_dl.csv    # train split
    ├── spa_val.csv   # val split
    └── spa_test.csv  # test split
```

**CREMA-D:**
```
$DATA_ROOT/
├── Image-01-FPS/<id>/    # video frames (.jpg)
├── Audio-1004/<id>.pkl   # pre-extracted audio features
├── stat.csv
├── train.csv
└── test.csv
```

**AVE / VGGSound:**
```
$DATA_ROOT/
├── Image-01-FPS-SE/<id>/   # video frames
├── Audio-1004-SE/<id>.pkl  # pre-extracted audio features
├── trainSet.txt
├── valSet.txt
└── testSet.txt
```

## Training

Each dataset folder contains a `run.sh` with all hyperparameters. Set `DATA_ROOT` and run:

```bash
# RAVDESS — bilevel OT distillation (image student)
cd ravdess
DATA_ROOT=/path/to/ravdess METHOD=bilevel STU_TYPE=0 bash run.sh

# CREMA-D — CE baseline (audio student)
cd cremad
DATA_ROOT=/path/to/CREMA-D METHOD=ce STU_TYPE=1 bash run.sh

# AVE — vanilla KD
cd ave
DATA_ROOT=/path/to/AVE_dataset METHOD=vanillaKD bash run.sh

# VGGSound — C²KD baseline
cd vggsound
DATA_ROOT=/path/to/VGGSound bash run.sh
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--method_type` | Training method: `ce`, `bilevel`, `sumall`, `vanillaKD`, `feadistill`, `reviewkd`, `norm` | `ce` |
| `--stu-type` | Student modality: `0`=image, `1`=audio | `0` |
| `--image_arch` | Image backbone: `resnet18`, `resnet50`, `vit_b_16`, `vit_l_16` | `resnet18` |
| `--audio_arch` | Audio backbone: `resnet18`, `resnet50`, `vit_s_16`, `vit_l_16` | `resnet18` |
| `--pre_train` | Pre-train teacher before distillation | `0` |
| `--metric` | OT cost metric: `l1`, `l2`, `cosine`, `chordal` | `cosine` |

### W&B Logging

Set `WANDB_API_KEY` in your environment to enable experiment tracking:

```bash
export WANDB_API_KEY=your_key_here
export WANDB_MODE=online   # or 'offline' / 'disabled'
```

## Methods

| Method key | Description |
|------------|-------------|
| `ce` | Cross-entropy baseline (no distillation) |
| `bilevel` | **Ours** — bilevel OT-based distillation |
| `sumall` | OT feature alignment + label alignment |
| `vanillaKD` | Vanilla KD (Hinton et al., 2015) |
| `feadistill` | Feature distillation via OT |
| `reviewkd` | ReviewKD (Chen et al., CVPR 2021) |
| `norm` | NORM (ICLR 2023) |

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{ucmkd2026,
  title     = {Cross-Modal Knowledge Distillation without Paired Data: Theoretical Foundations and Algorithms},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
