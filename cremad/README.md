# CREMA-D — Cross-Modal Knowledge Distillation

Unpaired audio-visual knowledge distillation on the [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) emotion dataset (6 classes).

## Structure

```
cremad/
├── main_overlap_tag.py     # training entry point
├── run.sh                  # experiment launcher
└── utils/
    ├── CremadDataset.py    # dataset
    ├── helper.py           # training functions + data loaders
    ├── model_res.py        # ImageNet / AudioNet backbones (ResNet, ViT)
    ├── backbone.py         # ResNet building blocks (used by model_res.py)
    └── module.py           # C²KD proxy heads (Tea, Stu, TeaViT, StuViT)
```

The shared KD losses (ReviewKD, NORM) live in `../kd_losses.py`.

## Data layout

```
$DATA_ROOT/
├── Image-01-FPS/<id>/   — video frames (.jpg)
├── Audio-1004/<id>.pkl  — pre-extracted audio features
├── stat.csv             — class list
├── train.csv            — train split (id, class)
└── test.csv             — test split  (id, class)
```

## Setup

```bash
pip install torch torchvision timm wandb pot scipy pillow
```

## Running

```bash
# Default: audio student, reviewkd method
DATA_ROOT=/path/to/CREMA-D bash run.sh

# CE baseline, image student
DATA_ROOT=/path/to/CREMA-D STU_TYPE=0 METHOD=ce bash run.sh

# Bilevel OT distillation
DATA_ROOT=/path/to/CREMA-D METHOD=bilevel bash run.sh
```

Key env vars:

| Variable | Default | Description |
|---|---|---|
| `DATA_ROOT` | **required** | Path to CREMA-D data directory |
| `CKPT_DIR` | `./ckpts` | Checkpoint save directory |
| `LOG_DIR` | `./logs` | Log file directory |
| `GPU` | `0` | CUDA device index |
| `STU_TYPE` | `1` | `0` = image student, `1` = audio student |
| `IMAGE_ARCH` | `resnet18` | `resnet18` \| `resnet50` \| `vit_b_16` |
| `AUDIO_ARCH` | `resnet18` | `resnet18` \| `resnet50` \| `vit_s_16` |
| `METHOD` | `reviewkd` | `ce` \| `bilevel` \| `sumall` \| `vanillaKD` \| `feadistill` \| `reviewkd` \| `norm` |
| `WANDB_API_KEY` | — | Set to authenticate with W&B |
