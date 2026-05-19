# RAVDESS — Cross-Modal Knowledge Distillation

Unpaired audio-visual knowledge distillation on the [RAVDESS](https://zenodo.org/record/1188976) emotion dataset (8 classes).

## Structure

```
ravdess/
├── main_overlap_tag.py     # training entry point
├── run.sh                  # experiment launcher
└── utils/
    ├── RavvdessDataset.py  # dataset (supports challenge splits)
    ├── helper.py           # training functions + data loaders
    ├── model_res.py        # ImageNet / AudioNet backbones (ResNet, ViT)
    └── module.py           # C²KD proxy heads (Tea, Stu, TeaViT, StuViT)
```

The shared KD losses (ReviewKD, NORM) live in `../kd_losses.py`.

## Data layout

```
$DATA_ROOT/
├── aud_features/   # .npy mel spectrogram features
├── vid_features/   # image frames
└── data_file/
    ├── spa_dl.csv   # train split
    ├── spa_val.csv  # val split
    └── spa_test.csv # test split
```

## Setup

```bash
pip install torch torchvision timm wandb pot geomloss scipy
```

## Running

```bash
# Single run (CE baseline)
DATA_ROOT=/path/to/ravdess bash run.sh

# Bilevel OT distillation
DATA_ROOT=/path/to/ravdess METHOD=bilevel bash run.sh

# All cost metrics in parallel
DATA_ROOT=/path/to/ravdess RUN_MODE=all_metrics bash run.sh

# Challenging splits
DATA_ROOT=/path/to/ravdess RUN_MODE=challenging bash run.sh
```

Key env vars:

| Variable | Default | Description |
|---|---|---|
| `DATA_ROOT` | **required** | Path to ravdess data directory |
| `CKPT_DIR` | `./ckpts` | Checkpoint save directory |
| `GPU` | `0` | CUDA device index |
| `STU_TYPE` | `0` | `0` = image student, `1` = audio student |
| `IMAGE_ARCH` | `vit_l_16` | `resnet18` \| `resnet50` \| `vit_b_16` \| `vit_l_16` |
| `AUDIO_ARCH` | `vit_s_16` | `resnet18` \| `resnet50` \| `vit_s_16` \| `vit_l_16` |
| `METHOD` | `ce` | `ce` \| `bilevel` \| `sumall` \| `vanillaKD` \| `feadistill` \| `reviewkd` \| `norm` \| `cost_metric` |
| `WANDB_API_KEY` | — | Set to authenticate with W&B |

## Challenge splits

`RavvdessDataset` supports three distribution-mismatch challenges (train only):

- **Marginal mismatch** — audio and image see disjoint sample pools per class
- **Domain shift** — extra augmentations simulate modality domain gap
- **Label imbalance** — one modality has long-tail class frequencies

Use `--challenge --challenge_preset [clean|mild|moderate|hard]` or individual flags.
