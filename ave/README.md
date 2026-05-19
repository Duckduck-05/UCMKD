# AVE — Cross-Modal Knowledge Distillation

Unpaired audio-visual knowledge distillation on the [AVE](https://github.com/YapengTian/AVE-ECCV18) dataset (28 classes).

## Structure

```
ave/
├── main_overlap_tag.py     # training entry point
├── run.sh                  # experiment launcher
└── utils/
    ├── AVEDataset.py       # dataset
    ├── helper.py           # training functions + data loaders
    ├── model_res.py        # ImageNet / AudioNet backbones (ResNet, ViT)
    └── module.py           # C²KD proxy heads (Tea, Stu, TeaViT, StuViT)
```

The shared KD losses (ReviewKD, NORM) live in `../kd_losses.py`.

## Data layout

```
$DATA_ROOT/
├── Image-01-FPS-SE/<id>/   — video frames
├── Audio-1004-SE/<id>.pkl  — pre-extracted audio features
├── trainSet.txt
├── valSet.txt
└── testSet.txt
```

## Setup

```bash
pip install torch torchvision timm wandb pot scipy pillow
```

## Running

```bash
DATA_ROOT=/path/to/AVE_dataset bash run.sh
DATA_ROOT=/path/to/AVE_dataset METHOD=bilevel bash run.sh
```

Key env vars:

| Variable | Default | Description |
|---|---|---|
| `DATA_ROOT` | **required** | Path to AVE dataset directory |
| `CKPT_DIR` | `./ckpts` | Checkpoint save directory |
| `GPU` | `0` | CUDA device index |
| `STU_TYPE` | `0` | `0` = image student, `1` = audio student |
| `IMAGE_ARCH` | `vit_s_16` | `resnet18` \| `resnet50` \| `vit_b_16` \| `vit_s_16` \| `vit_l_16` |
| `AUDIO_ARCH` | `vit_l_16` | `resnet18` \| `resnet50` \| `vit_s_16` \| `vit_l_16` |
| `METHOD` | `ce` | `ce` \| `bilevel` \| `sumall` \| `vanillaKD` \| `feadistill` \| `reviewkd` \| `norm` |
| `WANDB_API_KEY` | — | Set to authenticate with W&B |
