# VGGSound — Cross-Modal Knowledge Distillation

Unpaired audio-visual knowledge distillation on the [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) dataset (50 classes).

## Structure

```
vggsound/
├── main_overlap_tag.py     # training entry point
├── run.sh                  # experiment launcher
├── train.csv / test.csv    # full split files
├── train_tiny.csv / test_tiny.csv  # small-scale splits for debugging
└── utils/
    ├── VGGSoundDataset.py  # dataset
    ├── helper.py           # training functions + data loaders
    ├── model_res.py        # ImageNet / AudioNet backbones (ResNet)
    └── module.py           # C²KD proxy heads (Tea, Stu)
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
pip install torch torchvision wandb scipy librosa pillow
```

## Running

```bash
DATA_ROOT=/path/to/VGGSound bash run.sh

# Audio student
DATA_ROOT=/path/to/VGGSound STU_TYPE=1 bash run.sh
```

Key env vars:

| Variable | Default | Description |
|---|---|---|
| `DATA_ROOT` | **required** | Path to VGGSound dataset directory |
| `CKPT_DIR` | `./ckpts` | Checkpoint save directory |
| `GPU` | `0` | CUDA device index |
| `STU_TYPE` | `0` | `0` = image student, `1` = audio student |
| `NUM_FRAME` | `3` | Number of video frames per sample |
| `WANDB_API_KEY` | — | Set to authenticate with W&B |
