import torch
import torchvision as tv
from torchvision import transforms
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import random
from PIL import Image

# Convenience: pre‑built difficulty presets
CHALLENGE_PRESETS = {
    "clean": dict(
        marginal_mismatch=False,
        domain_shift=False,
        label_imbalance=False,
    ),
    "mild": dict(
        marginal_mismatch=True, marginal_ratio=0.7,
        domain_shift=True, domain_shift_level=0.3,
        label_imbalance=True, imbalance_factor=5.0,
    ),
    "moderate": dict(
        marginal_mismatch=True, marginal_ratio=0.5,
        domain_shift=True, domain_shift_level=0.5,
        label_imbalance=True, imbalance_factor=10.0,
    ),
    "hard": dict(
        marginal_mismatch=True, marginal_ratio=0.3,
        domain_shift=True, domain_shift_level=0.8,
        label_imbalance=True, imbalance_factor=50.0,
    ),
    # --- single‑challenge ablations ---
    "marginal_only": dict(
        marginal_mismatch=True, marginal_ratio=0.5,
        domain_shift=False,
        label_imbalance=False,
    ),
    "domain_only": dict(
        marginal_mismatch=False,
        domain_shift=True, domain_shift_level=0.5,
        label_imbalance=False,
    ),
    "imbalance_only": dict(
        marginal_mismatch=False,
        domain_shift=False,
        label_imbalance=True, imbalance_factor=10.0,
    ),
}



class _SpecAugment:
    """Frequency + time masking (SpecAugment-lite) applied after ToTensor+Normalize."""
    def __init__(self, freq_mask_ratio: float = 0.15, time_mask_ratio: float = 0.20, num_masks: int = 2):
        self.freq_mask_ratio = freq_mask_ratio
        self.time_mask_ratio = time_mask_ratio
        self.num_masks = num_masks

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, H, W = x.shape
        x = x.clone()
        for _ in range(self.num_masks):
            f = random.randint(0, max(1, int(H * self.freq_mask_ratio)))
            f0 = random.randint(0, max(0, H - f))
            x[:, f0:f0 + f, :] = 0.0
        for _ in range(self.num_masks):
            t = random.randint(0, max(1, int(W * self.time_mask_ratio)))
            t0 = random.randint(0, max(0, W - t))
            x[:, :, t0:t0 + t] = 0.0
        return x


class RavvdessDataset(Dataset):
    def __init__(self, csv_path, audio_dir, image_dir, mode='train',
                 challenge: bool = False,
                 challenge_preset: str = None,
                 marginal_mismatch: bool = False, marginal_ratio: float = 0.5,
                 domain_shift: bool = False, domain_shift_level: float = 0.5,
                 label_imbalance: bool = False, imbalance_modality: str = 'audio',
                 imbalance_factor: float = 10.0, seed: int = 42):
        """
        CSV columns: [audio_name, image_name, label]

        Challenge knobs (train only):
          challenge_preset : 'clean' | 'mild' | 'moderate' | 'hard' | ablation keys
          marginal_mismatch: disjoint audio/image pools per class
          domain_shift     : extra corruption augmentations
          label_imbalance  : long-tail sampling for one modality
        """
        if challenge and challenge_preset and challenge_preset in CHALLENGE_PRESETS:
            preset = CHALLENGE_PRESETS[challenge_preset]
            marginal_mismatch  = preset.get('marginal_mismatch',  marginal_mismatch)
            marginal_ratio     = preset.get('marginal_ratio',      marginal_ratio)
            domain_shift       = preset.get('domain_shift',        domain_shift)
            domain_shift_level = preset.get('domain_shift_level',  domain_shift_level)
            label_imbalance    = preset.get('label_imbalance',     label_imbalance)
            imbalance_factor   = preset.get('imbalance_factor',    imbalance_factor)

        self.df = pd.read_csv(csv_path, header=None)
        self.audio_dir = audio_dir
        self.image_dir = image_dir
        self.mode = mode

        self.audio_files = self.df.iloc[:, 0].values
        self.image_files = self.df.iloc[:, 1].values

        unique_classes = sorted(self.df.iloc[:, 2].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
        self.labels = [self.class_to_idx[l] for l in self.df.iloc[:, 2].values]

        self.indices_per_class = defaultdict(list)
        if mode == 'train':
            for idx, lbl in enumerate(self.labels):
                self.indices_per_class[lbl].append(idx)

        # Challenge 1: Marginal Mismatch
        self._marginal_mismatch = marginal_mismatch and (mode == 'train')
        self._audio_pool = defaultdict(list)
        self._image_pool = defaultdict(list)
        if self._marginal_mismatch:
            rng = random.Random(seed)
            for cls, idxs in self.indices_per_class.items():
                shuffled = idxs.copy()
                rng.shuffle(shuffled)
                split = max(1, int(len(shuffled) * marginal_ratio))
                self._audio_pool[cls] = shuffled[:split] or shuffled
                self._image_pool[cls] = shuffled[split:] or shuffled
        else:
            for cls, idxs in self.indices_per_class.items():
                self._audio_pool[cls] = idxs
                self._image_pool[cls] = idxs

        # Challenge 2: Domain Shift
        self._domain_shift = domain_shift and (mode == 'train')
        self._domain_shift_level = domain_shift_level
        if self._domain_shift:
            self._audio_noise_std = 0.05 + 0.20 * domain_shift_level

        # Challenge 3: Label Imbalance
        self._label_imbalance = label_imbalance and (mode == 'train')
        self._imbalance_modality = imbalance_modality
        self._imbalanced_pool = None
        if self._label_imbalance:
            num_classes = len(unique_classes)
            rng_i = random.Random(seed + 1)
            order = list(range(num_classes))
            rng_i.shuffle(order)
            probs = {cls: (1.0 / imbalance_factor) ** (rank / max(num_classes - 1, 1))
                     for rank, cls in enumerate(order)}
            rng_s = random.Random(seed + 2)
            self._imbalanced_pool = defaultdict(list)
            for cls in range(num_classes):
                src = self._audio_pool[cls] if imbalance_modality == 'audio' else self._image_pool[cls]
                keep = max(1, int(len(src) * probs[cls]))
                self._imbalanced_pool[cls] = rng_s.sample(src, min(keep, len(src)))

        # Transforms
        if mode == 'train':
            self.aud_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[-10.59], std=[85.66]),
                _SpecAugment(),
            ])
        else:
            self.aud_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[-10.59], std=[85.66]),
            ])

        if mode == 'train':
            img_layers = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
            if self._domain_shift:
                lvl = self._domain_shift_level
                img_layers += [
                    transforms.ColorJitter(brightness=0.2+0.4*lvl, contrast=0.2+0.4*lvl,
                                           saturation=0.2+0.3*lvl, hue=0.05+0.10*lvl),
                    transforms.RandomGrayscale(p=0.1+0.2*lvl),
                    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2+0.3*lvl),
                ]
            img_layers += [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
            if self._domain_shift:
                img_layers.append(transforms.RandomErasing(p=0.1+0.2*self._domain_shift_level, scale=(0.02, 0.15)))
            self.img_transform = transforms.Compose(img_layers)
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        target_label = self.labels[index]

        if self.mode == 'train':
            if self._label_imbalance and self._imbalance_modality == 'audio':
                pool = self._imbalanced_pool[target_label]
            else:
                pool = self._audio_pool[target_label]
            audio_idx = random.choice(pool)
        else:
            audio_idx = index

        audio_np = np.load(os.path.join(self.audio_dir, self.audio_files[audio_idx]))
        if audio_np.ndim == 3 and audio_np.shape[2] == 3:
            audio_np = audio_np[:, :, ::-1].copy()
        if self._domain_shift:
            audio_np = audio_np + np.random.randn(*audio_np.shape).astype(audio_np.dtype) * self._audio_noise_std
        audio = self.aud_transform(audio_np)

        if self.mode == 'train':
            if self._label_imbalance and self._imbalance_modality == 'image':
                img_idx = random.choice(self._imbalanced_pool[target_label])
            elif self._marginal_mismatch:
                img_idx = random.choice(self._image_pool[target_label])
            else:
                img_idx = index
        else:
            img_idx = index

        image = Image.open(os.path.join(self.image_dir, self.image_files[img_idx])).convert('RGB')
        image = self.img_transform(image)

        return {'audio': audio, 'image': image, 'label': target_label}
