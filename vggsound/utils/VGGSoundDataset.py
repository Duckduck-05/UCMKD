import os
import pickle
import random
from collections import defaultdict

import librosa
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VGGSound(Dataset):
    """VGGSound audio-visual dataset with unpaired sampling for cross-modal KD.

    At training time, audio and image are sampled independently from the same
    class (unpaired). At val/test time, the original paired samples are used.

    Expects data_root to contain:
        Image-01-FPS-SE/<id>/   — video frames
        Audio-1004-SE/<id>.pkl  — pre-extracted audio features
        trainSet.txt / valSet.txt / testSet.txt  — split files (class&id&...)
    """

    def __init__(self, args, mode: str = 'train'):
        self.fps = 1
        self.num_frame = args.num_frame
        self.mode = mode
        self.image, self.audio, self.label = [], [], []

        data_root = args.data_root
        audio_dir = os.path.join(data_root, 'Audio-1004-SE')
        split_files = {
            'train': os.path.join(data_root, 'trainSet.txt'),
            'test':  os.path.join(data_root, 'testSet.txt'),
            'val':   os.path.join(data_root, 'valSet.txt'),
        }

        # Build class list from test split (consistent across all splits)
        classes = []
        with open(split_files['test'], 'r') as f:
            for line in f:
                cls = line.split('&')[0]
                if cls not in classes:
                    classes.append(cls)
        class_dict = {c: i for i, c in enumerate(classes)}

        # Load file list for the requested split
        with open(split_files[mode], 'r') as f:
            for line in f:
                parts = line.split('&')
                audio_path  = os.path.join(audio_dir, parts[1] + '.pkl')
                visual_path = os.path.join(data_root, f'Image-{self.fps:02d}-FPS-SE', parts[1])
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    if audio_path not in self.audio:
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[parts[0]])

        # Index samples by class for unpaired training sampling
        self.indices_per_class = defaultdict(list)
        for idx, lbl in enumerate(self.label):
            self.indices_per_class[lbl].append(idx)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        target_label = self.label[idx]

        # Unpaired: sample audio from the same class, independent of image index
        audio_idx = random.choice(self.indices_per_class[target_label]) if self.mode == 'train' else idx

        spectrogram = pickle.load(open(self.audio[audio_idx], 'rb'))
        spectrogram = np.resize(spectrogram, (257, 1024))

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        image_samples = os.listdir(self.image[idx])
        images = torch.zeros((self.num_frame, 3, 224, 224))
        for i in range(self.num_frame):
            fname = image_samples[i % len(image_samples)]
            images[i] = transform(Image.open(os.path.join(self.image[idx], fname)).convert('RGB'))

        return {'audio': spectrogram, 'image': images.permute(1, 0, 2, 3), 'label': target_label}
