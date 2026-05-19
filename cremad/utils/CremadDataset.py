import csv
import os
import pickle
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CremadDataset(Dataset):
    """CREMA-D audio-visual dataset.

    Expects data_root to contain:
        Image-01-FPS/<id>/   — video frames (.jpg)
        Audio-1004/<id>.pkl  — pre-extracted audio features
        stat.csv             — class list
        train.csv / test.csv — split files (id, class)
    """

    def __init__(self, mode: str = 'train', data_root: str = None):
        assert data_root is not None, 'data_root must be provided'
        self.mode = mode
        self.visual_path = os.path.join(data_root, 'Image-01-FPS')
        self.audio_path  = os.path.join(data_root, 'Audio-1004')

        with open(os.path.join(data_root, 'stat.csv'), encoding='UTF-8-sig') as f:
            classes = [row[0] for row in csv.reader(f)]
        self.classes = sorted(classes)

        csv_file = os.path.join(data_root, 'train.csv' if mode == 'train' else 'test.csv')
        self.data, self.data2class = [], {}
        with open(csv_file) as f:
            for item in csv.reader(f):
                aid = os.path.join(self.audio_path,  item[0] + '.pkl')
                vid = os.path.join(self.visual_path, item[0])
                if item[1] in classes and os.path.exists(aid) and os.path.exists(vid):
                    self.data.append(item[0])
                    self.data2class[item[0]] = item[1]

        self._img_transform_train = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._img_transform_eval = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        label = self.classes.index(self.data2class[datum])

        # Audio
        fbank = torch.tensor(
            pickle.load(open(os.path.join(self.audio_path, datum + '.pkl'), 'rb'))
        ).float().unsqueeze(0)

        # Visual — pick 2 frames
        folder = os.path.join(self.visual_path, datum)
        all_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
        pick_num = 2
        transf = self._img_transform_train if self.mode == 'train' else self._img_transform_eval
        if self.mode == 'train':
            selected = random.sample(all_files, min(pick_num, len(all_files)))
        else:
            mid = len(all_files) // 2
            selected = all_files[max(0, mid - 1):mid + 1]

        images = torch.cat([
            transf(Image.open(os.path.join(folder, f)).convert('RGB')).unsqueeze(0)
            for f in selected
        ])

        return images, fbank, label
