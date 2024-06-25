import torch

import pandas as pd
import os
from torchvision.io import read_image
from torch.utils.data import Dataset

class FerPlus(Dataset):
    def __init__(self, set, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels.loc[self.img_labels['Usage'] == set]
        self.img_labels = self.img_labels.loc[~pd.isna(self.img_labels['Image name'])]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        map = {
            'Training': 'FER2013Train',
            'PrivateTest': 'FER2013Test',
            'PublicTest': 'FER2013Valid',
        }
        img_path = os.path.join(self.img_dir, map[self.img_labels.iloc[idx, 0]], self.img_labels.iloc[idx, 1])
        image = (read_image(img_path) / 255).to(torch.float32)
        label = torch.tensor(self.img_labels.iloc[idx, 3] / 10, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label