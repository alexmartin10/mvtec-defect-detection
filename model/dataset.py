import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import decode_image

class ImageDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img = list(Path(self.root_dir).rglob("*.png"))

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = decode_image(self.img[idx])
        image = image / 255
        if self.transform:
            image = self.transform(image)
        return image
