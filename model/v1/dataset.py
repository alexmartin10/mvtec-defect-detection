import torch
from pathlib import Path
from torchvision.transforms import v2
from torchvision.io import decode_image


class ImageDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = v2.Compose([
            v2.Resize((256, 256)),
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.img = list(Path(self.root_dir).rglob("*.png"))

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = decode_image(self.img[idx])
        if self.transform:
            image = self.transform(image)
        return image