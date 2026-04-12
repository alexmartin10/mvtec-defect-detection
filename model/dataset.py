import torch
from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import Dataset
from torchvision.io import decode_image

class ImageDataset(Dataset):
    def __init__(
        self,
        root_dir: str
    ):
        self.root_dir = root_dir
        self.transform = v2.Compose([
            v2.Resize((256, 256)),
            v2.ToDtype(torch.float32, scale=True)
        ])
        self.image_paths = list(Path(self.root_dir).rglob("*.png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = decode_image(self.image_paths[idx])
        return self.transform(image)