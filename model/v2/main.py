import torch
from dataset import ImageDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import torchvision.models as models
from pathlib import Path

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')

backbone = models.resnet18(weights='DEFAULT')

def make_hook(name, features):
    def hook_fn(module, input, output):
        try:
            t = features[name]
            features[name] = torch.cat((t, output)) #to avoid having only the last batch
        except KeyError:
            features[name] = output
    return hook_fn

#implementing random subsampling
def random_subsample(m, ratio):
    n_samples = int(len(m) * ratio)
    indices = torch.randperm(len(m))[:n_samples]
    return m[indices]

def get_patch_features(dataset):

    features = {}
    handle2 = backbone.layer2.register_forward_hook(make_hook('layer2', features))
    handle3 = backbone.layer3.register_forward_hook(make_hook('layer3', features))

    backbone.to(device)
    backbone.eval()
    with torch.no_grad():
        for img in dataset:
            img = img.to(device)
            backbone(img.unsqueeze(0))
            if torch.accelerator.is_available():
                torch.accelerator.empty_cache()

    f2 = features['layer2']
    f3 = features['layer3']
    f3_up = F.interpolate(f3, size=(32, 32), mode='bilinear', align_corners=False)
    combined = torch.cat((f2, f3_up), dim=1)
    B, C, H, W = combined.size()

    handle2.remove()
    handle3.remove()

    return combined.permute(0, 2, 3, 1).reshape(-1, C)

def get_score_dataset(d):
    scores = []
    f = {}
    handle2 = backbone.layer2.register_forward_hook(make_hook('layer2', f))
    handle3 = backbone.layer3.register_forward_hook(make_hook('layer3', f))
    backbone.to(device)
    backbone.eval()

    for image in d:
        with torch.no_grad():
            image = image.to(device)
            _ = backbone(image.unsqueeze(0))
            if torch.accelerator.is_available():
                torch.accelerator.empty_cache() 

        f2 = f['layer2'][-1].unsqueeze(0)
        f3 = f['layer3'][-1].unsqueeze(0)

        f3_up = F.interpolate(f3, size=(32, 32), mode='bilinear', align_corners=False)
        combined = torch.cat([f2, f3_up], dim=1)  # (1, 384, 32, 32)
        
        # patches de l'image test
        patches = combined.permute(0, 2, 3, 1).reshape(-1, 384)  # (1024, 384)
        
        # distance de chaque patch test vers la banque mémoire
        dists = torch.cdist(patches.to(device), memory_bank.to(device))  # (1024, 214016)
        
        # distance minimale pour chaque patch
        min_dists = dists.min(dim=1).values  # (1024,)
        
        # score = distance maximale parmi tous les patches
        score = min_dists.max().item()

        scores.append(score)
    
    handle2.remove()
    handle3.remove()

    return scores

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_BASE_DIR = BASE_DIR / "data" / "bottle" / "bottle"

train_data = ImageDataset(DATA_BASE_DIR/"train"/"good")
memory_bank = get_patch_features(train_data)
memory_bank = random_subsample(memory_bank, ratio=0.1)

data_test_good = ImageDataset(DATA_BASE_DIR/"test"/"good")
data_test_broken_large = ImageDataset(DATA_BASE_DIR/"train"/"broken_large")
data_test_broken_small = ImageDataset(DATA_BASE_DIR/"test"/"broken_small")
data_test_contamination = ImageDataset(DATA_BASE_DIR/"test"/"contamination")

score_good = get_score_dataset(data_test_good)
score_broken_large = get_score_dataset(data_test_broken_large)
score_broken_small = get_score_dataset(data_test_broken_small)
score_contamination = get_score_dataset(data_test_contamination)

score_training = get_score_dataset(train_data)

threshold = np.percentile(score_training, 95)

labels = [0] * len(score_good) \
    + [1] * (
       len(score_broken_small) \
        + len(score_broken_large) \
        + len(score_contamination)
        )

all_scores = score_good + score_broken_small + score_broken_large + score_contamination

print(f"ROC AUC score : {roc_auc_score(labels, all_scores)}")

torch.save(
    {
    'memory_bank': memory_bank,
    'threshold': threshold
    },
    'patchcore.pt'
    )