import torch
import logging
import numpy as np
import torch.nn.functional as F
import torchvision.models as models

from pathlib import Path
from sklearn.metrics import roc_auc_score

from dataset import ImageDataset
from sampler import RandomSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("patchcore.log")
    ]
)
logger = logging.getLogger(__name__)

torch.manual_seed(42)

def make_hook(name, features):
    def hook_fn(module, input, output):
        features[name] = output
    return hook_fn

def extract_features(image, features, backbone, device):
    with torch.no_grad():
        image = image.to(device)
        backbone(image.unsqueeze(0))
        if torch.accelerator.is_available():
            torch.accelerator.empty_cache()
    
    f2 = features['layer2']
    f3 = features['layer3']
    f3_up = F.interpolate(f3, size=(32, 32), mode='bilinear', align_corners=False)
    
    combined = torch.cat((f2, f3_up), dim=1)
    _, C, _, _ = combined.size() 

    return combined.permute(0, 2, 3, 1).reshape(-1, C)

def get_patch_features(dataset, backbone, device):
    backbone.to(device)
    backbone.eval()

    features = {}
    handle2 = backbone.layer2.register_forward_hook(make_hook('layer2', features))
    handle3 = backbone.layer3.register_forward_hook(make_hook('layer3', features))

    all_patches = []
    for image in dataset:
        image_features = extract_features(image, features, backbone, device)
        all_patches.append(image_features)
    
    handle2.remove()
    handle3.remove()

    return torch.cat(all_patches, dim=0)

def get_score_dataset(dataset, memory_bank, backbone, device):
    backbone.to(device)
    backbone.eval()

    scores = []
    features = {}
    handle2 = backbone.layer2.register_forward_hook(make_hook('layer2', features))
    handle3 = backbone.layer3.register_forward_hook(make_hook('layer3', features))

    for image in dataset:       
        # patches de l'image test
        patches = extract_features(image, features, backbone, device)
        
        # distance de chaque patch test vers la banque mémoire
        dists = torch.cdist(patches.to(device), memory_bank.to(device))  # (1024, 214016)
        
        # distance minimale pour chaque patch
        min_dists = dists.amin(dim=1)  # (1024,)
        
        # score = distance maximale parmi tous les patches
        score = min_dists.amax().item()

        scores.append(score)
    
    handle2.remove()
    handle3.remove()

    return scores

def auroc_score(score_good, score_broken):
    if len(score_good) == 0 or len(score_broken) == 0:
        raise ValueError("At least one argument is empty")
    labels = [0] * len(score_good) + [1] * len(score_broken)
    scores = score_good + score_broken
    return roc_auc_score(labels, scores)

def main():
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')

    backbone = models.resnet18(weights='DEFAULT')

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_BASE_DIR = BASE_DIR / "data" / "bottle" / "bottle"

    sampler = RandomSampler(ratio=0.1)

    train_data = ImageDataset(DATA_BASE_DIR/"train"/"good")
    memory_bank = get_patch_features(train_data, backbone, device)
    memory_bank = sampler.sample(memory_bank)

    data_test_good = ImageDataset(DATA_BASE_DIR/"test"/"good")
    data_test_broken_large = ImageDataset(DATA_BASE_DIR/"test"/"broken_large")
    data_test_broken_small = ImageDataset(DATA_BASE_DIR/"test"/"broken_small")
    data_test_contamination = ImageDataset(DATA_BASE_DIR/"test"/"contamination")

    score_good = get_score_dataset(data_test_good, memory_bank, backbone, device)
    score_broken_large = get_score_dataset(data_test_broken_large, memory_bank, backbone, device)
    score_broken_small = get_score_dataset(data_test_broken_small, memory_bank, backbone, device)
    score_contamination = get_score_dataset(data_test_contamination, memory_bank, backbone, device)

    score_training = get_score_dataset(train_data, memory_bank, backbone, device)

    threshold = np.percentile(score_training, 95)

    global_auroc_score = auroc_score(score_good, score_broken_small + score_broken_large + score_contamination)
    logger.info(f"ROC AUC score : {global_auroc_score}")

    # AUROC broken_large
    broken_large_auroc_score = auroc_score(score_good, score_broken_large)
    logger.info(f"ROC AUC score broken_large : {broken_large_auroc_score}")

    # AUROC broken_small
    broken_small_auroc_score = auroc_score(score_good, score_broken_small)
    logger.info(f"ROC AUC score broken_small : {broken_small_auroc_score}")

    # AUROC contamination
    contamination_auroc_score = auroc_score(score_good, score_contamination)
    logger.info(f"ROC AUC score broken_contamination : {contamination_auroc_score}")


    torch.save(
        {
        'memory_bank': memory_bank,
        'threshold': threshold
        },
        'patchcore.pt'
        )
    
if __name__ == "__main__":
    main()