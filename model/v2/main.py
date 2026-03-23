import torch
from dataset import ImageDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

backbone = models.resnet18(weights='DEFAULT')

def make_hook(name, features):
  def hook_fn(module, input, output):
    try:
      t = features[name]
      features[name] = torch.cat((t, output)) #to avoid having only the last batch
    except KeyError:
      features[name] = output
  return hook_fn

def get_patch_features(dataset):

    features = {}
    backbone.layer2.register_forward_hook(make_hook('layer2', features))
    backbone.layer3.register_forward_hook(make_hook('layer3', features))

    backbone.to(device)
    backbone.eval()
    with torch.no_grad():
        for img in dataset:
            img = img.to(device)
            backbone(img.unsqueeze(0))
            torch.cuda.empty_cache()

    f2 = features['layer2']
    f3 = features['layer3']
    f3_up = F.interpolate(f3, size=(32, 32), mode='bilinear', align_corners=False)
    combined = torch.cat((f2, f3_up), dim=1)
    B, C, H, W = combined.size()
    return combined.permute(0, 2, 3, 1).reshape(-1, C)



#implementing random subsampling
def random_subsample(m, ratio):
    n_samples = int(len(m) * ratio)
    indices = torch.randperm(len(m))[:n_samples]
    return m[indices]

def get_score_dataset(d):
  scores = []
  f = {}
  backbone.layer2.register_forward_hook(make_hook('layer2', f))
  backbone.layer3.register_forward_hook(make_hook('layer3', f))
  backbone.to(device)
  backbone.eval()

  for image in d:
    with torch.no_grad():
      image = image.to(device)
      _ = backbone(image.unsqueeze(0))
      torch.cuda.empty_cache()

    f2 = f['layer2'][-1].unsqueeze(0)
    f3 = f['layer3'][-1].unsqueeze(0)

    f3_up = F.interpolate(f3, size=(32, 32), mode='bilinear', align_corners=False)
    combined = torch.cat([f2, f3_up], dim=1)  # (1, 384, 32, 32)
    
    # patches de l'image test
    patches = combined.permute(0, 2, 3, 1).reshape(-1, 384)  # (1024, 384)
    
    # distance de chaque patch test vers la banque mémoire
    dists = torch.cdist(patches, memory_bank)  # (1024, 214016)
    
    # distance minimale pour chaque patch
    min_dists = dists.min(dim=1).values  # (1024,)
    
    # score = distance maximale parmi tous les patches
    score = min_dists.max().item()

    scores.append(score)
  return scores

train_data = ImageDataset("../../data/bottle/bottle/train/good")
memory_bank = get_patch_features(train_data)
memory_bank = random_subsample(memory_bank, ratio=0.1)

dg = ImageDataset("../../data/bottle/bottle/test/good")
dbl = ImageDataset("../../data/bottle/bottle/test/broken_large")
dbs = ImageDataset("../../data/bottle/bottle/test/broken_small")
dc = ImageDataset("../../data/bottle/bottle/test/contamination")

sg = get_score_dataset(dg)
sbl = get_score_dataset(dbl)
sbs = get_score_dataset(dbs)
sc = get_score_dataset(dc)

score_training = get_score_dataset(train_data)

threshold = np.percentile(score_training, 95)

labels = [0] * len(sg) + [1] * (len(sbs)+len(sbl)+len(sc))
result = np.concat((
    np.int32(sg>threshold),
    np.int32(sbs>threshold),
    np.int32(sbl>threshold),
    np.int32(sc>threshold)
))

roc_auc_score(labels, result)

torch.save(
    {
    'memory_bank': memory_bank,
    'threshold': threshold
    },
    'patchcore.pt'
    )