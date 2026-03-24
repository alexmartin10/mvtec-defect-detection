import torch
import torch.nn.functional as F
from torchvision.transforms import v2
import torchvision.models as models

class PatchCore:
    def __init__(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.memory_bank = checkpoint['memory_bank']
        self.threshold = checkpoint['threshold']
        
        self.backbone = models.resnet18(weights='DEFAULT')
        self.backbone.eval()

        self.features = {}
        handle2 = self.backbone.layer2.register_forward_hook(self.make_hook('layer2', self.features))
        handle3 = self.backbone.layer3.register_forward_hook(self.make_hook('layer3', self.features))

        self.transform = v2.Compose([
            v2.Resize((256, 256)),
            v2.ToDtype(torch.float32, scale=True)
        ])

    @staticmethod
    def make_hook(name, features):
        def hook_fn(module, input, output):
            features[name] = output
        return hook_fn
    
    def predict(self, image):

        image = self.transform(image)
        with torch.no_grad():
            _ = self.backbone(image.unsqueeze(0))

            f2 = self.features['layer2'][-1].unsqueeze(0)
            f3 = self.features['layer3'][-1].unsqueeze(0)

            f3_up = F.interpolate(f3, size=(32, 32), mode='bilinear', align_corners=False)
            combined = torch.cat([f2, f3_up], dim=1)  # (1, 384, 32, 32)
            
            patches = combined.permute(0, 2, 3, 1).reshape(-1, 384) 
            dists = torch.cdist(patches, self.memory_bank)
            min_dists = dists.min(dim=1).values
            score = min_dists.max().item()
        return {
            "score": round(score, 4),
            "anomaly": bool(score>self.threshold),
            "threshold": round(float(self.threshold), 4)
        }