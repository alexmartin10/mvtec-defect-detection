import torch

class GreedyCoresetSampler:
    def __init__(
        self,
        ratio:float,
        dimension_linear_projection=128
    ):
        if not 0 < ratio < 1:
            raise ValueError("Ratio must be a number between 0 and 1")
        self.ratio = ratio
        self.dimension_linear_projection = dimension_linear_projection
    
    def _reduce_feature(
        self,
        feature:torch.Tensor
    ):
        if feature.shape[1] == self.dimension_linear_projection:
            return feature
        elif feature.shape[1] < self.dimension_linear_projection:
            return feature
        else:
            linear_projection = torch.nn.Linear(
                feature.shape[1],
                self.dimension_linear_projection,
                bias=False
            )
            return linear_projection(feature)

    def sample(
        self, features: torch.Tensor
    ):
        subset_length = int(len(features) * self.ratio)
        feat = self._reduce_feature(features)

        # Remainig indices in tensor to subsample
        remaining_indices = list(range(len(features)))

        origin = torch.zeros((1, feat.shape[1]))
        d = torch.cdist(feat, origin)  # (N, 1)

        selected = []

        for _ in range(subset_length):
            # Finding point to add to coreset
            m = torch.amin(d, dim=1)          
            local_i = torch.argmax(m).item()  

            # Finding associated original associated index
            original_i = remaining_indices[local_i]
            selected.append(original_i)

            new_element = feat[local_i].unsqueeze(0)  # (1, D)
            feat = torch.cat((feat[:local_i], feat[local_i+1:]), dim=0)
            d    = torch.cat((d[:local_i],    d[local_i+1:]),    dim=0)
            remaining_indices.pop(local_i)

            # Update distance matrix to avoid calculating the whole each time
            dist_to_new = torch.cdist(feat, new_element)  # (N-1, 1)
            d = torch.cat((d, dist_to_new), dim=1)

        mask = torch.zeros(len(features), dtype=torch.bool)
        mask[selected] = True
        return features[mask]


class RandomSampler:
    def __init__(self, ratio):
        self.ratio = ratio

    def sample(self, features):
        n_samples = int(len(features) * self.ratio)
        indices = torch.randperm(len(features))[:n_samples]
        return features[indices]