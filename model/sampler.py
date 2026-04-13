import torch

class GreedyCoresetSampler:
    """Greedy coreset subsampling algorithm for memory bank reduction.
 
    Iteratively selects the most representative patches by maximizing
    the minimum distance to already selected points. This ensures
    the resulting subset covers the feature space as uniformly as possible.
 
    Args:
        ratio: Fraction of patches to keep. Must be strictly between 0 and 1.
        dimension_linear_projection: Target dimensionality for feature reduction
            before sampling. Reduces computational cost of distance calculations.
            Defaults to 128.
 
    Raises:
        ValueError: If ratio is not strictly between 0 and 1.
    """
 
    def __init__(
        self,
        ratio: float,
        dimension_linear_projection: int =128
    ):
        if not 0 < ratio < 1:
            raise ValueError("Ratio must be a number between 0 and 1")
        self.ratio = ratio
        self.dimension_linear_projection = dimension_linear_projection
        self.linear_projection = None
    
    def _reduce_feature(
        self,
        feature:torch.Tensor
    ) -> torch.Tensor:
        """Project features to a lower-dimensional space for faster sampling.
 
        If the feature dimension is already at or below the target, returns
        the features unchanged. The projection matrix is initialized once
        and reused across calls.
 
        Args:
            feature: Feature tensor of shape (N, D).
 
        Returns:
            Projected feature tensor of shape (N, dimension_linear_projection)
            or the original tensor if no projection is needed.
        """
        if feature.shape[1] <= self.dimension_linear_projection:
            return feature
        if self.linear_projection is None:
            self.linear_projection = torch.nn.Linear(
                feature.shape[1],
                self.dimension_linear_projection,
                bias=False
            )
        with torch.no_grad():
            return self.linear_projection(feature)


    def sample(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """Select a representative coreset subset from the given features.
 
        Args:
            features: Feature tensor of shape (N, D).
 
        Returns:
            Subsampled feature tensor of shape (ratio * N, D).
        """
        subset_length = int(len(features) * self.ratio)
        feat = self._reduce_feature(features)

        remaining_indices = list(range(len(features)))

        origin = torch.zeros((1, feat.shape[1]))
        d = torch.cdist(feat, origin)  # (N, 1)

        selected = []

        for _ in range(subset_length):
            # Select the point farthest from all already selected points
            m = torch.amin(d, dim=1)          
            local_i = torch.argmax(m).item()  

            original_i = remaining_indices[local_i]
            selected.append(original_i)

            new_element = feat[local_i].unsqueeze(0)  # (1, D)
            feat = torch.cat((feat[:local_i], feat[local_i+1:]), dim=0)
            d    = torch.cat((d[:local_i],    d[local_i+1:]),    dim=0)
            remaining_indices.pop(local_i)

            # Incrementally update distances to avoid recomputing the full matrix
            dist_to_new = torch.cdist(feat, new_element)  # (N-1, 1)
            d = torch.cat((d, dist_to_new), dim=1)

        mask = torch.zeros(len(features), dtype=torch.bool)
        mask[selected] = True
        return features[mask]


class RandomSampler:
    """Random subsampling of a feature tensor.
 
    Selects a random subset of patches from the memory bank.
    Faster than greedy coreset sampling with minimal performance loss.
 
    Args:
        ratio: Fraction of patches to keep. Must be strictly between 0 and 1.
    """

    def __init__(
        self,
        ratio: float
    ):
        self.ratio = ratio

    def sample(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """Randomly sample a subset of the given features.
 
        Args:
            features: Feature tensor of shape (N, D).
 
        Returns:
            Subsampled feature tensor of shape (ratio * N, D).
        """
        n_samples = int(len(features) * self.ratio)
        indices = torch.randperm(len(features))[:n_samples]
        return features[indices]