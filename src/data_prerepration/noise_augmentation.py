import torch


class NoiseAugmentation:
    def __init__(self, mean=0.0, std=0.01):
        """
        Args:
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.
        """
        self.mean = mean
        self.std = std

    def __call__(self, point_cloud):
        """Add Gaussian noise to the point cloud."""
        noise = torch.normal(self.mean, self.std, size=point_cloud.shape)
        return point_cloud + noise
