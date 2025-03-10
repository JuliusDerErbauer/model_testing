import torch
from torch.utils.data import Dataset
import numpy as np


class GripperSingleFrameDataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        Args:
            file_path (str): Path to the .npy file containing multiple point clouds.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_path = file_path
        self.transform = transform
        self.point_clouds = np.load(file_path, allow_pickle=True)

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx]
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        point_cloud = point_cloud.permute(1, 0)

        # Apply any additional transformations (if specified)
        if self.transform:
            point_cloud = self.transform(point_cloud)

        return point_cloud
