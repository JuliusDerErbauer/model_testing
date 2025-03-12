import torch
from torch.utils.data import Dataset
import numpy as np


def standardize_bbox(pcl):
    # Flatten the frames and points into one dimension (N_total, 3)
    pcl_flat = pcl.reshape(-1, pcl.shape[-1])

    mins = np.amin(pcl_flat, axis=0)  # Global min for each coordinate, shape: (3,)
    maxs = np.amax(pcl_flat, axis=0)  # Global max for each coordinate, shape: (3,)
    center = (mins + maxs) / 2.  # Global center, shape: (3,)
    scale = np.amax(maxs - mins)  # Scalar scale factor
    print("Center: {}, Scale: {}".format(center, scale))

    result = ((pcl - center) / scale).astype(np.float32)
    return result


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
        standardize_bbox(point_cloud)
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        point_cloud = point_cloud.permute(1, 0)

        # Apply any additional transformations (if specified)
        if self.transform:
            point_cloud = self.transform(point_cloud)

        return point_cloud
