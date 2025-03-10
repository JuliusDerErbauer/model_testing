import torch
import os
import numpy as np
from torch.utils.data import Dataset


class GripperTimeSeriesDataset(Dataset):
    def __init__(self, root_dir, sequence_length=10, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform

        # List all .npy files (each representing a time series)
        self.files = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir)) if f.endswith('.npy')]

    def load_point_clouds(self, file_path):
        # Load the .npy file as a numpy array
        return np.load(file_path, allow_pickle=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        point_clouds = self.load_point_clouds(file_path)

        # Select a sequence of frames (sequence_length)
        # Make sure there's enough data to select the sequence
        if len(point_clouds) < self.sequence_length:
            raise IndexError("Not enough frames in this time series.")

        # Create time sequence slices (T, N, 3)
        start_idx = np.random.randint(0, len(point_clouds) - self.sequence_length)
        end_idx = start_idx + self.sequence_length
        sequence = point_clouds[start_idx:end_idx]

        # Convert to tensor
        sequence = torch.tensor(sequence)  # Shape (T, N, 3)

        if self.transform:
            sequence = self.transform(sequence)

        return sequence


if __name__ == "__main__":
    dataset = GripperTimeSeriesDataset(
        "/Users/julianheines/PycharmProjects/object_deformation/src/data/generated/training_data_linear_random"
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
