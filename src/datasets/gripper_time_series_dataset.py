import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random


class GripperTimeSeriesDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (str): Path to the directory containing .npy files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform
        self.files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')])
        self.file_frame_counts = {f: self._num_frames(f) for f in self.files}

    def _num_frames(self, file_path):
        """Loads only the shape of the file to count frames without loading full data."""
        return np.load(file_path, mmap_mode='r').shape[0]

    def set_frame_idx(self, idx):
        self.frame_idx = idx

    def __len__(self):
        """Estimated total number of (frame, next_frame) pairs across all files."""
        return sum(v - 1 for v in self.file_frame_counts.values())

    def __getitem__(self, idx):
        """Randomly selects a file and a frame from that file."""
        file_path = random.choice(self.files)
        num_frames = self.file_frame_counts[file_path]

        if num_frames < 2:
            return self.__getitem__(idx)  # Skip files with <2 frames

        frame_idx = random.randint(0, num_frames - 2)  # Select a frame except the last one

        if self.frame_idx:
            frame_idx = self.frame_idx

        # Load only the required file and extract the needed frames
        data = np.load(file_path, allow_pickle=True)
        current_frame = data[frame_idx]  # Shape: (num_points, 3)
        next_frame = data[frame_idx + 1]  # Shape: (num_points, 3)

        # Convert to PyTorch tensors and reshape if necessary
        current_frame = torch.tensor(current_frame, dtype=torch.float32).permute(1, 0)  # Shape: (3, num_points)
        next_frame = torch.tensor(next_frame, dtype=torch.float32).permute(1, 0)  # Shape: (3, num_points)

        if self.transform:
            current_frame = self.transform(current_frame)
            next_frame = self.transform(next_frame)

        return current_frame, next_frame
