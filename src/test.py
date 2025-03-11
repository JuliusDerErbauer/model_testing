import os

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
import numpy as np

from model.simple_pointnet2_autoencoder import SimplePointnet2Autoencoder

from datasets.gripper_single_frame_dataset import GripperSingleFrameDataset

MODEL_PATH = "best_model_v1.pth"
OUTPUT_PATH = "model_outputs/model_output_v01.npy"


def test_model(model, dataloader, device, output_path):
    model.eval()
    results = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            output = model(data)
            output_np = output.permute(0, 2, 1).cpu().numpy()
            results.append(output_np)

    # Convert to NumPy array and save
    results_np = np.array(results[0])
    np.save(output_path, results_np)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    dataset = GripperSingleFrameDataset(
        "data/random_data_0.npy"
    )
    dataset = Subset(dataset, [21])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = SimplePointnet2Autoencoder()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))

    model.to(device)

    test_model(model, dataloader, device, OUTPUT_PATH)
