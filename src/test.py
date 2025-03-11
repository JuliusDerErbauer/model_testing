import os

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
import numpy as np

from model.simple_pointnet2_autoencoder import SimplePointnet2Autoencoder

from datasets.gripper_single_frame_dataset import GripperSingleFrameDataset
from datasets.gripper_time_series_dataset import GripperTimeSeriesDataset

MODEL_PATH = "wheights/next_step_model_v2.pth"
DATA_PATH = "data/training_data_linear_random"
OUTPUT_PATH = "model_outputs/model_output_v02.npy"


def test_model(model, dataloader, device, output_path):
    model.eval()
    results = []

    with torch.no_grad():
        for x, y in dataloader:
            results.append(x[0])
            x = x.to(device)
            output, feat = model(x)
            output = output + x
            output_np = output.cpu().numpy()
            results.append(y[0])
            results.append(output_np[0])

    # Convert to NumPy array and save
    results_np = np.array(results)
    results_np = results_np.swapaxes(1, 2)

    np.save(output_path, results_np)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    dataset = GripperTimeSeriesDataset(
        DATA_PATH
    )
    dataset = Subset(dataset, [18])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = SimplePointnet2Autoencoder()
    model.load_state_dict(torch.load(MODEL_PATH))

    model.to(device)

    test_model(model, dataloader, device, OUTPUT_PATH)
