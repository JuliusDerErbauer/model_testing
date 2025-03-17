import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
import numpy as np

from model.pointnet_simple import PointCloudAE
from task import next_step_prediction_task, reconstruction_task

from model.simple_pointnet2_autoencoder import SimplePointnet2Autoencoder
from model.pointnet_autoencoder import PCAutoEncoder

from datasets.gripper_single_frame_dataset import GripperSingleFrameDataset
from datasets.gripper_time_series_dataset import GripperTimeSeriesDataset

MODEL_PATH = "weights/reconstruction_chamfer_pointnet_model_v1.pth"
DATA_PATH = "data/random_data_0.npy"
OUTPUT_PATH = "model_outputs/reconstruction_output_chamfer_v01.npy"
TASK = reconstruction_task


# IT IS: X, Y, Y_pred

def test_model(model, dataloader, device, output_path, task):
    model.eval()
    results = []

    with torch.no_grad():
        for data in dataloader:
            (inputs, outputs, targets) = task(data, model, device)
            output_np = outputs.cpu().numpy()
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            results.append(inputs_np[0])
            results.append(targets_np[0])
            results.append(output_np[0])

    # Convert to NumPy array and save
    results_np = np.array(results)
    results_np = results_np.swapaxes(1, 2)

    np.save(output_path, results_np)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    dataset = GripperSingleFrameDataset(
        DATA_PATH
    )
    dataset = Subset(dataset, [18])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = PointCloudAE(1024, 128)
    model.load_state_dict(torch.load(MODEL_PATH))

    model.to(device)

    test_model(model, dataloader, device, OUTPUT_PATH, TASK)

