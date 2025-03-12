import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
import numpy as np

from model.simple_pointnet2_autoencoder import SimplePointnet2Autoencoder
from model.pointnet_autoencoder import PCAutoEncoder

from datasets.gripper_single_frame_dataset import GripperSingleFrameDataset
from datasets.gripper_time_series_dataset import GripperTimeSeriesDataset

MODEL_PATH = "wheights/next_step_model_pointnet_v0.pth"
DATA_PATH = "/Users/julianheines/PycharmProjects/object_deformation/src/data/generated/training_data_all_arms_same"
OUTPUT_PATH = "model_outputs/pointnet_series_output_v00.npy"


# IT IS: X, Y, Y_pred

def test_model(model, dataloader, device, output_path, num_step=100):
    model.eval()
    results = []

    with torch.no_grad():
        for x, y in dataloader:
            results.append(x[0])
            x = x.to(device)
            for i in range(num_step):
                output, feat = model(x)
                output = output + x
                x = output
                output_np = output.cpu().numpy()
                results.append(output_np[0])
                output.to(device)


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
    dataset.set_frame_idx(1)
    dataset = Subset(dataset, [0])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = PCAutoEncoder()
    model.load_state_dict(torch.load(MODEL_PATH))

    model.to(device)

    test_model(model, dataloader, device, OUTPUT_PATH)
