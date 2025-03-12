import os.path

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim
import torch
from tqdm import tqdm
from chamfer_distance import ChamferDistance

from datasets.gripper_time_series_dataset import GripperTimeSeriesDataset
from datasets.gripper_single_frame_dataset import GripperSingleFrameDataset
from data_prerepration.noise_augmentation import NoiseAugmentation
from model.simple_pointnet2_autoencoder import SimplePointnet2Autoencoder
from model.pointnet_autoencoder import PCAutoEncoder


class ChamferLoss:
    def __init__(self):
        self.loss = ChamferDistance()

    def forward(self, x):
        return self.loss(x)[0]


def next_step_prediction_task(data, model, device):
    inputs = data[0]
    targets = data[1]

    inputs = inputs.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs, feature = model(inputs)  # Forward pass through the model
    outputs = outputs + inputs
    return (inputs, outputs, targets)


def reconstruction_task(data, model, device):
    inputs = data

    inputs = inputs.to(device)
    targets = inputs
    outputs, feature = model(inputs)

    return (inputs, outputs, targets)


MODEL_PATH = "wheights/reconstruction_full_model_v1.pth"
DATA_PATH = "data/random_data_0.npy"
EPOCHS = 50
BATCH_SIZE = 20
NUM_POINT_CLOUDS = 300
SPLIT = 0.2
NUM_POINTS_TRAIN = int(NUM_POINT_CLOUDS * (1 - SPLIT))

NUM_POINTS_VAL = int(NUM_POINT_CLOUDS * SPLIT)
TASK = reconstruction_task


# best_model_v1 -> 68.7471 training loss (arround 67 training loss)
# next_stept_model_pointnetv2 ->


def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, task, device, epochs=10):
    best_val_loss = float('inf')  # To store the best validation loss
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_naive_loss = 0.0

        # Training Loop
        for data in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()  # Zero gradients from previous step

            (inputs, outputs, targets) = task(data, model, device)

            loss = criterion(outputs, targets)  # Compute the loss
            naive_loss = criterion(inputs, targets)

            running_loss += loss.item()
            running_naive_loss += naive_loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = running_loss / len(train_dataloader)  # Average training loss
        print(f"Train Loss: {avg_train_loss:.4f}")

        avg_naive_loss = running_naive_loss / len(train_dataloader)
        print(f"Naive Loss {avg_naive_loss:.4f}")

        # Evaluate after each epoch
        val_loss = evaluate_model(model, val_dataloader, criterion, task, device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)  # Save model weights


def evaluate_model(model, val_dataloader, criterion, task, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    running_naive_loss = 0.0
    with torch.no_grad():  # No gradient calculation during evaluation
        for data in val_dataloader:
            (inputs, outputs, targets) = task(data, model, device)
            loss = criterion(outputs, targets)
            naive_loss = criterion(inputs, targets)

            running_loss += loss.item()
            running_naive_loss += naive_loss.item()

    avg_val_loss = running_loss / len(val_dataloader)
    print(f"Naive Loss eval: {running_naive_loss / len(val_dataloader):.4f}")
    return avg_val_loss


def main():
    device = None
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    device = torch.device(device)

    data_augmentation_pipeline = NoiseAugmentation()

    dataset = GripperSingleFrameDataset(
        DATA_PATH,
        transform=data_augmentation_pipeline
    )

    indices = list(range(len(dataset)))

    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=17)

    train_subset = Subset(dataset, train_indices)
    subset_indices = torch.randperm(len(train_subset))[:NUM_POINTS_TRAIN]
    train_subset = Subset(train_subset, subset_indices)

    val_subset = Subset(dataset, val_indices)
    subset_indices = torch.randperm(len(val_subset))[:NUM_POINTS_VAL]
    val_subset = Subset(val_subset, subset_indices)

    train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimplePointnet2Autoencoder()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    criterion = ChamferLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, TASK,
                device, epochs=EPOCHS)


if __name__ == "__main__":
    main()
