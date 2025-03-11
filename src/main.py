import os.path

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim
import torch
from tqdm import tqdm

from datasets.gripper_time_series_dataset import GripperTimeSeriesDataset
from data_prerepration.noise_augmentation import NoiseAugmentation
from model.simple_pointnet2_autoencoder import SimplePointnet2Autoencoder

MODEL_PATH = "wheights/next_step_model_v2.pth"
DATA_PATH = "data/training_data_linear_random"
EPOCHS = 10
BATCH_SIZE = 20
NUM_POINTS_TRAIN = 300
SPLIT = 0.25
NUM_POINTS_VAL = int(NUM_POINTS_TRAIN * SPLIT)

# best_model_v1 -> 68.7471 training loss (arround 67 training loss)
# full_model_v0 ->

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, device, epochs=10):
    best_val_loss = float('inf')  # To store the best validation loss
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Training Loop
        for data in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()  # Zero gradients from previous step
            inputs = data[0]  # assuming the data contains the point cloud or inputs
            targets = data[1]  # your target data, or could be labels

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)  # Forward pass through the model
            loss = criterion(outputs, targets)  # Compute the loss

            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            print(loss.item())

        avg_train_loss = running_loss / len(train_dataloader)  # Average training loss
        print(f"Train Loss: {avg_train_loss:.4f}")

        # Evaluate after each epoch
        val_loss = evaluate_model(model, val_dataloader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)  # Save model weights


def evaluate_model(model, val_dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    with torch.no_grad():  # No gradient calculation during evaluation
        for data in val_dataloader:
            inputs = data[0]
            targets = data[1]
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

    avg_val_loss = running_loss / len(val_dataloader)
    return avg_val_loss


if __name__ == "__main__":
    device = None
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    device = torch.device(device)

    data_augmentation_pipeline = NoiseAugmentation()

    dataset = GripperTimeSeriesDataset(
        DATA_PATH,
        transform=data_augmentation_pipeline
    )

    indices = list(range(len(dataset)))

    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=17)

    train_subset = Subset(dataset, train_indices)
    subset_indices = torch.randperm(len(train_subset))[:NUM_POINTS_TRAIN]
    train_subset = Subset(train_subset, subset_indices)

    val_subset = Subset(dataset, val_indices)
    subset_indices = torch.randperm(len(train_subset))[:NUM_POINTS_VAL]
    val_subset = Subset(val_subset, subset_indices)

    train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimplePointnet2Autoencoder()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, device, epochs=EPOCHS)
