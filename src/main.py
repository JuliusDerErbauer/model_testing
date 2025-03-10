import os.path

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim
import torch
from tqdm import tqdm

from src.datasets.gripper_single_frame_dataset import GripperSingleFrameDataset
from src.data_prerepration.noise_augmentation import NoiseAugmentation
from src.model.simple_pointnet2_autoencoder import SimplePointnet2Autoencoder

MODEL_PATH = "wheights/best_model_v3.pth"


# best_model_v1 -> 68.7471 training loss (arround 67 training loss)
# best_model_v2 ->

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, device, epochs=10):
    best_val_loss = float('inf')  # To store the best validation loss
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Training Loop
        for data in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()  # Zero gradients from previous step
            inputs = data  # assuming the data contains the point cloud or inputs
            targets = data  # your target data, or could be labels

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

        avg_train_loss = running_loss / len(train_dataloader)  # Average training loss
        print(f"Train Loss: {avg_train_loss:.4f}")

        # Evaluate after each epoch
        val_loss = evaluate_model(model, val_dataloader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")  # Save model weights


def evaluate_model(model, val_dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    with torch.no_grad():  # No gradient calculation during evaluation
        for data in val_dataloader:
            inputs = data
            targets = data
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
        "cuda"
    else:
        device = "cpu"
    device = torch.device(device)

    data_augmentation_pipeline = transforms.Compose(
        [NoiseAugmentation()]
    )

    dataset = GripperSingleFrameDataset(
        "/Users/julianheines/PycharmProjects/object_deformation/src/data/generated/random_data_0.npy",
        transform=data_augmentation_pipeline
    )

    train_dataset, eval_dataset = train_test_split(dataset, test_size=0.2, random_state=17)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.25, random_state=17)

    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=20, shuffle=False)

    model = SimplePointnet2Autoencoder()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, device, epochs=10)
