import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataloader import BraTSDataset
from src.preprocessing import preprocess
from src.model import UNet3D


# ==== EDIT THIS PATH FOR YOUR MACHINE ====
DATA_DIR = r"E:\brain tumor segmentation and analysis using cnn\datasets\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
# ==========================================

NUM_EPOCHS = 10
BATCH_SIZE = 1  # bump to 2 later if GPU has enough VRAM — test with 1 first


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)

        num_classes = preds.shape[1]
        targets_onehot = torch.nn.functional.one_hot(targets.long(), num_classes)
        targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()

        intersection = (preds * targets_onehot).sum(dim=(2, 3, 4))
        union = preds.sum(dim=(2, 3, 4)) + targets_onehot.sum(dim=(2, 3, 4))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def train():
    all_patients = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])
    train_ids, val_ids = train_test_split(all_patients, test_size=0.2, random_state=42)

    train_dataset = BraTSDataset(DATA_DIR, transform=preprocess)
    train_dataset.patient_dirs = train_ids

    val_dataset = BraTSDataset(DATA_DIR, transform=preprocess)
    val_dataset.patient_dirs = val_ids

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet3D(in_channels=4, num_classes=4).to(device)
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_model.pt")
            print("Saved new best model.")


if __name__ == "__main__":
    train()