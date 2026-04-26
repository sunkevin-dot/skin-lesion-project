import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F

import os
os.makedirs("models", exist_ok=True)

from dataset import SkinDataset, train_transform, val_transform


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_dataset = SkinDataset("splits/train.csv", transform=train_transform)
val_dataset = SkinDataset("splits/val.csv", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Replace final layer with dropout + linear
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, 1)
)
model = model.to(device)

# Loss + optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 5  # start small for testing
best_auc = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to binary predictions
    preds = [1 if p > 0.5 else 0 for p in all_probs]

    # Metrics
    accuracy = accuracy_score(all_labels, preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    print(f"Epoch {epoch+1}, Val Loss: {val_loss / len(val_loader):.4f}")
    print(f"Epoch {epoch+1}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), "models/best_model.pth")
        print("Saved new best model!")
    