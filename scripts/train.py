import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from dataset import SkinDataset, train_transform, val_transform

# ------------------------
# SETTINGS
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 7
USE_AUGMENTATION = True # change to False for baseline

model_name = "aug" if USE_AUGMENTATION else "baseline"

# ------------------------
# MODEL BUILDER (IMPORTANT)
# ------------------------
def build_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 1)
    )
    return model

# ------------------------
# DATA
# ------------------------
train_dataset = SkinDataset(
    "splits/train.csv",
    transform=train_transform if USE_AUGMENTATION else val_transform
)

val_dataset = SkinDataset("splits/val.csv", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ------------------------
# MODEL
# ------------------------
model = build_model().to(device)

labels = train_dataset.df["label"]

num_pos = (labels == 1).sum()
num_neg = (labels == 0).sum()

pos_weight_value = num_neg / num_pos

print(f"Positive samples: {num_pos}")
print(f"Negative samples: {num_neg}")
print(f"Computed pos_weight: {pos_weight_value:.4f}")

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight_value]).to(device)
)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ------------------------
# TRACKING
# ------------------------
train_losses = []
val_losses = []
val_aucs = []
val_accs = []

best_auc = 0

# ------------------------
# TRAINING LOOP
# ------------------------
for epoch in range(EPOCHS):

    # ----- TRAIN -----
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

    train_losses.append(total_loss)

    # ----- VALIDATION -----
    model.eval()
    val_loss = 0
    all_probs = []
    all_labels = []

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

    preds = [1 if p > 0.5 else 0 for p in all_probs]

    accuracy = accuracy_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_probs)

    val_losses.append(val_loss / len(val_loader))
    val_aucs.append(auc)
    val_accs.append(accuracy)

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {total_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"AUC: {auc:.4f}, Accuracy: {accuracy:.4f}\n")

    # Save best model
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), f"models/best_model_{model_name}.pth")
        print("Saved new best model!\n")

# ------------------------
# SAVE TRAINING HISTORY
# ------------------------
history = pd.DataFrame({
    "epoch": list(range(1, EPOCHS + 1)),
    "train_loss": train_losses,
    "val_loss": val_losses,
    "val_auc": val_aucs,
    "val_accuracy": val_accs
})

history.to_csv(f"results/training_history_{model_name}.csv", index=False)

print(f"Training complete. Saved history for {model_name}")
    