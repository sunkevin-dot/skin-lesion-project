import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import pandas as pd
from sklearn.metrics import roc_auc_score

from dataset import SkinDataset, train_transform, val_transform

# ------------------------
# SETTINGS
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_MODELS = 3
EPOCHS = 7
USE_AUGMENTATION = True  # set False for baseline ensemble

model_tag = "aug" if USE_AUGMENTATION else "baseline"

# ------------------------
# MODEL BUILDER (must match train.py)
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

# Ensure save dir exists
os.makedirs("models/ensemble", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ------------------------
# TRAIN MULTIPLE MODELS
# ------------------------
for m in range(NUM_MODELS):
    print(f"\nTraining model {m+1}/{NUM_MODELS}")

    model = build_model().to(device)

    labels = train_dataset.df["label"]

    num_pos = (labels == 1).sum()
    num_neg = (labels == 0).sum()

    pos_weight_value = num_neg / num_pos

    print(f"Computed pos_weight: {pos_weight_value:.4f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value]).to(device)
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_auc = 0
    history = []

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

        # ----- VALIDATE -----
        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images).squeeze()
                probs = torch.sigmoid(outputs)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        auc = roc_auc_score(all_labels, all_probs)

        history.append([epoch+1, total_loss, auc])

        print(f"Model {m+1}, Epoch {epoch+1}, Train Loss: {total_loss:.4f}, AUC: {auc:.4f}")

        # Save best model
        if auc > best_auc:
            best_auc = auc
            save_path = f"models/ensemble/model_{model_tag}_{m}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved model {m} (best so far)")

    # Save training history for this model
    hist_df = pd.DataFrame(history, columns=["epoch", "train_loss", "val_auc"])
    hist_df.to_csv(f"results/ensemble_history_{model_tag}_{m}.csv", index=False)

print("\nAll ensemble models trained!")