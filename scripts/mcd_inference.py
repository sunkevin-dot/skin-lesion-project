import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn

from dataset import SkinDataset, val_transform

# ------------------------
# SETTINGS
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = 50  # number of stochastic passes

# Choose model
USE_AUGMENTED = True   # True → augmented, False → baseline

# Choose dataset
USE_TEST = False        # True → ISIC test, False → validation

model_tag = "aug" if USE_AUGMENTED else "baseline"
dataset_csv = "splits/isic_test.csv" if USE_TEST else "splits/val.csv"
output_file = f"results/mcd_predictions_{model_tag}_{'test' if USE_TEST else 'val'}.csv"

# ------------------------
# MODEL BUILDER
# ------------------------
def build_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 1)
    )
    return model

# ------------------------
# LOAD MODEL
# ------------------------
model = build_model()
model.load_state_dict(torch.load(f"models/best_model_{model_tag}.pth"))
model.to(device)

# IMPORTANT: enable dropout at inference
model.train()

# ------------------------
# LOAD DATA
# ------------------------
dataset = SkinDataset(dataset_csv, transform=val_transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ------------------------
# INFERENCE
# ------------------------
all_means = []
all_vars = []
all_labels = []

for images, labels in loader:
    images = images.to(device)

    preds = []

    for _ in range(T):
        outputs = model(images)
        prob = torch.sigmoid(outputs)
        preds.append(prob.item())

    preds = np.array(preds)

    all_means.append(preds.mean())
    all_vars.append(preds.var())
    all_labels.append(labels.item())

# ------------------------
# SAVE RESULTS
# ------------------------
df = pd.DataFrame({
    "image_path": dataset.df["image_path"],
    "mean_prediction": all_means,
    "uncertainty": all_vars,
    "true_label": all_labels
})

df.to_csv(output_file, index=False)

print(f"Saved MCD results to {output_file}")
print("Average prediction:", np.mean(all_means))
print("Average uncertainty:", np.mean(all_vars))