import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
import os

os.makedirs("results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# MODEL (MATCH TRAINING)
# ------------------------
def build_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 1)
    )
    return model

model = build_model()
model.load_state_dict(torch.load("models/best_model_aug.pth"))  # use best model
model.to(device)
model.eval()

# ------------------------
# TRANSFORM
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ------------------------
# LOAD PREDICTIONS
# ------------------------
df = pd.read_csv("results/ensemble_predictions_aug_test.csv")
df["pred_label"] = (df["mean_prediction"] > 0.5).astype(int)

# ------------------------
# AUTO SELECT CASES
# ------------------------
fn = df[(df["true_label"]==1) & (df["pred_label"]==0)].iloc[0]
fp = df[(df["true_label"]==0) & (df["pred_label"]==1)].iloc[0]
tp = df[(df["true_label"]==1) & (df["pred_label"]==1)].iloc[0]
tn = df[(df["true_label"]==0) & (df["pred_label"]==0)].iloc[0]

cases = [fn, fp, tp, tn]
titles = ["False Negative", "False Positive", "True Positive", "True Negative"]

# ------------------------
# PLOT
# ------------------------
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, row in enumerate(cases):

    image_path = row["image_path"]
    true_label = int(row["true_label"])
    pred_label = int(row["pred_label"])
    prob = row["mean_prediction"]

    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((224, 224))

    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    # Forward
    output = model(input_tensor)

    # Backward
    model.zero_grad()
    output.backward()

    # Saliency
    saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
    saliency = np.max(saliency, axis=0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    r = i // 2
    c = (i % 2) * 2

    # Original
    axes[r, c].imshow(image_resized)
    axes[r, c].set_title(f"{titles[i]}\nTrue:{true_label} Pred:{pred_label}")
    axes[r, c].axis("off")

    # Saliency
    axes[r, c+1].imshow(image_resized)
    axes[r, c+1].imshow(saliency, cmap="hot", alpha=0.5)
    axes[r, c+1].set_title(f"Saliency (p={prob:.2f})")
    axes[r, c+1].axis("off")

plt.tight_layout()
plt.savefig("results/saliency_test_combined.png")
plt.show()

print("Saved combined saliency figure")