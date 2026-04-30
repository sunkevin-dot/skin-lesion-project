import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
import os

# Create results folder
os.makedirs("results/test_saliency", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build model (must match training)
model = models.resnet18(pretrained=False)

model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.fc.in_features, 1)
)

model.load_state_dict(torch.load("models/best_model.pth"))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load ISIC predictions
df = pd.read_csv("results/ensemble_predictions.csv")

# Add predicted label
df["pred_label"] = (df["mean_prediction"] > 0.5).astype(int)

# 🔥 CHOOSE AN INDEX (change this!)
index = 0

row = df.iloc[index]
image_path = row["image_path"]
true_label = int(row["true_label"])
pred_label = int(row["pred_label"])
prob = row["mean_prediction"]

# Determine case type
if true_label == 1 and pred_label == 0:
    case_type = "False Negative"
elif true_label == 0 and pred_label == 1:
    case_type = "False Positive"
else:
    case_type = "Correct"

# Load image
image = Image.open(image_path).convert("RGB")
image_resized = image.resize((224, 224))

input_tensor = transform(image).unsqueeze(0).to(device)
input_tensor.requires_grad = True

# Forward pass
output = model(input_tensor)

# Backward pass
model.zero_grad()
output.backward()

# Saliency
saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
saliency = np.max(saliency, axis=0)

# Normalize
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

# Plot
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(image_resized)
plt.title(f"True: {true_label}, Pred: {pred_label}\n{case_type}")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(image_resized)
plt.imshow(saliency, cmap="hot", alpha=0.5)
plt.title(f"Saliency (Prob={prob:.3f})")
plt.axis("off")

# Save
save_path = f"results/test_saliency/saliency_test_{index}.png"
plt.savefig(save_path)
plt.show()

print(f"Saved to {save_path}")