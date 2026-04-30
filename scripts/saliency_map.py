import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
import os

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model (same architecture as training)
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

# Load dataset CSV
df = pd.read_csv("results/mcd_predictions.csv")

# 🔥 CHANGE THIS INDEX TO TEST DIFFERENT CASES
index = 8

sample_path = df.iloc[index]["image_path"]
true_label = int(df.iloc[index]["true_label"])

# Load image
image = Image.open(sample_path).convert("RGB")

# Resize for BOTH model and visualization (fix mismatch)
image_resized = image.resize((224, 224))

# Prepare input tensor
input_tensor = transform(image).unsqueeze(0).to(device)
input_tensor.requires_grad = True

# Use stored prediction instead of recomputing
prob = df.iloc[index]["mean_prediction"]
pred_label = 1 if prob > 0.5 else 0

# Forward pass (needed for gradient)
output = model(input_tensor)

# Backward pass
model.zero_grad()
output.backward()

# Determine case type
if true_label == 1 and pred_label == 0:
    case_type = "False Negative (Missed Melanoma)"
elif true_label == 0 and pred_label == 1:
    case_type = "False Positive"
else:
    case_type = "Correct"



# Generate saliency map
saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
saliency = np.max(saliency, axis=0)

# Normalize saliency for visualization
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

# Plot
plt.figure(figsize=(12,5))

# Original image with labels
plt.subplot(1,2,1)
plt.imshow(image_resized)
plt.title(f"Original Image\nTrue: {true_label}, Pred: {pred_label}\n{case_type}")
plt.axis("off")

# Saliency overlay
plt.subplot(1,2,2)
plt.imshow(image_resized)
plt.imshow(saliency, cmap="hot", alpha=0.5)
plt.title("Saliency Map Overlay")
plt.axis("off")

# Save figure
save_path = f"results/saliency_{index}.png"
plt.savefig(save_path)
plt.show()

print(f"Saved saliency map to {save_path}")
print(f"Prediction probability: {prob:.4f}")
print(f"Case type: {case_type}")