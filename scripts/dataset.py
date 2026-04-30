import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ------------------------
# TRANSFORMS
# ------------------------

# Training transform WITH augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# Validation / test transform (NO augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ------------------------
# DATASET CLASS
# ------------------------

class SkinDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image path and label
        img_path = self.df.iloc[idx]["image_path"]
        label = self.df.iloc[idx]["label"]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label