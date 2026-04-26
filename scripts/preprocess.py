import os
import pandas as pd
from sklearn.model_selection import train_test_split

print("Starting preprocessing...")

# Path to your dataset
DATA_DIR = "data/HAM10000"
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

# Load metadata
print("Loading metadata...")
df = pd.read_csv(METADATA_PATH)

# Convert labels: melanoma = 1, others = 0
print("Converting labels...")
df["label"] = df["dx"].apply(lambda x: 1 if x == "mel" else 0)

# Find image paths
def get_image_path(image_id):
    for folder in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        path = os.path.join(DATA_DIR, folder, image_id + ".jpg")
        if os.path.exists(path):
            return path
    return None

print("Linking image paths...")
df["image_path"] = df["image_id"].apply(get_image_path)

# Remove missing images
df = df[df["image_path"].notnull()]

# Create splits: 70% train, 15% val, 15% calib
print("Creating splits...")
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

val_df, calib_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# Save splits
print("Saving splits...")
os.makedirs("splits", exist_ok=True)

train_df[["image_path", "label"]].to_csv("splits/train.csv", index=False)
val_df[["image_path", "label"]].to_csv("splits/val.csv", index=False)
calib_df[["image_path", "label"]].to_csv("splits/calib.csv", index=False)

# Print summary
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Calib:", len(calib_df))
print("Preprocessing complete!")