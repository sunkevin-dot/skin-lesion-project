import os
import pandas as pd
from sklearn.model_selection import train_test_split

print("Starting preprocessing...")

# ------------------------
# PATHS
# ------------------------
DATA_DIR = "data/HAM10000"
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

# ------------------------
# LOAD METADATA
# ------------------------
print("Loading metadata...")
df = pd.read_csv(METADATA_PATH)

# ------------------------
# LABEL CONVERSION
# ------------------------
print("Converting labels...")
df["label"] = df["dx"].apply(lambda x: 1 if x == "mel" else 0).astype(int)

# ------------------------
# FIND IMAGE PATHS
# ------------------------
def get_image_path(image_id):
    for folder in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        path = os.path.join(DATA_DIR, folder, image_id + ".jpg")
        if os.path.exists(path):
            return path
    return None

print("Linking image paths...")
df["image_path"] = df["image_id"].apply(get_image_path)

# ------------------------
# CLEAN DATA
# ------------------------
print("Removing missing images...")
df = df[df["image_path"].notnull()]
df = df.reset_index(drop=True)

# ------------------------
# PRINT DATASET DISTRIBUTION
# ------------------------
print("\nFull dataset class distribution:")
print(df["label"].value_counts())

# Optional: save full dataset
os.makedirs("splits", exist_ok=True)
df.to_csv("splits/full_dataset.csv", index=False)

# ------------------------
# TRAIN / VAL / CALIB SPLIT
# ------------------------
print("\nCreating splits...")

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

# ------------------------
# SAVE SPLITS
# ------------------------
print("Saving splits...")

train_df[["image_path", "label"]].to_csv("splits/train.csv", index=False)
val_df[["image_path", "label"]].to_csv("splits/val.csv", index=False)
calib_df[["image_path", "label"]].to_csv("splits/calib.csv", index=False)

# ------------------------
# VERIFY SPLITS
# ------------------------
print("\nSplit sizes:")
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Calib:", len(calib_df))

print("\nSplit class distributions:")
print("Train:\n", train_df["label"].value_counts())
print("Val:\n", val_df["label"].value_counts())
print("Calib:\n", calib_df["label"].value_counts())

print("\nPreprocessing complete!")