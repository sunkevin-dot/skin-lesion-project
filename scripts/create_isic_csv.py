import pandas as pd

# Load ISIC test labels
df = pd.read_csv("data/ISIC_2019/ISIC_2019_Test_GroundTruth.csv")

# Convert to binary label (melanoma vs others)
df["label"] = df["MEL"]

# Create image path
df["image_path"] = df["image"].apply(
    lambda x: f"data/ISIC_2019/{x}.jpg"
)

# Keep only needed columns
df_final = df[["image_path", "label"]]

# Save
df_final.to_csv("splits/isic_test.csv", index=False)

print("Created splits/isic_test.csv")