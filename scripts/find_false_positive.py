import pandas as pd

# Load predictions
df = pd.read_csv("results/mcd_predictions.csv")

# Add predicted label
df["pred_label"] = (df["mean_prediction"] > 0.5).astype(int)

# Find false positives (benign predicted as malignant)
false_positives = df[(df["true_label"] == 0) & (df["pred_label"] == 1)]

print("Number of false positives:", len(false_positives))
print("\nSample false positives:\n")
print(false_positives.head(10))