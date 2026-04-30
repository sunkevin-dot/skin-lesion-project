import pandas as pd

df = pd.read_csv("results/mcd_predictions.csv")

# Add predicted label
df["pred_label"] = (df["mean_prediction"] > 0.5).astype(int)

# Find incorrect predictions
incorrect = df[df["true_label"] != df["pred_label"]]

print(incorrect.head(10))