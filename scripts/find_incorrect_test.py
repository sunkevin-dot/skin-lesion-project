import pandas as pd

df = pd.read_csv("results/ensemble_predictions.csv")

df["pred_label"] = (df["mean_prediction"] > 0.5).astype(int)

incorrect = df[df["true_label"] != df["pred_label"]]

print("Number of incorrect samples:", len(incorrect))
print(incorrect.head(10))