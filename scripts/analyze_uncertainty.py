import pandas as pd
import numpy as np

# ------------------------
# SETTINGS
# ------------------------
METHOD = "ensemble"  # "mcd", "ensemble", "mcd_ensemble"
MODEL_TAG = "aug"    # "baseline" or "aug"
DATASET = "test"     # "val" or "test"

file_map = {
    "mcd": f"results/mcd_predictions_{MODEL_TAG}_{DATASET}.csv",
    "ensemble": f"results/ensemble_predictions_{MODEL_TAG}_{DATASET}.csv",
    "mcd_ensemble": f"results/mcd_ensemble_predictions.csv"
}

file_path = file_map[METHOD]

# ------------------------
# LOAD DATA
# ------------------------
df = pd.read_csv(file_path)

# predicted label
df["pred_label"] = (df["mean_prediction"] > 0.5).astype(int)

# correctness
df["correct"] = df["true_label"] == df["pred_label"]

# ------------------------
# UNCERTAINTY METRICS
# ------------------------

# Variance-based uncertainty
correct_uncertainty = df[df["correct"]]["uncertainty"].mean()
incorrect_uncertainty = df[~df["correct"]]["uncertainty"].mean()

# Entropy
df["entropy"] = -(
    df["mean_prediction"] * np.log(df["mean_prediction"] + 1e-8) +
    (1 - df["mean_prediction"]) * np.log(1 - df["mean_prediction"] + 1e-8)
)

correct_entropy = df[df["correct"]]["entropy"].mean()
incorrect_entropy = df[~df["correct"]]["entropy"].mean()

# ------------------------
# PRINT RESULTS
# ------------------------
print("\n===== UNCERTAINTY ANALYSIS =====")
print(f"Method: {METHOD} | Model: {MODEL_TAG} | Dataset: {DATASET}\n")

print(f"Average uncertainty (correct):   {correct_uncertainty:.6f}")
print(f"Average uncertainty (incorrect): {incorrect_uncertainty:.6f}\n")

print(f"Average entropy (correct):       {correct_entropy:.6f}")
print(f"Average entropy (incorrect):     {incorrect_entropy:.6f}\n")

# ------------------------
# TOP UNCERTAIN SAMPLES
# ------------------------
top_uncertain = df.sort_values("uncertainty", ascending=False).head(10)

print("Top 10 most uncertain samples:")
print(top_uncertain[["mean_prediction", "uncertainty", "true_label", "pred_label", "correct"]])

# ------------------------
# SAVE SUMMARY (for tables)
# ------------------------
summary = pd.DataFrame({
    "method": [METHOD],
    "model": [MODEL_TAG],
    "dataset": [DATASET],
    "uncertainty_correct": [correct_uncertainty],
    "uncertainty_incorrect": [incorrect_uncertainty],
    "entropy_correct": [correct_entropy],
    "entropy_incorrect": [incorrect_entropy]
})

summary_file = f"results/uncertainty_summary_{METHOD}_{MODEL_TAG}_{DATASET}.csv"
summary.to_csv(summary_file, index=False)

print(f"\nSaved summary to {summary_file}")