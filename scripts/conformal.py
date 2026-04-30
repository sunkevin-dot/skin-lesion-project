import pandas as pd
import numpy as np

# ------------------------
# SETTINGS
# ------------------------
METHOD = "ensemble"   # "mcd", "ensemble", "mcd_ensemble"
MODEL_TAG = "aug"     # "baseline" or "aug"
DATASET = "test"      # "val" or "test"

alpha = 0.1  # 90% coverage

file_map = {
    "mcd": f"results/mcd_predictions_{MODEL_TAG}_{DATASET}.csv",
    "ensemble": f"results/ensemble_predictions_{MODEL_TAG}_{DATASET}.csv",
    "mcd_ensemble": f"results/mcd_ensemble_predictions.csv"
}

calib_file_map = {
    "mcd": f"results/mcd_predictions_{MODEL_TAG}_val.csv",
    "ensemble": f"results/ensemble_predictions_{MODEL_TAG}_val.csv",
    "mcd_ensemble": f"results/mcd_ensemble_predictions.csv"
}

test_file = file_map[METHOD]
calib_file = calib_file_map[METHOD]

# ------------------------
# LOAD DATA
# ------------------------
calib_df = pd.read_csv(calib_file)
test_df = pd.read_csv(test_file)

# ------------------------
# NONCONFORMITY SCORE
# ------------------------
# s(x) = 1 - probability(true class)
calib_scores = []

for _, row in calib_df.iterrows():
    p = row["mean_prediction"]
    y = row["true_label"]

    score = 1 - p if y == 1 else p
    calib_scores.append(score)

calib_scores = np.array(calib_scores)

# ------------------------
# THRESHOLD
# ------------------------
threshold = np.quantile(calib_scores, 1 - alpha)

print(f"\nConformal threshold: {threshold:.4f}")

# ------------------------
# APPLY TO TEST SET
# ------------------------
prediction_sets = []
correct_flags = []

for _, row in test_df.iterrows():
    p = row["mean_prediction"]
    y = row["true_label"]

    # scores for both classes
    score_0 = p
    score_1 = 1 - p

    pred_set = []

    if score_0 <= threshold:
        pred_set.append(0)

    if score_1 <= threshold:
        pred_set.append(1)

    prediction_sets.append(pred_set)

    # check coverage
    correct_flags.append(y in pred_set)

# ------------------------
# METRICS
# ------------------------
coverage = np.mean(correct_flags)
set_sizes = [len(s) for s in prediction_sets]
avg_set_size = np.mean(set_sizes)
ambiguity_rate = np.mean([1 if len(s) > 1 else 0 for s in prediction_sets])

# ------------------------
# PRINT RESULTS
# ------------------------
print("\n===== CONFORMAL RESULTS =====")
print(f"Method: {METHOD} | Model: {MODEL_TAG} | Dataset: {DATASET}\n")

print(f"Coverage: {coverage:.4f}")
print(f"Average set size: {avg_set_size:.4f}")
print(f"Ambiguity rate: {ambiguity_rate:.4f}")

# ------------------------
# SAVE RESULTS
# ------------------------
summary = pd.DataFrame({
    "method": [METHOD],
    "model": [MODEL_TAG],
    "dataset": [DATASET],
    "coverage": [coverage],
    "avg_set_size": [avg_set_size],
    "ambiguity_rate": [ambiguity_rate]
})

summary_file = f"results/conformal_summary_{METHOD}_{MODEL_TAG}_{DATASET}.csv"
summary.to_csv(summary_file, index=False)

print(f"\nSaved conformal results to {summary_file}")