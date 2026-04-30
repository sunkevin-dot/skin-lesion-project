import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# ------------------------
# SETTINGS
# ------------------------
MODEL_TAG = "aug"   # "baseline" or "aug"
DATASET = "test"    # "val" or "test"

df = pd.read_csv(f"results/ensemble_predictions_{MODEL_TAG}_{DATASET}.csv")

# Predicted labels
df["pred_label"] = (df["mean_prediction"] > 0.5).astype(int)

y_true = df["true_label"].values
y_pred = df["pred_label"].values
y_prob = df["mean_prediction"].values

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)  # sensitivity
f1 = f1_score(y_true, y_pred, zero_division=0)

# Specificity
tn = np.sum((y_true == 0) & (y_pred == 0))
fp = np.sum((y_true == 0) & (y_pred == 1))
specificity = tn / (tn + fp + 1e-8)

# AUC
auc = roc_auc_score(y_true, y_prob)

# Print results
print("\n===== RESULTS =====")
print(f"Model: {MODEL_TAG} | Dataset: {DATASET}\n")

print(f"Accuracy:    {accuracy:.4f}")
print(f"AUC:         {auc:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Sensitivity: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score:    {f1:.4f}")