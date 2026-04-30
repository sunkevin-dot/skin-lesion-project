import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve

os.makedirs("results", exist_ok=True)

MODEL_TAG = "aug"  # or "baseline"

# ------------------------
# LOAD DATA FUNCTION
# ------------------------
def load_data(dataset):
    files = {
        "mcd": f"results/mcd_predictions_{MODEL_TAG}_{dataset}.csv",
        "ensemble": f"results/ensemble_predictions_{MODEL_TAG}_{dataset}.csv"
    }

    data = {}
    for k, f in files.items():
        if os.path.exists(f):
            df = pd.read_csv(f)
            df["pred_label"] = (df["mean_prediction"] > 0.5).astype(int)
            data[k] = df

    return data

# ------------------------
# GENERATE FIGURES + TABLES
# ------------------------
def generate_outputs(data, dataset):

    # ------------------------
    # METRIC FUNCTION
    # ------------------------
    def compute_metrics(df):
        y_true = df["true_label"]
        y_pred = df["pred_label"]
        y_prob = df["mean_prediction"]

        acc = np.mean(y_true == y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = auc(fpr, tpr)

        precision = np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_pred == 1), 1)
        recall = np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1)

        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        specificity = tn / (tn + fp + 1e-8)

        f1 = 2 * precision * recall / max((precision + recall), 1e-8)

        return [acc, auc_score, precision, recall, specificity, f1]

    # ------------------------
    # MODEL COMPARISON TABLE
    # ------------------------
    table = []
    for k, df in data.items():
        table.append(compute_metrics(df))

    metrics_df = pd.DataFrame(
        table,
        columns=["Accuracy","AUC","Precision","Sensitivity","Specificity","F1"],
        index=data.keys()
    )

    metrics_df.to_csv(f"results/model_comparison_{dataset}.csv")

    # ------------------------
    # UNCERTAINTY TABLE
    # ------------------------
    uncertainty_table = pd.DataFrame({
        "Method": list(data.keys()),
        "Mean Uncertainty": [df["uncertainty"].mean() for df in data.values()],
        "Std Uncertainty": [df["uncertainty"].std() for df in data.values()]
    })

    uncertainty_table.to_csv(f"results/uncertainty_table_{dataset}.csv", index=False)

    # ------------------------
    # ROC CURVE
    # ------------------------
    plt.figure()
    for name, df in data.items():
        fpr, tpr, _ = roc_curve(df["true_label"], df["mean_prediction"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.title(f"ROC Curve ({dataset})")
    plt.savefig(f"results/roc_curve_{dataset}.png")
    plt.close()

    # ------------------------
    # CONFUSION MATRIX
    # ------------------------
    if "ensemble" in data:
        cm = confusion_matrix(
            data["ensemble"]["true_label"],
            data["ensemble"]["pred_label"]
        )

        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix ({dataset})")
        plt.savefig(f"results/confusion_matrix_{dataset}.png")
        plt.close()

    # ------------------------
    # PRECISION-RECALL CURVE
    # ------------------------
    plt.figure()
    for name, df in data.items():
        precision, recall, _ = precision_recall_curve(
            df["true_label"], df["mean_prediction"]
        )
        plt.plot(recall, precision, label=name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title(f"PR Curve ({dataset})")
    plt.savefig(f"results/pr_curve_{dataset}.png")
    plt.close()

    # ------------------------
    # UNCERTAINTY HISTOGRAM
    # ------------------------
    plt.figure()
    for name, df in data.items():
        plt.hist(df["uncertainty"], bins=50, alpha=0.5, label=name)

    plt.legend()
    plt.title(f"Uncertainty Distribution ({dataset})")
    plt.savefig(f"results/uncertainty_hist_{dataset}.png")
    plt.close()

    # ------------------------
    # UNCERTAINTY VS CORRECTNESS
    # ------------------------
    if "ensemble" in data:
        df = data["ensemble"]
        df["correct"] = df["true_label"] == df["pred_label"]

        correct = df[df["correct"]]["uncertainty"]
        incorrect = df[~df["correct"]]["uncertainty"]

        plt.figure()
        plt.boxplot([correct, incorrect], tick_labels=["Correct","Incorrect"])
        plt.title(f"Uncertainty vs Correctness ({dataset})")
        plt.savefig(f"results/uncertainty_vs_correct_{dataset}.png")
        plt.close()

    # ------------------------
    # MCD vs ENSEMBLE UNCERTAINTY
    # ------------------------
    plt.figure()
    vals = []
    labels = []

    for name, df in data.items():
        vals.append(df["uncertainty"])
        labels.append(name)

    plt.boxplot(vals, tick_labels=labels)
    plt.title(f"Uncertainty Comparison ({dataset})")
    plt.savefig(f"results/uncertainty_compare_{dataset}.png")
    plt.close()


# ------------------------
# RUN FOR BOTH DATASETS
# ------------------------
print("\nGenerating VALIDATION outputs...")
val_data = load_data("val")
generate_outputs(val_data, "val")

print("\nGenerating TEST outputs...")
test_data = load_data("test")
generate_outputs(test_data, "test")

# ------------------------
# LEARNING CURVES
# ------------------------
if os.path.exists("results/training_history_baseline.csv") and os.path.exists("results/training_history_aug.csv"):

    base = pd.read_csv("results/training_history_baseline.csv")
    aug = pd.read_csv("results/training_history_aug.csv")

    plt.figure(figsize=(12,8))

    # ------------------------
    # 1. TRAIN vs VAL LOSS
    # ------------------------
    plt.subplot(2,2,1)
    plt.plot(base["epoch"], base["train_loss"], label="Baseline Train")
    plt.plot(base["epoch"], base["val_loss"], label="Baseline Val")
    plt.title("Baseline Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(aug["epoch"], aug["train_loss"], label="Aug Train")
    plt.plot(aug["epoch"], aug["val_loss"], label="Aug Val")
    plt.title("Augmented Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # ------------------------
    # 2. AUC
    # ------------------------
    plt.subplot(2,2,3)
    plt.plot(base["epoch"], base["val_auc"], label="Baseline")
    plt.plot(aug["epoch"], aug["val_auc"], label="Augmented")
    plt.title("Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()

    # ------------------------
    # 3. ACCURACY
    # ------------------------
    plt.subplot(2,2,4)
    plt.plot(base["epoch"], base["val_accuracy"], label="Baseline")
    plt.plot(aug["epoch"], aug["val_accuracy"], label="Augmented")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/learning_curve.png")
    plt.close()

# ------------------------
# ENSEMBLE TRAINING CURVES
# ------------------------
import matplotlib.pyplot as plt
import pandas as pd
import os

ensemble_files = [
    "results/ensemble_history_aug_0.csv",
    "results/ensemble_history_aug_1.csv",
    "results/ensemble_history_aug_2.csv"
]

histories = []

# Load all histories
for i, file in enumerate(ensemble_files):
    if os.path.exists(file):
        df = pd.read_csv(file)
        df["model"] = f"Model {i+1}"
        histories.append(df)

if len(histories) > 0:

    plt.figure(figsize=(12,5))

    # ------------------------
    # 1. TRAINING LOSS
    # ------------------------
    plt.subplot(1,2,1)
    for df in histories:
        plt.plot(df["epoch"], df["train_loss"], label=df["model"].iloc[0])

    plt.title("Ensemble Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # ------------------------
    # 2. VALIDATION AUC
    # ------------------------
    plt.subplot(1,2,2)
    for df in histories:
        plt.plot(df["epoch"], df["val_auc"], label=df["model"].iloc[0])

    plt.title("Ensemble Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/ensemble_training_curves.png")
    plt.close()

    print("Saved ensemble training curves")
    
print("\nAll figures and tables generated successfully!")