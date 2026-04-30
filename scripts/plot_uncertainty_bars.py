import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

# ------------------------
# LOAD DATA
# ------------------------
mcd = pd.read_csv("results/uncertainty_summary_mcd_aug_val.csv")
ens = pd.read_csv("results/uncertainty_summary_ensemble_aug_val.csv")

# Extract values
methods = ["MCD", "Ensemble"]

unc_correct = [
    mcd["uncertainty_correct"][0],
    ens["uncertainty_correct"][0]
]

unc_incorrect = [
    mcd["uncertainty_incorrect"][0],
    ens["uncertainty_incorrect"][0]
]

ent_correct = [
    mcd["entropy_correct"][0],
    ens["entropy_correct"][0]
]

ent_incorrect = [
    mcd["entropy_incorrect"][0],
    ens["entropy_incorrect"][0]
]

x = np.arange(len(methods))

# ------------------------
# PLOT 1: UNCERTAINTY
# ------------------------
plt.figure()

plt.bar(x, unc_correct, label="Correct")
plt.bar(x, unc_incorrect, bottom=unc_correct, label="Incorrect")

plt.xticks(x, methods)
plt.ylabel("Uncertainty")
plt.title("Uncertainty Comparison (Validation)")
plt.legend()

plt.savefig("results/uncertainty_bar_val.png")
plt.close()

# ------------------------
# PLOT 2: ENTROPY
# ------------------------
plt.figure()

plt.bar(x, ent_correct, label="Correct")
plt.bar(x, ent_incorrect, bottom=ent_correct, label="Incorrect")

plt.xticks(x, methods)
plt.ylabel("Entropy")
plt.title("Entropy Comparison (Validation)")
plt.legend()

plt.savefig("results/entropy_bar_val.png")
plt.close()

print("Bar plots saved!")