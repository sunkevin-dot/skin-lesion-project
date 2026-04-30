import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

# ------------------------
# LOAD DATA
# ------------------------
mcd = pd.read_csv("results/conformal_summary_mcd_aug_val.csv")
ens = pd.read_csv("results/conformal_summary_ensemble_aug_val.csv")

# Combine into one table
df = pd.DataFrame({
    "Method": ["MCD", "Deep Ensemble"],
    "Coverage": [mcd["coverage"][0], ens["coverage"][0]],
    "Avg Set Size": [mcd["avg_set_size"][0], ens["avg_set_size"][0]],
    "Ambiguity Rate": [mcd["ambiguity_rate"][0], ens["ambiguity_rate"][0]]
})

# ------------------------
# SAVE TABLE
# ------------------------
df.to_csv("results/conformal_comparison_val.csv", index=False)

print("\nConformal Comparison Table:")
print(df)

# ------------------------
# BAR PLOT
# ------------------------
x = np.arange(len(df["Method"]))
width = 0.25

plt.figure(figsize=(8,5))

plt.bar(x - width, df["Coverage"], width, label="Coverage")
plt.bar(x, df["Avg Set Size"], width, label="Avg Set Size")
plt.bar(x + width, df["Ambiguity Rate"], width, label="Ambiguity Rate")

plt.xticks(x, df["Method"])
plt.ylabel("Value")
plt.title("Conformal Prediction Comparison (Validation)")
plt.legend()

plt.savefig("results/conformal_bar_val.png")
plt.close()

print("Saved bar chart and table!")