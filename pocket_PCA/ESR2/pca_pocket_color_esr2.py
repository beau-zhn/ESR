import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import re

# === LOAD POCKET DATA ===
pocket_path = r"C:\Users\a1611\Desktop\BioInfo\pocket_PCA\ESR2\pocket_descriptors_ESR2_dynamic.csv"
df = pd.read_csv(pocket_path)

# === LOAD CLUSTERS FILE ===
cluster_file = r"C:\Users\a1611\Desktop\BioInfo\pocket_PCA\ESR2\clusters_ESR2.txt"

cluster_map = {}  # pdb -> cluster number

with open(cluster_file, "r", encoding="utf-8") as f:
    current_cluster = None
    for line in f:
        line = line.strip()

        # detect lines like "CLUSTER 1"
        m = re.match(r"CLUSTER\s+(\d+)", line)
        if m:
            current_cluster = int(m.group(1))
            continue

        # detect lines like "- 1err.pdb"
        if line.startswith("- "):
            pdb_name = line.replace("- ", "").strip()
            cluster_map[pdb_name] = current_cluster

# add cluster numbers to df
df["cluster"] = df["pdb"].map(cluster_map)

# remove rows without cluster assignment
df = df.dropna(subset=["cluster"])
df["cluster"] = df["cluster"].astype(int)

print("Loaded clusters for:", len(df), "structures")

# === PCA FEATURES ===
X = df[["radius", "compactness", "openness"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]
df["PC3"] = X_pca[:, 2]

df.to_csv("pocket_PCA_ESR2_colored.csv", index=False)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Saved pocket_PCA_ESR2_colored.csv")

# === COLOR MAP FOR CLUSTERS ===
colors = {
    1: "red",
    2: "blue",
    3: "green",
    4: "orange"
}

# === 2D PCA PLOT ===
plt.figure(figsize=(8,6))
for c in [1, 2, 3, 4]:
    subset = df[df["cluster"] == c]
    if len(subset) == 0:
        continue
    plt.scatter(subset["PC1"], subset["PC2"], s=15, alpha=0.7,
                label=f"Cluster {c}", color=colors.get(c, "black"))

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of ESR2 Pocket Geometry (colored by clusters)")
plt.legend()
plt.grid(True)
plt.show()

# === 3D PCA PLOT ===
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

for c in [1, 2, 3, 4]:
    subset = df[df["cluster"] == c]
    if len(subset) == 0:
        continue
    ax.scatter(subset["PC1"], subset["PC2"], subset["PC3"], s=20, alpha=0.7,
               label=f"Cluster {c}", color=colors.get(c, "black"))

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("3D Pocket Geometry PCA – ESR2 (Cluster-colored)")
ax.legend()

plt.show()
