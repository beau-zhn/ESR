import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# === LOAD POCKET DESCRIPTORS ===
csv_path = r"/pocket_descriptors_ESR1_dynamic.csv"
df = pd.read_csv(csv_path)

# Features for PCA
X = df[["radius", "compactness", "openness"]].values

# === NORMALIZATION ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === PCA (3 components) ===
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)

# Save PCA results
df["PC1"] = X_pca[:,0]
df["PC2"] = X_pca[:,1]
df["PC3"] = X_pca[:,2]

df.to_csv("pocket_PCA_ESR1.csv", index=False)
print("Saved PCA results → pocket_PCA_ESR1.csv")

# === 2D PCA PLOT ===
plt.figure(figsize=(8,6))
plt.scatter(df["PC1"], df["PC2"], s=15, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of ESR1 Pocket Geometry (radius, compactness, openness)")
plt.grid(True)
plt.show()

# === 3D PCA PLOT ===
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["PC1"], df["PC2"], df["PC3"], s=15, alpha=0.7)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("3D Pocket Geometry PCA – ESR1")
plt.show()
