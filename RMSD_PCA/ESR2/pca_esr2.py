import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ============================
# LOAD MATRIX
# ============================

drmsd_path = r"C:\Users\a1611\Desktop\BioInfo\ESR_2\out_ESR2\drmsd_matrix.csv"
df = pd.read_csv(drmsd_path, index_col=0)
labels = df.index.tolist()

M = df.values.astype(float)

# Replace NaNs like before
col_means = np.nanmean(M, axis=0)
inds = np.where(np.isnan(M))
M[inds] = np.take(col_means, inds[1])

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(M)

# ============================
# PCA
# ============================

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)

# ============================
# PLOT PCA 2D
# ============================

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], s=12, alpha=0.7)
plt.title("PCA of ESR2 conformational space (dRMSD vectors)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

# ============================
# PLOT PCA 3D
# ============================

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], s=12, alpha=0.7)
ax.set_title("PCA 3D – ESR2 conformational space")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.show()
