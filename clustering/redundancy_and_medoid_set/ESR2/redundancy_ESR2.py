import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
import re
import matplotlib.pyplot as plt

# === PATHS ===
drmsd_path   = r"C:\Users\a1611\Desktop\BioInfo\ESR_2\out_ESR2\drmsd_matrix.csv"
clusters_txt = r"C:\Users\a1611\Desktop\BioInfo\pocket_PCA\ESR2\clusters_ESR2.txt"

curve_out_csv   = r"C:\Users\a1611\Desktop\BioInfo\redundancy\ESR2\redundancy_ESR2_curve.csv"
medoids_out_csv = r"C:\Users\a1611\Desktop\BioInfo\redundancy\ESR2\ESR2_medoids.csv"
curve_png_div   = r"C:\Users\a1611\Desktop\BioInfo\redundancy\ESR2\ESR2_diversity_curve.png"
curve_png_cov   = r"C:\Users\a1611\Desktop\BioInfo\redundancy\ESR2\ESR2_coverage_error_curve.png"

# === LOAD dRMSD MATRIX ===
df = pd.read_csv(drmsd_path, index_col=0)
labels = df.index.tolist()
M = df.values.astype(float)

# Diagonal = 0, NaN fix
np.fill_diagonal(M, 0.0)
col_means = np.nanmean(M, axis=0)
inds = np.where(np.isnan(M))
M[inds] = np.take(col_means, inds[1])
M[np.isnan(M)] = np.nanmedian(M)

N = M.shape[0]
assert N == len(labels)

# upper triangle for global stats
iu = np.triu_indices(N, k=1)
global_mean = float(np.mean(M[iu]))
print(f"Global mean dRMSD: {global_mean:.3f} Å")

# === 2. LOAD CLUSTERS & ENTROPY ===
cluster_map = {}   # pdb -> cluster id
cluster_sizes = {} # cluster id -> count

with open(clusters_txt, "r", encoding="utf-8") as f:
    current_cluster = None
    for line in f:
        line = line.strip()

        m = re.match(r"CLUSTER\s+(\d+)", line)
        if m:
            current_cluster = int(m.group(1))
            continue

        if line.startswith("- "):
            pdb_name = line.replace("- ", "").strip()
            cluster_map[pdb_name] = current_cluster

# vector of cluster ids in same order as labels[]
clusters = []
for pdb in labels:
    c = cluster_map.get(pdb, None)
    clusters.append(c)
    if c is not None:
        cluster_sizes[c] = cluster_sizes.get(c, 0) + 1

clusters = np.array(clusters, dtype=float)

# Shannon entropy over cluster distribution
valid_clusters = clusters[~np.isnan(clusters)]
unique, counts = np.unique(valid_clusters, return_counts=True)
p = counts / counts.sum()
H = -np.sum(p * np.log2(p))
effective_states = 2 ** H

print("\n=== Cluster entropy (ESR2) ===")
print("Cluster counts:", dict(zip(unique.astype(int), counts)))
print(f"Shannon entropy H = {H:.3f} bits")
print(f"Effective number of conformational states ≈ {effective_states:.2f}")

# === 3. GREEDY DIVERSITY / COVERAGE CURVE (CUMULATIVE) ===

# 1) старт: глобальный медоид
mean_dist = np.mean(M, axis=1)  # среднее расстояние каждой структуры до остальных
start_idx = int(np.argmin(mean_dist))
selected = [start_idx]

# precompute for speed
all_indices = np.arange(N)

subset_ranks = []
subset_pdbs = []
coverage_errors = []
normalized_diversity = []

# helper: recompute coverage error
def compute_coverage_error(selected_indices):
    sel = np.array(selected_indices, dtype=int)
    # расстояние от каждой структуры до ближайшей в selected
    d_min = np.min(M[:, sel], axis=1)
    return float(np.mean(d_min))

# step 1
ce = compute_coverage_error(selected)
norm_div = 1.0 - ce / global_mean
subset_ranks.append(1)
subset_pdbs.append(labels[start_idx])
coverage_errors.append(ce)
normalized_diversity.append(norm_div)

# 2) итеративно добавляем
for k in range(2, N + 1):
    remaining = np.setdiff1d(all_indices, np.array(selected, dtype=int))

    best_j = None
    best_min_dist = -1.0

    # выбираем структуру, которая максимально далеко от текущего набора
    for j in remaining:
        d_to_sel = M[j, selected]
        min_d = float(np.min(d_to_sel))
        if min_d > best_min_dist:
            best_min_dist = min_d
            best_j = j

    selected.append(best_j)

    ce = compute_coverage_error(selected)
    norm_div = 1.0 - ce / global_mean

    subset_ranks.append(k)
    subset_pdbs.append(labels[best_j])
    coverage_errors.append(ce)
    normalized_diversity.append(norm_div)

    if k in [1, 2, 5, 10, 20, 50, 100, N]:
        print(f"k = {k:3d}: coverage_error = {ce:.3f}, normalized_diversity = {norm_div:.3f}")

# save curve CSV
curve_df = pd.DataFrame({
    "rank": subset_ranks,
    "pdb": subset_pdbs,
    "coverage_error": coverage_errors,
    "normalized_diversity": normalized_diversity
})
curve_df.to_csv(curve_out_csv, index=False)
print(f"\nSaved redundancy curve → {curve_out_csv}")

# === PLOTS FOR DIVERSITY CURVE ===

# 1) normalized_diversity vs rank
plt.figure(figsize=(8,6))
plt.plot(curve_df["rank"], curve_df["normalized_diversity"])
plt.xlabel("Number of selected structures (rank)")
plt.ylabel("Normalized diversity (1 - coverage_error / global_mean)")
plt.title("ESR2 cumulative conformational diversity")
plt.grid(True)
plt.tight_layout()
plt.savefig(curve_png_div, dpi=300)
print(f"Saved diversity curve plot → {curve_png_div}")
plt.show()

# 2) coverage_error vs rank
plt.figure(figsize=(8,6))
plt.plot(curve_df["rank"], curve_df["coverage_error"])
plt.xlabel("Number of selected structures (rank)")
plt.ylabel("Coverage error (mean min dRMSD to selected set)")
plt.title("ESR2 coverage error vs number of representatives")
plt.grid(True)
plt.tight_layout()
plt.savefig(curve_png_cov, dpi=300)
print(f"Saved coverage error plot → {curve_png_cov}")
plt.show()

# === 4. MINIMAL NON-REDUNDANT SET: CLUSTER MEDOIDS ===

cluster_medoids = []

for c in sorted(cluster_sizes.keys()):
    idxs = [i for i, pdb in enumerate(labels) if cluster_map.get(pdb, None) == c]
    idxs = np.array(idxs, dtype=int)
    size = len(idxs)
    if size == 0:
        continue

    subM = M[np.ix_(idxs, idxs)]
    # среднее расстояние внутри кластера (по строкам)
    mean_intra = np.mean(subM, axis=1)
    medoid_local_idx = int(np.argmin(mean_intra))  # индекс внутри subM
    medoid_global_idx = idxs[medoid_local_idx]
    medoid_pdb = labels[medoid_global_idx]

    # второй представитель (самый далеко от медоида внутри кластера)
    dist_to_medoid = subM[medoid_local_idx, :]
    far_local_idx = int(np.argmax(dist_to_medoid))
    far_global_idx = idxs[far_local_idx]
    far_pdb = labels[far_global_idx]

    cluster_medoids.append({
        "cluster": c,
        "size": size,
        "medoid_pdb": medoid_pdb,
        "backup_pdb": far_pdb,
        "mean_intra_cluster_dRMSD": float(np.mean(subM[np.triu_indices(size, k=1)]))
    })

    print(f"\nCLUSTER {c}: size={size}")
    print(f"  Medoid: {medoid_pdb}")
    print(f"  Backup: {far_pdb}")

medoids_df = pd.DataFrame(cluster_medoids)
medoids_df.to_csv(medoids_out_csv, index=False)
print(f"\nSaved medoids → {medoids_out_csv}")

print("\nDONE: ESR2 redundancy, entropy, medoids and plots computed.")
