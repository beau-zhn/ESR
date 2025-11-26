import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

# === Paths to matrices ===
drmsd_path  = r"/ESR_2/out_ESR2/drmsd_matrix.csv"
common_path = r"/ESR_2/out_ESR2/common_counts.csv"

# === Output file ===
out_file = "../cluster_report.txt"
fout = open(out_file, "w", encoding="utf-8")

def write(s=""):
    print(s)
    fout.write(s + "\n")

# === Read distance matrix ===
df = pd.read_csv(drmsd_path, index_col=0)
labels = df.index.tolist()
M = df.values.astype(float)

# === Read common_CA matrix (optional) ===
try:
    common_df = pd.read_csv(common_path, index_col=0)
    C = common_df.values.astype(float)
except FileNotFoundError:
    C = None

# Set diagonal = 0
np.fill_diagonal(M, 0.0)

# === NaN handling ===
col_means = np.nanmean(M, axis=0)
inds = np.where(np.isnan(M))
M[inds] = np.take(col_means, inds[1])
M[np.isnan(M)] = np.nanmedian(M)

# === Condensed upper triangle for linkage ===
iu = np.triu_indices(M.shape[0], k=1)
condensed = M[iu]

# === Ward clustering ===
Z = linkage(condensed, method="ward")

k = 4  # number of clusters
clusters = fcluster(Z, t=k, criterion="maxclust")

# === Group indices by cluster ===
cluster_dict = {}
for idx, c in enumerate(clusters):
    cluster_dict.setdefault(c, []).append(idx)

# === Global statistics ===
all_dists = M[iu]
global_mean = float(np.mean(all_dists))
global_median = float(np.median(all_dists))

write(f"Global mean dRMSD (all pairs): {global_mean:.3f} Å")
write(f"Global median dRMSD: {global_median:.3f} Å")

if C is not None:
    all_common = C[iu]
    global_common_mean = float(np.mean(all_common))
    write(f"Global mean common_CA: {global_common_mean:.1f}")
else:
    global_common_mean = None

# === Per-cluster analysis ===
for c in sorted(cluster_dict.keys()):
    idxs = np.array(cluster_dict[c])
    size = len(idxs)

    write("\n" + "=" * 60)
    write(f"CLUSTER {c}")
    write(f"Size: {size} structures")
    write("Structures:")
    for lab in [labels[i] for i in idxs]:
        write(f"  - {lab}")

    # Intra-cluster dRMSD
    if size > 1:
        subM = M[np.ix_(idxs, idxs)]
        iu_sub = np.triu_indices(size, k=1)
        intra_dists = subM[iu_sub]
        intra_mean = float(np.mean(intra_dists))
        intra_median = float(np.median(intra_dists))
    else:
        intra_mean = np.nan
        intra_median = np.nan

    # Inter-cluster dRMSD
    other_idxs = np.setdiff1d(np.arange(M.shape[0]), idxs)
    if len(other_idxs) > 0:
        interM = M[np.ix_(idxs, other_idxs)]
        inter_mean = float(np.mean(interM))
        inter_median = float(np.median(interM))
    else:
        inter_mean = np.nan
        inter_median = np.nan

    # Common_CA (optional)
    if C is not None:
        subC = C[np.ix_(idxs, idxs)]
        iu_subC = np.triu_indices(size, k=1)
        if size > 1:
            intra_common = float(np.mean(subC[iu_subC]))
        else:
            intra_common = np.nan
    else:
        intra_common = None

    write("\nMetrics:")
    if not np.isnan(intra_mean):
        write(f"  Mean intra-cluster dRMSD: {intra_mean:.3f} Å")
        write(f"  Median intra-cluster dRMSD: {intra_median:.3f} Å")
    else:
        write("  Intra-cluster dRMSD: only one structure (no pairs).")

    if not np.isnan(inter_mean):
        write(f"  Mean inter-cluster dRMSD: {inter_mean:.3f} Å")
        write(f"  Median inter-cluster dRMSD: {inter_median:.3f} Å")

    if intra_common is not None:
        write(f"  Mean intra-cluster common_CA: {intra_common:.1f}")
        write(f"  Global mean common_CA: {global_common_mean:.1f}")

    # === Interpretation ===
    write("\nInterpretation:")
    if size == 1:
        write("  • This cluster contains a single structure — likely an outlier conformation.")
    else:
        # compactness
        if intra_mean < global_mean * 0.6:
            tight = "very compact (structures nearly identical)"
        elif intra_mean < global_mean * 0.9:
            tight = "moderately compact (structures similar)"
        else:
            tight = "loose (structures vary within the cluster)"

        # separation
        if inter_mean > global_mean * 1.1:
            sep = "strongly separated from other clusters"
        elif inter_mean > global_mean * 0.9:
            sep = "moderately separated from others"
        else:
            sep = "weakly separated (overlapping conformations)"

        write(f"  • Cluster compactness: {tight}.")
        write(f"  • Cluster separation: {sep}.")

        if intra_common is not None:
            if intra_common > global_common_mean * 1.1:
                write("  • Higher-than-average shared Cα residues → well-aligned, similar domain length.")
            elif intra_common < global_common_mean * 0.9:
                write("  • Lower-than-average shared Cα → domain truncation or alignment differences.")
            else:
                write("  • Shared Cα count is near global average.")

fout.close()
print(f"\n\nReport saved to: {out_file}")
