import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

# Output file
output_path = r"C:\Users\a1611\Desktop\BioInfo\out\clusters_ESR1.txt"
out = []

def w(text=""):
    """Write to list and print."""
    out.append(text)
    print(text)

# === FILE PATHS ===
drmsd_path  = r"C:\Users\a1611\Desktop\BioInfo\out\drmsd_matrix.csv"
common_path = r"C:\Users\a1611\Desktop\BioInfo\out\common_counts.csv"

# === Load matrices ===
df = pd.read_csv(drmsd_path, index_col=0)
labels = df.index.tolist()
M = df.values.astype(float)

# Load common_CA matrix if exists
try:
    common_df = pd.read_csv(common_path, index_col=0)
    C = common_df.values.astype(float)
except FileNotFoundError:
    C = None

# Diagonal = 0
np.fill_diagonal(M, 0.0)

# Handle NaN (same as heatmap)
col_means = np.nanmean(M, axis=0)
inds = np.where(np.isnan(M))
M[inds] = np.take(col_means, inds[1])
M[np.isnan(M)] = np.nanmedian(M)

# Condensed matrix
iu = np.triu_indices(M.shape[0], k=1)
condensed = M[iu]

# Ward clustering
Z = linkage(condensed, method='ward')
k = 4
clusters = fcluster(Z, t=k, criterion='maxclust')

cluster_dict = {}
for idx, c in enumerate(clusters):
    cluster_dict.setdefault(c, []).append(idx)

# Global stats
global_mean = float(np.mean(M[iu]))
global_median = float(np.median(M[iu]))

w(f"Global mean dRMSD: {global_mean:.3f} Å")
w(f"Global median dRMSD: {global_median:.3f} Å")

if C is not None:
    global_common_mean = float(np.mean(C[iu]))
    w(f"Global mean common_CA: {global_common_mean:.1f}")
else:
    global_common_mean = None
w()

# === Cluster analysis ===
for c in sorted(cluster_dict.keys()):
    idxs = np.array(cluster_dict[c])
    size = len(idxs)

    # Intra-cluster distances
    if size > 1:
        subM = M[np.ix_(idxs, idxs)]
        iu_sub = np.triu_indices(size, k=1)
        intra_mean = float(np.mean(subM[iu_sub]))
        intra_median = float(np.median(subM[iu_sub]))
    else:
        intra_mean = intra_median = np.nan

    # Inter-cluster distances
    others = np.setdiff1d(np.arange(M.shape[0]), idxs)
    if len(others) > 0:
        interM = M[np.ix_(idxs, others)]
        inter_mean = float(np.mean(interM))
        inter_median = float(np.median(interM))
    else:
        inter_mean = inter_median = np.nan

    # Common CA
    if C is not None and size > 1:
        subC = C[np.ix_(idxs, idxs)]
        intra_common = float(np.mean(subC[iu_sub]))
    else:
        intra_common = None

    # Labels
    cluster_labels = [labels[i] for i in idxs]

    # Output
    w("="*70)
    w(f"CLUSTER {c}")
    w(f"Size: {size}")
    w("Structures:")
    for lab in cluster_labels:
        w(f"  - {lab}")

    w("\nNumbers:")
    w(f"  Mean intra-cluster dRMSD: {intra_mean}")
    w(f"  Mean inter-cluster dRMSD: {inter_mean}")

    if intra_common is not None:
        w(f"  Mean intra-cluster common_CA: {intra_common}")
        w(f"  Global common_CA: {global_common_mean}")

    # Interpretation (English)
    w("\nInterpretation:")
    if size == 1:
        w("  • Single isolated structure; unique conformation.")
    else:
        if intra_mean < global_mean * 0.6:
            tight = "very compact (nearly identical backbones)"
        elif intra_mean < global_mean * 0.9:
            tight = "compact (clearly similar)"
        else:
            tight = "loose (substantial internal variation)"

        if inter_mean > global_mean * 1.1:
            sep = "strongly separated from other clusters"
        elif inter_mean > global_mean * 0.9:
            sep = "moderately separated from other clusters"
        else:
            sep = "weakly separated (partial overlap)"

        w(f"  • The cluster is {tight}.")
        w(f"  • It is {sep}.")

        if intra_common is not None:
            if intra_common > global_common_mean * 1.1:
                w("  • High residue overlap: structures share a long aligned construct.")
            elif intra_common < global_common_mean * 0.9:
                w("  • Low residue overlap: likely shorter constructs or different domain boundaries.")
            else:
                w("  • Residue overlap is typical for this dataset.")

    w()

# === Save file ===
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))

print(f"\nSaved to: {output_path}")
