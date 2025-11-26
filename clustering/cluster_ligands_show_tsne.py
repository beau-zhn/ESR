#!/usr/bin/env python3
# cluster_ligands_show_tsne.py
# Generates TSNE+KMeans visualizations for ligand clusters (clean legends, no text clutter).

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

from rdkit import Chem
from rdkit.Chem import DataStructs, MACCSkeys, rdFingerprintGenerator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------
DEFAULT_SMILES_CANDIDATES = ["SMILES", "SMILES_NoH", "Canonical_SMILES"]

def load_df_autosmiles(xlsx_path: str, smiles_col: str | None):
    """Load Excel and detect SMILES column, auto-repair if file is CSV-inside-Excel."""
    df = pd.read_excel(xlsx_path, sheet_name=0, header=0)

    # handle Excel file containing one column of comma-separated text
    if df.shape[1] == 1:
        col0 = df.columns[0]
        only_col = df.iloc[:, 0].astype(str)
        if "," in col0 or only_col.iloc[0].count(",") >= 2:
            print("[info] Excel sheet contains CSV-style text, reparsing...")
            lines = ([col0] if "," in col0 else []) + only_col.tolist()
            df = pd.read_csv(StringIO("\n".join(lines)))

    chosen = None
    if smiles_col and smiles_col in df.columns:
        chosen = smiles_col
    else:
        for c in DEFAULT_SMILES_CANDIDATES:
            if c in df.columns:
                chosen = c
                break
    if not chosen:
        raise ValueError(f"Could not find a SMILES column. Columns={list(df.columns)}")

    df = df.dropna(subset=[chosen]).copy()
    df["__SMILES__"] = (
        df[chosen]
        .astype(str)
        .str.replace("\u00A0", " ", regex=False)
        .str.strip()
    )
    df["mol"] = df["__SMILES__"].apply(Chem.MolFromSmiles)
    before = len(df)
    df = df[df["mol"].notnull()].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[warn] Dropped {dropped} rows with invalid SMILES.")
    return df, chosen

# ----------------------------------------------------------------------
# Fingerprint builders
def fp_morgan(mol, radius=2, nBits=1024):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    return gen.GetFingerprint(mol)

def fp_maccs(mol):
    return MACCSkeys.GenMACCSKeys(mol)

def fp_rdkit(mol, nBits=1024):
    return Chem.RDKFingerprint(mol, fpSize=nBits)

def fp_to_numpy_bitarray(fp) -> np.ndarray:
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def build_matrix(mols, fp_name: str) -> np.ndarray:
    if fp_name == "Morgan":
        fps = [fp_morgan(m) for m in mols]
    elif fp_name == "MACCS":
        fps = [fp_maccs(m) for m in mols]
    elif fp_name == "RDKit":
        fps = [fp_rdkit(m) for m in mols]
    else:
        raise ValueError(f"Unknown FP {fp_name}")
    return np.array([fp_to_numpy_bitarray(fp) for fp in fps], dtype=int)

# ---------------- TSNE on binary fingerprints ----------------
def run_tsne_binary(X: np.ndarray, random_state=42, perplexity=30, metric="jaccard"):
    """
    Run t-SNE on binary fingerprints. For Jaccard/Hamming we compute a
    precomputed distance matrix and set init='random' (required).
    Also clip perplexity to a valid range for small datasets.
    """
    n = X.shape[0]
    # perplexity must be < n and typically <= (n-1)/3
    max_ok = max(5, (n - 1) // 3)
    perp = float(min(perplexity, max_ok))

    if metric in ("jaccard", "hamming"):
        # boolean for Jaccard/Hamming
        D = pairwise_distances(X.astype(bool), metric=metric)
        tsne = TSNE(
            n_components=2,
            metric="precomputed",
            perplexity=perp,
            init="random",            # <-- required for precomputed
            random_state=random_state,
            # if your sklearn is older, use a number for learning_rate:
            learning_rate=200.0
        )
        emb = tsne.fit_transform(D)
    else:
        tsne = TSNE(
            n_components=2,
            metric="euclidean",
            perplexity=perp,
            init="pca",
            random_state=random_state,
            learning_rate=200.0
        )
        emb = tsne.fit_transform(X.astype(float))
    return emb
    """
    For binary fingerprint matrices, TSNE with Jaccard distance usually works well.
    We compute a precomputed distance matrix then run TSNE(metric='precomputed').
    """
    # ensure boolean for Jaccard/Hamming
    if metric in ("jaccard", "hamming"):
        D = pairwise_distances(X.astype(bool), metric=metric)
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, max(5, (len(X) - 1) // 3)),  # keep valid range
            learning_rate="auto",
            metric="precomputed",
            init="pca",
            random_state=random_state,
            # n_iter=1000,
            verbose=0,
        )
        emb = tsne.fit_transform(D)
    else:
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, max(5, (len(X) - 1) // 3)),
            learning_rate="auto",
            metric="euclidean",
            init="pca",
            random_state=random_state,
            # n_iter=1000,
            verbose=0,
        )
        emb = tsne.fit_transform(X.astype(float))
    return emb

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Show all clusters on TSNE grids.")
    parser.add_argument("--xlsx", required=True, help="Path to .xlsx file with ligands")
    parser.add_argument("--smiles_col", default=None, help="SMILES column name (auto-detected if omitted)")
    parser.add_argument("--k", nargs="+", type=int, default=[3, 5, 7, 10], help="List of K values")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0, help="TSNE perplexity (5–50 typical)")
    parser.add_argument("--outdir", default="outputs_tsne")
    args = parser.parse_args()

    df, chosen = load_df_autosmiles(args.xlsx, args.smiles_col)
    print(f"[info] Using SMILES column: {chosen}")
    print(f"[info] Valid molecules: {len(df)}")

    out_base = Path(args.outdir)
    ensure_dir(out_base)

    fingerprints = ["Morgan", "MACCS", "RDKit"]
    K_values = list(dict.fromkeys(args.k))

    fp_mats, fp_embs = {}, {}
    for fp in fingerprints:
        print(f"[info] Building {fp} fingerprints...")
        X = build_matrix(df["mol"], fp)
        fp_mats[fp] = X
        print(f"[info] {fp} matrix: {X.shape}")
        print(f"[info] Running TSNE ({fp})...")
        fp_embs[fp] = run_tsne_binary(X, random_state=args.random_state, perplexity=args.perplexity, metric="jaccard")

    n_rows, n_cols = len(fingerprints), len(K_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.3*n_rows), squeeze=False)
    summary_rows = []

    for r, fp in enumerate(fingerprints):
        X, emb = fp_mats[fp], fp_embs[fp]
        fp_dir = out_base / fp
        ensure_dir(fp_dir)
        pd.DataFrame(emb, columns=["TSNE1", "TSNE2"]).to_csv(fp_dir / f"{fp}_tsne_embedding.csv", index=False)

        for c, k in enumerate(K_values):
            print(f"[info] KMeans {fp} (K={k})...")
            kmeans = KMeans(n_clusters=k, random_state=args.random_state, n_init="auto")
            labels = kmeans.fit_predict(X)

            try:
                sil = silhouette_score(X, labels, metric="euclidean")
            except Exception:
                sil = np.nan

            ax = axes[r, c]
            scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels, s=20, cmap='tab10', alpha=0.9, linewidths=0)

            uniq = np.unique(labels)
            cluster_counts = {cl: int((labels == cl).sum()) for cl in uniq}

            # choose descriptor for legend
            if "Chemical_Family" in df.columns:
                meta_col = "Chemical_Family"
            elif "QueryGene" in df.columns:
                meta_col = "QueryGene"
            else:
                meta_col = None

            cluster_name = {}
            if meta_col:
                for cl in uniq:
                    vals = (
                        df.loc[labels == cl, meta_col]
                        .astype(str)
                        .replace({"nan": ""})
                        .value_counts()
                        .index[:1]
                        .tolist()
                    )
                    cluster_name[cl] = vals[0] if vals else ""
            else:
                cluster_name = {cl: "" for cl in uniq}

            # legend: id — name (n=..)
            handles = []
            for cl in uniq:
                color = scatter.cmap(scatter.norm(cl))
                label_txt = f"{cl}"
                if cluster_name.get(cl):
                    label_txt += f" — {cluster_name[cl]}"
                label_txt += f" (n={cluster_counts[cl]})"
                handles.append(Line2D([0], [0], marker='o', linestyle='', color=color, label=label_txt))
            leg = ax.legend(handles=handles, title="Clusters",
                            bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
            ax.add_artist(leg)

            ax.set_title(f"{fp} — TSNE — K={k} | silhouette={sil:.3f}")
            ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2")
            ax.set_xticks([]); ax.set_yticks([])

            # save per-panel PNG
            panel_png = fp_dir / f"{fp}_TSNE_K{k}.png"
            fig_tmp, ax_tmp = plt.subplots(figsize=(6, 5))
            ax_tmp.scatter(emb[:, 0], emb[:, 1], c=labels, s=20, cmap='tab10')
            ax_tmp.set_title(f"{fp} — TSNE — K={k} | silhouette={sil:.3f}")
            ax_tmp.set_xlabel("t-SNE-1"); ax_tmp.set_ylabel("t-SNE-2")
            ax_tmp.set_xticks([]); ax_tmp.set_yticks([])
            fig_tmp.tight_layout(); fig_tmp.savefig(panel_png, dpi=220); plt.close(fig_tmp)

            labeled = df.drop(columns=["mol"]).copy()
            labeled[f"{fp}_K{k}"] = labels
            labels_csv = fp_dir / f"{fp}_labels_K{k}.csv"
            labeled.to_csv(labels_csv, index=False)

            summary_rows.append({
                "Fingerprint": fp, "K": k, "N": len(df),
                "Silhouette": float(sil) if sil == sil else None,
                "Labels_CSV": str(labels_csv), "TSNE_Panel_PNG": str(panel_png)
            })

    fig.tight_layout()
    grid_png = out_base / "ALL_FPs_ALL_K_TSNE_grid.png"
    fig.savefig(grid_png, dpi=240)
    print(f"[done] Saved grid: {grid_png.resolve()}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_base / "clustering_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[done] Summary: {summary_csv.resolve()}")

    plt.show()

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
