#!/usr/bin/env python3
# cluster_ligands_show_family.py
# UMAP + KMeans for ligands, coloured by NR Chemical_Family
# with 3 fingerprints (Morgan, MACCS, RDKit).

# python cluster_ligands_show_family_copy.py \
#   --xlsx "ligands_with_final_chemical_classificationESR1.xlsx" \
#   --smiles_col SMILES \
#   --k 3 5 7 10 \
#   --outdir outputs_family

import argparse
from pathlib import Path
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import DataStructs, MACCSkeys, rdFingerprintGenerator

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from umap import UMAP
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------
DEFAULT_SMILES_CANDIDATES = ["SMILES", "SMILES_NoH", "Canonical_SMILES"]

def load_df_autosmiles(xlsx_path: str, smiles_col: Optional[str]):
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
    # keep as in your "correct" code
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

def run_umap_binary(X: np.ndarray, random_state=42):
    reducer = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="jaccard",
        random_state=random_state
    )
    return reducer.fit_transform(X)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Show all clusters on UMAP grids (colored by NR family).")
    parser.add_argument("--xlsx", required=True, help="Path to .xlsx file with ligands")
    parser.add_argument("--smiles_col", default=None, help="SMILES column name (auto-detected if omitted)")
    parser.add_argument("--k", nargs="+", type=int, default=[3, 5, 7, 10], help="List of K values")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--outdir", default="outputs_family")
    args = parser.parse_args()

    df, chosen = load_df_autosmiles(args.xlsx, args.smiles_col)
    print(f"[info] Using SMILES column: {chosen}")
    print(f"[info] Valid molecules: {len(df)}")

    if "Chemical_Family" not in df.columns:
        raise ValueError("Column 'Chemical_Family' not found in the Excel file.")

    out_base = Path(args.outdir)
    ensure_dir(out_base)

    # 3 fingerprints
    fingerprints = ["Morgan", "MACCS", "RDKit"]
    K_values = list(dict.fromkeys(args.k))  # unique, keep order

    fp_mats, fp_embs = {}, {}
    for fp in fingerprints:
        print(f"[info] Building {fp} fingerprints...")
        X = build_matrix(df["mol"], fp)
        fp_mats[fp] = X
        print(f"[info] {fp} matrix: {X.shape}")
        print(f"[info] Running UMAP ({fp})...")
        fp_embs[fp] = run_umap_binary(X, random_state=args.random_state)

    n_rows, n_cols = len(fingerprints), len(K_values)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4.3 * n_rows),
        squeeze=False
    )
    summary_rows = []

    # fixed colours for 3 NR families, grey for anything else
    family_color_map = {
        "Steroids":    "#1f77b4",  # blue
        "Fatty acids": "#ed1717",  # red
        "Orphan":      "#2ca02c",  # green
    }

    for r, fp in enumerate(fingerprints):
        X, emb = fp_mats[fp], fp_embs[fp]
        fp_dir = out_base / fp
        ensure_dir(fp_dir)

        # save embedding
        pd.DataFrame(emb, columns=["UMAP1", "UMAP2"]).to_csv(
            fp_dir / f"{fp}_umap_embedding.csv",
            index=False
        )

        for c, k in enumerate(K_values):
            print(f"[info] KMeans {fp} (K={k})...")
            kmeans = KMeans(
                n_clusters=k,
                random_state=args.random_state,
                n_init="auto"
            )
            labels = kmeans.fit_predict(X)

            try:
                sil = silhouette_score(X, labels, metric="euclidean")
            except Exception:
                sil = np.nan

            ax = axes[r, c]

            # families + colours
            families = df["Chemical_Family"].astype(str).fillna("Unknown")
            point_colors = families.map(
                lambda f: family_color_map.get(f, "#7f7f7f")  # grey for others
            )

            # main scatter coloured by NR family
            scatter = ax.scatter(
                emb[:, 0],
                emb[:, 1],
                c=list(point_colors),
                s=25,
                alpha=0.9,
                linewidths=0,
            )

            # optional: mark cluster centroids with numbers (same as in your first code idea)
            uniq = np.unique(labels)
            # for cl in uniq:
            #     pts = emb[labels == cl]
            #     cx, cy = pts.mean(axis=0)
            #     ax.text(
            #         cx, cy, str(cl),
            #         fontsize=11,
            #         fontweight="bold",
            #         color="black",
            #         ha="center",
            #         va="center",
            #         bbox=dict(
            #             boxstyle="round,pad=0.2",
            #             fc="white",
            #             ec="black",
            #             alpha=0.6
            #         ),
            #     )

            # legend: one entry per family present
            uniq_fams = sorted(families.unique())
            handles = [
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    color=family_color_map.get(f, "#7f7f7f"),
                    label=f
                )
                for f in uniq_fams
            ]
            legend = ax.legend(
                handles=handles,
                title="NR Family",
                loc="upper right",
                fontsize=9,
                frameon=True
            )

            ax.set_title(f"{fp} — K={k} | silhouette={sil:.3f}")
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            ax.set_xticks([])
            ax.set_yticks([])

            # save each panel separately (also coloured by family)
            panel_png = fp_dir / f"{fp}_UMAP_K{k}.png"
            fig_tmp, ax_tmp = plt.subplots(figsize=(6, 5))

            ax_tmp.scatter(
                emb[:, 0],
                emb[:, 1],
                c=list(point_colors),
                s=25,
                alpha=0.9,
                linewidths=0,
            )

            # centroids + numbers in standalone panel
            for cl in uniq:
                pts = emb[labels == cl]
                cx, cy = pts.mean(axis=0)
                ax_tmp.text(
                    cx, cy, str(cl),
                    fontsize=11,
                    fontweight="bold",
                    color="black",
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="white",
                        ec="black",
                        alpha=0.6
                    ),
                )

            handles_tmp = [
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    color=family_color_map.get(f, "#7f7f7f"),
                    label=f
                )
                for f in uniq_fams
            ]
            ax_tmp.legend(
                handles=handles_tmp,
                title="NR Family",
                loc="upper right",
                fontsize=8,
                frameon=True
            )

            ax_tmp.set_title(f"{fp} — K={k} | silhouette={sil:.3f}")
            ax_tmp.set_xlabel("UMAP-1")
            ax_tmp.set_ylabel("UMAP-2")
            ax_tmp.set_xticks([])
            ax_tmp.set_yticks([])

            fig_tmp.tight_layout()
            fig_tmp.savefig(panel_png, dpi=220)
            plt.close(fig_tmp)

            # save labels CSV with cluster IDs
            labeled = df.drop(columns=["mol"]).copy()
            labeled[f"{fp}_K{k}"] = labels
            labels_csv = fp_dir / f"{fp}_labels_K{k}.csv"
            labeled.to_csv(labels_csv, index=False)

            summary_rows.append({
                "Fingerprint": fp,
                "K": k,
                "N": len(df),
                "Silhouette": float(sil) if sil == sil else None,
                "Labels_CSV": str(labels_csv),
                "UMAP_Panel_PNG": str(panel_png),
            })

    fig.tight_layout()
    grid_png = out_base / "ALL_FPs_ALL_K_UMAP_grid.png"
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
