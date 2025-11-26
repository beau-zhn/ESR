#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan .pdb/.ent files recursively, read SEQRES length per chain,
collect unique ligands, and annotate:
  - seqres_len < 50          -> "inconsistent structure"
  - presence of ligand 'ZN'  -> "dna based"

Output CSV columns:
  pdb_id, path, chain, seqres_len, ligands, comments
"""

import sys, os, re, csv
from collections import defaultdict

# Standard amino acids (+ common PDB variants we treat as amino)
AMINO_STD = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "MSE",  # Selenomethionine (count like MET)
    "SEC",  # Selenocysteine
    "PYL"   # Pyrrolysine
}

# Solvents/buffers/salts NOT to report as ligands (ZN is intentionally NOT here)
NON_LIGAND = {
    "HOH","DOD","H2O","WAT",
    "GOL","MPD","PEG","PGE","EDO","PG4","IPA","EOH","ACE","ACET","FMT",
    "SO4","PO4","NO3","TRS","TES","HEP","MES","MOPS","BME",
    "CL","K","NA","MG","CA","CU","MN","IOD","I","BR","CS","NI","CO","CD","SR","RB","F"
}

def iter_pdb_files(paths):
    """Yield .pdb/.ent files recursively from given files/dirs."""
    exts = (".pdb", ".ent")
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    if fn.lower().endswith(exts):
                        yield os.path.join(root, fn)
        else:
            if p.lower().endswith(exts):
                yield p

def pdb_id_from_name(path):
    base = os.path.basename(path)
    m = re.match(r"([A-Za-z0-9]{4})\.(pdb|ent)$", base, re.IGNORECASE)
    return (m.group(1).upper() if m else os.path.splitext(base)[0][:4].upper())

def parse_pdb(path):
    """
    Returns:
      seqres_len_by_chain: dict(chain -> int) biological length from SEQRES
      ligands:             set of unique ligand 3-letter codes (HETATM, filtered)
    """
    seqres_len_by_chain = defaultdict(int)
    ligands = set()

    # PDB fixed-columns:
    # - record name: 1-6
    # - SEQRES: chainID at col 12 (index 11), residue list from col 20 (index 19)
    # - HETATM: resname 18-20
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(line) < 6:
                continue
            rec = line[:6]
            if rec == "SEQRES":
                chain = line[11:12].strip()
                # residues are whitespace-separated 3-letter tokens from col 20 onwards
                tokens = [t.strip().upper() for t in line[19:].split() if t.strip()]
                # count only amino acids (treat MSE/SEC/PYL as amino)
                seqres_len_by_chain[chain] += sum(1 for t in tokens if t in AMINO_STD)

            elif rec == "HETATM":
                resname = line[17:20].strip().upper()
                if resname and resname not in NON_LIGAND:
                    ligands.add(resname)

    return dict(seqres_len_by_chain), ligands

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_all_pdb_seq.py <pdb_or_dir> [more_paths...]")
        sys.exit(1)

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["pdb_id","path","chain","seqres_len","ligands","comments"])

    for path in iter_pdb_files(sys.argv[1:]):
        pid = pdb_id_from_name(path)
        seqres, ligs = parse_pdb(path)

        # если нет SEQRES вовсе — всё равно выведем строку без цепи
        if not seqres:
            comments = []
            if "ZN" in ligs:
                comments.append("dna based")
            comments.append("inconsistent structure")  # длина неизвестна/нулевая
            writer.writerow([
                pid, path, "", 0, ";".join(sorted(ligs)), "; ".join(comments)
            ])
            continue

        # по всем цепям
        lig_str = ";".join(sorted(ligs))
        has_zn = ("ZN" in ligs)

        for chain in sorted(seqres.keys()):
            s_len = seqres.get(chain, 0)
            comments = []
            if has_zn:
                comments.append("dna based")
            if s_len < 50:
                comments.append("inconsistent structure")

            writer.writerow([
                pid, path, chain, s_len, lig_str, "; ".join(comments)
            ])

if __name__ == "__main__":
    main()
