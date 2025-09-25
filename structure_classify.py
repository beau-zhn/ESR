#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-row summary per PDB, ONLY for Estrogen receptor chain(s):

- Detect ER chains from COMPND: MOLECULE contains "ESTROGEN RECEPTOR".
- Compute SEQRES length per chain (prefer authoritative numRes on SEQRES serNum=1; fallback token count).
- Collect unique ligands (keep ZN).
- Choose chain if multiple ER chains: A > longest > lexicographic.
- Skip files with no ER chain.

CSV columns: pdb_id,path,chain,seqres_len,ligands,comments
"""

import sys, os, re, csv
from collections import defaultdict

# solvents/buffers not treated as ligands (ZN intentionally NOT excluded)
NON_LIGAND = {
    "HOH","DOD","H2O","WAT",
    "GOL","MPD","PEG","PGE","EDO","PG4","IPA","EOH","ACE","ACET","FMT",
    "SO4","PO4","NO3","TRS","TES","HEP","MES","MOPS","BME",
    "CL","K","NA","MG","CA","CU","MN","IOD","I","BR","CS","NI","CO","CD","SR","RB","F"
}

def iter_pdb_files(paths):
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

# ----- COMPND parsing: map chain -> molecule name -----
def parse_compnd_chain_map(path):
    """
    Returns dict(chain_id -> molecule_name) extracted from COMPND records.
    Handles multi-line 'COMPND' blocks with key:value; pairs separated by ';'.
    """
    compnd_text = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("COMPND"):
                compnd_text.append(line[10:].rstrip())  # after column 11 content
    if not compnd_text:
        return {}

    # Join and split by semicolons, but keep commas in CHAIN lists
    joined = " ".join(compnd_text)
    # Normalize spaces
    joined = re.sub(r"\s+", " ", joined)
    # Split into statements ending with ';'
    parts = [p.strip() for p in joined.split(';') if p.strip()]

    chain_to_mol = {}
    current_mol = None
    current_chains = []

    for part in parts:
        if ':' in part:
            key, val = part.split(':', 1)
            key = key.strip().upper()
            val = val.strip()
            if key == "MOL_ID":
                # flush previous
                if current_mol and current_chains:
                    for ch in current_chains:
                        chain_to_mol[ch] = current_mol
                # reset block
                current_mol = None
                current_chains = []
            elif key == "MOLECULE":
                current_mol = val.upper()
            elif key == "CHAIN":
                # CHAIN can be "A, B, C"
                current_chains = [c.strip() for c in val.split(',') if c.strip()]
        # ignore other keys

    # flush the last block
    if current_mol and current_chains:
        for ch in current_chains:
            chain_to_mol[ch] = current_mol

    return chain_to_mol

# ----- SEQRES + ligands parsing -----
def parse_seqres_lengths_and_ligands(path):
    """
    Returns:
      lengths: dict(chain -> int) (prefer numRes from serNum=1, else token count)
      ligands: set of unique HETATM 3-letter codes (filtered)
    """
    lengths_declared = {}
    lengths_fallback = defaultdict(int)
    ligands = set()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(line) < 6:
                continue
            rec = line[:6]
            if rec == "SEQRES":
                chain  = line[11:12].strip()
                sernum = line[7:10].strip()
                numres = line[13:17].strip()
                if sernum == "1":
                    try:
                        lengths_declared[chain] = int(numres)
                    except ValueError:
                        pass
                # fallback: count any 3-letter alpha token
                toks = [t.strip().upper() for t in line[19:].split() if t.strip()]
                lengths_fallback[chain] += sum(1 for t in toks if len(t) == 3 and t.isalpha())
            elif rec == "HETATM":
                resname = line[17:20].strip().upper()
                if resname and resname not in NON_LIGAND:
                    ligands.add(resname)

    chains = set(lengths_declared) | set(lengths_fallback)
    lengths = {}
    for ch in chains:
        lengths[ch] = lengths_declared.get(ch, lengths_fallback.get(ch, 0))
    return lengths, ligands

def pick_estrogen_chain(er_chains, lengths):
    """
    Among ER chains, prefer A; else the longest; else lexicographic.
    """
    if not er_chains:
        return None
    if "A" in er_chains:
        return "A"
    # longest by SEQRES, then by ID
    return sorted(er_chains, key=lambda c: (-lengths.get(c, 0), c))[0]

def summarize_one_file(path):
    pid = pdb_id_from_name(path)
    chain_to_mol = parse_compnd_chain_map(path)
    lengths, ligs = parse_seqres_lengths_and_ligands(path)

    # select only chains whose MOLECULE mentions 'ESTROGEN RECEPTOR'
    er_chains = {ch for ch, mol in chain_to_mol.items()
                 if mol and "ESTROGEN RECEPTOR" in mol}

    chosen = pick_estrogen_chain(er_chains, lengths)
    if chosen is None:
        return None  # skip file: no estrogen receptor chain found

    seq_len = lengths.get(chosen, 0)
    lig_str = ";".join(sorted(ligs))
    comments = []
    if "ZN" in ligs:
        comments.append("dna based")
    if seq_len < 50:
        comments.append("inconsistent structure")

    return [pid, path, chosen, seq_len, lig_str, "; ".join(comments)]

def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_offline_any.py <pdb_or_dir> [more_paths...]", file=sys.stderr)
        sys.exit(1)

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["pdb_id","path","chain","seqres_len","ligands","comments"])

    for path in iter_pdb_files(sys.argv[1:]):
        row = summarize_one_file(path)
        if row is not None:  # only ER-containing files
            writer.writerow(row)

if __name__ == "__main__":
    main()

