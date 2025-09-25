#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, re, gzip
from collections import defaultdict

VERBOSE = True
FIELDS = ["pdb_id", "status", "holo_apo", "ligands", "seq_len_total_observed", "chain_lengths", "comments"]

# solvents/buffers to ignore (ZN intentionally NOT ignored)
NON_LIGAND = {
    "HOH","DOD","H2O","WAT",
    "GOL","MPD","PEG","PGE","EDO","PG4","IPA","EOH","ACE","ACET","FMT",
    "SO4","PO4","NO3","TRS","TES","HEP","MES","MOPS","BME",
    "CL","K","NA","MG","CA","CU","MN","IOD","I","BR","CS","NI","CO","CD","SR","RB","F"
}
# Canonical AAs + common proteinogenic variants you likely want to count as amino acids
AMINO = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "MSE","SEC","PYL","ASX","GLX","UNK",  # include common placeholders/variants if you want them counted
    "HSD","HSE","HSP","HIP","HID","HIE"  # histidine naming variants
}
NUCLEOTIDES = {
    "A","C","G","T","U","DA","DC","DG","DT","DU",
    "AMP","CMP","GMP","TMP","UMP","ADE","CYT","GUA","THY","URI",
    "M2G","OMG","PSU","5MC","1MA","H2U","2MG","7MG","OMC","5MU","5BU","I"
}

def log(s):
    if VERBOSE:
        sys.stdout.write(s + "\n"); sys.stdout.flush()

def list_pdb_files(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith((".pdb", ".pdb.gz")):
                out.append(os.path.join(dp, fn))
    return sorted(out)

def pdb_id_from_name(path):
    base = os.path.basename(path)
    m = re.match(r"([A-Za-z0-9]{4})\.pdb(\.gz)?$", base, re.IGNORECASE)
    return m.group(1).upper() if m else os.path.splitext(base)[0][:4].upper()

def open_text_maybe_gz(path):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def parse_pdb(path):
    """Return a row dict with ligands and per-chain sequence lengths from PDB ATOM records."""
    pdb_id = pdb_id_from_name(path)
    try:
        fh = open_text_maybe_gz(path)
    except Exception as e:
        return {
            "pdb_id": pdb_id, "status": "read_failed",
            "holo_apo": "", "ligands": "", "seq_len_total_observed": 0,
            "chain_lengths": "", "comments": f"failed to open: {e}"
        }

    ligs = set()
    # Per-chain residue sets (observed)
    atom_residues = defaultdict(set)     # chain -> {(resseq, icode)} unique residues observed
    atom_residues_named = defaultdict(set)  # if you want to track resname too
    # For span estimate
    min_resseq = defaultdict(lambda: None)
    max_resseq = defaultdict(lambda: None)

    def to_int(s):
        try: return int(s)
        except: return None

    with fh:
        for line in fh:
            if len(line) < 27:  # too short
                continue
            rec = line[0:6].strip().upper()
            if rec not in ("ATOM", "HETATM"):
                continue

            resname = line[17:20].strip().upper()
            chain   = line[21].strip()
            resseq  = line[22:26].strip()
            icode   = line[26].strip()

            if rec == "ATOM" and resname in AMINO:
                # Count unique residues by (resSeq, iCode)
                atom_residues[chain].add((resseq, icode))
                atom_residues_named[chain].add((resseq, icode, resname))
                # Track span
                n = to_int(resseq)
                if n is not None:
                    if min_resseq[chain] is None or n < min_resseq[chain]:
                        min_resseq[chain] = n
                    if max_resseq[chain] is None or n > max_resseq[chain]:
                        max_resseq[chain] = n

            elif rec == "HETATM":
                # Candidate ligand: exclude buffers/solvents/amino/nucleotide
                if (resname not in NON_LIGAND) and (resname not in AMINO) and (resname not in NUCLEOTIDES):
                    ligs.add(resname)

    # Per-chain observed length and span estimate
    chain_lengths = []
    total_observed = 0
    for ch in sorted(set(list(atom_residues.keys()))):
        observed = len(atom_residues[ch])
        total_observed += observed
        mn = min_resseq[ch]
        mx = max_resseq[ch]
        span = (mx - mn + 1) if (mn is not None and mx is not None) else observed
        chain_lengths.append(f"{ch}:{observed}obs/{span}span")

    comments = []
    if "ZN" in ligs:
        comments.append("DNA based")  # keep your original note; adjust if needed
    # mark small proteins if you like
    if total_observed < 50:
        comments.append("inconsistent size")

    row = {
        "pdb_id": pdb_id,
        "status": "ok",
        "holo_apo": ("holo" if ligs else "apo"),
        "ligands": ";".join(sorted(ligs)),
        "seq_len_total_observed": total_observed,
        "chain_lengths": ";".join(chain_lengths),  # e.g., "A:218obs/224span;B:221obs/226span;C:11obs/11span;D:11obs/11span"
        "comments": "; ".join(comments)
    }
    log(f"{pdb_id}: ligs=[{row['ligands']}] chains=({row['chain_lengths']})")
    return row

def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_offline_any.py <root_dir> [ids.txt]")
        sys.exit(1)
    root = sys.argv[1]
    if not os.path.isdir(root):
        print("Not a directory:", root); sys.exit(1)

    files = list_pdb_files(root)

    # optional filter by ids (2nd arg)
    if len(sys.argv) >= 3:
        ids_txt = open(sys.argv[2], "r", encoding="utf-8").read()
        wanted = set(s.upper() for s in re.findall(r"[A-Za-z0-9]{4}", ids_txt))
        files = [p for p in files if pdb_id_from_name(p) in wanted]

    if not files:
        print("No .pdb/.pdb.gz files found in:", root); sys.exit(1)

    rows = [parse_pdb(p) for p in files]

    with open("pdb_classified.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    log(f"Done. Wrote {len(rows)} rows to pdb_classified.csv")

if __name__ == "__main__":
    main()
