#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert all mmCIF files (.cif and .cif.gz) in a folder (recursively) to PDB.
Requires: gemmi  (pip install gemmi)

Usage:
  py convert_cif_to_pdb.py <src_dir> <dst_dir> [ids.txt]

- <src_dir>:  directory that contains .cif or .cif.gz (can have subfolders)
- <dst_dir>:  where to write .pdb files (will be created)
- [ids.txt]:  optional file with PDB IDs (one per line or any text containing 4-char IDs);
              if given, only those IDs will be converted.
"""

import os
import sys
import re
import gzip
import io
import gemmi

def read_ids_filter(ids_path: str):
    txt = open(ids_path, "r", encoding="utf-8", errors="ignore").read()
    return set(s.upper() for s in re.findall(r"[A-Za-z0-9]{4}", txt))

def find_cifs(src_dir: str):
    exts = (".cif", ".cif.gz")
    for root, _, files in os.walk(src_dir):
        for fn in files:
            if fn.lower().endswith(exts):
                yield os.path.join(root, fn)

def pdb_id_from_name(path: str):
    base = os.path.basename(path)
    # take first 4 word chars if standard name
    m = re.match(r"([A-Za-z0-9]{4})\.", base)
    return m.group(1).upper() if m else os.path.splitext(base)[0][:4].upper()

def load_cif_document(path: str) -> gemmi.cif.Document:
    # read CIF doc from plain or gz file
    if path.lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
            return gemmi.cif.read_string(fh.read())
    else:
        return gemmi.cif.read_file(path)

def convert_one(cif_path: str, out_dir: str) -> str:
    """Returns output .pdb path on success, raises on failure."""
    doc = load_cif_document(cif_path)
    # pick block: if multiple, prefer the one that has _atom_site
    if len(doc) == 1:
        block = doc[0]
    else:
        block = None
        for b in doc:
            if b.find_mmcif_category('_atom_site'):
                block = b
                break
        if block is None:
            block = doc[0]

    st = gemmi.make_structure_from_block(block)
    # nicer atom names/altlocs not needed; write minimal PDB is fine
    pdb_id = pdb_id_from_name(cif_path)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pdb_id}.pdb")
    st.write_minimal_pdb(out_path)
    return out_path

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]
    ids_filter = read_ids_filter(sys.argv[3]) if len(sys.argv) >= 4 else None

    if not os.path.isdir(src_dir):
        print("Source directory not found:", src_dir)
        sys.exit(2)
    os.makedirs(dst_dir, exist_ok=True)

    files = list(find_cifs(src_dir))
    if ids_filter:
        files = [p for p in files if pdb_id_from_name(p) in ids_filter]
    if not files:
        print("No .cif or .cif.gz files found.")
        sys.exit(0)

    n = len(files)
    ok = 0
    fail = 0
    for i, path in enumerate(sorted(files), 1):
        pdbid = pdb_id_from_name(path)
        try:
            outp = convert_one(path, dst_dir)
            ok += 1
            print(f"[{i}/{n}] {pdbid}  ->  {os.path.basename(outp)}")
        except Exception as e:
            fail += 1
            print(f"[{i}/{n}] {pdbid}  FAILED: {e}")

    print(f"Done: {ok} converted, {fail} failed. Output dir: {dst_dir}")

if __name__ == "__main__":
    main()
