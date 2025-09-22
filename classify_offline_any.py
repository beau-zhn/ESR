#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline classifier (recursive) for PDB/mmCIF files, including .gz:
- Supports: .pdb, .ent, .cif, .pdb.gz, .ent.gz, .cif.gz
- APO/HOLO = presence of real ligands (exclude water/ions/buffers)
- Secondary structure:
    * PDB:     HELIX/SHEET records
    * mmCIF:   _struct_conf / _struct_sheet_range  (accepts label_* or auth_* fields)
Outputs:
  pdb_classified.csv + apo.csv + holo.csv + alpha.csv + beta.csv + alpha_beta.csv

Run:
  py classify_offline_any.py <root_dir> [ids.txt]
"""

import os, sys, csv, time, re, gzip, io

# ---------- optional mmCIF support ----------
USE_GEMMI = False
try:
    import gemmi
    USE_GEMMI = True
except Exception:
    pass

VERBOSE = True
def log(s: str): sys.stdout.write(s + "\n"); sys.stdout.flush()

# ---------- dictionaries ----------
NON_LIGAND = {
    "HOH","DOD","H2O","WAT",
    "GOL","MPD","PEG","PGE","EDO","PG4","IPA","EOH","ACE","ACET","FMT",
    "SO4","PO4","NO3","TRS","TES","HEP","MES","MOPS","BME",
    "CL","K","NA","MG","CA","ZN","CU","MN","IOD","I","BR","CS","NI","CO","CD","SR","RB","F"
}
AMINO = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
         "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
NUCLEOTIDES = {
    "A","C","G","T","U","DA","DC","DG","DT","DU",
    "AMP","CMP","GMP","TMP","UMP","ADE","CYT","GUA","THY","URI",
    "M2G","OMG","PSU","5MC","1MA","H2U","2MG","7MG","OMC","5MU","5BU","I"
}

# thresholds for sec-struct class
ALPHA_MIN = 40.0
BETA_MIN  = 40.0
MUTUAL_CUTOFF = 25.0

FIELDS = ["pdb_id","status","holo_apo","ligands","secstruct","helix_pct","beta_pct"]

# ---------- helpers ----------
def read_ids_filter(path):
    txt = open(path, "r", encoding="utf-8").read()
    return set(s.upper() for s in re.findall(r"[A-Za-z0-9]{4}", txt))

def list_struct_files(root):
    exts = (".pdb",".ent",".cif",".pdb.gz",".ent.gz",".cif.gz")
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dp, fn))
    return sorted(out)

def pdb_id_from_name(path):
    base = os.path.basename(path)
    m = re.match(r"([A-Za-z0-9]{4})\.(pdb|ent|cif)(\.gz)?$", base, re.IGNORECASE)
    if m: return m.group(1).upper()
    return os.path.splitext(base)[0][:4].upper()

def open_text_any(path):
    if path.lower().endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

# ---------- PDB/ENT parsing ----------
def parse_pdb_stream(lines):
    ligs = set(); protein_res = set(); helix_len = 0; sheet_len = 0

    def s_int(s):
        try: return int(s.strip())
        except: return None

    for line in lines:
        if len(line) < 6: continue
        rec = line[:6]
        if rec.startswith("ATOM  "):
            resname = line[17:20].strip().upper()
            if resname in AMINO:
                chain = line[21:22]
                seq   = line[22:26].strip()
                if seq and seq not in ('.','?'):
                    protein_res.add((chain, seq))
        elif rec.startswith("HETATM"):
            resname = line[17:20].strip().upper()
            if resname and resname not in NON_LIGAND:
                ligs.add(resname)
        elif rec.startswith("HELIX "):
            L = s_int(line[71:76]) if len(line) >= 76 else None
            if L is not None:
                helix_len += max(0, L)
            else:
                bch = line[19:20]; ech = line[31:32]
                b   = s_int(line[21:25]); e = s_int(line[33:37])
                if bch == ech and b is not None and e is not None:
                    helix_len += max(0, e - b + 1)
        elif rec.startswith("SHEET "):
            bch = line[21:22]; ech = line[32:33]
            b   = s_int(line[22:26]); e = s_int(line[33:37])
            if bch == ech and b is not None and e is not None:
                sheet_len += max(0, e - b + 1)

    return ligs, len(protein_res), helix_len, sheet_len

def parse_pdb_file(path):
    with open_text_any(path) as f:
        return parse_pdb_stream(f)

# ---------- CIF parsing (robust via gemmi; accepts label_* or auth_*) ----------
def parse_cif_file(path):
    if not USE_GEMMI:
        return set(), 0, 0, 0

    # read with gemmi (supports string or file)
    try:
        if path.lower().endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
                doc = gemmi.cif.read_string(fh.read())
        else:
            doc = gemmi.cif.read_file(path)
        block = None
        if len(doc) == 1:
            block = doc[0]
        else:
            for b in doc:
                if b.find_mmcif_category('_atom_site'):
                    block = b; break
            if block is None:
                block = doc[0]
    except Exception:
        return set(), 0, 0, 0

    # --- atom_site: ligands + protein residues ---
    cat = block.find_mmcif_category('_atom_site')
    if not cat:
        return set(), 0, 0, 0
    tags = list(cat.tags)

    def col(name):
        try: return tags.index(name)
        except ValueError: return -1

    # prefer label_*; fall back to auth_* where needed
    i_comp = col('label_comp_id');   i_comp2 = col('auth_comp_id')
    i_asym = col('label_asym_id');   i_asym2 = col('auth_asym_id')
    i_seq  = col('label_seq_id');    i_seq2  = col('auth_seq_id')
    i_grp  = col('group_PDB')  # may be absent

    ligs = set()
    protein_res = set()

    for r in cat:
        comp = (r[i_comp] if i_comp >= 0 else (r[i_comp2] if i_comp2 >= 0 else '')).upper()
        if not comp:
            continue

        # count protein residues
        a_id = r[i_asym] if i_asym >= 0 else (r[i_asym2] if i_asym2 >= 0 else '')
        s_id = r[i_seq]  if i_seq  >= 0 else (r[i_seq2]  if i_seq2  >= 0 else '')
        if comp in AMINO and s_id not in ('.','?',''):
            protein_res.add((a_id, s_id))
            # don't "continue": HET rows can still be counted below separately

        # ligand detection: prefer HETATM if available; else "not amino & not nucleotide"
        is_het = (i_grp >= 0 and r[i_grp].upper() == 'HETATM')
        if is_het or (comp not in AMINO and comp not in NUCLEOTIDES):
            if comp not in NON_LIGAND:
                ligs.add(comp)

    # --- helices from _struct_conf ---
    helix_len = 0
    sc = block.find_mmcif_category('_struct_conf')
    if sc:
        T = list(sc.tags)
        def cc(name):
            try: return T.index(name)
            except ValueError: return -1

        i_type = cc('conf_type_id')
        i_len  = cc('pdbx_PDB_helix_length')

        # pairs of (asym,seq) options: prefer *auth*, else *label*
        beg_asym = cc('beg_auth_asym_id'); end_asym = cc('end_auth_asym_id')
        beg_seq  = cc('beg_auth_seq_id');  end_seq  = cc('end_auth_seq_id')
        if min(beg_asym, end_asym, beg_seq, end_seq) < 0:
            beg_asym = cc('beg_label_asym_id'); end_asym = cc('end_label_asym_id')
            beg_seq  = cc('beg_label_seq_id');  end_seq  = cc('end_label_seq_id')

        for r in sc:
            ctype = r[i_type] if i_type >= 0 else ''
            if i_type >= 0 and not ctype.upper().startswith('HELX'):
                continue
            if i_len >= 0 and r[i_len] not in ('.','?',''):
                try:
                    helix_len += int(r[i_len]); continue
                except Exception:
                    pass
            if min(beg_asym, end_asym, beg_seq, end_seq) >= 0:
                bch, ech = r[beg_asym], r[end_asym]
                b, e = r[beg_seq], r[end_seq]
                if bch == ech and b not in ('.','?','') and e not in ('.','?',''):
                    try:
                        helix_len += max(0, int(e) - int(b) + 1)
                    except Exception:
                        pass

    # --- sheets from _struct_sheet_range ---
    sheet_len = 0
    ss = block.find_mmcif_category('_struct_sheet_range')
    if ss:
        T = list(ss.tags)
        def dd(name):
            try: return T.index(name)
            except ValueError: return -1

        beg_asym = dd('beg_auth_asym_id'); end_asym = dd('end_auth_asym_id')
        beg_seq  = dd('beg_auth_seq_id');  end_seq  = dd('end_auth_seq_id')
        if min(beg_asym, end_asym, beg_seq, end_seq) < 0:
            beg_asym = dd('beg_label_asym_id'); end_asym = dd('end_label_asym_id')
            beg_seq  = dd('beg_label_seq_id');  end_seq  = dd('end_label_seq_id')

        for r in ss:
            if min(beg_asym, end_asym, beg_seq, end_seq) < 0:
                continue
            bch, ech = r[beg_asym], r[end_asym]
            b, e = r[beg_seq], r[end_seq]
            if bch == ech and b not in ('.','?','') and e not in ('.','?',''):
                try:
                    sheet_len += max(0, int(e) - int(b) + 1)
                except Exception:
                    pass

    return ligs, len(protein_res), helix_len, sheet_len

# ---------- shared ----------
def classify_secstruct(helix_res, sheet_res, protein_res):
    if protein_res <= 0:
        return "unknown", None, None
    if helix_res == 0 and sheet_res == 0:
        return "unknown", 0.0, 0.0
    h_pct = 100.0 * helix_res / protein_res
    b_pct = 100.0 * sheet_res / protein_res
    if h_pct >= ALPHA_MIN and b_pct < MUTUAL_CUTOFF:
        sec = "alpha"
    elif b_pct >= BETA_MIN and h_pct < MUTUAL_CUTOFF:
        sec = "beta"
    else:
        sec = "alpha_beta"
    return sec, h_pct, b_pct

def process_file(path):
    t0 = time.time()
    pdbid = pdb_id_from_name(path)
    low = path.lower()

    try:
        if low.endswith((".pdb", ".ent", ".pdb.gz", ".ent.gz")):
            ligs, nprot, hlen, blen = parse_pdb_file(path)
        elif low.endswith((".cif", ".cif.gz")):
            ligs, nprot, hlen, blen = parse_cif_file(path)
        else:
            return {"pdb_id": pdbid, "status":"skip","holo_apo":"","ligands":"",
                    "secstruct":"unknown","helix_pct":"","beta_pct":""}, time.time()-t0
    except Exception:
        return {"pdb_id": pdbid, "status":"read_failed","holo_apo":"","ligands":"",
                "secstruct":"unknown","helix_pct":"","beta_pct":""}, time.time()-t0

    holo_apo = "holo" if ligs else "apo"
    sec, h_pct, b_pct = classify_secstruct(hlen, blen, nprot)
    row = {
        "pdb_id": pdbid, "status":"ok",
        "holo_apo": holo_apo,
        "ligands": ";".join(sorted(ligs)),
        "secstruct": sec,
        "helix_pct": round(h_pct,1) if h_pct is not None else "",
        "beta_pct":  round(b_pct,1) if b_pct is not None else ""
    }
    return row, time.time()-t0

# ---------- main ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: py classify_offline_any.py <root_dir> [ids.txt]")
        sys.exit(1)

    root = sys.argv[1]
    if not os.path.isdir(root):
        print("Not a directory:", root); sys.exit(1)

    files = list_struct_files(root)
    if len(sys.argv) >= 3:
        filt = read_ids_filter(sys.argv[2])
        files = [p for p in files if pdb_id_from_name(p) in filt]

    if not files:
        print("No .pdb/.ent/.cif(.gz) files found in:", root); sys.exit(1)

    rows = []
    n = len(files)
    for i, path in enumerate(files, 1):
        row, dt = process_file(path)
        rows.append(row)
        if VERBOSE:
            log(f"[{i}/{n}] {row['pdb_id']} {row['holo_apo'].upper()} "
                f"ligs=[{row['ligands']}] sec={row['secstruct']} (H={row['helix_pct']}, B={row['beta_pct']}) {dt:.2f}s")

    with open("pdb_classified.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)

    def dump_subset(name, predicate):
        sub = [r for r in rows if r.get("status")=="ok" and predicate(r)]
        with open(f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader(); w.writerows(sub)

    dump_subset("apo",        lambda r: r["holo_apo"]=="apo")
    dump_subset("holo",       lambda r: r["holo_apo"]=="holo")
    dump_subset("alpha",      lambda r: r["secstruct"]=="alpha")
    dump_subset("beta",       lambda r: r["secstruct"]=="beta")
    dump_subset("alpha_beta", lambda r: r["secstruct"]=="alpha_beta")

    ok = sum(1 for r in rows if r.get("status")=="ok")
    fail = len(rows) - ok
    log(f"Done: {ok} ok, {fail} failed.")
    log("Output: pdb_classified.csv + apo.csv + holo.csv + alpha.csv + beta.csv + alpha_beta.csv")

if __name__ == "__main__":
    main()
