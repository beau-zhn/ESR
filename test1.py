#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test.py — выводит только er_state для лигандов на ГЛАВНОЙ ERα-цепи каждого PDB.

Порядок доказательств:
  0) MANUAL_ER_CLASS (офлайн, по PDB-коду лиганда)
  1) ChEMBL по InChIKey (CCD → RDKit → entry/nonpolymer)
  2) Если ChEMBL не помог: ChEMBL по NAME/SYNONYMS из CCD
  3) Если всё ещё неизвестно: эвристика по названию CCD
  4) Если всё ещё неизвестно: ключевые слова из RCSB title + PDBe summary

Примеры:
  py test.py ".\\data set\\pdb\\1A52.pdb" --verbose
  py test.py ".\\data set\\pdb" -o ".\\er_state.csv" --max-pdb 200 --verbose
"""

import os, re, time, json, argparse, sys
from collections import defaultdict
import urllib.request

# ---------------- config ----------------
HTTP_TIMEOUT = 20
HTTP_HEADERS = {"User-Agent": "ER-Alpha-State/1.3"}
CCD_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{code}"

NON_LIGAND = {
    "HOH","DOD","H2O","WAT","GOL","MPD","PEG","PGE","EDO","PG4","IPA","EOH","ACE","ACET","FMT",
    "SO4","PO4","NO3","TRS","TES","HEP","MES","MOPS","BME",
    "CL","K","NA","MG","CA","CU","MN","IOD","I","BR","CS","NI","CO","CD","SR","RB","F","AU",
    "ZN","CCS","ETC","ZTW","CME","BCT"
}
ER_BETA_NEGWORDS = ["BETA","ESR2","Q92731","NR3A2","MEMBER 2"]

# для выбора main chain
ESTROGEN_LIGANDS = {"OHT","4HT","E2","ES2","EST","E1","DES","TAM","TOR","RAL","BZA","FUL","ICI","ICI1","IF6","GNE","WAY"}

# алиасы только для вытаскивания инфы (например, OHT -> 4HT)
LIGAND_ALIASES = {"OHT":"4HT", "E2":"EST", "ES2":"EST"}

# --- OFFLINE fallback для частых ER-лигандов (минимум ложных) ---
MANUAL_ER_CLASS = {
    # стероиды/фитоэстрогены → агонисты
    "EST":"agonist","E2":"agonist","E1":"agonist","DES":"agonist","GEN":"agonist",
    # классические SERMs
    "TAM":"modulator (SERM)","4HT":"modulator (SERM)","OHT":"modulator (SERM)",
    "RAL":"modulator (SERM)","BZA":"modulator (SERM)","TOR":"modulator (SERM)",
    # серия бензоксатииинов (1XP*)
    "AIH":"modulator (SERM)","AIU":"modulator (SERM)",
    "AIJ":"modulator (SERM)","AIT":"modulator (SERM)","AEJ":"modulator (SERM)",
    # SERD
    "FUL":"antagonist","ICI":"antagonist","ICI1":"antagonist"
}

# ---------------- http utils ----------------
def _http_json(url, timeout=HTTP_TIMEOUT):
    req = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8","ignore"))

# ---------------- CCD / names ----------------
def fetch_ccd_info(code, tries=3):
    """return (smiles, inchikey, name, synonyms_list)"""
    url = CCD_URL.format(code=code)
    for i in range(tries):
        try:
            j = _http_json(url)
            chem = j.get("chem_comp", {}) or {}
            desc = j.get("rcsb_chem_comp_descriptor", {}) or {}
            smi  = chem.get("smiles") or chem.get("pdbx_smiles") or desc.get("smiles")
            ik   = chem.get("inchikey") or desc.get("inchi_key")
            name = (chem.get("name") or "").strip()
            syns = j.get("rcsb_chem_comp_synonyms", {})
            out = []
            if isinstance(syns, dict):
                for v in syns.values(): out += (v if isinstance(v, list) else [v])
            elif isinstance(syns, list):
                out += syns
            elif isinstance(syns, str):
                out.append(syns)
            out = sorted({str(s).strip() for s in out if s})
            return smi, ik, name, out
        except Exception:
            time.sleep(1.2*(i+1))
    return None, None, "", []

def fetch_ccd_with_alias(code):
    seen = set()
    for c in (code, LIGAND_ALIASES.get(code)):
        if not c or c in seen: continue
        smi, ik, name, syns = fetch_ccd_info(c)
        if smi or ik or name: return smi, ik, name, syns, c
        seen.add(c)
    return None, None, "", [], None

# эвристика по названию CCD
def guess_from_ccd_name_3class(name: str) -> str:
    n = (name or "").lower()
    if not n: return "unknown"
    if any(k in n for k in ("estradiol","oestradiol","estrone","estriol")):
        return "agonist"
    if any(k in n for k in ("genistein","daidzein","coumestrol","resveratrol")):
        return "agonist"
    if any(k in n for k in ("raloxifene","tamoxifen","bazedoxifene","arzoxifene","lasofoxifene",
                             "benzothiophen","benzoxathiin","benzoxathiin")):
        return "modulator (SERM)"
    if any(k in n for k in ("fulvestrant","degrader")):
        return "antagonist"
    return "unknown"

# ---------------- Entry → InChIKey ----------------
_ENTRY_IK_CACHE = {}
def entry_code_to_inchikey(pdb_id: str, throttle=0.15):
    pid = pdb_id.strip().upper()
    if pid in _ENTRY_IK_CACHE:
        return _ENTRY_IK_CACHE[pid]
    mapping = {}
    try:
        entry = _http_json(f"https://data.rcsb.org/rest/v1/core/entry/{pid}")
        for ref in entry.get("nonpolymer_entities") or []:
            ent_id = str(ref).strip("/").split("/")[-1]
            time.sleep(throttle)
            ej = _http_json(f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pid}/{ent_id}")
            chem = ej.get("chem_comp", {}) or {}
            code = (chem.get("id") or "").upper()
            desc = ej.get("rcsb_chem_comp_descriptor", {}) or {}
            ik = desc.get("inchi_key")
            if code and ik:
                mapping[code] = ik
    except Exception:
        mapping = {}
    _ENTRY_IK_CACHE[pid] = mapping
    return mapping

# локально получить IK из SMILES (если установлен rdkit)
def inchikey_from_smiles_local(smiles):
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        if not m: return None
        Chem.SanitizeMol(m)
        return Chem.MolToInchiKey(m)
    except Exception:
        return None

# ---------------- RCSB/PDBe texts ----------------
_RCSB_ENTRY_CACHE = {}
def rcsb_title(pid):
    pid = pid.upper()
    if pid in _RCSB_ENTRY_CACHE: return _RCSB_ENTRY_CACHE[pid]
    try:
        j = _http_json(f"https://data.rcsb.org/rest/v1/core/entry/{pid}")
        t = ((j.get("struct") or {}).get("title") or "").strip()
    except Exception:
        t = ""
    _RCSB_ENTRY_CACHE[pid] = t
    return t

def _norm(x):
    if x is None: return ""
    if isinstance(x, str): return x
    if isinstance(x, (list,tuple,set)): return ", ".join(_norm(t) for t in x if t)
    if isinstance(x, dict): return ", ".join(_norm(v) for v in x.values() if v)
    return str(x)

_PDBE_SUMMARY_CACHE = {}
def pdbe_summary(pid):
    pid_l = pid.lower()
    if pid_l in _PDBE_SUMMARY_CACHE: return _PDBE_SUMMARY_CACHE[pid_l]
    try:
        j = _http_json(f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/summary/{pid_l}")
        lst = j.get(pid_l, [])
        if not lst: txt = ""
        else:
            t = _norm(lst[0].get("title")); m = _norm(lst[0].get("experimental_method")); k = _norm(lst[0].get("keywords"))
            txt = " | ".join([s for s in (t,m,k) if s])
    except Exception:
        txt = ""
    _PDBE_SUMMARY_CACHE[pid_l] = txt
    return txt

def classify_texts_3class(*texts):
    blob = " | ".join([t for t in texts if t]).lower()
    if ("selective estrogen receptor modulator" in blob) or ("selective oestrogen receptor modulator" in blob) or (" serm" in blob):
        return "modulator (SERM)"
    if ("antagonist" in blob) or ("antiestrogen" in blob) or ("serd" in blob) or ("degrader" in blob):
        return "antagonist"
    if "agonist" in blob:
        return "agonist"
    if "corepressor" in blob:
        return "antagonist"
    if "coactivator" in blob and "peptide" in blob:
        return "agonist"
    return "unknown"

# ---------------- ChEMBL (optional) ----------------
try:
    from chembl_webresource_client.new_client import new_client
    _CHEMBL_OK = True
    chembl_mol  = new_client.molecule
    chembl_mech = new_client.mechanism
except Exception:
    _CHEMBL_OK = False

def chembl_by_ik(ik):
    if not (_CHEMBL_OK and ik): return None, []
    try:
        hits = chembl_mol.filter(inchi_key=ik)
        if not hits: return None, []
        cid = hits[0]["molecule_chembl_id"]
        mechs = chembl_mech.filter(molecule_chembl_id=cid)
        return cid, mechs
    except Exception:
        return None, []

def chembl_by_name(q):
    if not (_CHEMBL_OK and q): return None, []
    try:
        hits = chembl_mol.search(q)
        if not hits: return None, []
        cid = hits[0]["molecule_chembl_id"]
        mechs = chembl_mech.filter(molecule_chembl_id=cid)
        return cid, mechs
    except Exception:
        return None, []

def classify_mechs_3class(mechs):
    if not mechs: return "unknown"
    txt = " | ".join([(str(m.get('action_type') or "")+" "+str(m.get('mechanism_of_action') or "")).lower() for m in mechs])
    if ("selective estrogen receptor modulator" in txt) or ("selective oestrogen receptor modulator" in txt) or (" serm" in txt):
        return "modulator (SERM)"
    if ("inverse agonist" in txt) or ("antagonist" in txt) or ("antiestrogen" in txt) or ("serd" in txt) or ("degrader" in txt):
        return "antagonist"
    if ("partial agonist" in txt) or ("full agonist" in txt) or (" agonist" in txt):
        return "agonist"
    if ("corepressor" in txt):
        return "antagonist"
    if ("coactivator" in txt and "peptide" in txt):
        return "agonist"
    return "unknown"

# ---------------- PDB parsing ----------------
def parse_pdb_compnd_chain_map(path):
    comp = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("COMPND"):
                comp.append(line[10:].rstrip())
    if not comp: return {}
    joined = re.sub(r"\s+"," "," ".join(comp))
    parts  = [p.strip() for p in joined.split(";") if p.strip()]
    chain2mol = {}
    cur_mol, cur_chains = None, []
    for part in parts:
        if ":" not in part: continue
        k, v = part.split(":",1)
        k = k.strip().upper(); v = v.strip()
        if k == "MOL_ID":
            if cur_mol and cur_chains:
                for ch in cur_chains: chain2mol[ch] = cur_mol
            cur_mol, cur_chains = None, []
        elif k == "MOLECULE":
            cur_mol = v.upper()
        elif k == "CHAIN":
            cur_chains = [c.strip() for c in v.split(",") if c.strip()]
    if cur_mol and cur_chains:
        for ch in cur_chains: chain2mol[ch] = cur_mol
    return chain2mol

def detect_er_alpha_chains_pdb(path):
    er = set()
    for ch, mol in parse_pdb_compnd_chain_map(path).items():
        if "ESTROGEN RECEPTOR" in mol and not any(b in mol for b in ER_BETA_NEGWORDS):
            er.add(ch)
    return er

def parse_pdb_seqres_len(path):
    declared = {}
    fallback = defaultdict(int)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("SEQRES"): continue
            chain  = line[11:12].strip()
            sernum = line[7:10].strip()
            numres = line[13:17].strip()
            if sernum == "1":
                try: declared[chain] = int(numres)
                except: pass
            toks = [t.strip().upper() for t in line[19:].split() if t.strip()]
            fallback[chain] += sum(1 for t in toks if len(t)==3 and t.isalpha())
    lengths = {}
    for ch in set(declared)|set(fallback):
        lengths[ch] = declared.get(ch, fallback.get(ch,0))
    return lengths

def parse_pdb_ligands_by_chain(path):
    by_chain = defaultdict(set)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = line[:6]
            if rec == "HET   ":
                res = line[7:10].strip().upper()
                ch  = line[12:13]
            elif rec == "HETATM":
                res = line[17:20].strip().upper()
                ch  = line[21:22]
            else:
                continue
            if res and res not in NON_LIGAND and ch.strip():
                by_chain[ch].add(res)
    return {k: sorted(v) for k, v in by_chain.items()}

def choose_main_er_chain(er_chains, lengths, ligs_by_chain):
    if not er_chains: return None
    cands = [ch for ch in er_chains if any(l in ESTROGEN_LIGANDS for l in ligs_by_chain.get(ch, []))]
    pool = cands or list(er_chains)
    pool.sort(key=lambda c: (-lengths.get(c,0), c))
    pool.sort(key=lambda c: (c != 'A', -lengths.get(c,0), c))
    return pool[0]

# ---------------- core ----------------
def states_for_file_main_chain(pdb_path, verbose=False):
    pid = os.path.basename(pdb_path)[:4].upper()
    er_chains = detect_er_alpha_chains_pdb(pdb_path)
    if not er_chains:
        return [], f"{pid}: ERα chains not found"

    lengths = parse_pdb_seqres_len(pdb_path)
    ligs_by_chain = parse_pdb_ligands_by_chain(pdb_path)
    main_ch = choose_main_er_chain(er_chains, lengths, ligs_by_chain)
    if not main_ch:
        return [], f"{pid}: main ERα chain not resolved"

    if verbose:
        reason = "estrogen ligand present" if any(l in ESTROGEN_LIGANDS for l in ligs_by_chain.get(main_ch, [])) else "fallback by length/name"
        print(f"{pid}: main ERα chain -> {main_ch} ({reason})")
        ligs_here = ligs_by_chain.get(main_ch, [])
        print(f"Ligands on {main_ch}: {ligs_here if ligs_here else '—'}")

    title = rcsb_title(pid)
    pdbe  = pdbe_summary(pid)

    rows = []
    for code in sorted(ligs_by_chain.get(main_ch, [])):
        code = code.upper()

        # 0) офлайн-словарь
        state = MANUAL_ER_CLASS.get(code, "unknown")
        src = "manual" if state != "unknown" else "none"

        # 1) CCD / alias
        smi, ik, name, syns, used = fetch_ccd_with_alias(code)

        # 1a) RDKit → IK
        if state == "unknown" and smi and not ik:
            ik = inchikey_from_smiles_local(smi)

        # 1b) entry → IK (иногда CCD пуст)
        if state == "unknown" and not ik:
            alias = LIGAND_ALIASES.get(code, code)
            ik = entry_code_to_inchikey(pid).get(alias)

        # 2) ChEMBL (IK → mechs; затем name/synonyms)
        if state == "unknown" and _CHEMBL_OK:
            cid, mechs = (None, [])
            if ik:
                cid, mechs = chembl_by_ik(ik)
            if not cid:
                for q in [name] + syns + [code]:
                    if not q: continue
                    cid, mechs = chembl_by_name(q)
                    if cid: break
            s2 = classify_mechs_3class(mechs)
            if s2 != "unknown":
                state, src = s2, "chembl"

        # 3) эвристика по названию CCD
        if state == "unknown":
            s3 = guess_from_ccd_name_3class(name)
            if s3 != "unknown":
                state, src = s3, "ccd_name"

        # 4) ключевые слова из RCSB/PDBe
        if state == "unknown":
            s4 = classify_texts_3class(title, pdbe)
            if s4 != "unknown":
                state, src = s4, "text"

        if verbose:
            print(f"  {pid} {main_ch} {code} -> {state} [{src}]")
        rows.append({"pdb_id": pid, "chain": main_ch, "ligand_code": code, "er_state": state})
    return rows, ""

# ---------------- CLI ----------------
def iter_pdb_paths(src):
    if os.path.isfile(src) and src.lower().endswith((".pdb",".ent")):
        yield src; return
    for dp, _, fns in os.walk(src):
        for fn in fns:
            if fn.lower().endswith((".pdb",".ent")):
                yield os.path.join(dp, fn)

def main():
    ap = argparse.ArgumentParser(description="Output er_state for ligands on MAIN ERα chain (CSV).")
    ap.add_argument("src", help="PDB file or folder")
    ap.add_argument("-o","--out", default=None, help="save CSV (default: stdout)")
    ap.add_argument("--max-pdb", type=int, default=None, help="limit number of files")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    paths = list(iter_pdb_paths(args.src))
    if not paths:
        raise FileNotFoundError(f"No PDB files at: {args.src}")
    if args.max_pdb: paths = paths[:args.max_pdb]

    all_rows = []
    for i, p in enumerate(paths, 1):
        rows, msg = states_for_file_main_chain(p, verbose=args.verbose)
        if msg:
            print(msg); continue
        all_rows.extend(rows)
        if args.verbose:
            print(f"[{i}/{len(paths)}] {os.path.basename(p)[:4].upper()} -> {len(rows)} row(s)")

    import csv
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as fw:
            w = csv.DictWriter(fw, fieldnames=["pdb_id","chain","ligand_code","er_state"])
            w.writeheader(); w.writerows(all_rows)
        print(f"[OK] saved: {args.out}")
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=["pdb_id","chain","ligand_code","er_state"])
        w.writeheader(); w.writerows(all_rows)

if __name__ == "__main__":
    main()
