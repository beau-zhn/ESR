#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERα-only (PDB only) -> ligands + SMILES/InChIKey + Action type + Bioactivity

Creates:
  - er_files_summary_alpha.csv
  - ligands_master_alpha.csv   (lig_code, smiles, inchikey, chembl_id, er_action)
  - ligand_bioactivity.csv     (chembl_id, lig_code, target, target_chembl_id, type, n, median_nM, min_nM, max_nM)
  - ccd_missing.csv            (если CCD не дал SMILES)
  - ccd_debug.log

Usage:
  py er_alpha_pdb_only.py "<folder_or_single_pdb>"
"""

import os, sys, re, time
from collections import defaultdict

import requests
import pandas as pd
from tqdm import tqdm

# ============================== CONFIG ====================================

# Не считаем лигандами (ZN умышленно не в списке, он нужен как флаг "dna based")
NON_LIGAND = {
    "HOH","DOD","H2O","WAT",
    "GOL","MPD","PEG","PGE","EDO","PG4","IPA","EOH","ACE","ACET","FMT",
    "SO4","PO4","NO3","TRS","TES","HEP","MES","MOPS","BME",
    "CL","K","NA","MG","CA","CU","MN","IOD","I","BR","CS","NI","CO","CD","SR","RB","F",
    "AU"
}
# Лиганды-эстрогены/СЕРМы для приоритета выбора цепи
ESTROGEN_LIGANDS = {"OHT","4HT","E2","ES2","EST","E1","DES","TAM","TOR","RAL","BZA","FUL","ICI","ICI1","IF6","GNE","WAY"}
# Ключевые слова, по которым исключаем ERβ
ER_BETA_NEGWORDS = ["BETA","ESR2","Q92731","NR3A2","MEMBER 2"]

# Алиасы (PDB -> CCD id)
LIGAND_ALIASES = {
    "OHT": "4HT",   # 4-hydroxytamoxifen
    "E2":  "EST",   # estradiol (E2/ES2)
    "ES2": "EST",
}

HTTP_TIMEOUT = 20
HTTP_HEADERS = {"User-Agent": "ER-Alpha-Project/1.0"}
CCD_URL      = "https://data.rcsb.org/rest/v1/core/chemcomp/{code}"

# UniProt интересующих таргетов
ER_UNIPROTS = {
    "ESR1": "P03372",   # Estrogen receptor alpha
    "ESR2": "Q92731",   # Estrogen receptor beta
}
# Жёсткие (надёжные) ChEMBL target IDs для ESR1/ESR2
KNOWN_ER_TARGETS = {"P03372": "CHEMBL206", "Q92731": "CHEMBL242"}
# Известные СЕРМы (если механизм пуст, помечаем как SERM)
KNOWN_SERMS = {"OHT","4HT","TAM","TOR","RAL","BZA","LAS"}
# Фолбэк: PDB-код -> понятное имя для поиска по ChEMBL
LIGAND_NAME_FALLBACK = {
    "OHT": "4-hydroxytamoxifen",
    "4HT": "4-hydroxytamoxifen",
    "TAM": "tamoxifen",
    "EST": "estradiol",
    "E2":  "estradiol",
    "ES2": "estradiol",
    "RAL": "raloxifene",
    "BZA": "bazedoxifene",
}

# Фолбэк: PDB-код -> известный ChEMBL ID (минимальный набор)
KNOWN_LIGAND_TO_CHEMBL = {
    "OHT": "CHEMBL489",   # Afimoxifene (4-hydroxytamoxifen)
    "4HT": "CHEMBL489",
    "TAM": "CHEMBL18",
}

# фильтр по названию таргета (без кейса)
ER_NAME_KEYWORDS = ("estrogen receptor", )

# какие типы собирать (+ RBA)
BIO_TYPES_PRIMARY = ("IC50", "Ki", "EC50")
BIO_TYPES_EXTRA   = ("Kd", "AC50", "RBA")  # RBA часто в M
ALLOWED_TYPES     = {t.upper() for t in BIO_TYPES_PRIMARY + BIO_TYPES_EXTRA}

# ============================== PDB UTILS =================================

def iter_pdb_paths(root_or_file):
    p = root_or_file
    if os.path.isfile(p) and p.lower().endswith((".pdb",".ent")):
        yield p; return
    for dp, _, fns in os.walk(p):
        for fn in fns:
            if fn.lower().endswith((".pdb",".ent")):
                yield os.path.join(dp, fn)

def pdb_id_from_path(path):
    base = os.path.basename(path)
    m = re.match(r"([A-Za-z0-9]{4})\.", base)
    return (m.group(1).upper() if m else base[:4].upper())

def parse_pdb_compnd_chain_map(path):
    compnd_lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("COMPND"):
                compnd_lines.append(line[10:].rstrip())
    if not compnd_lines:
        return {}
    joined = " ".join(compnd_lines)
    joined = re.sub(r"\s+", " ", joined)
    parts  = [p.strip() for p in joined.split(';') if p.strip()]

    chain_to_mol = {}
    current_mol, current_chains = None, []
    for part in parts:
        if ':' not in part:
            continue
        key, val = part.split(':',1)
        key = key.strip().upper(); val = val.strip()
        if key == "MOL_ID":
            if current_mol and current_chains:
                for ch in current_chains: chain_to_mol[ch] = current_mol
            current_mol, current_chains = None, []
        elif key == "MOLECULE":
            current_mol = val.upper()
        elif key == "CHAIN":
            current_chains = [c.strip() for c in val.split(',') if c.strip()]
    if current_mol and current_chains:
        for ch in current_chains: chain_to_mol[ch] = current_mol
    return chain_to_mol

def detect_er_alpha_chains_pdb(path):
    chain2mol = parse_pdb_compnd_chain_map(path)
    er = set()
    for ch, mol in chain2mol.items():
        if "ESTROGEN RECEPTOR" in mol and not any(b in mol for b in ER_BETA_NEGWORDS):
            er.add(ch)
    return er

def parse_pdb_seqres_len(path):
    declared = {}
    fallback = defaultdict(int)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("SEQRES"):
                continue
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

def parse_pdb_ligands(path):
    ligs = set()
    ligs_by_chain = defaultdict(set)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            res = line[17:20].strip().upper()
            ch  = line[21:22]
            if res and res not in NON_LIGAND:
                ligs.add(res)
                if ch:
                    ligs_by_chain[ch].add(res)
    return ligs, {k:sorted(v) for k,v in ligs_by_chain.items()}

def choose_main_er_chain(er_chains, lengths, ligs_by_chain):
    if not er_chains: return None
    # приоритет: цепь с эстроген/СЕРМ лигандом
    cand = []
    for ch in er_chains:
        ligs = set(ligs_by_chain.get(ch, []))
        if any(l in ligs for l in ESTROGEN_LIGANDS):
            cand.append(ch)
    if cand:
        cand.sort(key=lambda c: (-lengths.get(c,0), c))
        return cand[0]
    if 'A' in er_chains:
        return 'A'
    return sorted(er_chains, key=lambda c: (-lengths.get(c,0), c))[0]

# ============================== CCD / SMILES ==============================

def fetch_ccd_smiles(code, max_tries=4):
    url = CCD_URL.format(code=code)
    for i in range(max_tries):
        try:
            r = requests.get(url, timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS)
            if r.status_code == 404:
                return None, None
            if r.ok:
                data = r.json()
                chem  = data.get("chem_comp", {})
                smiles = chem.get("smiles") or chem.get("pdbx_smiles")
                ik     = chem.get("inchikey")
                if not smiles:
                    desc   = data.get("rcsb_chem_comp_descriptor", {})
                    smiles = desc.get("smiles")
                    ik     = ik or desc.get("inchi_key")
                return smiles, ik
        except requests.RequestException:
            pass
        time.sleep(1.2*(i+1))
    return None, None

def fetch_ccd_smiles_with_alias(code):
    tried = []
    for cand in [code, LIGAND_ALIASES.get(code)]:
        if not cand or cand in tried:
            continue
        smi, ik = fetch_ccd_smiles(cand)
        if smi:
            return smi, ik, cand
        tried.append(cand)
    return None, None, None

def inchikey_from_smiles_local(smiles):
    """Fallback: вычисляем InChIKey локально (RDKit)."""
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        if not m:
            return None
        Chem.SanitizeMol(m)
        return Chem.MolToInchiKey(m)
    except Exception:
        return None

def load_overrides():
    """Опциональный файл ручных правок: ligand_overrides.csv
       колонки: lig_code,smiles,inchikey,er_action"""
    p = "ligand_overrides.csv"
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    m = {}
    for _, r in df.iterrows():
        code = str(r['lig_code']).strip().upper()
        smi  = str(r.get('smiles') or '').strip()
        ik   = str(r.get('inchikey') or '').strip() or None
        act  = str(r.get('er_action') or '').strip()
        if code and smi:
            m[code] = (smi, ik, act if act else None)
    return m

# =========================== ChEMBL CLIENT API ============================

try:
    from chembl_webresource_client.new_client import new_client
    _CHEMBL_OK = True
    chembl_target    = new_client.target
    chembl_molecule  = new_client.molecule
    chembl_activity  = new_client.activity
    chembl_mechanism = new_client.mechanism
except Exception:
    _CHEMBL_OK = False

def chembl_resolve_esr_targets():
    """{'P03372':'CHEMBL206', 'Q92731':'CHEMBL242'} (уточняется через API, есть фолбэк)."""
    out = KNOWN_ER_TARGETS.copy()
    if not _CHEMBL_OK:
        return out
    try:
        q = chembl_target.filter(target_components__accession="P03372")
        if q: out["P03372"] = q[0]['target_chembl_id']
    except Exception: pass
    try:
        q = chembl_target.filter(target_components__accession="Q92731")
        if q: out["Q92731"] = q[0]['target_chembl_id']
    except Exception: pass
    return out

def chembl_find_mols_by_inchikey(inchikey):
    if not _CHEMBL_OK or not inchikey:
        return []
    try:
        return chembl_molecule.filter(inchi_key=inchikey)
    except Exception:
        return []

def chembl_find_by_name(name):
    """Поиск молекулы по названию (строка поиска ChEMBL). Возвращает список хитов."""
    if not _CHEMBL_OK or not name:
        return []
    try:
        return new_client.molecule.search(name)
    except Exception:
        return []

def chembl_mechanisms_for_mol(chembl_id):
    if not _CHEMBL_OK or not chembl_id:
        return []
    try:
        return chembl_mechanism.filter(molecule_chembl_id=chembl_id)
    except Exception:
        return []

def classify_action_from_mechs(mechs):
    txt = " | ".join(
        [(m.get('action_type') or "")+" "+(m.get('mechanism_of_action') or "") for m in mechs]
    ).lower()
    if "partial agonist" in txt: return "partial agonist"
    if "inverse agonist" in txt:  return "inverse agonist"
    if "full agonist" in txt or "agonist" in txt: return "agonist"
    if "antagonist" in txt and "inverse" not in txt: return "antagonist"
    if "serm" in txt or "selective estrogen receptor modulator" in txt or "modulator" in txt:
        return "modulator"
    return "unknown"

# -------- Bioactivity collector (any ER target by name) --------

# -------- Bioactivity collector (any ER target by name) + debug --------
def chembl_bioactivity_block_any_er(chembl_id):
    """
    Возвращает dict: {TYPE(UPPER): [values_nM]} для всех активностей молекулы,
    где target_pref_name содержит "estrogen receptor".
    Пишет краткий лог в bio_debug.log.
    """
    def _to_nm(val, units):
        if val is None: return None
        try: v = float(val)
        except: return None
        u = (units or "").lower()
        if not u or u in ("nm","nanomolar"): return v
        if u in ("pm","picomolar"):          return v * 0.001
        if u in ("um","micromolar"):         return v * 1000.0
        if u in ("mm","millimolar"):         return v * 1_000_000.0
        if u in ("m","molar"):               return v * 1e9     # RBA и др.
        return None

    def _good_rel(rel):
        rel = (rel or "").strip()
        return rel in ("", "=", "<", "<=", "~", "≈")

    res = {t: [] for t in ALLOWED_TYPES}
    n_rows = 0
    n_kept = 0

    if not _CHEMBL_OK:
        return res

    try:
        rows = chembl_activity.filter(molecule_chembl_id=chembl_id)[:1000]
    except Exception:
        rows = []

    n_rows = len(rows)
    for a in rows:
        pref = (a.get("target_pref_name") or "").lower()
        if not any(k in pref for k in ER_NAME_KEYWORDS):
            continue
        typ = (a.get("standard_type") or "").upper()
        if typ not in res:
            continue
        v = _to_nm(a.get("standard_value"), a.get("standard_units"))
        if v is None:
            continue
        if not _good_rel(a.get("standard_relation")):
            continue
        res[typ].append(v); n_kept += 1

    # pChEMBL -> surrogate IC50 если совсем пусто
    if not any(res.values()):
        tmp = []
        for a in rows:
            pref = (a.get("target_pref_name") or "").lower()
            if not any(k in pref for k in ER_NAME_KEYWORDS):
                continue
            p = a.get("pchembl_value")
            if not p: continue
            try: p = float(p)
            except: continue
            nm = (10**(-p)) * 1e9
            tmp.append(nm)
        if tmp:
            res["IC50"] += tmp

    # лог
    with open("bio_debug.log", "a", encoding="utf-8") as fh:
        fh.write(f"{chembl_id}: total_rows={n_rows}, kept_after_name_filter={n_kept}; "
                 f"nonempty_types={[k for k,v in res.items() if v]}\n")
    return res

# -------- REST fallback for ESR1/ESR2 if name-filter is empty --------
def fetch_bioactivity_rest_by_target(chembl_id, target_chembl_id, types=("IC50","Ki","EC50","Kd","AC50","RBA")):
    """
    Прямой REST-запрос к ChEMBL: molecule + конкретный target_chembl_id.
    Возвращает {TYPE: [nM]} с конвертацией единиц и допуском relations (=, <, <=, ~, ≈).
    """
    BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    def _to_nm(val, units):
        if val is None: return None
        try: v = float(val)
        except: return None
        u = (units or "").lower()
        if not u or u in ("nm","nanomolar"): return v
        if u in ("pm","picomolar"):          return v * 0.001
        if u in ("um","micromolar"):         return v * 1000.0
        if u in ("mm","millimolar"):         return v * 1_000_000.0
        if u in ("m","molar"):               return v * 1e9
        return None
    def _good_rel(rel):
        rel = (rel or "").strip()
        return rel in ("", "=", "<", "<=", "~", "≈")

    out = {t.upper(): [] for t in types}
    # пагинация
    url = BASE
    params = {
        "molecule_chembl_id": chembl_id,
        "target_chembl_id": target_chembl_id,
        "limit": 1000,
    }
    try:
        r = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        if r.ok:
            data = r.json()
            acts = data.get("activities", [])
            for a in acts:
                typ = (a.get("standard_type") or "").upper()
                if typ not in out: continue
                if not _good_rel(a.get("standard_relation")): continue
                v = _to_nm(a.get("standard_value"), a.get("standard_units"))
                if v is None: continue
                out[typ].append(v)
    except requests.RequestException:
        pass
    return out

# ================================ MAIN ====================================

def main():
    if len(sys.argv) < 2:
        print('Usage: python er_alpha_pdb_only.py "<folder_or_single_pdb>"', file=sys.stderr)
        sys.exit(1)
    src = sys.argv[1]

    # ---------- 1) Обход PDB, выбор ERα-цепи, базовый отчёт ----------
    per_file_rows = []
    for path in tqdm(list(iter_pdb_paths(src)), desc="Scanning PDBs"):
        pid = pdb_id_from_path(path)
        er_chains = detect_er_alpha_chains_pdb(path)
        if not er_chains:
            continue

        lengths = parse_pdb_seqres_len(path)
        ligs_global, ligs_by_chain = parse_pdb_ligands(path)
        main_ch = choose_main_er_chain(er_chains, lengths, ligs_by_chain)
        if not main_ch:
            continue

        seq_len = lengths.get(main_ch, 0)
        comments = []
        if "ZN" in ligs_global:
            comments.append("dna based")
        if seq_len < 50:
            comments.append("inconsistent structure")

        per_file_rows.append({
            'pdb_id': pid, 'path': path, 'format': 'pdb',
            'chain': main_ch, 'seqres_len': seq_len,
            'ligands_global': ";".join(sorted(ligs_global)),
            'ligands_er_chain': ";".join(sorted(ligs_by_chain.get(main_ch, []))),
            'comments': "; ".join(comments)
        })

    df_files = pd.DataFrame(per_file_rows).drop_duplicates()
    if df_files.empty:
        print("No ERα chains found in given PDBs.")
        return
    df_files.to_csv("er_files_summary_alpha.csv", index=False)

    # ---------- 2) Лиганды ER-цепей: SMILES / IK / ChEMBL / action ----------
    lig_codes = sorted(set(";".join(df_files['ligands_er_chain']).split(";")) - {""})
    df_ligs = pd.DataFrame({'lig_code': lig_codes})

    overrides = load_overrides()
    smiles_list, ik_list, chembl_ids, actions = [], [], [], []
    ccd_missing, debug = [], []

    for code in tqdm(df_ligs['lig_code'], desc="Annotating ligands"):
        code_up = str(code).upper()

        smi = ik = act_override = None
        if code_up in overrides:
            smi, ik, act_override = overrides[code_up]
            debug.append(f"{code_up}: override")

        if not smi:
            smi, ik, used = fetch_ccd_smiles_with_alias(code_up)
            if smi: debug.append(f"{code_up}: CCD via {used}")
            else:   debug.append(f"{code_up}: CCD miss")

        if smi and not ik:
            ik_local = inchikey_from_smiles_local(smi)
            if ik_local:
                ik = ik_local
                debug.append(f"{code_up}: IK via RDKit")
            else:
                debug.append(f"{code_up}: IK RDKit miss")

        if not smi:
            ccd_missing.append(code_up)

        smiles_list.append(smi)
        ik_list.append(ik)

        # --- ChEMBL molecule (InChIKey → name search → known map) ---
        cid = None

        # 1) InChIKey
        mols = chembl_find_mols_by_inchikey(ik) if ik else []
        if mols:
            cid = mols[0].get('molecule_chembl_id')

        # 2) Name search (включая алиасы)
        if not cid:
            qname = LIGAND_NAME_FALLBACK.get(code_up) or code_up
            hits = chembl_find_by_name(qname)

            # если ищем OHT/4HT — приоритезируем CHEMBL489 и CHEMBL486
            pick = None
            if code_up in {"OHT","4HT"} and hits:
                for h in hits:
                    if h.get('molecule_chembl_id') in ("CHEMBL489", "CHEMBL486"):
                        pick = h; break

            # иначе/если не найдено — берём тот, где имя/синонимы содержат запрос
            if not pick and hits:
                key = qname.lower()
                for h in hits:
                    pref = (h.get('pref_name') or '').lower()
                    syns = [(s.get('synonyms') or '').lower()
                            for s in (h.get('molecule_synonyms') or [])]
                    if key in pref or any(key in s for s in syns):
                        pick = h; break

            # последний вариант — первый хит
            if not pick and hits:
                pick = hits[0]

            if pick:
                cid = pick.get('molecule_chembl_id')
                debug.append(f"{code_up}: ChEMBL via name '{qname}' -> {cid}")

        # 3) Жёсткая карта — самый последний вариант
        if not cid and code_up in KNOWN_LIGAND_TO_CHEMBL:
            cid = KNOWN_LIGAND_TO_CHEMBL[code_up]
            debug.append(f"{code_up}: ChEMBL via KNOWN map -> {cid}")

        chembl_ids.append(cid)

        # Action: override > mechanisms > SERM fallback
        if act_override:
            act = act_override
        else:
            mechs = chembl_mechanisms_for_mol(cid) if cid else []
            act = classify_action_from_mechs(mechs)
            if act == "unknown" and code_up in KNOWN_SERMS:
                act = "modulator (SERM)"
        actions.append(act)

    with open("ccd_debug.log", "w", encoding="utf-8") as fh:
        fh.write("\n".join(debug))
    if ccd_missing:
        pd.Series(sorted(set(ccd_missing)), name="lig_code").to_csv("ccd_missing.csv", index=False)

    df_ligs['smiles']    = smiles_list
    df_ligs['inchikey']  = ik_list
    df_ligs['chembl_id'] = chembl_ids
    df_ligs['er_action'] = actions
    df_ligs.to_csv("ligands_master_alpha.csv", index=False)

    # ---------- 3) Биоактивности (любой таргет, где имя содержит "estrogen receptor") ----------
    bio_rows = []
    for _, row in df_ligs.iterrows():
        cid = row.get('chembl_id')
        if not cid or not isinstance(cid, str):
            continue
        lig_code = str(row['lig_code'])

        acts = chembl_bioactivity_block_any_er(cid)

        # если по имени таргета пусто — доберём REST-запросами по ESR1/ESR2
        if not any(acts.values()):
            tgt_map = {"P03372": "CHEMBL206", "Q92731": "CHEMBL242"}
            for tchem in tgt_map.values():
                add = fetch_bioactivity_rest_by_target(cid, tchem)
                for k, vs in add.items():
                    if vs:
                        acts.setdefault(k, [])
                        acts[k].extend(vs)
        for typ, vals in acts.items():
            if not vals:
                continue
            s = pd.Series(sorted(vals))
            bio_rows.append({
                "chembl_id": cid,
                "lig_code": lig_code,
                "target": "ER (name match)",
                "target_chembl_id": "",  # может быть смесью разных ER-мишеней
                "type": typ,             # IC50/KI/EC50/KD/AC50/RBA (UPPER; все в nM)
                "n": int(s.count()),
                "median_nM": float(s.median()),
                "min_nM": float(s.min()),
                "max_nM": float(s.max())
            })

    pd.DataFrame(bio_rows).to_csv("ligand_bioactivity.csv", index=False)

    print("Done:")
    print(" - er_files_summary_alpha.csv")
    print(" - ligands_master_alpha.csv")
    print(" - ligand_bioactivity.csv")
    if os.path.exists("ccd_missing.csv"): print(" - ccd_missing.csv")
    if os.path.exists("ccd_debug.log"):   print(" - ccd_debug.log")

if __name__ == "__main__":
    main()
