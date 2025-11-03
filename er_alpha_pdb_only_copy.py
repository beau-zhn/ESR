#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERα-only (PDB only) -> ligands + SMILES/InChIKey + Action type + Bioactivity (robust, no-hang)

Creates:
  - er_files_summary_alpha.csv
  - ligands_master_alpha.csv
  - ligand_bioactivity.csv
  - ccd_missing.csv (если CCD не дал SMILES)
  - ccd_debug.log
  - cache_ccd.json, cache_chembl_mech.json, cache_chembl_mol.json

Usage:
  py er_alpha_pdb_only.py "<folder_or_single_pdb>" [--net full|lite|off] [--checkpoint 25]
Defaults:
  --net full      : максимум данных (дольше, больше онлайн)
  --net lite      : короткие таймауты/меньше попыток (быстрее/стабильнее)
  --net off       : полностью оффлайн (только оверрайды/известные карты)
  --checkpoint 25 : каждые 25 лигандов писать промежуточный CSV
"""

import os, sys, re, time, json, argparse
from collections import defaultdict

import requests
import pandas as pd
from tqdm import tqdm

# ============================== CONFIG ====================================

NON_LIGAND = {
    "HOH","DOD","H2O","WAT",
    "GOL","MPD","PEG","PGE","EDO","PG4","IPA","EOH","ACE","ACET","FMT",
    "SO4","PO4","NO3","TRS","TES","HEP","MES","MOPS","BME",
    "CL","K","NA","MG","CA","CU","MN","IOD","I","BR","CS","NI","CO","CD","SR","RB","F",
    "AU"
}
ESTROGEN_LIGANDS = {"OHT","4HT","E2","ES2","EST","E1","DES","TAM","TOR","RAL","BZA","FUL","ICI","ICI1","IF6","GNE","WAY"}
ER_BETA_NEGWORDS = ["BETA","ESR2","Q92731","NR3A2","MEMBER 2"]

LIGAND_ALIASES = {"OHT":"4HT", "E2":"EST", "ES2":"EST"}

# сетевые параметры по умолчанию (меняются режимом --net)
HTTP_TIMEOUT = 40
HTTP_RETRIES = 4
HTTP_HEADERS = {"User-Agent": "ER-Alpha-Project/1.2"}
CCD_URL      = "https://data.rcsb.org/rest/v1/core/chemcomp/{code}"
CHEMBL_MOLS  = "https://www.ebi.ac.uk/chembl/api/data/molecule"
CHEMBL_MECH  = "https://www.ebi.ac.uk/chembl/api/data/mechanism"
CHEMBL_ACT   = "https://www.ebi.ac.uk/chembl/api/data/activity"

ER_NAME_KEYWORDS = ("estrogen receptor", )

BIO_TYPES_PRIMARY = ("IC50", "Ki", "EC50")
BIO_TYPES_EXTRA   = ("Kd", "AC50", "RBA")

# ESR1 / ESR2 (для строгого таргета можно добавить при желании)
KNOWN_ER_TARGETS = {"P03372": "CHEMBL206", "Q92731": "CHEMBL242"}

KNOWN_SERMS = {"OHT","4HT","TAM","TOR","RAL","BZA","LAS"}
LIGAND_NAME_FALLBACK = {
    "OHT":"4-hydroxytamoxifen","4HT":"4-hydroxytamoxifen","TAM":"tamoxifen",
    "EST":"estradiol","E2":"estradiol","ES2":"estradiol","RAL":"raloxifene","BZA":"bazedoxifene",
}
KNOWN_LIGAND_TO_CHEMBL = {"OHT":"CHEMBL489","4HT":"CHEMBL489","TAM":"CHEMBL18"}

# Caches
CCD_CACHE_FILE    = "cache_ccd.json"
MECH_CACHE_FILE   = "cache_chembl_mech.json"
MOL_CACHE_FILE    = "cache_chembl_mol.json"

def _load_cache(path):
    try:
        with open(path,"r",encoding="utf-8") as fh: return json.load(fh)
    except Exception:
        return {}
def _save_cache(path, obj):
    try:
        with open(path,"w",encoding="utf-8") as fh: json.dump(obj, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

CCD_CACHE  = _load_cache(CCD_CACHE_FILE)   # code -> [smiles, inchikey] | None
MECH_CACHE = _load_cache(MECH_CACHE_FILE)  # chembl_id -> mechanisms[]
MOL_CACHE  = _load_cache(MOL_CACHE_FILE)   # "inchikey:..."|"name:..." -> molecules[]

def _sleep(s=0.25): time.sleep(s)

# ============================== PDB UTILS =================================

def iter_pdb_paths(root_or_file):
    p = root_or_file
    if os.path.isfile(p) and p.lower().endswith((".pdb",".ent")):
        yield p; return
    for dp,_,fns in os.walk(p):
        for fn in fns:
            if fn.lower().endswith((".pdb",".ent")):
                yield os.path.join(dp,fn)

def pdb_id_from_path(path):
    base = os.path.basename(path)
    m = re.match(r"([A-Za-z0-9]{4})\.", base)
    return (m.group(1).upper() if m else base[:4].upper())

def parse_pdb_compnd_chain_map(path):
    compnd=[]
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            if line.startswith("COMPND"):
                compnd.append(line[10:].rstrip())
    if not compnd: return {}
    joined=re.sub(r"\s+"," "," ".join(compnd))
    parts=[p.strip() for p in joined.split(';') if p.strip()]
    chain2mol={}
    cur_mol=None; cur_ch=[]
    for part in parts:
        if ':' not in part: continue
        k,v=part.split(':',1); k=k.strip().upper(); v=v.strip()
        if k=="MOL_ID":
            if cur_mol and cur_ch:
                for ch in cur_ch: chain2mol[ch]=cur_mol
            cur_mol=None; cur_ch=[]
        elif k=="MOLECULE":
            cur_mol=v.upper()
        elif k=="CHAIN":
            cur_ch=[c.strip() for c in v.split(',') if c.strip()]
    if cur_mol and cur_ch:
        for ch in cur_ch: chain2mol[ch]=cur_mol
    return chain2mol

def detect_er_alpha_chains_pdb(path):
    chain2mol=parse_pdb_compnd_chain_map(path)
    er=set()
    for ch,mol in chain2mol.items():
        if "ESTROGEN RECEPTOR" in mol and not any(b in mol for b in ER_BETA_NEGWORDS):
            er.add(ch)
    return er

def parse_pdb_seqres_len(path):
    declared={}; fallback=defaultdict(int)
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            if not line.startswith("SEQRES"): continue
            chain=line[11:12].strip()
            ser=line[7:10].strip()
            num=line[13:17].strip()
            if ser=="1":
                try: declared[chain]=int(num)
                except: pass
            toks=[t.strip().upper() for t in line[19:].split() if t.strip()]
            fallback[chain]+=sum(1 for t in toks if len(t)==3 and t.isalpha())
    out={}
    for ch in set(declared)|set(fallback):
        out[ch]=declared.get(ch,fallback.get(ch,0))
    return out

def parse_pdb_ligands(path):
    ligs=set(); ligs_by=defaultdict(set)
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            if not line.startswith("HETATM"): continue
            res=line[17:20].strip().upper()
            ch =line[21:22]
            if res and res not in NON_LIGAND:
                ligs.add(res)
                if ch: ligs_by[ch].add(res)
    return ligs,{k:sorted(v) for k,v in ligs_by.items()}

def choose_main_er_chain(er_chains, lengths, ligs_by):
    if not er_chains: return None
    cand=[ch for ch in er_chains if any(l in set(ligs_by.get(ch,[])) for l in ESTROGEN_LIGANDS)]
    if cand:
        cand.sort(key=lambda c:(-lengths.get(c,0), c)); return cand[0]
    if 'A' in er_chains: return 'A'
    return sorted(er_chains, key=lambda c:(-lengths.get(c,0), c))[0]

# ============================== NET HELPERS ===============================

def make_net_profile(net_mode):
    global HTTP_TIMEOUT, HTTP_RETRIES
    if net_mode=="full":
        HTTP_TIMEOUT=40; HTTP_RETRIES=4
    elif net_mode=="lite":
        HTTP_TIMEOUT=8;  HTTP_RETRIES=1
    else:  # off
        HTTP_TIMEOUT=3;  HTTP_RETRIES=0

def requests_get(url, params=None):
    if requests_get.offline: return None
    last=None
    tries=max(1, HTTP_RETRIES)
    for i in range(tries):
        try:
            r=requests.get(url, params=params, timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS)
            if r.ok: _sleep(0.2); return r
            last=r
        except requests.RequestException as e:
            last=e
        _sleep(0.6*(i+1))
    return last
requests_get.offline=False

# ============================== CCD / SMILES ==============================

def fetch_ccd_smiles(code):
    if code in CCD_CACHE:
        v=CCD_CACHE[code]; return (v or [None,None])
    url=CCD_URL.format(code=code)
    r=requests_get(url)
    if isinstance(r, requests.Response) and r.status_code==404:
        CCD_CACHE[code]=None; _save_cache(CCD_CACHE_FILE, CCD_CACHE); return None, None
    if isinstance(r, requests.Response) and r.ok:
        try:
            data=r.json()
            chem=data.get("chem_comp",{})
            smiles=chem.get("smiles") or chem.get("pdbx_smiles")
            ik     =chem.get("inchikey")
            if not smiles:
                desc=data.get("rcsb_chem_comp_descriptor",{})
                smiles=desc.get("smiles")
                ik    =ik or desc.get("inchi_key")
        except Exception:
            smiles,ik=None,None
        CCD_CACHE[code]=[smiles, ik] if smiles else None
        _save_cache(CCD_CACHE_FILE, CCD_CACHE)
        return smiles, ik
    CCD_CACHE[code]=None; _save_cache(CCD_CACHE_FILE, CCD_CACHE); return None, None

def fetch_ccd_smiles_with_alias(code):
    tried=set()
    for cand in [code, LIGAND_ALIASES.get(code)]:
        if not cand or cand in tried: continue
        smi,ik=fetch_ccd_smiles(cand)
        if smi: return smi,ik,cand
        tried.add(cand)
    return None, None, None

def inchikey_from_smiles_local(smiles):
    try:
        from rdkit import Chem
        m=Chem.MolFromSmiles(smiles)
        if not m: return None
        Chem.SanitizeMol(m)
        return Chem.MolToInchiKey(m)
    except Exception:
        return None

# =========================== ChEMBL (REST) ================================

def chembl_find_mols_by_inchikey(ik):
    if not ik: return []
    key=f"inchikey:{ik}"
    if key in MOL_CACHE: return MOL_CACHE[key]
    r=requests_get(CHEMBL_MOLS, params={"inchikey":ik, "format":"json"})
    if isinstance(r, requests.Response) and r.ok:
        res=r.json().get("molecules",[])
        MOL_CACHE[key]=res; _save_cache(MOL_CACHE_FILE, MOL_CACHE)
        return res
    MOL_CACHE[key]=[]; _save_cache(MOL_CACHE_FILE, MOL_CACHE); return []

def chembl_find_by_name(name):
    if not name: return []
    key=f"name:{name.lower()}"
    if key in MOL_CACHE: return MOL_CACHE[key]
    # REST по имени делаем шире: pref_name__icontains + molecule_synonyms__synonyms__icontains
    r=requests_get(CHEMBL_MOLS, params={"pref_name__icontains":name, "format":"json", "limit":200})
    items=[]
    if isinstance(r, requests.Response) and r.ok:
        items += r.json().get("molecules",[])
    # син-матч (вторая попытка)
    if not items:
        r2=requests_get(CHEMBL_MOLS, params={"molecule_synonyms__synonyms__icontains":name, "format":"json", "limit":200})
        if isinstance(r2, requests.Response) and r2.ok:
            items += r2.json().get("molecules",[])
    MOL_CACHE[key]=items; _save_cache(MOL_CACHE_FILE, MOL_CACHE); return items

def chembl_mechanisms_for_mol(chembl_id):
    if not chembl_id: return []
    if chembl_id in MECH_CACHE: return MECH_CACHE[chembl_id]
    r=requests_get(CHEMBL_MECH, params={"molecule_chembl_id":chembl_id, "format":"json"})
    mechs=[]
    if isinstance(r, requests.Response) and r.ok:
        mechs=r.json().get("mechanisms",[])
    MECH_CACHE[chembl_id]=mechs; _save_cache(MECH_CACHE_FILE, MECH_CACHE); return mechs

def classify_action_from_mechs(mechs):
    txt=" | ".join([(m.get('action_type') or "")+" "+(m.get('mechanism_of_action') or "") for m in mechs]).lower()
    if "partial agonist" in txt: return "partial agonist"
    if "inverse agonist"  in txt: return "inverse agonist"
    if "full agonist" in txt or "agonist" in txt: return "agonist"
    if "antagonist" in txt and "inverse" not in txt: return "antagonist"
    if "serm" in txt or "selective estrogen receptor modulator" in txt or "modulator" in txt: return "modulator"
    return "unknown"

def _to_nm(val, units):
    if val is None: return None
    try: v=float(val)
    except: return None
    u=(units or "").lower()
    if not u or u in ("nm","nanomolar"): return v
    if u in ("pm","picomolar"): return v*0.001
    if u in ("um","micromolar"): return v*1000.0
    if u in ("mm","millimolar"): return v*1_000_000.0
    if u in ("m","molar"): return v*1e9
    return None

def _good_rel(rel):
    rel=(rel or "").strip()
    return rel in ("","=","<","<=")

def chembl_bioactivity_block_any_er(chembl_id):
    res={t:[] for t in set(BIO_TYPES_PRIMARY)|set(BIO_TYPES_EXTRA)}
    r=requests_get(CHEMBL_ACT, params={"molecule_chembl_id":chembl_id, "format":"json", "limit":1000})
    if not (isinstance(r, requests.Response) and r.ok): return res
    rows=r.json().get("activities",[])
    for a in rows:
        pref=(a.get("target_pref_name") or "").lower()
        if not any(k in pref for k in ER_NAME_KEYWORDS): continue
        typ=(a.get("standard_type") or "").upper()
        if typ not in res: continue
        v=_to_nm(a.get("standard_value"), a.get("standard_units"))
        if v is None: continue
        if not _good_rel(a.get("standard_relation")): continue
        res[typ].append(v)
    if not any(res.values()):
        tmp=[]
        for a in rows:
            pref=(a.get("target_pref_name") or "").lower()
            if not any(k in pref for k in ER_NAME_KEYWORDS): continue
            p=a.get("pchembl_value")
            if not p: continue
            try: p=float(p)
            except: continue
            tmp.append((10**(-p))*1e9)
        if tmp: res["IC50"]=res.get("IC50",[])+tmp
    return res

# ================================ MAIN ====================================

def load_overrides():
    p="ligand_overrides.csv"
    if not os.path.exists(p): return {}
    df=pd.read_csv(p)
    m={}
    for _,r in df.iterrows():
        code=str(r['lig_code']).strip().upper()
        smi =str(r.get('smiles') or '').strip()
        ik  =str(r.get('inchikey') or '').strip() or None
        act =str(r.get('er_action') or '').strip() or None
        if code and smi: m[code]=(smi, ik, act)
    return m

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("--net", choices=["full","lite","off"], default="full")
    parser.add_argument("--checkpoint", type=int, default=25)
    args=parser.parse_args()

    # профили сети
    make_net_profile(args.net)
    requests_get.offline = (args.net=="off")

    # ---------- 1) Scan PDB ----------
    per_file=[]
    paths=list(iter_pdb_paths(args.src))
    for path in tqdm(paths, desc="Scanning PDBs"):
        pid=pdb_id_from_path(path)
        er_ch=detect_er_alpha_chains_pdb(path)
        if not er_ch: continue
        lengths=parse_pdb_seqres_len(path)
        ligs_global, ligs_by=parse_pdb_ligands(path)
        main_ch=choose_main_er_chain(er_ch, lengths, ligs_by)
        if not main_ch: continue
        seq_len=lengths.get(main_ch,0)
        comments=[]
        if "ZN" in ligs_global: comments.append("dna based")
        if seq_len<50: comments.append("inconsistent structure")
        per_file.append({
            "pdb_id":pid,"path":path,"format":"pdb","chain":main_ch,"seqres_len":seq_len,
            "ligands_global":";".join(sorted(ligs_global)),
            "ligands_er_chain":";".join(sorted(ligs_by.get(main_ch,[]))),
            "comments":"; ".join(comments)
        })
    df_files=pd.DataFrame(per_file).drop_duplicates()
    if df_files.empty:
        print("No ERα chains found in given PDBs."); return
    df_files.to_csv("er_files_summary_alpha.csv", index=False)

    # ---------- 2) Ligands annotate ----------
    lig_codes=sorted(set(";".join(df_files['ligands_er_chain']).split(";"))-{""})
    df_ligs=pd.DataFrame({"lig_code":lig_codes})

    overrides=load_overrides()
    smiles, inchikeys, chembl_ids, actions = [], [], [], []
    ccd_missing=[]; debug=[]

    ck_every=max(1, args.checkpoint)
    for i,code in enumerate(tqdm(df_ligs['lig_code'], desc="Annotating ligands"), start=1):
        code_up=str(code).upper()
        # печать текущего для наглядности
        print(f"[{i}/{len(df_ligs)}] {code_up}")

        smi=ik=act_override=None
        if code_up in overrides:
            smi,ik,act_override=overrides[code_up]; debug.append(f"{code_up}: override")

        if not smi and args.net!="off":
            smi,ik,used=fetch_ccd_smiles_with_alias(code_up)
            if smi: debug.append(f"{code_up}: CCD via {used}")
            else:   debug.append(f"{code_up}: CCD miss")

        if smi and not ik:
            ik_local=inchikey_from_smiles_local(smi)
            if ik_local: ik=ik_local; debug.append(f"{code_up}: IK via RDKit")
            else:        debug.append(f"{code_up}: IK RDKit miss")

        if not smi:
            ccd_missing.append(code_up)

        smiles.append(smi); inchikeys.append(ik)

        cid=None
        if args.net!="off":
            # inchikey -> chembl
            mols=chembl_find_mols_by_inchikey(ik) if ik else []
            if mols:
                cid=(mols[0].get('molecule_chembl_id')
                     or (mols[0].get('molecule') or {}).get('molecule_chembl_id'))
            # name fallback
            if not cid:
                qname=LIGAND_NAME_FALLBACK.get(code_up) or code_up
                hits=chembl_find_by_name(qname)
                pick=None
                if code_up in {"OHT","4HT"} and hits:
                    for h in hits:
                        hid=h.get('molecule_chembl_id') or (h.get('molecule') or {}).get('molecule_chembl_id')
                        if hid in ("CHEMBL489","CHEMBL486"): pick=h; break
                if not pick and hits: pick=hits[0]
                if pick:
                    cid=(pick.get('molecule_chembl_id')
                         or (pick.get('molecule') or {}).get('molecule_chembl_id'))
                    debug.append(f"{code_up}: ChEMBL via name '{qname}' -> {cid}")
        if not cid and code_up in KNOWN_LIGAND_TO_CHEMBL:
            cid=KNOWN_LIGAND_TO_CHEMBL[code_up]; debug.append(f"{code_up}: KNOWN map -> {cid}")

        chembl_ids.append(cid)

        # action
        if act_override:
            act=act_override
        elif args.net!="off" and cid:
            mechs=chembl_mechanisms_for_mol(cid)
            act=classify_action_from_mechs(mechs)
            if act=="unknown" and code_up in KNOWN_SERMS: act="modulator (SERM)"
        else:
            act="modulator (SERM)" if code_up in KNOWN_SERMS else "unknown"
        actions.append(act or "unknown")

        # чекпоинт
        if i%ck_every==0 or i==len(df_ligs):
            tmp=pd.DataFrame({"lig_code":df_ligs['lig_code'][:i],
                              "smiles":smiles[:i],
                              "inchikey":inchikeys[:i],
                              "chembl_id":chembl_ids[:i],
                              "er_action":actions[:i]})
            tmp.to_csv("ligands_master_alpha.tmp.csv", index=False)

    with open("ccd_debug.log","w",encoding="utf-8") as fh:
        fh.write("\n".join(debug))
    if ccd_missing:
        pd.Series(sorted(set(ccd_missing)), name="lig_code").to_csv("ccd_missing.csv", index=False)

    df_ligs['smiles']=smiles
    df_ligs['inchikey']=inchikeys
    df_ligs['chembl_id']=chembl_ids
    df_ligs['er_action']=actions
    df_ligs['er_action']=df_ligs['er_action'].fillna('unknown')
    df_ligs.to_csv("ligands_master_alpha.csv", index=False)

    # ---------- 3) Bioactivity (name-match ER) ----------
    bio=[]
    if args.net=="off":
        print("Network OFF: skipping bioactivity.")
    else:
        for _,row in df_ligs.iterrows():
            cid=row.get('chembl_id')
            if not cid or not isinstance(cid,str): continue
            lig_code=str(row['lig_code'])
            acts=chembl_bioactivity_block_any_er(cid)
            for typ,vals in acts.items():
                if not vals: continue
                s=pd.Series(sorted(vals))
                bio.append({
                    "chembl_id":cid,"lig_code":lig_code,
                    "target":"ER (name match)","target_chembl_id":"",
                    "type":typ,"n":int(s.count()),
                    "median_nM":float(s.median()),
                    "min_nM":float(s.min()),"max_nM":float(s.max())
                })
    pd.DataFrame(bio).to_csv("ligand_bioactivity.csv", index=False)

    print("Done:")
    print(" - er_files_summary_alpha.csv")
    print(" - ligands_master_alpha.csv")
    print(" - ligand_bioactivity.csv")
    if os.path.exists("ligands_master_alpha.tmp.csv"): print(" - ligands_master_alpha.tmp.csv")
    if os.path.exists("ccd_missing.csv"): print(" - ccd_missing.csv")
    if os.path.exists("ccd_debug.log"):   print(" - ccd_debug.log")

if __name__=="__main__":
    main()
