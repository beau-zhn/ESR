from rdkit import Chem
import os


pdb_dir = "data set/cif to pdb/"
out_sdf = "ligands.sdf"

unique_ligands = {}
for pdb_file in os.listdir(pdb_dir):
    if not pdb_file.endswith(".pdb"):
        continue
    pdb_path = os.path.join(pdb_dir, pdb_file)
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
    if mol is None:
        continue

    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    for frag in frags:
        heavy = frag.GetNumHeavyAtoms()
        if 5 < heavy < 80:  # small molecule filter
            pdb_id = pdb_file.replace(".pdb", "")
            # Keep only first ligand per PDB
            if pdb_id not in unique_ligands:
                unique_ligands[pdb_id] = frag
            break  # stop after first ligand

# Write them to SDF
writer = Chem.SDWriter(out_sdf)
for pdb_id, mol in unique_ligands.items():
    mol.SetProp("_Name", pdb_id)
    writer.write(mol)
writer.close()

print(f"Saved {len(unique_ligands)} unique ligands.")
