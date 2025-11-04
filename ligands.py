from rdkit import Chem
import os


pdb_dir = "data set/cif to pdb"
out_sdf = "ligands.sdf"

ligands = []
names = []

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
        # Keep small molecules but skip peptides, waters, ions
        if 5 < heavy < 80:
            frag.SetProp("_Name", pdb_file.replace(".pdb", ""))
            ligands.append(frag)
            names.append(pdb_file.replace(".pdb", ""))


writer = Chem.SDWriter(out_sdf)
for mol in ligands:
    writer.write(mol)
writer.close()

print(f"Extracted {len(ligands)} ligands from {len(os.listdir(pdb_dir))} PDB files.")