import os
import numpy as np
from Bio.PDB import PDBParser
import pandas as pd

# === FOLDER WITH ALL PDB FILES ===
pdb_folder = r"C:\Users\a1611\Desktop\BioInfo\ESR_2"

parser = PDBParser(QUIET=True)

results = []

POCKET_CUTOFF = 8.0  # Å, radius around ligand centroid

for pdb_file in os.listdir(pdb_folder):
    if not pdb_file.lower().endswith(".pdb"):
        continue

    filepath = os.path.join(pdb_folder, pdb_file)
    structure = parser.get_structure("rec", filepath)
    model = structure[0]

    # ---- ligand atoms (HETATM, not water) ----
    ligand_atoms = []
    for chain in model:
        for res in chain:
            if res.id[0] != ' ' and res.resname not in ["HOH"]:
                for atom in res:
                    ligand_atoms.append(atom.get_coord())

    # skip apo structures (no ligand)
    if len(ligand_atoms) == 0:
        continue

    ligand_centroid = np.mean(np.array(ligand_atoms), axis=0)

    # ---- Cα within cutoff: pocket ----
    pocket_coords = []

    for chain in model:
        for res in chain:
            if res.id[0] == ' ' and 'CA' in res:
                ca_coord = res['CA'].get_coord()
                dist = np.linalg.norm(ca_coord - ligand_centroid)
                if dist <= POCKET_CUTOFF:
                    pocket_coords.append(ca_coord)

    if len(pocket_coords) == 0:
        continue

    pocket_coords = np.array(pocket_coords)
    distances = np.linalg.norm(pocket_coords - ligand_centroid, axis=1)

    pocket_radius = float(np.mean(distances))
    compactness = float(np.std(distances))
    openness = float(np.max(distances))

    results.append({
        "pdb": pdb_file,
        "radius": pocket_radius,
        "compactness": compactness,
        "openness": openness,
        "n_pocket_res": len(pocket_coords)
    })

df = pd.DataFrame(results)
df.to_csv("pocket_descriptors_ESR2_dynamic.csv", index=False)

print("Total processed structures:", len(df))
print(df.head())
