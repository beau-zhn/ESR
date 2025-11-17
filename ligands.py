from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import os
from collections import defaultdict

# Configuration
pdb_dir = "data set/pdb beta files"
out_sdf = "ligands.sdf"
out_log = "ligand_extraction_log.txt"

# Common crystallization agents and buffers to exclude
EXCLUDE_SMILES = {
    'O',           # Water
    'C(C)O',       # Ethanol
    'C(CO)O',      # Ethylene glycol
    'OCCOCCOCCOC', # PEG fragments
    '[Na+]', '[K+]', '[Cl-]', '[Mg+2]', '[Ca+2]', '[Zn+2]',  # Common ions
    'C(=O)O',      # Formate
    'CC(=O)O',     # Acetate
    'OS(=O)(=O)O', # Sulfate
}

# Common ligand names in ESR1 structures (for verification)
KNOWN_ESR_LIGANDS = {
    'EST', 'E2', 'EDC', 'OHT', '4HT', 'RAL', 'TAM',  # Common codes
    'DES', 'HEX', 'EQU', 'GW5'
}

ligands = []
names = []
ligand_info = defaultdict(list)
extraction_log = []

print("Starting ligand extraction...")
extraction_log.append("=== Ligand Extraction Log ===\n")

pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]

for idx, pdb_file in enumerate(pdb_files, 1):
    pdb_path = os.path.join(pdb_dir, pdb_file)
    pdb_id = pdb_file.replace(".pdb", "")
    
    if idx % 50 == 0:
        print(f"Processing {idx}/{len(pdb_files)}...")
    
    try:
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
        if mol is None:
            extraction_log.append(f"{pdb_id}: Failed to parse PDB\n")
            continue
        
        # Get molecular fragments
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        
        structure_ligands = []
        for frag_idx, frag in enumerate(frags):
            heavy = frag.GetNumHeavyAtoms()
            
            # First filter: size range
            if not (5 < heavy < 80):
                continue
            
            # Try to sanitize the molecule
            try:
                Chem.SanitizeMol(frag)
            except:
                extraction_log.append(f"{pdb_id}: Fragment {frag_idx} failed sanitization\n")
                continue
            
            # Get SMILES for filtering
            try:
                smiles = Chem.MolToSmiles(frag)
            except:
                continue
            
            # Exclude common crystallization agents and buffers
            if smiles in EXCLUDE_SMILES:
                extraction_log.append(f"{pdb_id}: Excluded common buffer/salt: {smiles}\n")
                continue
            
            # Additional filters
            num_bonds = frag.GetNumBonds()
            num_atoms = frag.GetNumAtoms()
            
            # Skip if too simple (likely fragments or ions)
            if num_bonds < 5:
                continue
            
            # Calculate some basic descriptors for filtering
            try:
                mol_wt = Descriptors.MolWt(frag)
                num_rings = Descriptors.RingCount(frag)
                
                # Drug-like properties for ESR1 ligands (fairly permissive)
                if mol_wt < 100 or mol_wt > 800:
                    continue
                    
            except:
                continue
            
            # Keep this ligand
            ligand_name = f"{pdb_id}"
            if len(structure_ligands) > 0:
                ligand_name += f"_lig{len(structure_ligands)+1}"
            
            frag.SetProp("_Name", ligand_name)
            frag.SetProp("PDB_ID", pdb_id)
            frag.SetProp("HeavyAtoms", str(heavy))
            frag.SetProp("MolWt", f"{mol_wt:.2f}")
            frag.SetProp("SMILES", smiles)
            
            ligands.append(frag)
            names.append(ligand_name)
            structure_ligands.append(ligand_name)
            
            ligand_info[pdb_id].append({
                'name': ligand_name,
                'heavy_atoms': heavy,
                'mol_wt': mol_wt,
                'smiles': smiles
            })
        
        if structure_ligands:
            extraction_log.append(f"{pdb_id}: Extracted {len(structure_ligands)} ligand(s)\n")
        else:
            extraction_log.append(f"{pdb_id}: No ligands found\n")
            
    except Exception as e:
        extraction_log.append(f"{pdb_id}: Error - {str(e)}\n")
        continue

# Write ligands to SDF file
print(f"\nWriting {len(ligands)} ligands to {out_sdf}...")
writer = Chem.SDWriter(out_sdf)
for mol in ligands:
    writer.write(mol)
writer.close()

# Write extraction log
with open(out_log, 'w') as f:
    f.writelines(extraction_log)

# Print summary statistics
print(f"\n=== Extraction Summary ===")
print(f"Total PDB files processed: {len(pdb_files)}")
print(f"Total ligands extracted: {len(ligands)}")
print(f"Structures with ligands: {len(ligand_info)}")
print(f"Structures without ligands: {len(pdb_files) - len(ligand_info)}")
print(f"\nLigands per structure statistics:")
ligs_per_struct = [len(v) for v in ligand_info.values()]
if ligs_per_struct:
    print(f"  Mean: {sum(ligs_per_struct)/len(ligs_per_struct):.2f}")
    print(f"  Min: {min(ligs_per_struct)}")
    print(f"  Max: {max(ligs_per_struct)}")

# Save a visual sample of extracted ligands
print(f"\nGenerating sample visualization...")
sample_size = min(20, len(ligands))
img = Draw.MolsToGridImage(
    ligands[:sample_size], 
    molsPerRow=5, 
    legends=[m.GetProp("_Name") for m in ligands[:sample_size]],
    subImgSize=(200, 200)
)
img.save("ligand_sample.png")
print(f"Saved sample visualization to ligand_sample.png")

print(f"\nLog saved to {out_log}")
print("Extraction complete!")