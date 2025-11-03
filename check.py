import requests
from chembl_webresource_client.new_client import new_client

pdb_id = "1A52"

# 1️⃣ Get entry metadata to find ligand entity IDs
entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
entry = requests.get(entry_url).json()

ligand_entities = entry["rcsb_entry_container_identifiers"].get("nonpolymer_entity_ids", [])

if not ligand_entities:
    print("❌ No ligands found in this PDB entry.")
else:
    print(f"✅ Ligand entity IDs found: {ligand_entities}")

    for lig_id in ligand_entities:
        # 2️⃣ Fetch chemical info for each ligand
        ligand_url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{lig_id}"
        ligand_data = requests.get(ligand_url).json()

        chem_name = ligand_data["rcsb_nonpolymer_entity_container_identifiers"]["name"]
        smiles = ligand_data["rcsb_nonpolymer_entity_container_identifiers"].get("smiles", None)

        print(f"\nLigand name: {chem_name}")
        if smiles:
            print(f"SMILES: {smiles}")
        else:
            print("No SMILES available from RCSB.")

        # 3️⃣ Look up this molecule in ChEMBL
        molecule = new_client.molecule
        activity = new_client.activity
        mechanism = new_client.mechanism

        results = []
        if smiles:
            results = molecule.filter(molecule_structures__canonical_smiles__exact=smiles)
        if not results:
            results = molecule.filter(pref_name__icontains=chem_name)

        if results:
            mol = results[0]
            chembl_id = mol['molecule_chembl_id']
            print(f"🔹 ChEMBL ID: {chembl_id}")

            # 4️⃣ Mechanism of action
            mech_data = mechanism.filter(molecule_chembl_id=chembl_id)
            for m in mech_data:
                print(f"Target: {m['target_name']}")
                print(f"Action type: {m['mechanism_of_action']}")

            # 5️⃣ Bioactivity values
            acts = activity.filter(molecule_chembl_id=chembl_id).only(
                ['standard_type', 'standard_value', 'standard_units']
            )
            print("\nBioactivity data (first few entries):")
            for a in acts[:5]:
                print(f"{a['standard_type']}: {a['standard_value']} {a['standard_units']}")
        else:
            print("❌ Ligand not found in ChEMBL.")
