import pandas as pd
import numpy as np
import re
import requests
import json
from urllib.parse import urljoin

# --- Configuration ---
# *** Make sure to update this line to your actual Excel file path ***
INPUT_FILE = 'esr1_structures_sample.xlsx'
OUTPUT_FILE = 'enriched_project_data.xlsx'
# Target is Estrogen Receptor alpha
ESR1_TARGET_CHEMBL_ID = 'CHEMBL206' 
CHEMBL_API_BASE = 'https://www.ebi.ac.uk/chembl/api/data/'

# --- API Function using requests ---

def get_chembl_data(pdb_ligand_code, target_chembl_id):
    """Retrieves structure and bioactivity data from ChEMBL REST API."""
    
    results = {
        'CHEMBL_ID': 'N/A',
        'SMILES': 'N/A',
        'InChIKey': 'N/A',
        'Action_Type': 'N/A (check MOA)',
        'pChEMBL_Value': np.nan,
        'Standard_Type': 'N/A'
    }

    # 1. Search for the molecule by PDB code
    mol_url = urljoin(CHEMBL_API_BASE, 'molecule.json')
    params = {
        'molecule_structures__pdb_code__iexact': pdb_ligand_code,
        'only': 'molecule_chembl_id,canonical_smiles,inchi_key'
    }
    
    try:
        mol_response = requests.get(mol_url, params=params, timeout=10)
        mol_response.raise_for_status() 
        mol_data = mol_response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Error (Molecule search) for {pdb_ligand_code}: {e}")
        return results

    if not mol_data.get('molecules'):
        print(f"⚠️ No ChEMBL molecule found for PDB code: {pdb_ligand_code}")
        return results

    chembl_data = mol_data['molecules'][0]
    chembl_id = chembl_data.get('molecule_chembl_id')
    results.update({
        'CHEMBL_ID': chembl_id,
        'SMILES': chembl_data.get('canonical_smiles', 'N/A'),
        'InChIKey': chembl_data.get('inchi_key', 'N/A')
    })
    
    if not chembl_id:
        return results

    # 2. Search for bioactivity data
    activity_url = urljoin(CHEMBL_API_BASE, 'activity.json')
    activity_params = {
        'target_chembl_id': target_chembl_id,
        'molecule_chembl_id': chembl_id,
        'standard_type__in': 'IC50,Ki,EC50',
        'only': 'standard_type,pchembl_value',
        'limit': 100 
    }
    
    try:
        act_response = requests.get(activity_url, params=activity_params, timeout=10)
        act_response.raise_for_status()
        act_data = act_response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Error (Activity search) for {chembl_id}: {e}")
        return results

    # Process bioactivity data to find the best pChEMBL value
    best_pchembl = -1
    best_activity = {}
    
    for activity in act_data.get('activities', []):
        try:
            pchembl = float(activity.get('pchembl_value'))
            if pchembl > best_pchembl:
                best_pchembl = pchembl
                best_activity = activity
        except (TypeError, ValueError):
            continue
            
    if best_pchembl > 0:
        results['pChEMBL_Value'] = best_pchembl
        results['Standard_Type'] = best_activity.get('standard_type', 'N/A')
    
    return results

# --- Main Processing Function ---

def process_data(input_path, output_path, target_chembl_id):
    """Main function to run the data enrichment pipeline."""
    
    try:
        # Assuming your input is an Excel file
        df = pd.read_excel(input_path)
    except Exception as e:
        print(f"❌ Error reading input file: {e}")
        return
        
    # 1. Extract Unique Ligand Codes (Robust Splitting)
    all_ligands = df['ligands'].dropna().astype(str).str.split(r'[;,\s]+').explode().str.strip().unique()
    
    # Exclude common cofactors, ions, and solvents
    EXCLUDE_LIGANDS = {'HOH', 'SO4', 'CL', 'NA', 'EDO', 'GOL', '', 'CCS', 'AU'} 
    unique_ligand_codes = [
        ligand for ligand in all_ligands 
        if ligand not in EXCLUDE_LIGANDS and len(ligand) == 3
    ]
    
    print(f"Found unique ligands to process: {unique_ligand_codes}")

    # 2. Query ChEMBL for each unique 3-letter ligand
    ligand_data_map = {}
    for code in unique_ligand_codes:
        print(f"Processing ligand code: {code}...")
        ligand_data_map[code] = get_chembl_data(code, target_chembl_id)
        
    print("\nChEMBL data retrieval complete.")

    # 3. Apply the new data back to the main DataFrame
    new_columns = ['CHEMBL_ID', 'SMILES', 'InChIKey', 'Action_Type', 'pChEMBL_Value', 'Standard_Type']
    for col in new_columns:
        df[col] = np.nan

    for index, row in df.iterrows():
        ligands_str = str(row['ligands'])
        if ligands_str and ligands_str != 'nan':
            # Identify valid ligands in the current row
            row_ligands = [
                l for l in re.split(r'[;,\s]+', ligands_str) 
                if l not in EXCLUDE_LIGANDS and len(l) == 3
            ]
            
            if row_ligands:
                main_ligand_code = row_ligands[0]
                data = ligand_data_map.get(main_ligand_code, {})
                
                # Update the row with the fetched data
                for col in new_columns:
                    df.loc[index, col] = data.get(col, np.nan)

    # 4. Save the enriched data
    df.to_excel(output_path, index=False)
    print(f"\n✅ Data enrichment complete! Results saved to: {output_path}")
    
if __name__ == '__main__':
    process_data(INPUT_FILE, OUTPUT_FILE, ESR1_TARGET_CHEMBL_ID)