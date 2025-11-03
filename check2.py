import requests
import time
import pandas as pd

class ChemblLigandFinder:
    def __init__(self):
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.er_targets = ["CHEMBL206", "CHEMBL207", "CHEMBL4080"]  # ESR1, ESR2, and ER complex

    def safe_get(self, url, params=None):
        """Perform safe GET with retries."""
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException:
            return {}

    def get_bioactivities(self, chembl_id: str):
        """Retrieve bioactivity data for given molecule."""
        activities = []
        for target in self.er_targets:
            url = f"{self.base_url}/activity.json"
            params = {"molecule_chembl_id": chembl_id, "target_chembl_id": target, "limit": 1000}
            data = self.safe_get(url, params)
            activities.extend(data.get("activities", []))
            time.sleep(0.2)

        # If none found — retry without target filter
        if not activities:
            url = f"{self.base_url}/activity.json"
            params = {"molecule_chembl_id": chembl_id, "limit": 1000}
            data = self.safe_get(url, params)
            activities.extend(data.get("activities", []))
        return activities

    def get_mechanism(self, chembl_id: str):
        """Retrieve mechanism of action."""
        url = f"{self.base_url}/mechanism.json"
        params = {"molecule_chembl_id": chembl_id}
        data = self.safe_get(url, params)
        return data.get("mechanisms", [])

    def get_indications(self, chembl_id: str):
        """Retrieve clinical indication data."""
        url = f"{self.base_url}/drug_indication.json"
        params = {"molecule_chembl_id": chembl_id, "limit": 100}
        data = self.safe_get(url, params)
        return [x.get("mesh_heading") for x in data.get("drug_indications", []) if x.get("mesh_heading")]

    def determine_er_action_type(self, activities, mechanisms, indications):
        """Infer action type using keywords."""
        keywords = {
            "agonist": ["agonist", "activator", "stimulator"],
            "antagonist": ["antagonist", "inhibitor", "blocker"],
            "modulator": ["modulator", "partial agonist"]
        }
        text = " ".join(
            str(v).lower()
            for v in [activities, mechanisms, indications]
        )
        for k, kws in keywords.items():
            if any(w in text for w in kws):
                return k.capitalize()
        return "Unknown"

    def process_ligand(self, lig_code, smiles, inchikey, chembl_id):
        """Main data integration for one ligand."""
        print(f"\nProcessing {lig_code} ({chembl_id})...")

        activities = self.get_bioactivities(chembl_id)
        mechanisms = self.get_mechanism(chembl_id)
        indications = self.get_indications(chembl_id)

        action_type = self.determine_er_action_type(activities, mechanisms, indications)

        ic50_values = [a.get("standard_value") + " " + (a.get("standard_units") or "")
                       for a in activities if a.get("standard_type") == "IC50" and a.get("standard_value")]
        ki_values = [a.get("standard_value") + " " + (a.get("standard_units") or "")
                     for a in activities if a.get("standard_type") == "Ki" and a.get("standard_value")]

        result = {
            "lig_code": lig_code,
            "chembl_id": chembl_id,
            "er_action_type": action_type,
            "ic50_values": "; ".join(ic50_values[:5]),
            "ki_values": "; ".join(ki_values[:5]),
            "num_activities": len(activities),
            "indications": "; ".join(indications) if indications else "N/A"
        }

        print(f"  Found {len(activities)} activities, {len(mechanisms)} mechanisms, {len(indications)} indications")
        print(f"  Action type: {action_type}")

        return result


if __name__ == "__main__":
    ligands = [
        ("E2", "", "", "CHEMBL278512"),  # Estradiol
        ("CL", "", "", "CHEMBL1216078"), # Chloride ion (may have limited ChEMBL data)
        ("0CZ", "", "", "CHEMBL6329")    # Trifluoromethyl compound (example)
    ]

    finder = ChemblLigandFinder()
    results = []

    for lig_code, smiles, inchikey, chembl_id in ligands:
        results.append(finder.process_ligand(lig_code, smiles, inchikey, chembl_id))
        time.sleep(0.5)

    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("FINAL RESULTS:")
    print("="*80)
    print(df)
    df.to_csv("ligands_with_bioactivity.csv", index=False)
    print("\nDetailed results saved to 'ligands_with_bioactivity.csv'")
