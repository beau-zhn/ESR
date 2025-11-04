from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


suppl = Chem.SDMolSupplier("ligands.sdf")
mols = [m for m in suppl if m is not None]
names = [m.GetProp("_Name") if m.HasProp("_Name") else f"Ligand_{i}" for i,m in enumerate(mols)]

fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]

n = len(fps)
sim = np.zeros((n,n))
for i in range(n):
    for j in range(i,n):
        s = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        sim[i,j] = sim[j,i] = s

sim_df = pd.DataFrame(sim, index=names, columns=names)
sim_df.to_csv("tanimoto_similarity_matrix.csv")

plt.figure(figsize=(10,8))
sns.heatmap(sim_df, cmap="viridis")
plt.title("Ligand Tanimoto Similarity Matrix (ECFP4)")
plt.tight_layout()
plt.show()