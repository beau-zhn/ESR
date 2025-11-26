# How to Navigate

## We organised several folders with code we sed and theur results. 




# ESR1 Structural Diversity and Redundancy Assessment

## Objective
Analyze the dataset of **474 estrogen receptor α (ESR1) crystal structures** to:
- Determine structural diversity  
- Identify truly unique conformations  
- Assess whether additional structures provide new biological insights  

Validation will be performed against a comparison set of **37 ESR2 structures**.  

---

## Target Data
- **474 ESR1 crystal structures** from the PDB  
- Structures bound to diverse ligands (agonists, antagonists, SERMs)  
- Multiple crystal forms and experimental conditions  
- **37 ESR2 structures** for cross-receptor comparison  

---

## Approach
1. Cluster ESR1 structures by conformational similarity and ligand types  
2. Identify structural outliers and unique conformational states  
3. Calculate information content and redundancy metrics  
4. Create a minimal representative structural set  

---

## Analysis Components

### 1. Conformational Clustering Analysis
- All-atom **RMSD matrix calculation**  
- **Hierarchical clustering** by structural similarity  
- **Principal component analysis (PCA)** of conformational space  
- Identification of conformational clusters  

### 2. Ligand–Structure Relationship
- **Ligand chemical similarity analysis**  
  - Tanimoto similarity matrices (ECFP/Morgan fingerprints via RDKit)  
- **Structure–activity mapping**  
  - Overlay ligand similarity with RMSD matrices  
  - Map IC50/binding affinity data (ChEMBL/literature) onto clusters  
- **Pocket analysis**  
  - P2RANK for cavity detection/volume calculation  
  - SURFNET / 3V for shape complementarity  
  - Pocket descriptors (ASA, depth, electrostatics)  
  - PCA on descriptors to identify ligand-class binding modes  
- **Selectivity determinants**  
  - ESR1 vs ESR2 comparison  
  - Map selectivity-determining mutations  
  - Residue-level conformational differences  

### 3. Redundancy Assessment
- **Information theory metrics**  
  - Shannon entropy of conformational states  
  - Mutual information between ligand types and conformations  
  - Information gain of new structures  
- **Diminishing returns analysis**  
  - Cumulative diversity vs. number of structures  
  - Marginal information gain curves  
  - Identify saturation point of redundancy  
- **Subset selection**  
  - Greedy/maximally diverse selection  
  - Set cover algorithms for state coverage  
  - K-medoids for representative structures  

### 4. Quality Control
- **Resolution/refinement assessment**  
  - Parse R-factors, R-free, resolution from PDB metadata  
  - Correlate with structural outliers  
- **B-factor and mobility analysis**  
  - Identify flexible vs. rigid regions  
  - Distinguish real differences from noise  
- **Crystal packing effects**  
  - Contact area calculation  
  - Compare NMR and crystal-derived conformations  

---

## Expected Outcomes
1. Definitive **ESR1 conformational landscape map**  
2. **Minimal essential structural set** representing full diversity  
3. Guidelines for **future ESR1 structural studies**  
4. Deeper understanding of **ligand-induced conformational changes**  

---

## Skills Developed
- Large-scale **structural dataset management**  
- **Clustering & dimensionality reduction** techniques  
- Structure–activity relationship (SAR) analysis  
- **Information theory** applications in structural biology  
- Data mining & pattern recognition  
- Scientific **visualization & interpretation**  

