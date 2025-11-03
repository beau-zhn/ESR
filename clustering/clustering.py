import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. Define the Ligand Data and Structural Input (SMILES) ---
# You MUST include a valid SMILES string for every unique ligand_code in your CSV.
LIGAND_SMILES = {
    '0CZ': 'c1cc(ccc1C(c2ccc(cc2)O)(C(F)(F)F)C(F)(F)F)O',
    '0D1': 'c1cc(ccc1C(=C(Cl)Cl)c2ccc(cc2)O)O',
    '0L8': 'c1cc(ccc1c2c(cc(cc2Br)CO)c3ccoc3)O',
    '15Q': 'c1cc(c(cc1c2ccc(c(c2Cl)C=NO)O)F)O',
    '17M': 'CC12CCc3c4ccc(cc4ccc3C1CCC2(C)O)O',
    '1GJ': 'c1ccc(cc1)Cn2c(c3cccc(c3n2)C(F)(F)F)c4ccc(cc4O)O',
    '1GM': 'CCn1c(c2cccc(c2n1)C(F)(F)F)c3ccc(cc3O)O',
    '1GQ': 'CCCCn1c2c(cccc2C(F)(F)F)c(n1)c3ccc(cc3O)O',
    '1GR': 'CC(C)Cn1c2c(cccc2C(F)(F)F)c(n1)c3ccc(cc3O)O',
    '1GS': 'CC(C)Cn1c(c2cccc(c2n1)C(F)(F)F)c3ccc(cc3O)O',
    '1GT': 'C=CCCn1c2c(cccc2C(F)(F)F)c(n1)c3ccc(cc3O)O',
    '1GU': 'C=CCCn1c(c2cccc(c2n1)C(F)(F)F)c3ccc(cc3O)O',
    '1GV': 'Cc1cc(ccc1c2ccc(s2)c3ccc(cc3C)O)O',
    '1HP': 'Cn1c2cc(cnc2nc1N)c3ccc(cc3)O',
    '244': 'c1cc(ccc1c2cc3cc(cc(c3o2)CC#N)O)O',
    '27G': 'CCCCOC(=O)c1ccccc1C(=O)OCc2ccccc2',
    '27H': 'CC1=CCC2(CCC(C2C(C1)OC(=O)c3ccc(cc3)O)(C(C)C)O)C',
    '27J': 'CC1CCCC(CCCCCc2cc(cc(c2C(=O)O1)O)O)O',
    '27K': 'CCCCOC(=O)c1ccc(cc1)O',
    '27L': 'CC(C)(C)CC(C)(C)c1ccc(cc1)O',
    '27M': 'c1cc(c(cc1O)O)C(=O)c2ccc(cc2O)O',
    '27N': 'c1cc(ccc1C(c2ccc(cc2)O)C(Cl)(Cl)Cl)O',
    '29S': 'Cc1c2cc(ccc2n(c1c3ccc(cc3)O)Cc4ccc(cc4)OCCN5CCCCCC5)O',
    '2I9': 'CN(C)C(=O)C=CCNCCOc1ccc(cc1)Oc2c3ccc(cc3sc2c4ccc(cc4)O)O',
    '2OH': 'CC(C)(c1ccc(cc1)O)c2ccc(cc2)O',
    '369': 'Cc1cc2cc(ccc2c(c1c3cccc(c3)O)Oc4ccc(cc4)O)O',
    '36J': 'CC1CCCC(CCCC=Cc2cc(cc(c2C(=O)O1)O)O)O',
    '36M': 'CCCOC(=O)c1ccc(cc1)O',
    '3YJ': 'CCCN(C)C(=O)CCCCCCCCCCC1Cc2cc(ccc2C3C1C4CCC(C4(CC3)C)O)O',
    '458': 'CC1=CCC2(CC1C(OC2)c3ccc(cc3)O)CO',
    '459': 'CC1CCC2(COC(C1C2C)c3ccc(cc3)O)CO',
    '4OH': 'CC12CCC3c4ccc(cc4CCC3C1C(C(C2O)O)O)O',
    '4Q7': 'COc1ccccc1OS(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)O)c5ccc(cc5)O',
    '4Q9': 'Cc1ccccc1OS(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)O)c5ccc(cc5)O',
    '53Q': 'CCN(CC)CCOc1ccc(cc1)C(=C(c2ccccc2)Cl)c3ccccc3',
    '5C4': 'CC(=C(c1ccc(cc1)O)c2ccc(cc2)O)c3cccc(c3)Nc4ccccc4',
    '5C6': 'CCC(=C(c1ccc(cc1)O)c2ccc(cc2)O)c3cccc(c3)Nc4cccc5c4cccc5',
    '5C7': 'CC(=C(c1ccc(cc1)O)c2ccc(cc2)O)c3ccc(cc3)C(C)(C)C',
    '5C8': 'CC(=C(c1ccc(cc1)O)c2ccc(cc2)O)c3ccc4ccccc4c3',
    '5C9': 'CCC(=C(c1ccc(cc1)O)c2ccc(cc2)O)c3cccc(c3)Nc4cccc(c4)C',
    '5CC': 'CCC(=C(c1ccc(cc1)O)c2ccc(cc2)O)c3cccc(c3)Nc4ccccc4',
    '5CE': 'CC12CCC(CC1CCC2O)c3ccc(cc3C(F)(F)F)O',
    '5CJ': 'CC12CCC(CC1CCC2O)c3cc(c(c(c3F)F)O)F',
    '5CK': 'CC12CCC(CC1CCC2O)c3ccc(c(c3F)F)O',
    '5CQ': 'Cc1cc(ccc1C2CCC3(C(C2)CCC3O)C)O',
    '5DG': 'CC(=C(c1ccc(cc1)O)c2ccc(cc2)O)c3cccc(c3)Nc4ccc(cc4)F',
    '5DH': 'c1ccc(cc1)C(=C(c2ccc(cc2)O)c3ccc(cc3)O)C#N',
    '5DJ': 'c1ccc(cc1)C=C(c2ccc(cc2)O)c3ccc(cc3)O',
    '5ES': 'c1cc(cc(c1)Nc2ccc(cc2)F)C=C(c3ccc(cc3)O)c4ccc(cc4)O',
    '5ET': 'c1cc(c(cc1O)Cl)c2ccc(s2)c3ccc(cc3Cl)O',
    '5EU': 'Cc1cc(sc1c2ccc(cc2Cl)O)c3ccc(cc3Cl)O',
    '5FS': 'Cc1cc(ccc1C2=CS(=O)(=O)C=C2c3ccc(cc3C)O)O',
    '5FT': 'Cc1cc(ccc1O)C2=C(C3C(CC2S3=O)S(=O)(=O)Oc4ccccc4)c5ccc(c(c5)C)O',
    '5FV': 'Cc1cc(ccc1C2=C(C3C(CC2S3=O)S(=O)(=O)Oc4ccccc4)c5ccc(cc5C)O)O',
    '5FY': 'Cc1cc(ccc1C2=C(C3C(CC2S3=O)S(=O)(=O)Oc4ccc(cc4)O)c5ccc(cc5C)O)O',
    '5G2': 'c1cc(c(cc1O)Cl)C2=CS(=O)(=O)C=C2c3ccc(cc3Cl)O',
    '5G3': 'c1ccc(c(c1)N=C(c2ccc(cc2)O)c3ccc(cc3O)O)Cl',
    '5G4': 'Cc1ccccc1N=C(c2ccc(cc2)O)c3ccc(cc3O)O',
    '5G5': 'c1ccc(c(c1)N=C(c2ccc(cc2)O)c3ccc(cc3)O)Cl',
    '5G6': 'c1ccc(cc1)N=C(c2ccc(cc2)O)c3ccc(cc3)O',
    '5G7': 'Cc1ccccc1N=C(c2ccc(cc2)O)c3ccc(cc3)O',
    '5HW': 'CC1CCCC(=C(c2ccc(cc2)O)c3ccc(cc3)O)C1',
    '5HX': 'c1ccc(cc1)C(=C2C3CCCC2CCC3)c4ccc(cc4)O',
    '5HZ': 'c1cc(ccc1C(=C2C3CCCC2CCC3)c4ccc(cc4)O)O',
    '5J0': 'c1cc(cc(c1)O)C(=C2C3CCCC2CCC3)c4ccc(cc4)O',
    '5J1': 'c1cc(ccc1C(c2ccc(cc2)O)C3C4CCCC3CCC4)O',
    '5J2': 'c1cc(ccc1C(=Nc2ccc(cc2)F)c3ccc(cc3O)O)O',
    '5JX': 'c1cc(ccc1C(=C2CCC(CC2)CF)c3ccc(cc3)O)O',
    '5JY': 'CCC1CCC(=C(c2ccc(cc2)O)c3ccc(cc3)O)CC1',
    '5K0': 'CC1CCC(=C(c2ccc(cc2)O)c3ccc(cc3)O)CC1',
    '5K1': 'CSC1CCCC(=C(c2ccc(cc2)O)c3ccc(cc3)O)C1',
    '5K2': 'c1ccc2c(c1)CCC(=C(c3ccc(cc3)O)c4ccc(cc4)O)C2',
    '5K4': 'CCC1CCCC(=C(c2ccc(cc2)O)c3ccc(cc3)O)C1',
    '5K5': 'c1cc(ccc1C(=C2CCC3CCCCC3C2)c4ccc(cc4)O)O',
    '5K7': 'COC(=O)CC1CCC(=C(c2ccc(cc2)O)c3ccc(cc3)O)CC1',
    '5K8': 'COC(=O)C(C1CCCC(=C(c2ccc(cc2)O)c3ccc(cc3)O)C1)C(=O)OC',
    '5KA': 'c1cc(ccc1C(=C2CCC(CC2)CCOCCCCF)c3ccc(cc3)O)O',
    '5KB': 'c1ccc(cc1)C2CCCC(=C(c3ccc(cc3)O)c4ccc(cc4)O)C2',
    '5KD': 'COc1ccc(cc1)C2CCCC(=C(c3ccc(cc3)O)c4ccc(cc4)O)C2',
    '5KE': 'c1cc(ccc1C2CCCC(=C(c3ccc(cc3)O)c4ccc(cc4)O)C2)O',
    '5KF': 'c1cc(ccc1C(=C2CCCC(C2)CCO)c3ccc(cc3)O)O',
    '5KG': 'c1cc(ccc1C(=C2CCC(CC2)CCO)c3ccc(cc3)O)O',
    '5OR': 'c1cc(ccc1c2c(n3cc(ccc3n2)O)I)O',
    '5OS': 'c1cc(c(cc1O)Cl)c2cocc2c3ccc(cc3Cl)O',
    '5P1': 'c1cc(ccc1c2c(n3cc(ccc3n2)O)C(F)(F)F)O',
    '5VP': 'c1ccc(cc1)c2csc3c2c(nc(n3)Cl)NCc4ccc(cc4)O',
    '5YR': 'CCCN(C)C(=O)CCCCCCCCCSC1Cc2cc(ccc2C3C1C4CCC(C4(CC3)C)O)O',
    '61Z': 'c1ccc(cc1)c2csc3c2c(nc(n3)Cl)N4Cc5ccc(cc5C4)O',
    '689': 'CC1C=C(C2C(C1(COC2c3ccc(cc3)O)CO)C)C',
    '6WL': 'Cn1c2c(cccc2C(F)(F)F)c(n1)c3ccc(cc3O)O',
    '6WM': 'CC12CCC3c4ccc(cc4CCC3C1CC(C2O)Cc5ccccc5)O',
    '6WN': 'CC12CCC3c4ccc(cc4CCC3C1CC(=Cc5ccccc5)C2=O)O',
    '6WP': 'c1ccc2c(c1)cccc2OS(=O)(=O)C3CC4C(=C(C3O4)c5ccc(cc5)O)c6ccc(cc6)O',
    '6WQ': 'c1cc(ccc1C(=C2CCc3c2ccc(c3)Br)c4ccc(cc4)O)O',
    '6WR': 'c1cc(c(cc1O)Cl)C2=CCC3C(C2)CCC3O',
    '6WS': 'c1cc(ccc1C(=C(Cl)Cl)c2ccc(cc2)Cl)Cl',
    '6WU': 'CC12CC=C(CC1CCC2O)c3cc(c(cc3F)O)F',
    '6WV': 'CC12CCC3c4ccc(cc4CCC3C1CCC2Nc5ccccc5)O',
    '6WW': 'CC12CCC3c4ccc(cc4CCC3C1CCC2N(C)c5ccccc5)O',
    '73I': 'c1cc(ccc1C2=C(C3C(CC2O3)S(=O)(=O)N(CC(F)(F)F)c4ccc(cc4)O)c5ccc(cc5)O)O',
    '77I': 'COc1ccc(cc1)N(CC(F)(F)F)S(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)OCCCN5CCCCC5)c6ccc(cc6)O',
    '77W': 'CC1(c2ccc(cc2CCN1c3ccc(cc3)F)O)c4ccc(cc4)C=CC(=O)O',
    '782': 'CC(C)c1ccc(cc1)N2CCc3cc(ccc3C2(C)c4ccc(cc4)C=CC(=O)O)O',
    '7AI': 'c1cc(ccc1C2=C(C3C(CC2O3)S(=O)(=O)N(CC(F)(F)F)c4ccc(cc4)OCCN5CCCCC5)c6ccc(cc6)O)O',
    '7E1': 'c1cc(cc(c1)Cl)OS(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)C=CCCCC(=O)O)c5ccc(cc5)O',
    '7E3': 'c1cc(cc(c1)Cl)OS(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)OCCCCC(=O)O)c5ccc(cc5)O',
    '7EB': 'CN(C)CCOc1ccc(cc1)C2=C(C3CC(C2O3)S(=O)(=O)Oc4ccc(cc4)Br)c5ccc(cc5)O',
    '7EC': 'c1cc(ccc1C2=C(C3C(CC2O3)S(=O)(=O)Oc4ccc(cc4)Br)c5ccc(cc5)OCCN6CCCCC6)O',
    '7ED': 'CCC(c1ccc(cc1)O)C(c2ccc(cc2)O)C(=O)OCCCCCCCCOC(=O)C(c3ccc(cc3)O)C(CC)c4ccc(cc4)O',
    '7EE': 'CC12CCC3c4ccc(cc4CCC3C1CCC2(C#Cc5ccc(cc5)N)O)O',
    '7EF': 'CC(=CCn1c(c2cccc(c2n1)C(F)(F)F)c3ccc(cc3O)O)C',
    '7EG': 'c1cc(ccc1c2ccc(c(c2c3ccc(cc3)O)C=NO)O)O',
    '7EH': 'Cn1c2c(cc(cn2)c3ccccc3)nc1N',
    '7EI': 'COc1ccc(cc1)N(CC(F)(F)F)S(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)OCCN5CCCCC5)c6ccc(cc6)O',
    '7EL': 'c1cc(ccc1c2ccc(c(c2Cl)O)C=NO)O',
    '7EM': 'c1cc(ccc1c2cc(sc2c3ccc(cc3)O)c4ccc(cc4)O)O',
    '7EN': 'c1cc(c(cc1O)Cl)C2=C(S(=O)C=C2)c3ccc(cc3Cl)O',
    '7EO': 'c1cc(c(cc1O)Cl)C2=CC=C(S2=O)c3ccc(cc3Cl)O',
    '7EQ': 'c1cc(c(cc1O)F)C2=CC=C(S2=O)c3ccc(cc3F)O',
    '7ER': 'c1cc(c(cc1O)F)C2=CS(=O)(=O)C=C2c3ccc(cc3F)O',
    '7ES': 'c1cc(c(cc1O)F)c2ccsc2c3ccc(cc3F)O',
    '7EV': 'c1cc(ccc1C2=C(C3CC(C2O3)S(=O)(=O)Oc4ccc(cc4)Br)c5ccc(cc5)OCCCCC(=O)O)O',
    '7FD': 'CCOC(=O)C=Cc1ccc(cc1)C(=C2CCCCC2)c3ccc(cc3)O',
    '7FG': 'CCOC(=O)C=Cc1ccc(cc1)C(=C2C3CCCC2CCC3)c4ccc(cc4)O',
    '7FJ': 'c1ccc2c(c1)CC(=C(c3ccc(cc3)O)c4ccc(cc4)O)C2',
    '7FL': 'c1cc(ccc1C(=C2CCCCCC2)c3ccc(cc3)O)O',
    '7FO': 'c1cc(ccc1C2=C(C3C(CC2O3)S(=O)(=O)Oc4ccc(cc4)I)c5ccc(cc5)O)O',
    '7FP': 'CC(=O)Nc1ccc(cc1)OS(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)O)c5ccc(cc5)O',
    '7FQ': 'CC12CCC3c4ccc(cc4CCC3C1CC(C2O)Cc5ccc(cc5)OC)O',
    '7FR': 'CC(C)c1ccc(cc1)N=C2CCC3C2(CCC4C3CCc5c4ccc(c5)O)C',
    '7FS': 'CC(C)c1ccc(cc1)NC2CCC3C2(CCC4C3CCc5c4ccc(c5)O)C',
    '7FZ': 'Cc1cc(cc(c1O)C)c2ccc3c(c2)CCC3O',
    '7G0': 'CC12CCC(CC1CCC2O)c3ccc(cc3)O',
    '7G1': 'CC12CCC3(CCc4c3ccc(c4)O)CC1CCC2O',
    '7G2': 'c1cc(c(cc1c2ccc(c(c2)F)O)C=NO)O',
    '7G3': 'c1cc(ccc1c2ccc(c(c2)C=NO)O)O',
    '7G5': 'c1cc(ccc1C2=C(CS(=O)(=O)C2)c3ccc(cc3)O)O',
    '7I0': 'CCC(=C(c1ccc(cc1)C=CC(=O)O)c2ccc3c(c2)c[nH]n3)c4ccc(cc4Cl)F',
    '7I5': 'c1cc(ccc1C=CC(=O)O)N(CC(F)(F)F)S(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)O)c5ccc(cc5)O',
    '7I9': 'COc1ccc(cc1)N(CC(F)(F)F)S(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)OCCCCN5CCCCC5)c6ccc(cc6)O',
    '7J9': 'c1cc(ccc1C2=C(C3C(CC2O3)S(=O)(=O)Oc4ccc(cc4)Br)c5ccc(cc5)OCCCCCC(=O)O)O',
    '7JY': 'c1cc(cc(c1)Cl)OS(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)OCCCCCCC(=O)O)c5ccc(cc5)O',
    '7K6': 'c1cc(ccc1C2=C(C3C(CC2O3)S(=O)(=O)Oc4ccc(cc4)Br)c5ccc(cc5)OCCCCCCC(=O)O)O',
    '7KL': 'c1cc(cc(c1)Cl)OS(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)C=CC(=O)O)c5ccc(cc5)O',
    '7M1': 'c1ccc(cc1)OS(=O)(=O)c2ccc(c(c2)c3ccc(cc3)O)c4ccc(cc4)O',
    '7M4': 'c1cc(ccc1C=CCCCC(=O)O)C2=C(C3CC(C2O3)S(=O)(=O)Oc4ccc(cc4)Br)c5ccc(cc5)O',
    '7M7': 'c1cc(ccc1c2ccc(cc2c3ccc(cc3)O)S(=O)(=O)Oc4ccc(cc4)Br)O',
    '7OI': 'c1ccc(cc1)COc2ccc(cc2)N(CC(F)(F)F)S(=O)(=O)C3CC4C(=C(C3O4)c5ccc(cc5)O)c6ccc(cc6)O',
    '7OR': 'COc1ccc(cc1)N(CC(F)(F)F)S(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)OCc5ccccc5)c6ccc(cc6)O',
    '7Q5': 'COC(=O)C=Cc1ccc(cc1)N(CC(F)(F)F)S(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)O)c5ccc(cc5)O',
    '7QN': 'Cc1c(ccc2c1CCN(C2c3ccc(cc3)C=CC(=O)O)CC(C)C)O',
    '85M': 'CC1c2ccc(cc2OC(C1c3ccc(cc3)O)c4ccc(cc4)OCCN5CCCC5)O',
    '85Z': 'Cc1cc(ccc1C2=C(c3ccc(cc3OC2=O)O)Cc4ccc(cc4)C=CC(=O)O)F',
    '86V': 'CC1CCN(C1)CCOc2ccc(cc2)C3C(=C(c4ccc(cc4O3)O)C)c5ccc(cc5)O',
    '86Y': 'CC1CCN(C1)CCOc2ccc(cc2)C3C(=C(c4ccc(cc4O3)O)C)c5ccc(cc5)O',
    '98L': 'c1ccc(cc1)c2csc3c2c(nc(n3)Cl)NCc4cccc(c4)O',
    '9XY': 'CCC(=C(c1ccc(cc1)O)c2ccc(cc2)OCCNC)c3ccccc3',
    'A48': 'B(c1c(cc(cc1C)C)C)(c2c(cc(cc2C)C)C)N(CC(F)(F)F)c3ccc(cc3)O',
    'AEJ': 'c1ccc(cc1)N2CCc3cc(ccc3C2c4ccc(cc4)N5CCN6CCCCC6C5)O',
    'AIH': 'CC1CN(CC1C)CCOc2ccc(cc2)C3C(Sc4cc(ccc4O3)O)c5ccc(cc5)O',
    'AIJ': 'CC(COc1ccc(cc1)C2C(Sc3cc(ccc3O2)O)c4ccc(cc4)O)N5CCCC5',
    'AIT': 'CC(COc1ccc(cc1)C2C(Sc3cc(ccc3O2)O)c4ccc(cc4)O)N5CCCC5',
    'AIU': 'CC1CN(CC1C)CCOc2ccc(cc2)C3C(Sc4cc(ccc4O3)O)c5ccc(cc5)O',
    'BCT': 'C(=O)(O)[O-]',
    'C3D': 'c1ccc(cc1)C2CCc3cc(ccc3C2c4ccc(cc4)OCCN5CCCC5)O',
    'C6V': 'CC(c1cc(ccc1c2c(c3ccc(cc3s2)O)Oc4ccc(cc4)C=CC(=O)O)F)(F)F',
    'C8L': 'C1N2C3C4N(C2=O)CN5C6C7N(C5=O)CN8C9C2N(C8=O)CN5C8C%10N(C5=O)CN5C%11C%12N(C5=O)CN5C%13C%14N(C5=O)CN5C%15C%16N(C5=O)CN5C%17C(N1C5=O)N1CN3C(=O)N4CN6C(=O)N7CN9C(=O)N2CN8C(=O)N%10CN%11C(=O)N%12CN%13C(=O)N%14CN%15C(=O)N%16CN%17C1=O',
    'CCS': 'C(C(C(=O)O)N)SCC(=O)O',
    'CM3': 'CC1c2c(ccc(c2F)O)OC(C1c3ccc(cc3)O)c4ccc(cc4)OCCN5CCCCC5',
    'CM4': 'CC1c2cc(ccc2OC(C1c3ccc(cc3)O)c4ccc(cc4)OCCN5CCCC5)O',
    'CME': 'C(CSSCC(C(=O)O)N)O',
    'CSO': 'C(C(C(=O)O)N)SO',
    'CUE': 'c1cc2c(cc1O)oc-3c2C(=O)Oc4c3ccc(c4)O',
    'DC8': 'c1cc(ccc1C2C3CC(CC3c4cc(ccc4O2)O)(F)F)O',
    'DES': 'CCC(=C(CC)c1ccc(cc1)O)c2ccc(cc2)O',
    'DMS': 'CS(=O)C',
    'DRQ': 'CCC=Cc1cc2c(cc1O)CCC3C2CCC4(C3CCC4O)C',
    'E4D': 'c1cc(ccc1C2C(Oc3ccc(cc3S2)O)c4ccc(cc4)OCCN5CCCCC5)O',
    'EED': 'CC12CC(C3c4ccc(cc4CCC3C1CCC2O)O)COC',
    'EES': 'c1cc(ccc1n2c(c3cc(ccc3n2)O)Cl)O',
    'EEU': 'CC12CCC3c4ccc(cc4CCC3C1CCC2(C#Cc5cc(nc(c5)CN(CC(=O)O)CC(=O)O)CN(CC(=O)O)CC(=O)O)O)O',
    'EI1': 'CCc1c2cc(ccc2nn1c3ccc(cc3)O)O',
    'EMY': 'c1ccc(cc1)CC(=O)Sc2cc(nc(n2)Cl)NCc3ccc(cc3)O',
    'ESE': 'CC12CCC(CC1CCC2O)c3ccc(cc3)O',
    'ESL': 'CC12CCC3c4ccc(cc4CCC3C1CC(C2O)O)O',
    'EST': 'CC12CCC3c4ccc(cc4CCC3C1CCC2O)O',
    'ETC': 'CCC1Cc2cc(ccc2C3=C1c4ccc(cc4CC3CC)O)O',
    'EU': '[Eu+2]',
    'EZT': 'CC12CCC3c4ccc(cc4CCC3C1CCC2(C=Cc5ccccc5C(F)(F)F)O)O',
    'F3D': 'CCC(=C(c1ccc(cc1)OCCNCCCC(=O)N(C)C)c2ccc3c(c2)cn[nH]3)c4ccccc4',
    'FNJ': 'CCC(c1ccc(cc1)O)C(c2ccccc2)c3ccc(cc3)O',
    'FSV': 'c1cc(c(cc1C=Cc2cc(cc(c2)O)O)F)O',
    'FYS': 'CCC(c1ccc(cc1)O)C(c2ccc(cc2)O)c3ccc(cc3)O',
    'G8Y': 'c1cc(ccc1C2C3=C(CCOc4c3ccc(c4)O)c5ccc(cc5O2)O)OCCN6CC(C6)CF',
    'G9J': 'CC1=C(C(Oc2c1cc(cc2)O)c3ccc(cc3)I)c4cccc(c4)O',
    'GEN': 'c1cc(ccc1C2=COc3cc(cc(c3C2=O)O)O)O',
    'GQD': 'CC1Cc2cc(ccc2C(N1CC(C)C)c3ccc(cc3)C=CC(=O)O)O',
    'GW5': 'CCC(=C(c1ccccc1)c2ccc(cc2)C=CC(=O)O)c3ccccc3',
    'GZI': 'Cc1c(c(cnc1OCCNCCCF)F)C2c3c(c4ccccc4[nH]3)CC(N2CC(C)C(=O)O)C',
    'H09': 'Cc1c(ccc(c1C2c3c(c4ccccc4[nH]3)CC(N2CC(CO)(F)F)C)F)OCCNCCCF',
    'H8W': 'CC(C)CN1CCc2c(ccc3c2cn[nH]3)C1c4ccc(cc4)CCC(=O)O',
    'HZ3': 'COC(=O)C1=C(C2C(=C(C1O2)c3ccc(cc3)O)c4ccc(cc4)O)C(=O)OC',
    'I0G': 'c1cc(ccc1C2C3CCCC3c4cc(ccc4O2)O)O',
    'I0V': 'CCNCCc1ccc(cc1)CN(CC)c2cc(ccc2C3CCc4cc(ccc4C3)O)OC',
    'IAT': 'c1cc(ccc1C2=C(C3C(CC2O3)S(=O)(=O)Oc4ccc(cc4)n5cncn5)c6ccc(cc6)O)O',
    'IOG': 'CC(CCc1ccc(cc1)O)NC(=O)Cc2c3ccc(cc3[nH]c2c4ccccc4)OCCN5CCCCC5',
    'IOK': 'CC(CCc1ccc(cc1)O)NC(=O)Cc2c3ccccc3[nH]c2c4ccccc4',
    'J0W': 'CC(C)CN1CCc2cc(ccc2C1(C)c3ccc(cc3)C=CC(=O)O)O',
    'J2Z': 'CC12CCC3c4ccc(cc4CCC3C1CC(C2=O)O)O',
    'J3Z': 'CC12CCC3c4ccc(cc4CCC3C1CCC2=O)O',
    'JJ3': 'COCc1cc(cc2c1OC(C3C2CCC3)c4ccc(cc4)O)O',
    'KE9': 'CC1Cc2c3ccccc3[nH]c2C(N1CC(C)(C)F)c4c(cc(cc4F)C=CC(=O)O)F',
    'KN0': 'c1ccc(cc1)Cn2c3c(cccc3C(F)(F)F)c(n2)c4ccc(cc4O)O',
    'KN1': 'C=CCn1c2c(cccc2C(F)(F)F)c(n1)c3ccc(cc3O)O',
    'KN2': 'c1cc2c(cc1O)[nH]nc2c3ccc(cc3O)O',
    'KN3': 'CC(=CCn1c2c(cccc2C(F)(F)F)c(n1)c3ccc(cc3O)O)C',
    'L4G': 'CS(=O)(=O)c1ccc(cc1)c2ccc3cc(ccc3c2Oc4ccc(cc4)OCCN5CCCCC5)O',
    'L5B': 'c1cc(ccc1C2=C(CCCc3c2ccc(c3)C(=O)O)c4ccc(cc4Cl)Cl)OC5CCN(C5)CCCF',
    'L84': 'c1cc(ccc1C2=C(C3C(CC2O3)S(=O)(=O)N(CC(F)(F)F)c4ccc(cc4)OCCCN5CCCCC5)c6ccc(cc6)O)O',
    'LLB': 'CC1CCN(CC1)CCOc2ccc(cc2)C(=O)c3c4ccc(cc4sc3c5ccc(cc5)O)O',
    'LLC': 'c1cc(ccc1c2c(c3ccc(cc3s2)O)C(=O)c4ccc(cc4)OCCN5CCCC5)O',
    'LRQ': 'CC(C)(CN1C2CCCC1(c3c(c4ccccc4[nH]3)C2)c5c(cc(cc5F)C=CC(=O)O)F)F',
    'LVH': 'CC1Cc2c3ccccc3[nH]c2C(N1CC(C)(C)F)(C)c4c(cc(cc4F)C=CC(=O)O)F',
    'LYQ': 'c1cc(c(cc1O)Cl)C2C3CCCC3c4cc(ccc4N2)S(=O)(=O)N',
    'MLZ': 'CNCCCCC(C(=O)O)N',
    'ND1': 'CN(C)C(=O)C=CCNCCOc1ccc(cn1)C(=C(CC(F)(F)F)c2ccccc2)c3ccc4c(c3)c([nH]n4)F',
    'NYU': 'CC12CC(C3c4ccc(cc4CCC3C1CCC2O)O)c5ccc(cc5)OCCN(C)C',
    'OB1': 'c1ccc(cc1)NS(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)O)c5ccc(cc5)O',
    'OB2': 'CN(c1ccccc1)S(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)O)c5ccc(cc5)O',
    'OB3': 'CN(c1ccccc1Cl)S(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)O)c5ccc(cc5)O',
    'OB5': 'CCN(c1ccc(cc1)OC)S(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)O)c5ccc(cc5)O',
    'OB7': 'CCN(c1ccc(cc1)Cl)S(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)O)c5ccc(cc5)O',
    'OB8': 'CCN(c1ccc2ccccc2c1)S(=O)(=O)C3CC4C(=C(C3O4)c5ccc(cc5)O)c6ccc(cc6)O',
    'OB9': 'c1ccc(cc1)N(CC(F)(F)F)S(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)O)c5ccc(cc5)O',
    'OBB': 'c1cc(cc(c1)Br)OS(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)O)c5ccc(cc5)O',
    'OBC': 'c1ccc(c(c1)OS(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)O)c5ccc(cc5)O)F',
    'OBH': 'c1cc(ccc1C2=C(C3C(CC2O3)S(=O)(=O)OC4C=CCC=C4)c5ccc(cc5)O)O',
    'OBM': 'c1cc(ccc1C2=C(C3C(CC2O3)S(=O)(=O)Oc4ccc(cc4)Br)c5ccc(cc5)O)O',
    'OBT': 'c1cc(ccc1C2=C(C3C(CC2O3)S(=O)(=O)N(CC(F)(F)F)c4ccc(cc4)Cl)c5ccc(cc5)O)O',
    'ODE': 'CCOC(=O)C1C(C2C(=C(C1O2)c3ccc(cc3)O)c4ccc(cc4)O)C(=O)OCC',
    'ODY': 'CC1=C(C(Oc2c1cccc2O)c3ccc(cc3)OCCN4CC(C4)CF)c5cccc(c5)O',
    'OFB': 'c1cc(cc(c1)F)OS(=O)(=O)C2CC3C(=C(C2O3)c4ccc(cc4)O)c5ccc(cc5)O',
    'OGJ': 'CC1=C(C(Oc2c1cc(cc2)O)c3ccc(cc3)OCCN4CC(C4)CF)c5cccc(c5)O',
    'OHT': 'CCC(=C(c1ccc(cc1)O)c2ccc(cc2)OCCN(C)C)c3ccccc3',
    'PIQ': 'Cn1c2cc(cnc2nc1N)c3ccccc3',
    'PTI': 'c1ccc(cc1)N2CCc3cc(ccc3C2c4ccc(cc4)OCCN5CCCCC5)O',
    'Q97': 'CCC(=C(c1ccc(cc1)O)c2ccc(cc2)OCC)c3ccc(cc3)O',
    'QHG': 'CC(C)CN1CCc2cc(ccc2C1c3ccc(cc3)C=CC(=O)O)O',
    'QNE': 'CC1Cc2c(ccc3c2c[nH]n3)C(N1CC4(CC4)F)c5ccc(cc5OC)NC6CN(C6)CCCF',
    'QNH': 'CCCN1CC(C1)Nc2ccc(nc2)C3c4ccc5c(c4CC(N3CC(F)(F)F)C)cn[nH]5',
    'QNK': 'CC1Cc2c(ccc3c2c[nH]n3)C(N1CC4(CC4)F)c5ccc(cn5)NC6CN(C6)CCCF',
    'QSO': 'COc1ccc(cc1)C2=COc3cc(cc(c3C2=O)O)O',
    'QYM': 'CC12CC(C3c4ccc(cc4CCC3C1CCC2O)O)c5ccc(cc5)OCCN(C)C',
    'R3V': 'CC1CCN(C1)CCOc2ccc(cc2)C3c4ccc(cc4CCC3c5ccccc5)O',
    'RAL': 'c1cc(ccc1c2c(c3ccc(cc3s2)O)C(=O)c4ccc(cc4)OCCN5CCCCC5)O',
    'RL4': 'CC1CCCN1CCOc2ccc(cc2)C3c4ccc(cc4CCC3c5ccccc5)O',
    'SCH': 'CSSCC(C(=O)O)N',
    'STL': 'c1cc(ccc1C=Cc2cc(cc(c2)O)O)O',
    'T3O': 'CC1=CCC2CC1C(OC2(C)C)c3ccc(cc3)O',
    'TPO': 'CC(C(C(=O)O)N)OP(=O)(O)O',
    'TQF': 'CCN1CC(C1)Oc2ccc(cc2)C3c4ccc(cc4CC5(N3C(=O)c6ccccc6)CC5)O',
    'TS7': 'CCCN1CC(C1)Oc2ccc(cc2)C3c4ccc(cc4CC5(N3C(=O)c6ccccc6)CC5)O',
    'TT5': 'c1ccc(cc1)C(=O)N2C(c3ccc(cc3CC24CC4)O)c5ccc(cc5)OCCN6CCCCC6',
    'TTU': 'CCN1CCC(C1)Oc2ccc(cc2)C3c4ccc(cc4CC5(N3C(=O)c6ccccc6)CC5)O',
    'TU9': 'CCN1CCC(C1)COc2ccc(cc2)C3c4ccc(cc4CC5(N3C(=O)c6ccccc6)CC5)O',
    'TV3': 'CCCN1CCC(C1)COc2ccc(cc2)C3c4ccc(cc4CC5(N3C(=O)c6ccccc6)CC5)O',
    'TVF': 'CC1CCN(C1)CCOc2ccc(cc2)C3c4ccc(cc4CC5(N3C(=O)c6ccccc6)CC5)O',
    'TVL': 'CC1CCCN1CCOc2ccc(cc2)C3c4ccc(cc4CC5(N3C(=O)c6ccccc6)CC5)O',
    'TVX': 'CN(C)CCOc1ccc(cc1)C2c3ccc(cc3CC4(N2C(=O)c5ccccc5)CC4)O',
    'TW6': 'CNCCOc1ccc(cc1)C2c3ccc(cc3CC4(N2C(=O)c5ccccc5)CC4)O',
    'TWF': 'CCNCCOc1ccc(cc1)C2c3ccc(cc3CC4(N2C(=O)c5ccccc5)CC4)O',
    'TX9': 'CCCCCN1CCC(C1)COc2ccc(cc2)C3c4ccc(cc4CC5(N3C(=O)c6ccccc6)CC5)O',
    'TXK': 'CCCN1CCC(C1)CCc2ccc(cc2)C3c4ccc(cc4CC5(N3C(=O)c6ccccc6)CC5)O',
    'TZ3': 'CCCN1CCC(C1)CSc2ccc(cc2)C3c4ccc(cc4CC5(N3C(=O)c6ccccc6)CC5)O',
    'TZI': 'c1ccc(cc1)C(=O)N2C(c3ccc(cc3CC24CC4)O)c5ccc(cc5)OCC6CCN(C6)CCCF',
    'U6D': 'CC1Cc2c3ccccc3[nH]c2C(N1CC(C)(C)F)c4c(cc(cc4F)OCCN5CC(C5)CF)F',
    'V9J': 'c1cc(ccc1c2c3ccc(cc3sc2Oc4ccc(cc4)O)O)F',
    'VQI': 'Cc1cc2c(cc1O)CCN(C2c3ccc(cc3)C=CC(=O)O)CC(C)C',
    'WST': 'c1cc(c2c(c1)OC(C3C2CCC3)c4ccc(cc4)O)O',
    'WVE': 'c1cc(c(cc1O)Cl)C2C3CCCC3c4cc(ccc4N2)S(=O)(=O)N',
    'WVR': 'c1cc(ccc1C2C3CCCC3c4cc(ccc4N2)S(=O)(=O)N)O',
    'WVW': 'c1cc(ccc1C2C3CCCC3c4cc(ccc4N2)S(=O)(=O)N)O',
    'XBR': 'c1ccc2c(c1)c3c([nH]2)C(N(CC3)CCC(=O)O)c4ccc(cc4)Cl',
    'XDH': 'CC(C)(c1cc(c(c(c1)Cl)O)Cl)c2cc(c(c(c2)Cl)O)Cl',
    'YCM': 'C(C(C(=O)O)N)SCC(=O)N',
    'ZER': 'CC1CCCC(=O)CCCC=Cc2cc(cc(c2C(=O)O1)O)O',
    'ZN': '[Zn+2]',
    'ZNM': 'CC1Cc2c3ccccc3[nH]c2C(N1CC(CO)(F)F)c4c(cc(cc4F)NC5CN(C5)CCCF)F',
    'ZTW': 'c1cc(ccc1c2cc3ccc(cc3s2)O)O'
}


# --- 2. Load Data from CSV File ---
# 🚨 IMPORTANT: CHANGE THIS FILE PATH TO YOUR ACTUAL CSV FILE NAME
file_path = 'my.csv' 

if not os.path.exists(file_path):
    print(f"FATAL ERROR: The file '{file_path}' was not found. Please verify the file path and that the file is in the same directory as this script.")
    exit()
else:
    df_raw = pd.read_csv(file_path)
    print(f"Successfully loaded {len(df_raw)} records from {file_path}.")

# Check for the required column
if 'ligand_code' not in df_raw.columns:
    print("ERROR: CSV file must contain a column named 'ligand_code'. Exiting.")
    exit()

# --- 3. Prepare Unique Ligands for Feature Generation ---
unique_ligands = df_raw['ligand_code'].unique()

mols = []
valid_codes = []
for code in unique_ligands:
    smiles = LIGAND_SMILES.get(code)
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mols.append(mol)
            valid_codes.append(code)
        else:
            print(f"Warning: Invalid SMILES format provided for ligand code '{code}'. Skipping.")
    else:
        print(f"Warning: No SMILES defined in LIGAND_SMILES dictionary for code '{code}'. Skipping.")

if len(valid_codes) < 2:
    print("ERROR: Fewer than 2 unique ligands with valid SMILES were found. Clustering requires at least two points. Exiting.")
    exit()

# --- 4. Feature Generation (Morgan Fingerprints) ---
# Morgan Fingerprints encode the structural fragments (chemical features) of each molecule.
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
X_fingerprints = np.array([list(fp) for fp in fingerprints])

# --- 5. Clustering (K-Means) ---
# Since we know we are separating the steroid (EST) from the non-steroid (RAL), K=2 is appropriate.
K = 2 
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_fingerprints)

# Map the cluster IDs back to the ligand codes
cluster_results = pd.DataFrame({
    'Ligand': valid_codes,
    'Cluster_ID': cluster_labels
})

# --- 6. Integrate Results into Original DataFrame ---
cluster_map = cluster_results.set_index('Ligand')['Cluster_ID'].to_dict()
df_raw['Cluster_ID (Morgan FP)'] = df_raw['ligand_code'].map(cluster_map)

print("\n--- Structural Cluster Assignment ---")
print(cluster_results)
print("\n--- Final CSV Data Snippet with Cluster IDs ---")
print(df_raw[['pdb_id', 'ligand_code', 'er_state', 'Cluster_ID (Morgan FP)']])

# --- 7. Visualization (t-SNE) ---
# t-SNE reduces the 2048-dimensional fingerprint vector to 2D for plotting.
# The 'jaccard' metric is used here because it is mathematically equivalent to Tanimoto distance 
# for binary data (like Morgan Fingerprints).
perplexity_val = min(5, K - 1) if K > 1 else 1
tsne = TSNE(n_components=2, random_state=42, metric='jaccard', perplexity=perplexity_val, learning_rate='auto', init='random')
X_tsne = tsne.fit_transform(X_fingerprints)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=cluster_labels,
    cmap='viridis',
    s=150
)

# Add annotations to the plot
for i, code in enumerate(valid_codes):
    plt.annotate(
        code,
        (X_tsne[i, 0], X_tsne[i, 1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center'
    )


plt.title(f'Structural Clustering of ESR Ligands (K={K}) via t-SNE')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show() # This displays the visualization 
df_raw.groupby("Cluster_ID (Morgan FP)")["er_state"].value_counts()

# --- 8. Save the results (Uncomment to save) ---
# df_raw.to_csv('clustered_esr_output.csv', index=False)