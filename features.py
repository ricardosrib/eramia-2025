from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np 
import pandas as pd 

def main():
    df_raw = pd.read_csv('data/raw/b3db.tsv', sep='\t')
    df_features = compute_features(df_raw)
    df_features.to_csv('data/preprocess/b3db.csv', index=False)

def compute_features(df_raw:pd.DataFrame) -> pd.DataFrame:
    X = []
    y = []

    for _, row in df_raw.iterrows():
        fingerprints = smiles_to_fingerprints(row.SMILES)
        X.append(fingerprints)
        y.append(1 if row['BBB+/BBB-'] == 'BBB+' else 0)

    X = np.array(X)
    y = np.array(y)

    df_features = pd.DataFrame(X)
    df_features['label'] = y

    return df_features

def smiles_to_fingerprints(smiles:str) -> np.array:
    mol = Chem.MolFromSmiles(smiles)
    fingerprints = AllChem.GetMorganFingerprintAsBitVect(
        mol, useChirality=True, radius=2, nBits=1024, bitInfo={}
    )

    return np.array(fingerprints)

if __name__ == '__main__':
    main()