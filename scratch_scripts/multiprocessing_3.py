# trying to learn multiprocessing with pool, used it to parallelize fingerprint generation

import multiprocessing as mp
import pdb
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
import tqdm  # for progress bar
from functools import partial
import numpy as np

filename = '/Users/christian_g/python_projects/Data_Science_Projects/data/CYP2D6_smiles.csv'

# getting generator
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)

def get_fingerprint(smiles, mfpgen):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mfpgen.GetFingerprint(mol)

def get_fingerprint_with_mfpgen(smi):
        return get_fingerprint(smi, mfpgen)

def main():
    # retrieving smiles strings from file
    df = pd.read_csv(filename)
    smiles = df['Smiles'].values

    # repeat smiles 10 times
    smiles = np.repeat(smiles, 10)

    # non-parallelized
    start = time.time()
    fingerprints_not_par = [get_fingerprint(smi, mfpgen) for smi in smiles]
    print(f'Time for non-parallel: {time.time() - start}')

    # parallelized with map
    start = time.time()
    with mp.Pool() as pool:
        fingerprints_map = list(pool.map(get_fingerprint_with_mfpgen, smiles, chunksize=len(smiles)//mp.cpu_count()))

    print(f'Time for parallelized with map: {time.time() - start}')

    # parallelized with imap
    start = time.time()

    with mp.Pool() as pool:
        fingerprints_imap = list(pool.imap(get_fingerprint_with_mfpgen, smiles, chunksize=len(smiles)//mp.cpu_count()))

    print(f'Time for parallelized with imap: {time.time() - start}')

    if fingerprints_not_par == fingerprints_imap and fingerprints_not_par == fingerprints_map:
        print('Fingerprints are equal')
    else:
        print('Fingerprints are not equal')


if __name__ == "__main__":
    main()
