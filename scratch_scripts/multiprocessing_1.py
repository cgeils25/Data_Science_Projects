import multiprocessing as mp
import pandas as pd
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
from rdkit import Chem
import numpy as np
import time
import pdb

df = pd.read_csv('../data/CYP2D6_smiles.csv')
all_smiles = df.Smiles

# for testing
all_smiles = all_smiles[:100]


def generate_fps(smiles_list, queue):
    fps_ret = []
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        # pdb.set_trace()
        fp = mfpgen.GetFingerprint(mol)
        fp_as_np = np.array(fp)
        fps_ret.append(fp_as_np)

    if queue:
        queue.put(fps_ret)
        return
    return fps_ret


if __name__ == '__main__':

    # non-parallelized
    start = time.time()
    all_fps_nonpar = generate_fps(all_smiles, None)
    print(f'Total time for non-parallel: {time.time() - start}')

    smiles_p1, smiles_p2, smiles_p3, smiles_p4 = [
        all_smiles[(len(all_smiles) // 4) * i: (len(all_smiles) // 4) * (i + 1)] for i in range(4)]

    # parallelized
    start = time.time()

    # creating queue
    a_queue = mp.Queue()

    # defining process 1-4
    p1 = mp.Process(target=generate_fps, args=(smiles_p1, a_queue,))
    p2 = mp.Process(target=generate_fps, args=(smiles_p2, a_queue,))
    p3 = mp.Process(target=generate_fps, args=(smiles_p3, a_queue,))
    p4 = mp.Process(target=generate_fps, args=(smiles_p4, a_queue,))

    processes = [p1, p3]

    # starting processes
    for i, process in enumerate(processes):
        process.start()
        print(f'process {i+1} started')

    # pdb.set_trace()
    # tell program to wait for each process before exiting
    for process in processes:
        process.join()

    # pdb.set_trace()
    # retrieving fps from queue
    all_fps_par_unmerged = []
    while not a_queue.empty():  # ... because a queue is fifo idiot
        all_fps_par_unmerged.append(a_queue.get())

    # merging
    all_fps_par = [fp for sublist in all_fps_par_unmerged for fp in sublist]

    # pdb.set_trace()
    print(f'Total time for parallel: {time.time() - start}')

    # making sure I didn't mess it up
    print(f'Parallel and non-parallel outputs match: {all_fps_par == all_fps_nonpar}')
