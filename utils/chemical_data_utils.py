"""
Utilities for generating and transforming chemical data
"""

import time
from rdkit.Chem import Descriptors
from rdkit import Chem


def calc_all_molecular_descriptors(smiles_list: list):
    """
    Calculates all molecular descriptors for a list of smiles strings

    ** Convert output to a pandas dataframe and concatenate with original dataframe containing smiles strings

    :param smiles_list: list of smiles strings
    :return: dictionary names of descriptors as keys and lists of descriptors as values
    """
    start = time.time()
    all_descriptors_list = []
    for i, smiles in enumerate(smiles_list):
        this_smiles_descriptors = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(smiles))  # dict of descriptors
        all_descriptors_list.append(this_smiles_descriptors.values())
        if i == 0:  # get names of descriptors from first molecule
            all_descriptor_names = list(this_smiles_descriptors.keys())

    all_descriptors_concatenated = list(zip(*all_descriptors_list))

    all_descriptors = {}
    for i, descriptor_name in enumerate(all_descriptor_names):
        all_descriptors[descriptor_name] = all_descriptors_concatenated[i]

    del all_descriptors_list, all_descriptors_concatenated, all_descriptor_names

    print(f"Time to calc molecular descriptors: {time.time() - start}")

    return all_descriptors





