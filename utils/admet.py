from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.FilterCatalog import *

import pandas as pd


def calculate_admet_properties(smiles):
    """
    计算分子 ADMET 相关性质
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    molwt = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)

    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    rotatable = Lipinski.NumRotatableBonds(mol)

    ring_count = Lipinski.RingCount(mol)

    aromatic_rings = Lipinski.NumAromaticRings(mol)

    heavy_atoms = Lipinski.HeavyAtomCount(mol)

    fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)

    # Lipinski Rule
    lipinski_pass = 0

    if molwt <= 500:
        lipinski_pass += 1

    if logp <= 5:
        lipinski_pass += 1

    if hbd <= 5:
        lipinski_pass += 1

    if hba <= 10:
        lipinski_pass += 1

    lipinski_result = "Pass" if lipinski_pass == 4 else "Fail"

    # 简化版 bioavailability score
    bioavailability = round(
        (
            (1 if molwt < 500 else 0) +
            (1 if logp < 5 else 0) +
            (1 if tpsa < 140 else 0) +
            (1 if rotatable < 10 else 0)
        ) / 4,
        2
    )

    # CNS permeability
    cns = "High" if tpsa < 90 and logp > 2 else "Low"

    # PAINS Alert
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)

    catalog = FilterCatalog(params)

    pains_match = catalog.HasMatch(mol)

    pains_result = "Yes" if pains_match else "No"

    # 综合 ADMET Score
    score = (
        (1 if molwt < 500 else 0) +
        (1 if logp < 5 else 0) +
        (1 if tpsa < 140 else 0) +
        (1 if hbd <= 5 else 0) +
        (1 if hba <= 10 else 0) +
        (1 if rotatable < 10 else 0)
    ) / 6

    return {
        "SMILES": smiles,
        "MolWt": round(molwt, 2),
        "LogP": round(logp, 2),
        "TPSA": round(tpsa, 2),
        "HBD": hbd,
        "HBA": hba,
        "RotatableBonds": rotatable,
        "RingCount": ring_count,
        "AromaticRings": aromatic_rings,
        "HeavyAtomCount": heavy_atoms,
        "FractionCSP3": round(fraction_csp3, 2),
        "Lipinski_Passed": lipinski_pass,
        "Lipinski_Result": lipinski_result,
        "BioavailabilityScore": bioavailability,
        "CNS_Permeability": cns,
        "PAINS_Alert": pains_result,
        "ADMET_Score": round(score, 2)
    }


def batch_calculate_admet(smiles_list):

    results = []

    for smiles in smiles_list:

        try:
            result = calculate_admet_properties(smiles)

            if result is not None:
                results.append(result)

        except:
            continue

    return pd.DataFrame(results)