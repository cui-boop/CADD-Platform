import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


def calculate_basic_descriptors(smiles):
    """
    根据单个 SMILES 计算基础分子描述符。
    返回一个 dict，如果 SMILES 无效，则返回 None。
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    desc = {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "RingCount": Lipinski.RingCount(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "NumAromaticRings": Lipinski.NumAromaticRings(mol),
    }

    return desc


def build_descriptor_dataframe(df, smiles_col="smiles"):
    """
    批量计算描述符。
    输入：包含 smiles 列的数据框
    输出：描述符数据框 descriptor_df，以及保留下来的原始行 valid_df
    """
    descriptor_list = []
    valid_indices = []

    for idx, smiles in df[smiles_col].items():
        desc = calculate_basic_descriptors(smiles)

        if desc is not None:
            descriptor_list.append(desc)
            valid_indices.append(idx)

    descriptor_df = pd.DataFrame(descriptor_list)
    valid_df = df.loc[valid_indices].reset_index(drop=True)
    descriptor_df = descriptor_df.reset_index(drop=True)

    return descriptor_df, valid_df