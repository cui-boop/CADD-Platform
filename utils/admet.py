from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
import numpy as np

# =========================
# 分子基础属性
# =========================

def calculate_basic_properties(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    props = {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(mol),
        "AromaticRings": Descriptors.NumAromaticRings(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms()
    }

    return props


# =========================
# Lipinski Rule of 5
# =========================

def lipinski_rule(props):

    violations = 0

    if props["MolWt"] > 500:
        violations += 1
    if props["LogP"] > 5:
        violations += 1
    if props["HBD"] > 5:
        violations += 1
    if props["HBA"] > 10:
        violations += 1

    return {
        "violations": violations,
        "pass": violations <= 1
    }


# =========================
# Drug-likeness score（科研评分）
# =========================

def drug_likeness_score(props):

    score = 0

    # MolWt（越接近400越好）
    score += max(0, 1 - abs(props["MolWt"] - 400) / 400)

    # LogP（1~3最优）
    score += max(0, 1 - abs(props["LogP"] - 2) / 5)

    # TPSA（30~140最优）
    score += max(0, 1 - abs(props["TPSA"] - 90) / 100)

    # HBD/HBA
    score += 1 - min(props["HBD"], 5) / 5
    score += 1 - min(props["HBA"], 10) / 10

    # Rotatable bonds（越少越好）
    score += 1 - min(props["RotatableBonds"], 10) / 10

    return round(score / 6, 3)


# =========================
# ADMET 风险评分（科研版）
# =========================

def admet_risk_score(props):

    risk = 0

    if props["LogP"] > 5:
        risk += 0.2
    if props["TPSA"] > 140:
        risk += 0.2
    if props["MolWt"] > 500:
        risk += 0.2
    if props["RotatableBonds"] > 10:
        risk += 0.2
    if props["HBD"] > 5:
        risk += 0.1
    if props["HBA"] > 10:
        risk += 0.1

    return round(min(risk, 1.0), 3)


# =========================
# 总体 ADMET Score（最终评分）
# =========================

def admet_final_score(drug_score, risk_score):

    return round(drug_score * (1 - risk_score), 3)