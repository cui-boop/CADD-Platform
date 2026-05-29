
from pathlib import Path
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, Draw
    RDKIT_AVAILABLE = True
except Exception:
    Chem = Descriptors = Crippen = Lipinski = Draw = None
    RDKIT_AVAILABLE = False

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def rdkit_available():
    return RDKIT_AVAILABLE


def make_demo_design_candidates():
    return pd.DataFrame([
        {
            "compound_id": "CHEMBL138940",
            "smiles": "O=C(/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cn1)NCCN1CCOCC1",
            "target": "EGFR",
            "qsar_probability": 0.99,
            "admet_score": 0.78,
            "docking_score": -9.5,
            "docking_confidence": 0.79,
            "recommendation": "高推荐",
            "key_residues": "MET793;LYS745;ASP855",
            "binding_interpretation": "该分子与 EGFR 形成氢键、盐桥和 π-π 作用，结合模式较稳定。",
        },
        {
            "compound_id": "CHEMBL53711",
            "smiles": "CN(C)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1",
            "target": "EGFR",
            "qsar_probability": 1.00,
            "admet_score": 0.74,
            "docking_score": -10.0,
            "docking_confidence": 0.73,
            "recommendation": "高推荐",
            "key_residues": "MET793;LEU718;PHE856",
            "binding_interpretation": "该分子 docking score 较好，并存在氢键、疏水作用和 π-π 作用。",
        },
        {
            "compound_id": "CHEMBL3808884",
            "smiles": "COC(=O)c1cc2cc(NCc3cc(NC(=O)c4cc(-n5cnc(C)c5)cc(C(F)(F)F)c4)ccc3C)ncc2[nH]1",
            "target": "EGFR",
            "qsar_probability": 0.985,
            "admet_score": 0.70,
            "docking_score": -9.2,
            "docking_confidence": 0.66,
            "recommendation": "中等推荐",
            "key_residues": "MET793;VAL726;LEU844",
            "binding_interpretation": "该分子对 EGFR 有较好的 docking score，但结构偏复杂，后续应关注 ADMET。",
        },
    ])


def read_csv_if_exists(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def load_candidate_pool_from_results():
    dfs = {
        "final": read_csv_if_exists("results/final_ranking.csv"),
        "docking": read_csv_if_exists("results/docking_results.csv"),
        "admet": read_csv_if_exists("results/admet_results.csv"),
        "qsar": read_csv_if_exists("results/qsar_predictions.csv"),
    }
    base = None
    for key in ["final", "qsar", "docking"]:
        df = dfs[key]
        if df is not None and {"compound_id", "smiles"}.issubset(df.columns):
            base = df.copy()
            break
    if base is None:
        return pd.DataFrame()
    base["compound_id"] = base["compound_id"].astype(str)
    if "target" not in base.columns:
        base["target"] = "Unknown"
    for key, keep in [
        ("admet", ["compound_id", "admet_score", "MolWt", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds"]),
        ("docking", ["compound_id", "docking_score", "docking_confidence", "key_residues", "binding_interpretation"]),
        ("qsar", ["compound_id", "qsar_probability", "qsar_prediction"]),
    ]:
        df = dfs[key]
        if df is not None and "compound_id" in df.columns:
            cols = [c for c in keep if c in df.columns]
            if len(cols) > 1:
                tmp = df[cols].copy().drop_duplicates("compound_id")
                base = base.merge(tmp, on="compound_id", how="left", suffixes=("", f"_{key}"))
    return base.loc[:, ~base.columns.duplicated()].copy()


def calc_desc(smiles):
    if not RDKIT_AVAILABLE:
        return {"valid_smiles": "未检测", "MolWt": None, "LogP": None, "TPSA": None, "HBD": None, "HBA": None, "RotatableBonds": None, "canonical_smiles": smiles}
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {"valid_smiles": False, "MolWt": None, "LogP": None, "TPSA": None, "HBD": None, "HBA": None, "RotatableBonds": None, "canonical_smiles": smiles}
    return {
        "valid_smiles": True,
        "MolWt": round(Descriptors.MolWt(mol), 3),
        "LogP": round(Crippen.MolLogP(mol), 3),
        "TPSA": round(Descriptors.TPSA(mol), 3),
        "HBD": int(Lipinski.NumHDonors(mol)),
        "HBA": int(Lipinski.NumHAcceptors(mol)),
        "RotatableBonds": int(Lipinski.NumRotatableBonds(mol)),
        "canonical_smiles": Chem.MolToSmiles(mol),
    }


def analyze_parent_profile(row):
    def f(name):
        try:
            v = row.get(name)
            return None if pd.isna(v) else float(v)
        except Exception:
            return None
    notes = []
    if f("qsar_probability") is not None:
        notes.append("QSAR 活性概率较高，建议保留核心骨架。" if f("qsar_probability") >= 0.8 else "QSAR 活性概率一般，需增强活性相关结构特征。")
    if f("admet_score") is not None:
        notes.append("ADMET 表现尚可，优化时避免显著增加分子量和 LogP。" if f("admet_score") >= 0.6 else "ADMET score 偏低，应优先改善类药性。")
    if f("docking_score") is not None:
        notes.append("docking score 较好，可围绕关键相互作用做小幅结构优化。" if f("docking_score") <= -9 else "docking score 不够理想，可尝试增加氢键或疏水匹配。")
    if f("docking_confidence") is not None and f("docking_confidence") < 0.55:
        notes.append("docking confidence 偏低，说明结合能和相互作用支持不足。")
    if not notes:
        notes.append("当前信息有限，建议围绕外围取代基做保守类似物设计。")
    return notes


def _mol_to_smiles(mol):
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _replace_atom(smiles, from_num, to_num):
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == from_num:
            atom.SetAtomicNum(to_num)
            return _mol_to_smiles(rw.GetMol())
    return None


def _add_to_aromatic(smiles, atomic_num):
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 6 and atom.GetIsAromatic() and atom.GetTotalNumHs() > 0:
            idx = rw.AddAtom(Chem.Atom(atomic_num))
            rw.AddBond(atom.GetIdx(), idx, Chem.BondType.SINGLE)
            return _mol_to_smiles(rw.GetMol())
    return None


def _fallback_designs(smiles):
    out = []
    if "Cl" in smiles:
        out.append((smiles.replace("Cl", "F", 1), "卤素替换：Cl → F", "减小取代基体积，可能改善代谢稳定性。"))
    if "Br" in smiles:
        out.append((smiles.replace("Br", "Cl", 1), "卤素替换：Br → Cl", "降低分子量并尽量保留疏水作用。"))
    if "C(F)(F)F" in smiles:
        out.append((smiles.replace("C(F)(F)F", "F", 1), "降低疏水性：CF3 → F", "减少强疏水基团，改善潜在溶解性。"))
    return out


def generate_rule_based_designs(parent_row, max_designs=6, emphasis="均衡优化"):
    parent_id = str(parent_row.get("compound_id", "Parent"))
    smiles = str(parent_row.get("smiles", "")).strip()
    if not smiles:
        raise ValueError("缺少 smiles，无法进行分子设计。")
    parent_desc = calc_desc(smiles)
    issues = "；".join(analyze_parent_profile(parent_row))
    raw = []
    if RDKIT_AVAILABLE:
        raw += [
            (_replace_atom(smiles, 17, 9), "卤素替换：Cl → F", "减小取代基体积并微调电子效应。"),
            (_replace_atom(smiles, 35, 17), "卤素替换：Br → Cl", "降低分子量，同时尽量保留疏水相互作用。"),
            (_replace_atom(smiles, 35, 9), "卤素替换：Br → F", "明显降低分子量和疏水性。"),
            (_add_to_aromatic(smiles, 9), "芳香环引入 F", "增强代谢稳定性并微调结合口袋适配性。"),
            (_add_to_aromatic(smiles, 8), "芳香环引入 OH", "增加氢键供体或受体，有助于增强关键氢键。"),
            (_add_to_aromatic(smiles, 6), "芳香环引入 CH3", "增强疏水相互作用，但需关注 LogP。"),
        ]
    raw += _fallback_designs(smiles)
    rows, seen = [], set()
    for smi, strategy, improve in raw:
        if not smi or smi == smiles:
            continue
        desc = calc_desc(smi)
        key = desc.get("canonical_smiles") or smi
        if key in seen:
            continue
        seen.add(key)
        def delta(k):
            return None if desc.get(k) is None or parent_desc.get(k) is None else round(desc[k] - parent_desc[k], 3)
        rows.append({
            "design_id": f"D{len(rows)+1:03d}",
            "parent_compound_id": parent_id,
            "parent_smiles": smiles,
            "designed_smiles": smi,
            "canonical_smiles": desc.get("canonical_smiles"),
            "design_strategy": strategy,
            "expected_improvement": improve,
            "design_emphasis": emphasis,
            "valid_smiles": desc.get("valid_smiles"),
            "MolWt": desc.get("MolWt"),
            "LogP": desc.get("LogP"),
            "TPSA": desc.get("TPSA"),
            "HBD": desc.get("HBD"),
            "HBA": desc.get("HBA"),
            "RotatableBonds": desc.get("RotatableBonds"),
            "MolWt_delta": delta("MolWt"),
            "LogP_delta": delta("LogP"),
            "TPSA_delta": delta("TPSA"),
            "parent_profile_summary": issues,
        })
        if len(rows) >= max_designs:
            break
    if not rows:
        rows.append({"design_id":"D000","parent_compound_id":parent_id,"parent_smiles":smiles,"designed_smiles":"","canonical_smiles":"","design_strategy":"未生成新 SMILES","expected_improvement":"当前结构未匹配到自动规则，建议手动围绕关键残基和 ADMET 问题优化。","design_emphasis":emphasis,"valid_smiles":False,"parent_profile_summary":issues})
    return pd.DataFrame(rows)


def save_designed_molecules(df, output_path="results/designed_molecules.csv"):
    p = Path(output_path)
    p.parent.mkdir(exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def get_rdkit_mol_image(smiles, size=(300, 220)):
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)
