
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, Draw
    RDKIT_AVAILABLE = True
except Exception:
    Chem = Descriptors = Crippen = Lipinski = rdMolDescriptors = Draw = None
    RDKIT_AVAILABLE = False


RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

TOP10_DOCKING_PATH = RESULT_DIR / "top10_docking_candidates.csv"
DOCKING_RESULTS_PATH = RESULT_DIR / "docking_results.csv"

QSAR_RESULT_PATHS = [
    RESULT_DIR / "generated_qsar_predictions.csv",
    RESULT_DIR / "qsar_predictions.csv",
]

ADMET_RESULT_PATHS = [
    RESULT_DIR / "generated_druglikeness_predictions.csv",
    RESULT_DIR / "generated_screening_results.csv",
    RESULT_DIR / "admet_results.csv",
]

DESIGNED_FULL_OUTPUT = RESULT_DIR / "designed_molecules.csv"
DESIGNED_SIMPLE_OUTPUT = RESULT_DIR / "designed_molecules_for_prediction.csv"


def rdkit_available() -> bool:
    return RDKIT_AVAILABLE


def read_csv_if_exists(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None


def _norm_col(col: str) -> str:
    return str(col).strip().replace(" ", "_").replace("-", "_").replace("/", "_").lower()


def pick_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    direct = {str(c).strip(): c for c in df.columns}
    for cand in candidates:
        if cand in direct:
            return direct[cand]
    norm_map = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if smiles is None or pd.isna(smiles) or str(smiles).strip() == "":
        return None
    smi = str(smiles).strip()
    if not RDKIT_AVAILABLE:
        return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def calculate_descriptors(smiles: str) -> Dict:
    can = canonicalize_smiles(smiles)
    empty = dict(
        canonical_smiles="",
        valid_smiles=False,
        MolWt=None,
        LogP=None,
        TPSA=None,
        HBD=None,
        HBA=None,
        RotatableBonds=None,
        RingCount=None,
        AromaticRings=None,
        HeavyAtomCount=None,
        Rule_Score=None,
        Lipinski_Passed=None,
    )
    if can is None:
        return empty

    if not RDKIT_AVAILABLE:
        empty.update(canonical_smiles=can, valid_smiles=True)
        return empty

    mol = Chem.MolFromSmiles(can)
    molwt = round(Descriptors.MolWt(mol), 3)
    logp = round(Crippen.MolLogP(mol), 3)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 3)
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rot = int(Lipinski.NumRotatableBonds(mol))
    ring_count = int(rdMolDescriptors.CalcNumRings(mol))
    aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    rules = [molwt < 500, logp < 5, tpsa < 140, hbd <= 5, hba <= 10, rot < 10]
    rule_score = round(sum(int(x) for x in rules) / len(rules), 4)

    return dict(
        canonical_smiles=can,
        valid_smiles=True,
        MolWt=molwt,
        LogP=logp,
        TPSA=tpsa,
        HBD=hbd,
        HBA=hba,
        RotatableBonds=rot,
        RingCount=ring_count,
        AromaticRings=aromatic_rings,
        HeavyAtomCount=heavy_atoms,
        Rule_Score=rule_score,
        Lipinski_Passed=bool(rule_score >= 0.67),
    )


def _coalesce(df: pd.DataFrame, base_col: str, alt_col: str) -> pd.DataFrame:
    if base_col in df.columns and alt_col in df.columns:
        df[base_col] = df[base_col].combine_first(df[alt_col])
        df = df.drop(columns=[alt_col])
    elif alt_col in df.columns and base_col not in df.columns:
        df = df.rename(columns={alt_col: base_col})
    return df


def _standardize_qsar_file(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    compound_col = pick_existing_column(df, ["compound_id", "Compound_ID", "id", "ID"])
    smiles_col = pick_existing_column(df, ["smiles", "canonical_smiles", "SMILES"])
    target_col = pick_existing_column(df, ["target", "Target"])
    qsar_col = pick_existing_column(df, ["qsar_probability", "prob_active", "Active_Probability"])
    pred_col = pick_existing_column(df, ["qsar_prediction", "prediction", "Prediction"])
    if compound_col is None:
        return pd.DataFrame()
    out = pd.DataFrame({"compound_id": df[compound_col].astype(str)})
    if smiles_col:
        out["smiles"] = df[smiles_col].astype(str)
    if target_col:
        out["target"] = df[target_col].astype(str)
    if qsar_col:
        out["qsar_probability"] = pd.to_numeric(df[qsar_col], errors="coerce")
    if pred_col:
        out["qsar_prediction"] = df[pred_col].astype(str)
    return out.drop_duplicates("compound_id", keep="first")


def _standardize_admet_file(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    compound_col = pick_existing_column(df, ["compound_id", "Compound_ID", "id", "ID"])
    smiles_col = pick_existing_column(df, ["smiles", "canonical_smiles", "SMILES", "smiles_admet", "canonical_smiles_admet"])
    admet_col = pick_existing_column(df, ["成药性筛选分数", "ADMET_Score", "ADMET_Score_admet", "admet_score", "druglikeness_score"])
    if compound_col is None:
        return pd.DataFrame()
    out = pd.DataFrame({"compound_id": df[compound_col].astype(str)})
    if smiles_col:
        out["smiles"] = df[smiles_col].astype(str)
    if admet_col:
        out["admet_score"] = pd.to_numeric(df[admet_col], errors="coerce")
    for name in ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds", "Lipinski_Passed"]:
        col = pick_existing_column(df, [name, f"{name}_admet"])
        if col:
            out[name] = pd.to_numeric(df[col], errors="coerce") if name != "Lipinski_Passed" else df[col]
    return out.drop_duplicates("compound_id", keep="first")


def load_qsar_supplement() -> pd.DataFrame:
    frames = []
    for path in QSAR_RESULT_PATHS:
        df = read_csv_if_exists(path)
        if df is not None and not df.empty:
            std = _standardize_qsar_file(df)
            if not std.empty:
                frames.append(std)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("compound_id", keep="first")


def load_admet_supplement() -> pd.DataFrame:
    frames = []
    for path in ADMET_RESULT_PATHS:
        df = read_csv_if_exists(path)
        if df is not None and not df.empty:
            std = _standardize_admet_file(df)
            if not std.empty:
                frames.append(std)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("compound_id", keep="first")


def _standardize_docking_results(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    compound_col = pick_existing_column(df, ["compound_id", "Compound_ID", "id", "ID"])
    smiles_col = pick_existing_column(df, ["smiles", "canonical_smiles", "SMILES"])
    target_col = pick_existing_column(df, ["target", "Target", "protein", "receptor"])
    score_col = pick_existing_column(df, ["docking_score", "binding_affinity", "binding_affinity_kcal_mol", "vina_score", "score"])
    conf_col = pick_existing_column(df, ["docking_confidence", "confidence"])
    residues_col = pick_existing_column(df, ["key_residues", "residues"])
    interaction_col = pick_existing_column(df, ["interaction", "interactions"])
    if score_col is None:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["compound_id"] = df[compound_col].astype(str) if compound_col else ""
    if smiles_col:
        out["smiles"] = df[smiles_col].astype(str)
        out["canonical_smiles"] = out["smiles"].apply(canonicalize_smiles)
    else:
        out["smiles"] = ""
        out["canonical_smiles"] = ""
    out["target"] = df[target_col].astype(str) if target_col else ""
    out["docking_score"] = pd.to_numeric(df[score_col], errors="coerce")
    out["docking_confidence"] = pd.to_numeric(df[conf_col], errors="coerce") if conf_col else pd.NA
    out["key_residues"] = df[residues_col].astype(str) if residues_col else ""
    out["interaction"] = df[interaction_col].astype(str) if interaction_col else ""
    out = out.dropna(subset=["docking_score"]).copy()
    if out.empty:
        return pd.DataFrame()
    out = out.sort_values("docking_score", ascending=True)
    if out["compound_id"].astype(str).str.len().gt(0).any():
        return out.drop_duplicates("compound_id", keep="first")
    return out.drop_duplicates("canonical_smiles", keep="first")


def load_docking_supplement() -> pd.DataFrame:
    df = read_csv_if_exists(DOCKING_RESULTS_PATH)
    if df is None or df.empty:
        return pd.DataFrame()
    return _standardize_docking_results(df)


def standardize_candidate_file(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    raw = df.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    compound_col = pick_existing_column(raw, ["compound_id", "Compound_ID", "id", "ID"])
    smiles_col = pick_existing_column(raw, ["smiles", "canonical_smiles", "SMILES", "smiles_admet", "canonical_smiles_admet"])
    target_col = pick_existing_column(raw, ["target", "Target", "protein", "receptor"])
    docking_col = pick_existing_column(raw, ["docking_score", "binding_affinity", "vina_score", "score"])
    rank_col = pick_existing_column(raw, ["rank_for_docking", "rank", "Rank"])
    qsar_col = pick_existing_column(raw, ["qsar_probability", "prob_active", "Active_Probability"])
    pred_col = pick_existing_column(raw, ["qsar_prediction", "prediction", "Prediction"])
    admet_col = pick_existing_column(raw, ["成药性筛选分数", "ADMET_Score", "ADMET_Score_admet", "admet_score", "druglikeness_score"])
    if compound_col is None:
        raise ValueError("候选分子文件必须包含 compound_id 列。")
    out = pd.DataFrame()
    out["rank_for_docking"] = raw[rank_col] if rank_col else range(1, len(raw) + 1)
    out["compound_id"] = raw[compound_col].astype(str)
    out["smiles"] = raw[smiles_col].astype(str) if smiles_col else pd.NA
    out["target"] = raw[target_col].astype(str) if target_col else ""
    out["docking_score"] = pd.to_numeric(raw[docking_col], errors="coerce") if docking_col else pd.NA
    out["qsar_probability"] = pd.to_numeric(raw[qsar_col], errors="coerce") if qsar_col else pd.NA
    out["qsar_prediction"] = raw[pred_col].astype(str) if pred_col else ""
    out["admet_score"] = pd.to_numeric(raw[admet_col], errors="coerce") if admet_col else pd.NA
    return out


def enrich_candidates(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df is None or candidate_df.empty:
        return pd.DataFrame()
    df = candidate_df.copy()
    df["compound_id"] = df["compound_id"].astype(str)

    qsar_df = load_qsar_supplement()
    if not qsar_df.empty:
        df = df.merge(qsar_df, on="compound_id", how="left", suffixes=("", "_qsar"))
        for col in ["smiles", "target", "qsar_probability", "qsar_prediction"]:
            df = _coalesce(df, col, f"{col}_qsar")

    admet_df = load_admet_supplement()
    if not admet_df.empty:
        df = df.merge(admet_df, on="compound_id", how="left", suffixes=("", "_admet"))
        for col in ["smiles", "admet_score", "MolWt", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds", "Lipinski_Passed"]:
            df = _coalesce(df, col, f"{col}_admet")

    docking_df = load_docking_supplement()
    if not docking_df.empty:
        df = df.merge(
            docking_df[["compound_id", "target", "docking_score", "docking_confidence", "key_residues", "interaction"]],
            on="compound_id",
            how="left",
            suffixes=("", "_dock"),
        )
        for col in ["target", "docking_score", "docking_confidence", "key_residues", "interaction"]:
            df = _coalesce(df, col, f"{col}_dock")

    for col in ["target", "qsar_prediction", "key_residues", "interaction"]:
        if col not in df.columns:
            df[col] = ""
    for col in ["docking_score", "docking_confidence", "qsar_probability", "admet_score"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "smiles" not in df.columns:
        df["smiles"] = pd.NA

    df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
    desc_df = pd.DataFrame([calculate_descriptors(s) for s in df["smiles"]])
    for col in desc_df.columns:
        if col not in df.columns:
            df[col] = desc_df[col]
        elif col in ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds", "Lipinski_Passed"]:
            df[col] = df[col].combine_first(desc_df[col])

    df["rank_for_docking"] = pd.to_numeric(df["rank_for_docking"], errors="coerce")
    df["design_priority_score"] = df.apply(calculate_design_priority, axis=1)
    return df.sort_values(["rank_for_docking", "design_priority_score"], ascending=[True, False]).reset_index(drop=True)


def docking_score_to_norm(score) -> float:
    if pd.isna(score):
        return 0.5
    score = float(score)
    if score <= -10:
        return 1.0
    if score >= -5:
        return 0.0
    return round((-5 - score) / 5, 4)


def calculate_design_priority(row) -> float:
    q = 0.5 if pd.isna(row.get("qsar_probability", pd.NA)) else float(row.get("qsar_probability"))
    a = 0.5 if pd.isna(row.get("admet_score", pd.NA)) else float(row.get("admet_score"))
    d = docking_score_to_norm(row.get("docking_score", pd.NA))
    r = 0.5 if pd.isna(row.get("Rule_Score", pd.NA)) else float(row.get("Rule_Score"))
    score = 0.45 * q + 0.30 * a + 0.20 * d + 0.05 * r
    return round(max(0.0, min(1.0, score)), 4)


def load_top10_candidates(source_mode: str = "default", uploaded_file=None, manual_data: Optional[Dict] = None) -> Tuple[pd.DataFrame, str]:
    if source_mode == "upload":
        if uploaded_file is None:
            return pd.DataFrame(), ""
        raw = pd.read_csv(uploaded_file)
        return enrich_candidates(standardize_candidate_file(raw)), "用户上传 CSV"

    if source_mode == "manual":
        if not manual_data:
            return pd.DataFrame(), ""
        raw = pd.DataFrame([manual_data])
        return enrich_candidates(standardize_candidate_file(raw)), "手动输入候选分子"

    raw = read_csv_if_exists(TOP10_DOCKING_PATH)
    if raw is None or raw.empty:
        return pd.DataFrame(), str(TOP10_DOCKING_PATH)
    return enrich_candidates(standardize_candidate_file(raw)), str(TOP10_DOCKING_PATH)


def make_demo_top10_candidates() -> pd.DataFrame:
    demo = pd.DataFrame([
        {"rank_for_docking": 1, "compound_id": "GRU_GEN_0001", "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OC", "target": "EGFR", "docking_score": -9.2, "qsar_probability": 0.93, "qsar_prediction": "Active", "成药性筛选分数": 0.83},
        {"rank_for_docking": 2, "compound_id": "GRU_GEN_0002", "smiles": "CN(C)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1", "target": "EGFR", "docking_score": -8.8, "qsar_probability": 0.88, "qsar_prediction": "Active", "成药性筛选分数": 0.67},
        {"rank_for_docking": 3, "compound_id": "GRU_GEN_0003", "smiles": "O=C(Nc1ccc(F)c(Cl)c1)c1cc(-n2cnc(C)c2)ccc1C(F)(F)F", "target": "EGFR", "docking_score": -8.1, "qsar_probability": 0.79, "qsar_prediction": "Active", "成药性筛选分数": 0.67},
    ])
    return enrich_candidates(standardize_candidate_file(demo))


def diagnose_parent_candidate(row: Dict) -> List[str]:
    msgs = []
    cid = row.get("compound_id", "该分子")
    qsar = row.get("qsar_probability", pd.NA)
    admet = row.get("admet_score", pd.NA)
    docking = row.get("docking_score", pd.NA)
    if pd.notna(qsar):
        q = float(qsar)
        msgs.append(f"{cid} 的 QSAR 活性概率较高，适合作为结构优化母体。" if q >= 0.8 else ("QSAR 活性概率达到初筛阈值，但仍需要进一步优化。" if q >= 0.6 else "QSAR 活性概率偏低，不建议作为优先设计母体。"))
    else:
        msgs.append("未读取到 QSAR 活性概率，可在完成 QSAR 活性预测后重新加载。")
    if pd.notna(admet):
        a = float(admet)
        msgs.append("成药性筛选分数较高，设计时应避免破坏当前类药性。" if a >= 0.85 else ("成药性筛选分数达到推荐阈值，仍有进一步改善空间。" if a >= 0.65 else "成药性筛选分数偏低，应优先改善类药性。"))
    else:
        msgs.append("未读取到成药性筛选分数，可在完成成药性筛选后重新加载。")
    if pd.notna(docking):
        d = float(docking)
        msgs.append("docking score 较好，建议保留当前核心骨架并进行温和取代基优化。" if d <= -9 else ("docking score 中等，可尝试增强氢键、疏水作用或芳香相互作用。" if d <= -7 else "docking score 不够理想，后续设计应重点改善结合模式。"))
    else:
        msgs.append("未读取到 docking_score，可补充 docking_results.csv 或在读取框中输入。")
    for name, limit, hint in [
        ("MolWt", 500, "分子量偏高，建议减少大体积取代基或缩短侧链。"),
        ("LogP", 5, "LogP 偏高，建议降低强疏水基团。"),
        ("TPSA", 140, "TPSA 偏高，可能影响膜通透性。"),
        ("RotatableBonds", 10, "可旋转键较多，建议减少柔性侧链。"),
    ]:
        val = row.get(name, pd.NA)
        if pd.notna(val) and float(val) >= limit:
            msgs.append(hint)
    return msgs or ["该分子没有明显规则警告，可进行小幅类似物设计。"]


def _mol_to_smiles(mol):
    if mol is None or not RDKIT_AVAILABLE:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _replace_first_atom(smiles, from_atomic_num, to_atomic_num):
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == from_atomic_num:
            atom.SetAtomicNum(to_atomic_num)
            return _mol_to_smiles(rw.GetMol())
    return None


def _add_substituent_to_aromatic(smiles, atomic_num):
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetIsAromatic() and atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() > 0:
            idx = rw.AddAtom(Chem.Atom(atomic_num))
            rw.AddBond(atom.GetIdx(), idx, Chem.BondType.SINGLE)
            return _mol_to_smiles(rw.GetMol())
    return None


def _string_designs(smiles):
    s = str(smiles)
    out = []
    if "Cl" in s:
        out.append((s.replace("Cl", "F", 1), "卤素替换：Cl → F", "减小取代基体积并微调电子效应。"))
    if "Br" in s:
        out.append((s.replace("Br", "Cl", 1), "卤素替换：Br → Cl", "降低分子量并保留一定疏水性。"))
    if "C(F)(F)F" in s:
        out.append((s.replace("C(F)(F)F", "F", 1), "降低疏水性：CF3 → F", "减少强疏水基团，降低 LogP 风险。"))
    if "OC" in s:
        out.append((s.replace("OC", "O", 1), "缩短烷氧基侧链", "降低分子体积和疏水性。"))
    return out


def delta_comment(parent, new, name):
    if parent is None or new is None or pd.isna(parent) or pd.isna(new):
        return ""
    d = round(float(new) - float(parent), 3)
    if d == 0:
        return f"{name} 基本不变。"
    if name in ["MolWt", "LogP", "RotatableBonds"]:
        return f"{name} 下降 {abs(d)}，通常有利于类药性。" if d < 0 else f"{name} 上升 {d}，需复查风险。"
    if name == "TPSA":
        return f"TPSA 下降 {abs(d)}，可能有利于通透性。" if d < 0 else f"TPSA 上升 {d}，可能增强氢键但需关注通透性。"
    return f"{name} 变化 {d}。"


def generate_rule_based_designs(parent_row, max_designs=8, emphasis="均衡优化"):
    pid = str(parent_row.get("compound_id", "TOP10_PARENT"))
    psmi = str(parent_row.get("smiles", "")).strip()
    if not psmi:
        raise ValueError("母体分子缺少 smiles，无法生成设计分子。")
    pdesc = calculate_descriptors(psmi)
    diagnosis = "；".join(diagnose_parent_candidate(parent_row))
    raw = []
    if RDKIT_AVAILABLE:
        raw += [
            (_replace_first_atom(psmi, 17, 9), "卤素替换：Cl → F", "减小取代基体积并保持卤素取代特征。"),
            (_replace_first_atom(psmi, 35, 17), "卤素替换：Br → Cl", "降低分子量并保留一定疏水性。"),
            (_replace_first_atom(psmi, 35, 9), "卤素替换：Br → F", "明显降低分子量和疏水性。"),
            (_add_substituent_to_aromatic(psmi, 9), "芳香环引入 F", "改善代谢稳定性并微调疏水性。"),
            (_add_substituent_to_aromatic(psmi, 8), "芳香环引入 OH", "增加潜在氢键作用，但需关注 TPSA。"),
            (_add_substituent_to_aromatic(psmi, 7), "芳香环引入 NH2", "增加氢键供体/受体，可能增强结合。"),
            (_add_substituent_to_aromatic(psmi, 6), "芳香环引入 CH3", "增强疏水作用，但需关注 LogP。"),
        ]
    raw += _string_designs(psmi)
    follow_map = {
        "改善成药性": "优先重新进行 ADMET 筛选，并观察 MolWt、LogP、TPSA 和可旋转键是否改善。",
        "增强活性": "优先重新进行 QSAR 活性预测，并进一步结合 docking score 判断。",
        "为对接优化": "建议后续重点查看 docking score、关键残基和相互作用类型。",
        "降低 LogP / 分子量": "重点比较 LogP_delta 和 MolWt_delta 是否下降。",
    }
    follow = follow_map.get(emphasis, "建议重新进入 QSAR 和 ADMET 模块进行二次评价。")
    rows, seen = [], set()
    for smi, strategy, reason in raw:
        if not smi or smi == psmi:
            continue
        desc = calculate_descriptors(smi)
        can = desc.get("canonical_smiles") or ""
        if not can or can in seen:
            continue
        seen.add(can)
        did = f"DES_{pid}_{len(rows) + 1:03d}"
        comment_items = []
        for k in ["MolWt", "LogP", "TPSA", "RotatableBonds"]:
            item = delta_comment(pdesc.get(k), desc.get(k), k)
            if item:
                comment_items.append(item)
        rows.append({
            "design_id": did,
            "parent_compound_id": pid,
            "parent_rank_for_docking": parent_row.get("rank_for_docking", pd.NA),
            "parent_smiles": psmi,
            "designed_compound_id": did,
            "designed_smiles": can,
            "design_strategy": strategy,
            "design_reason": reason,
            "design_emphasis": emphasis,
            "property_change_comment": " ".join(comment_items),
            "expected_followup": follow,
            "valid_smiles": desc.get("valid_smiles"),
            "MolWt": desc.get("MolWt"),
            "LogP": desc.get("LogP"),
            "TPSA": desc.get("TPSA"),
            "HBD": desc.get("HBD"),
            "HBA": desc.get("HBA"),
            "RotatableBonds": desc.get("RotatableBonds"),
            "Rule_Score": desc.get("Rule_Score"),
            "Lipinski_Passed": desc.get("Lipinski_Passed"),
            "MolWt_delta": None if desc.get("MolWt") is None or pdesc.get("MolWt") is None else round(desc.get("MolWt") - pdesc.get("MolWt"), 3),
            "LogP_delta": None if desc.get("LogP") is None or pdesc.get("LogP") is None else round(desc.get("LogP") - pdesc.get("LogP"), 3),
            "TPSA_delta": None if desc.get("TPSA") is None or pdesc.get("TPSA") is None else round(desc.get("TPSA") - pdesc.get("TPSA"), 3),
            "RotatableBonds_delta": None if desc.get("RotatableBonds") is None or pdesc.get("RotatableBonds") is None else round(desc.get("RotatableBonds") - pdesc.get("RotatableBonds"), 3),
            "parent_qsar_probability": parent_row.get("qsar_probability", pd.NA),
            "parent_qsar_prediction": parent_row.get("qsar_prediction", ""),
            "parent_admet_score": parent_row.get("admet_score", pd.NA),
            "parent_docking_score": parent_row.get("docking_score", pd.NA),
            "parent_design_priority_score": parent_row.get("design_priority_score", pd.NA),
            "parent_diagnosis": diagnosis,
        })
        if len(rows) >= max_designs:
            break
    if not rows:
        rows.append({
            "design_id": f"DES_{pid}_000",
            "parent_compound_id": pid,
            "parent_smiles": psmi,
            "designed_compound_id": "",
            "designed_smiles": "",
            "design_strategy": "未生成新结构",
            "design_reason": "当前分子未匹配到可自动执行的轻量规则，建议人工围绕活性片段进行结构改造。",
            "design_emphasis": emphasis,
            "property_change_comment": "",
            "expected_followup": "可换用其他 Top 10 分子作为母体，或人工设计类似物。",
            "valid_smiles": False,
            "parent_diagnosis": diagnosis,
        })
    return pd.DataFrame(rows)


def save_designed_molecules(df):
    df.to_csv(DESIGNED_FULL_OUTPUT, index=False, encoding="utf-8-sig")
    valid = df[df["designed_compound_id"].astype(str).str.len().gt(0) & df["designed_smiles"].astype(str).str.len().gt(0)].copy()
    if valid.empty:
        simple = pd.DataFrame(columns=["compound_id", "smiles"])
    else:
        simple = valid[["designed_compound_id", "designed_smiles"]].rename(columns={"designed_compound_id": "compound_id", "designed_smiles": "smiles"})
    simple.to_csv(DESIGNED_SIMPLE_OUTPUT, index=False, encoding="utf-8-sig")
    return DESIGNED_FULL_OUTPUT, DESIGNED_SIMPLE_OUTPUT


def get_mol_image(smiles, size=(280, 210)):
    if not RDKIT_AVAILABLE:
        return None
    can = canonicalize_smiles(smiles)
    if can is None:
        return None
    mol = Chem.MolFromSmiles(can)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)
