# utils/scoring.py

from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def normalize_docking_score(score: float) -> float:
    """
    将 docking score 归一化到 0-1。
    docking score 一般为负值，越低越好。

    简单规则：
    score <= -10: 1.0
    score >= -5 : 0.0
    -10 到 -5 之间线性归一化
    """
    try:
        score = float(score)
    except Exception:
        return 0.0

    if score <= -10:
        return 1.0
    elif score >= -5:
        return 0.0
    else:
        return (-5 - score) / 5


def get_recommendation_level(total_score: float) -> str:
    """
    根据综合评分输出推荐等级。
    """
    if total_score >= 0.75:
        return "高推荐"
    elif total_score >= 0.50:
        return "中等推荐"
    else:
        return "低推荐"


def calculate_total_score(
    qsar_probability: float,
    admet_score: float,
    docking_score: float,
    w_qsar: float = 0.4,
    w_admet: float = 0.3,
    w_docking: float = 0.3
):
    """
    综合评分公式：

    total_score = 0.4 × QSAR 活性概率
                + 0.3 × ADMET score
                + 0.3 × docking_score_norm
    """
    qsar_probability = float(qsar_probability)
    admet_score = float(admet_score)
    docking_norm = normalize_docking_score(docking_score)

    total_score = (
        w_qsar * qsar_probability
        + w_admet * admet_score
        + w_docking * docking_norm
    )

    total_score = round(float(total_score), 4)
    recommendation = get_recommendation_level(total_score)

    return total_score, recommendation, round(docking_norm, 4)


def standardize_qsar_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一 QSAR 结果格式。
    需要包含：
    compound_id, smiles, target, qsar_probability
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required = ["compound_id", "smiles", "target", "qsar_probability"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"QSAR 结果缺少必要列：{missing}")

    df["qsar_probability"] = pd.to_numeric(df["qsar_probability"], errors="coerce")
    df = df.dropna(subset=["compound_id", "qsar_probability"])

    return df


def standardize_admet_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一 ADMET 结果格式。
    需要包含：
    compound_id, admet_score
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required = ["compound_id", "admet_score"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"ADMET 结果缺少必要列：{missing}")

    df["admet_score"] = pd.to_numeric(df["admet_score"], errors="coerce")
    df = df.dropna(subset=["compound_id", "admet_score"])

    return df


def standardize_docking_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一 docking 结果格式。
    需要包含：
    compound_id, docking_score
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required = ["compound_id", "docking_score"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Docking 结果缺少必要列：{missing}")

    df["docking_score"] = pd.to_numeric(df["docking_score"], errors="coerce")
    df = df.dropna(subset=["compound_id", "docking_score"])

    return df


def merge_and_score(qsar_df: pd.DataFrame, admet_df: pd.DataFrame, docking_df: pd.DataFrame) -> pd.DataFrame:
    """
    合并 QSAR、ADMET 和 docking 结果，并计算综合评分。
    """
    qsar_df = standardize_qsar_df(qsar_df)
    admet_df = standardize_admet_df(admet_df)
    docking_df = standardize_docking_df(docking_df)

    merged = qsar_df.merge(
        admet_df[["compound_id", "admet_score"]],
        on="compound_id",
        how="inner"
    )

    merged = merged.merge(
        docking_df[["compound_id", "docking_score"]],
        on="compound_id",
        how="inner"
    )

    if merged.empty:
        raise ValueError(
            "三个文件没有匹配到相同的 compound_id，请检查 QSAR、ADMET 和 docking 文件中的 compound_id 是否一致。"
        )

    results = []
    for _, row in merged.iterrows():
        total_score, recommendation, docking_norm = calculate_total_score(
            qsar_probability=row["qsar_probability"],
            admet_score=row["admet_score"],
            docking_score=row["docking_score"]
        )

        row_dict = row.to_dict()
        row_dict["docking_score_norm"] = docking_norm
        row_dict["total_score"] = total_score
        row_dict["recommendation"] = recommendation
        results.append(row_dict)

    final_df = pd.DataFrame(results)

    final_cols = [
        "compound_id",
        "smiles",
        "target",
        "qsar_probability",
        "admet_score",
        "docking_score",
        "docking_score_norm",
        "total_score",
        "recommendation"
    ]

    final_df = final_df[final_cols]
    final_df = final_df.sort_values("total_score", ascending=False).reset_index(drop=True)
    final_df["final_rank"] = range(1, len(final_df) + 1)

    return final_df


def save_final_ranking(final_df: pd.DataFrame, output_path="results/final_ranking.csv") -> Path:
    """
    保存最终综合评分结果。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def make_demo_qsar_df() -> pd.DataFrame:
    return pd.DataFrame({
        "compound_id": ["C001", "C002", "C003"],
        "smiles": [
            "CCOc1ccc(N)cc1",
            "CCN(CC)CC",
            "CCOC(=O)c1ccccc1"
        ],
        "target": ["EGFR", "EGFR", "EGFR"],
        "qsar_probability": [0.86, 0.62, 0.35],
        "qsar_prediction": ["Active", "Active", "Inactive"]
    })


def make_demo_admet_df() -> pd.DataFrame:
    return pd.DataFrame({
        "compound_id": ["C001", "C002", "C003"],
        "smiles": [
            "CCOc1ccc(N)cc1",
            "CCN(CC)CC",
            "CCOC(=O)c1ccccc1"
        ],
        "MolWt": [240.3, 101.2, 150.2],
        "LogP": [2.8, 1.1, 2.3],
        "TPSA": [65.2, 12.0, 26.3],
        "HBD": [2, 0, 0],
        "HBA": [5, 1, 2],
        "RotatableBonds": [4, 3, 3],
        "Lipinski_Passed": [4, 4, 4],
        "admet_score": [0.85, 0.78, 0.70]
    })


def make_demo_docking_df() -> pd.DataFrame:
    return pd.DataFrame({
        "compound_id": ["C001", "C002", "C003"],
        "smiles": [
            "CCOc1ccc(N)cc1",
            "CCN(CC)CC",
            "CCOC(=O)c1ccccc1"
        ],
        "target": ["EGFR", "EGFR", "EGFR"],
        "docking_score": [-9.2, -7.4, -6.1]
    })