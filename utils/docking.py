# utils/docking.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


DOCKING_SCORE_CANDIDATES = [
    "docking_score",
    "binding_affinity",
    "binding_affinity_kcal_mol",
    "binding_affinity_kcal/mol",
    "affinity",
    "score",
    "vina_score"
]


def normalize_column_name(col: str) -> str:
    """
    统一列名格式：
    例如 Binding Affinity kcal/mol -> binding_affinity_kcal_mol
    """
    col = str(col).strip().lower()
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.replace("/", "_")
    return col


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一 DataFrame 的列名。
    """
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    return df


def standardize_docking_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    将不同来源的 docking 结果统一成标准格式。
    必须至少包含：
    compound_id
    docking_score

    可选包含：
    smiles
    target
    pose_file
    interaction
    """
    df = normalize_columns(df)

    if "compound_id" not in df.columns:
        raise ValueError("docking_results.csv 必须包含 compound_id 列。")

    score_col = None
    for candidate in DOCKING_SCORE_CANDIDATES:
        if candidate in df.columns:
            score_col = candidate
            break

    if score_col is None:
        raise ValueError(
            "docking_results.csv 必须包含 docking_score 或 binding_affinity 等对接分数列。"
        )

    df = df.rename(columns={score_col: "docking_score"})

    df["docking_score"] = pd.to_numeric(df["docking_score"], errors="coerce")
    df = df.dropna(subset=["compound_id", "docking_score"])

    # docking score 通常越小越好，例如 -9.2 优于 -6.1
    df = df.sort_values("docking_score", ascending=True).reset_index(drop=True)
    df["docking_rank"] = range(1, len(df) + 1)

    return df


def load_docking_csv(file) -> pd.DataFrame:
    """
    读取上传的 docking CSV 文件，并标准化格式。
    file 可以是 Streamlit 上传文件，也可以是本地路径。
    """
    df = pd.read_csv(file)
    return standardize_docking_dataframe(df)


def save_docking_results(df: pd.DataFrame, output_path="results/docking_results.csv") -> Path:
    """
    保存标准化后的 docking 结果。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def make_docking_template() -> pd.DataFrame:
    """
    生成 docking_results.csv 模板。
    """
    return pd.DataFrame({
        "compound_id": ["C001", "C002", "C003"],
        "smiles": [
            "CCOc1ccc(N)cc1",
            "CCN(CC)CC",
            "CCOC(=O)c1ccccc1"
        ],
        "target": ["EGFR", "EGFR", "EGFR"],
        "docking_score": [-9.2, -8.1, -6.7],
        "pose_file": ["C001_pose.png", "C002_pose.png", "C003_pose.png"],
        "interaction": [
            "H-bond: MET793; hydrophobic interaction",
            "H-bond: LYS745",
            "hydrophobic interaction"
        ]
    })


def plot_docking_scores(df: pd.DataFrame, output_path="results/docking_score_plot.png"):
    """
    绘制 docking score 柱状图。
    注意：docking score 越低通常说明结合越稳定。
    """
    if df.empty:
        raise ValueError("没有可绘制的 docking 数据。")

    plot_df = df.sort_values("docking_score", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(plot_df["compound_id"].astype(str), plot_df["docking_score"])
    ax.set_xlabel("Compound ID")
    ax.set_ylabel("Docking score / Binding affinity (kcal/mol)")
    ax.set_title("Molecular Docking Score Ranking")
    ax.tick_params(axis="x", rotation=45)

    for i, value in enumerate(plot_df["docking_score"]):
        ax.text(i, value, f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top")

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, output_path