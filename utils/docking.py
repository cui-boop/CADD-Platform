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
    col = str(col).strip().lower()
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.replace("/", "_")
    return col


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    return df


def standardize_docking_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    if "compound_id" not in df.columns:
        raise ValueError("docking_results.csv 必须包含 compound_id 列。")

    if "target" not in df.columns:
        df["target"] = "Unknown"

    score_col = None
    for candidate in DOCKING_SCORE_CANDIDATES:
        if candidate in df.columns:
            score_col = candidate
            break

    if score_col is None:
        raise ValueError("docking_results.csv 必须包含 docking_score 或 binding_affinity 等对接分数列。")

    df = df.rename(columns={score_col: "docking_score"})

    if "smiles" not in df.columns:
        df["smiles"] = ""

    optional_count_cols = [
        "hbond_count",
        "hydrophobic_count",
        "pi_pi_count",
        "salt_bridge_count"
    ]

    for col in optional_count_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in ["key_residues", "interaction"]:
        if col not in df.columns:
            df[col] = ""

    df["compound_id"] = df["compound_id"].astype(str)
    df["target"] = df["target"].astype(str)
    df["docking_score"] = pd.to_numeric(df["docking_score"], errors="coerce")
    df = df.dropna(subset=["compound_id", "target", "docking_score"])

    df = df.sort_values(["target", "docking_score"], ascending=[True, True]).reset_index(drop=True)
    df["target_rank"] = df.groupby("target")["docking_score"].rank(method="first", ascending=True).astype(int)

    return df


def load_docking_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return standardize_docking_dataframe(df)


def save_docking_results(df: pd.DataFrame, output_path="results/docking_results.csv") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def normalize_docking_score(score: float) -> float:
    try:
        score = float(score)
    except Exception:
        return 0.0

    if score <= -10:
        return 1.0
    if score >= -5:
        return 0.0

    return (-5 - score) / 5


def calculate_interaction_score(row) -> float:
    hbond_score = min(float(row.get("hbond_count", 0)) / 3, 1.0)
    hydrophobic_score = min(float(row.get("hydrophobic_count", 0)) / 5, 1.0)
    pi_pi_score = min(float(row.get("pi_pi_count", 0)) / 2, 1.0)
    salt_bridge_score = min(float(row.get("salt_bridge_count", 0)) / 1, 1.0)

    interaction_score = (
        0.35 * hbond_score
        + 0.30 * hydrophobic_score
        + 0.20 * pi_pi_score
        + 0.15 * salt_bridge_score
    )

    return round(float(interaction_score), 4)


def calculate_docking_confidence(row) -> float:
    docking_score_norm = normalize_docking_score(row.get("docking_score", 0))
    interaction_score = calculate_interaction_score(row)

    confidence = 0.60 * docking_score_norm + 0.40 * interaction_score

    return round(float(confidence), 4)


def generate_binding_interpretation(row) -> str:
    compound_id = row.get("compound_id", "该分子")
    target = row.get("target", "该靶点")
    docking_score = float(row.get("docking_score", 0))
    hbond = int(row.get("hbond_count", 0))
    hydrophobic = int(row.get("hydrophobic_count", 0))
    pi_pi = int(row.get("pi_pi_count", 0))
    salt_bridge = int(row.get("salt_bridge_count", 0))
    key_residues = str(row.get("key_residues", "")).strip()

    parts = []

    if docking_score <= -9:
        parts.append(f"{compound_id} 与 {target} 的 docking score 较低，提示其可能具有较好的结合潜力。")
    elif docking_score <= -7:
        parts.append(f"{compound_id} 与 {target} 的 docking score 处于中等偏好范围，具有一定结合可能。")
    else:
        parts.append(f"{compound_id} 与 {target} 的 docking score 相对较高，结合稳定性可能有限。")

    interaction_desc = []

    if hbond > 0:
        interaction_desc.append(f"{hbond} 个氢键")
    if hydrophobic > 0:
        interaction_desc.append(f"{hydrophobic} 个疏水相互作用")
    if pi_pi > 0:
        interaction_desc.append(f"{pi_pi} 个 π-π 作用")
    if salt_bridge > 0:
        interaction_desc.append(f"{salt_bridge} 个盐桥")

    if interaction_desc:
        parts.append("该分子形成了" + "、".join(interaction_desc) + "，说明其结合模式不仅依赖结合能，也具有一定相互作用支撑。")
    else:
        parts.append("当前结果中未记录明确的关键相互作用，建议进一步检查对接构象或补充相互作用分析。")

    if key_residues and key_residues.lower() not in ["nan", "none"]:
        parts.append(f"关键残基包括：{key_residues}。")

    return "".join(parts)


def generate_optimization_suggestion(row) -> str:
    docking_score = float(row.get("docking_score", 0))
    hbond = int(row.get("hbond_count", 0))
    hydrophobic = int(row.get("hydrophobic_count", 0))
    pi_pi = int(row.get("pi_pi_count", 0))
    salt_bridge = int(row.get("salt_bridge_count", 0))
    confidence = float(row.get("docking_confidence", 0))

    suggestions = []

    if confidence >= 0.75:
        suggestions.append("该分子的综合对接置信度较高，建议优先保留核心骨架，并围绕外围取代基进行小幅优化。")
    elif confidence >= 0.50:
        suggestions.append("该分子的对接表现中等，可作为备选候选分子，建议结合 ADMET 和 QSAR 结果进一步判断。")
    else:
        suggestions.append("该分子的对接置信度较低，建议谨慎作为优先候选，可考虑重新优化结构或检查对接构象。")

    if hbond == 0:
        suggestions.append("当前未记录氢键，可考虑在不显著增加 TPSA 的前提下引入合适的氢键供体或受体。")

    if hydrophobic == 0 and docking_score > -8:
        suggestions.append("疏水相互作用不足，若结合口袋存在疏水区域，可尝试引入适度疏水取代基。")

    if pi_pi == 0:
        suggestions.append("若靶点口袋存在芳香残基，可考虑保留或优化芳香环以增强 π-π 堆积。")

    if salt_bridge > 0:
        suggestions.append("盐桥相互作用可能增强结合稳定性，但需要结合 ADMET 结果关注电荷对膜通透性的影响。")

    return " ".join(suggestions)


def analyze_docking_results(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_docking_dataframe(df)

    df["docking_score_norm"] = df["docking_score"].apply(normalize_docking_score).round(4)
    df["interaction_score"] = df.apply(calculate_interaction_score, axis=1)
    df["docking_confidence"] = df.apply(calculate_docking_confidence, axis=1)
    df["binding_interpretation"] = df.apply(generate_binding_interpretation, axis=1)
    df["optimization_suggestion"] = df.apply(generate_optimization_suggestion, axis=1)

    df = df.sort_values(["docking_confidence", "docking_score"], ascending=[False, True]).reset_index(drop=True)
    df["overall_rank"] = range(1, len(df) + 1)

    return df


def summarize_multi_target_results(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = analyze_docking_results(df)

    rows = []

    for compound_id, group in df.groupby("compound_id"):
        group = group.sort_values("docking_score", ascending=True)
        best = group.iloc[0]

        target_count = group["target"].nunique()
        mean_score = round(group["docking_score"].mean(), 4)
        mean_conf = round(group["docking_confidence"].mean(), 4)
        best_target = best["target"]
        best_score = round(float(best["docking_score"]), 4)

        if target_count >= 3 and mean_conf >= 0.65:
            interpretation = "该分子在多个靶点上均具有较好的 docking confidence，可能具有多靶点作用或药物重定位潜力。"
        elif target_count >= 2:
            interpretation = "该分子具有一定多靶点比较价值，建议重点关注其最佳结合靶点及选择性差异。"
        else:
            interpretation = "该分子目前仅包含单靶点结果，暂不能判断多靶点倾向。"

        if len(group) >= 2:
            sorted_scores = group["docking_score"].sort_values().tolist()
            selectivity_gap = round(sorted_scores[1] - sorted_scores[0], 4)
        else:
            selectivity_gap = None

        rows.append({
            "compound_id": compound_id,
            "target_count": target_count,
            "best_target": best_target,
            "best_docking_score": best_score,
            "mean_docking_score": mean_score,
            "mean_docking_confidence": mean_conf,
            "selectivity_gap": selectivity_gap,
            "multi_target_interpretation": interpretation
        })

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(
        ["mean_docking_confidence", "best_docking_score"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return summary_df


def make_smart_docking_template() -> pd.DataFrame:
    data = [
        {"compound_id": "CHEMBL53711", "smiles": "CN(C)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1", "target": "EGFR", "docking_score": -10.0, "hbond_count": 1, "hydrophobic_count": 3, "pi_pi_count": 1, "salt_bridge_count": 0, "key_residues": "MET793;LEU718;PHE856", "interaction": "H-bond; hydrophobic; pi-pi"},
        {"compound_id": "CHEMBL53711", "smiles": "CN(C)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1", "target": "HER2", "docking_score": -8.7, "hbond_count": 1, "hydrophobic_count": 2, "pi_pi_count": 1, "salt_bridge_count": 0, "key_residues": "MET801;LEU785", "interaction": "H-bond; hydrophobic"},
        {"compound_id": "CHEMBL53711", "smiles": "CN(C)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1", "target": "BACE1", "docking_score": -6.2, "hbond_count": 0, "hydrophobic_count": 1, "pi_pi_count": 0, "salt_bridge_count": 0, "key_residues": "VAL69", "interaction": "weak hydrophobic"},
        {"compound_id": "CHEMBL138940", "smiles": "O=C(/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cn1)NCCN1CCOCC1", "target": "EGFR", "docking_score": -9.5, "hbond_count": 2, "hydrophobic_count": 2, "pi_pi_count": 1, "salt_bridge_count": 1, "key_residues": "MET793;LYS745;ASP855", "interaction": "H-bond; salt bridge; pi-pi"},
        {"compound_id": "CHEMBL138940", "smiles": "O=C(/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cn1)NCCN1CCOCC1", "target": "HER2", "docking_score": -9.0, "hbond_count": 2, "hydrophobic_count": 1, "pi_pi_count": 1, "salt_bridge_count": 0, "key_residues": "MET801;LYS753", "interaction": "H-bond; pi-pi"},
        {"compound_id": "CHEMBL138940", "smiles": "O=C(/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cn1)NCCN1CCOCC1", "target": "BACE1", "docking_score": -5.8, "hbond_count": 0, "hydrophobic_count": 1, "pi_pi_count": 0, "salt_bridge_count": 0, "key_residues": "ILE118", "interaction": "weak hydrophobic"},
        {"compound_id": "CHEMBL3808884", "smiles": "COC(=O)c1cc2cc(NCc3cc(NC(=O)c4cc(-n5cnc(C)c5)cc(C(F)(F)F)c4)ccc3C)ncc2[nH]1", "target": "EGFR", "docking_score": -9.2, "hbond_count": 1, "hydrophobic_count": 3, "pi_pi_count": 0, "salt_bridge_count": 0, "key_residues": "MET793;VAL726;LEU844", "interaction": "H-bond; hydrophobic"},
        {"compound_id": "CHEMBL3808884", "smiles": "COC(=O)c1cc2cc(NCc3cc(NC(=O)c4cc(-n5cnc(C)c5)cc(C(F)(F)F)c4)ccc3C)ncc2[nH]1", "target": "HER2", "docking_score": -8.1, "hbond_count": 1, "hydrophobic_count": 2, "pi_pi_count": 0, "salt_bridge_count": 0, "key_residues": "MET801;VAL734", "interaction": "H-bond; hydrophobic"},
        {"compound_id": "CHEMBL3808884", "smiles": "COC(=O)c1cc2cc(NCc3cc(NC(=O)c4cc(-n5cnc(C)c5)cc(C(F)(F)F)c4)ccc3C)ncc2[nH]1", "target": "BACE1", "docking_score": -6.5, "hbond_count": 1, "hydrophobic_count": 1, "pi_pi_count": 0, "salt_bridge_count": 0, "key_residues": "ASP32", "interaction": "H-bond"},
    ]
    return pd.DataFrame(data)


def make_docking_template() -> pd.DataFrame:
    return make_smart_docking_template()


def plot_docking_scores(df: pd.DataFrame, selected_target=None, output_path="results/docking_score_plot.png"):
    if df is None or df.empty:
        raise ValueError("没有可绘制的 docking 数据。")

    df = analyze_docking_results(df)

    if selected_target is not None:
        plot_df = df[df["target"].astype(str) == str(selected_target)].copy()
    else:
        plot_df = df.copy()

    if plot_df.empty:
        raise ValueError("筛选后没有可绘制的数据。")

    plot_df = plot_df.sort_values("docking_score", ascending=True).copy()
    plot_df["label"] = plot_df["compound_id"].astype(str) + " / " + plot_df["target"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df["label"], plot_df["docking_score"])
    ax.set_xlabel("Compound / Target")
    ax.set_ylabel("Docking score (kcal/mol)")
    ax.set_title("Docking Score Ranking")
    ax.tick_params(axis="x", rotation=60)

    for i, value in enumerate(plot_df["docking_score"]):
        ax.text(i, value, f"{value:.2f}", ha="center", va="top")

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, output_path


def plot_multi_target_heatmap(df: pd.DataFrame, output_path="results/multi_target_docking_heatmap.png"):
    if df is None or df.empty:
        raise ValueError("没有可绘制的数据。")

    df = analyze_docking_results(df)

    pivot = df.pivot_table(
        index="compound_id",
        columns="target",
        values="docking_score",
        aggfunc="min"
    )

    fig, ax = plt.subplots(figsize=(8, max(4, 0.6 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title("Multi-target Docking Score Heatmap")
    ax.set_xlabel("Target")
    ax.set_ylabel("Compound ID")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, label="Docking score")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, output_path


def plot_confidence_by_target(df: pd.DataFrame, output_path="results/docking_confidence_by_target.png"):
    if df is None or df.empty:
        raise ValueError("没有可绘制的数据。")

    df = analyze_docking_results(df)

    summary = df.groupby("target")["docking_confidence"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary.index.astype(str), summary.values)
    ax.set_xlabel("Target")
    ax.set_ylabel("Mean docking confidence")
    ax.set_title("Mean Docking Confidence by Target")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 1)

    for i, value in enumerate(summary.values):
        ax.text(i, value, f"{value:.2f}", ha="center", va="bottom")

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, output_path