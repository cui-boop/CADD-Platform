import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

sys.path.append(PROJECT_ROOT)

from utils.admet import (
    calculate_admet_properties,
    batch_calculate_admet
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)

GENERATED_MOLECULES_PATH = RESULT_DIR / "generated_molecules.csv"
GENERATED_ADMET_PATH = RESULT_DIR / "generated_druglikeness_predictions.csv"
GENERATED_QSAR_PATH = RESULT_DIR / "generated_qsar_predictions.csv"
GENERATED_SCREENING_PATH = RESULT_DIR / "generated_screening_results.csv"
TOP10_DOCKING_PATH = RESULT_DIR / "top10_docking_candidates.csv"
ADMET_BATCH_PATH = RESULT_DIR / "druglikeness_screening_results.csv"


DISPLAY_RENAME_MAP = {
    "ADMET_Score": "成药性筛选分数",
    "ADMET_Score_admet": "成药性筛选分数",
    "passed_rule_count": "通过规则数",
    "passed_rule_count_admet": "通过规则数",
    "druglikeness_level": "成药性筛选等级",
    "druglikeness_level_admet": "成药性筛选等级",
}


def format_display_df(df):
    """
    仅用于页面展示：把内部兼容字段名转换为更准确的中文展示名。
    """
    if df is None:
        return df

    return df.rename(
        columns={k: v for k, v in DISPLAY_RENAME_MAP.items() if k in df.columns}
    )


def clean_druglikeness_output(df):
    """
    清理成药性筛选结果表格，避免同时出现 smiles / SMILES / canonical_smiles 等重复结构列。
    后续模块统一使用 smiles 列。
    """
    if df is None or df.empty:
        return df

    out_df = df.copy()

    if "SMILES" in out_df.columns:
        if "smiles" in out_df.columns:
            out_df = out_df.drop(columns=["SMILES"])
        else:
            out_df = out_df.rename(columns={"SMILES": "smiles"})

    if "canonical_smiles" in out_df.columns:
        if "smiles" in out_df.columns:
            out_df = out_df.drop(columns=["canonical_smiles"])
        else:
            out_df = out_df.rename(columns={"canonical_smiles": "smiles"})

    return out_df



def add_druglikeness_rule_summary(df):
    """
    根据 6 条经验规则计算通过规则数和筛选等级。
    成药性筛选分数 = 通过规则数 / 6。
    """
    if df is None or df.empty:
        return df

    out_df = df.copy()

    required_cols = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds"]
    if not all(col in out_df.columns for col in required_cols):
        return out_df

    for col in required_cols:
        out_df[col] = pd.to_numeric(out_df[col], errors="coerce")

    rule_df = pd.DataFrame({
        "MolWt_rule": out_df["MolWt"] < 500,
        "LogP_rule": out_df["LogP"] < 5,
        "TPSA_rule": out_df["TPSA"] < 140,
        "HBD_rule": out_df["HBD"] <= 5,
        "HBA_rule": out_df["HBA"] <= 10,
        "RotatableBonds_rule": out_df["RotatableBonds"] < 10,
    })

    out_df["passed_rule_count"] = rule_df.sum(axis=1).astype(int)

    def level(n):
        if n == 6:
            return "优秀"
        elif n == 5:
            return "较好"
        elif n == 4:
            return "中等"
        else:
            return "较弱"

    out_df["druglikeness_level"] = out_df["passed_rule_count"].apply(level)

    # 与原有内部字段保持兼容：分数 = 通过规则数 / 6
    out_df["ADMET_Score"] = (out_df["passed_rule_count"] / 6).round(2)

    return out_df


def _pick_existing_column(df, candidates):
    """
    从多个候选列名中选择第一个存在的列。
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def create_top10_docking_candidates():
    """
    从 generated_screening_results.csv 中筛选推荐进入分子对接的 Top 10 候选分子。

    筛选逻辑：
    1. 优先选择 qsar_prediction = Active、qsar_probability >= 0.60、通过规则数 >= 4/6 的分子；
    2. 在优先推荐分子中，先按 qsar_probability 从高到低排序，再按通过规则数从高到低排序；
    3. 如果满足优先条件的分子不足 10 个，则用剩余分子按 qsar_probability 和通过规则数继续补足。

    注意：这里不计算 docking score。该表只是推荐用户拿去做分子对接的候选分子列表。
    """
    if not GENERATED_SCREENING_PATH.exists():
        return None, None

    screening_df = pd.read_csv(GENERATED_SCREENING_PATH)

    compound_col = _pick_existing_column(
        screening_df,
        ["compound_id", "Compound_ID", "id", "ID"]
    )
    smiles_col = _pick_existing_column(
        screening_df,
        ["smiles", "canonical_smiles", "SMILES", "smiles_admet", "canonical_smiles_admet"]
    )
    qsar_col = _pick_existing_column(
        screening_df,
        ["qsar_probability", "prob_active"]
    )
    qsar_pred_col = _pick_existing_column(
        screening_df,
        ["qsar_prediction", "prediction"]
    )
    admet_col = _pick_existing_column(
        screening_df,
        ["ADMET_Score", "ADMET_Score_admet"]
    )

    if compound_col is None or smiles_col is None or qsar_col is None or admet_col is None:
        return None, None

    work_df = screening_df.copy()

    work_df[qsar_col] = pd.to_numeric(
        work_df[qsar_col],
        errors="coerce"
    )
    work_df[admet_col] = pd.to_numeric(
        work_df[admet_col],
        errors="coerce"
    )

    work_df = work_df.dropna(
        subset=[qsar_col, admet_col]
    ).copy()

    if work_df.empty:
        return None, None

    # 成药性筛选分数是离散规则分数，本质是通过规则数 / 6
    if "passed_rule_count" in work_df.columns:
        work_df["passed_rule_count"] = pd.to_numeric(
            work_df["passed_rule_count"],
            errors="coerce"
        ).fillna((work_df[admet_col] * 6).round()).astype(int)
    else:
        work_df["passed_rule_count"] = (work_df[admet_col] * 6).round().astype(int)

    def level(n):
        if n == 6:
            return "优秀"
        elif n == 5:
            return "较好"
        elif n == 4:
            return "中等"
        else:
            return "较弱"

    work_df["druglikeness_level"] = work_df["passed_rule_count"].apply(level)

    if qsar_pred_col is not None:
        active_mask = work_df[qsar_pred_col].astype(str).str.lower().eq("active")
    else:
        active_mask = work_df[qsar_col] >= 0.60

    work_df["recommended_for_docking"] = (
        active_mask
        & (work_df[qsar_col] >= 0.60)
        & (work_df["passed_rule_count"] >= 4)
    )

    work_df["selection_reason"] = work_df["recommended_for_docking"].map(
        {
            True: "优先推荐：预测为 Active，Active 概率较高，且至少满足 4/6 条成药性经验规则。",
            False: "备选：未完全达到优先推荐条件，但在当前候选集中综合排序靠前。"
        }
    )

    # 不引入人工 docking 分数；排序只基于 QSAR 概率和通过规则数
    work_df = work_df.sort_values(
        ["recommended_for_docking", qsar_col, "passed_rule_count"],
        ascending=[False, False, False]
    ).head(10).copy()

    work_df.insert(
        0,
        "rank_for_docking",
        range(1, len(work_df) + 1)
    )

    output_cols = [
        "rank_for_docking",
        compound_col,
        smiles_col,
        qsar_col,
    ]

    if qsar_pred_col is not None:
        output_cols.append(qsar_pred_col)

    output_cols.extend(
        [
            admet_col,
            "passed_rule_count",
            "druglikeness_level",
            "recommended_for_docking",
            "selection_reason"
        ]
    )

    optional_cols = [
        "MolWt", "MolWt_admet",
        "LogP", "LogP_admet",
        "TPSA", "TPSA_admet",
        "HBD", "HBD_admet",
        "HBA", "HBA_admet",
        "RotatableBonds", "RotatableBonds_admet",
        "Lipinski_Passed", "Lipinski_Passed_admet",
        "PAINS_Alert", "PAINS_Alert_admet"
    ]

    for col in optional_cols:
        if col in work_df.columns and col not in output_cols:
            output_cols.append(col)

    top10_df = work_df[output_cols].copy()

    rename_map = {
        compound_col: "compound_id",
        smiles_col: "smiles",
        qsar_col: "qsar_probability",
        admet_col: "成药性筛选分数"
    }

    if qsar_pred_col is not None:
        rename_map[qsar_pred_col] = "qsar_prediction"

    top10_df = top10_df.rename(columns=rename_map)

    top10_df.to_csv(
        TOP10_DOCKING_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    return TOP10_DOCKING_PATH, top10_df


def show_top10_docking_candidates(download_key):
    """
    展示推荐用于分子对接的 Top 10 候选分子，并提供下载按钮。
    """
    if not TOP10_DOCKING_PATH.exists():
        st.info("尚未生成 Top 10 分子对接候选分子。请先完成生成分子的批量活性预测和批量成药性筛选。")
        return

    top10_df = pd.read_csv(TOP10_DOCKING_PATH)

    st.subheader("推荐进入分子对接的 Top 10 候选分子")

    st.markdown(
        """
        优先选择 qsar_prediction = Active，且至少满足 4/6 条成药性经验规则的分子；
        在满足条件的分子中，先按 qsar_probability 从高到低排序，再按通过规则数从高到低排序。
        """
    )

    st.dataframe(
        format_display_df(top10_df),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="下载 Top 10 分子对接候选分子 CSV",
        data=format_display_df(top10_df).to_csv(index=False, encoding="utf-8-sig"),
        file_name="top10_docking_candidates.csv",
        mime="text/csv",
        use_container_width=True,
        key=download_key
    )


def update_generated_screening_results():
    """
    若已同时存在生成分子的 QSAR 活性预测结果与成药性筛选结果，
    则自动合并为 generated_screening_results.csv，供后续分子设计阶段使用。
    """
    if not GENERATED_QSAR_PATH.exists() or not GENERATED_ADMET_PATH.exists():
        return None

    qsar_df = pd.read_csv(GENERATED_QSAR_PATH)
    admet_df = pd.read_csv(GENERATED_ADMET_PATH)

    if "compound_id" in qsar_df.columns and "compound_id" in admet_df.columns:
        merged_df = qsar_df.merge(
            admet_df,
            on="compound_id",
            how="left",
            suffixes=("", "_admet")
        )
    elif "smiles" in qsar_df.columns and "smiles" in admet_df.columns:
        merged_df = qsar_df.merge(
            admet_df,
            on="smiles",
            how="left",
            suffixes=("", "_admet")
        )
    else:
        return None

    merged_df.to_csv(
        GENERATED_SCREENING_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    create_top10_docking_candidates()

    return GENERATED_SCREENING_PATH


st.set_page_config(
    page_title="成药性筛选",
    layout="wide"
)

st.title("💊 成药性筛选")

st.markdown("""
本模块用于对候选分子进行 **早期类药性与成药性初筛**。页面会基于 RDKit 计算分子的基础理化性质，
并结合 Lipinski Rule、Veber 相关经验规则和 PAINS 结构警示，对候选分子进行规则型筛选。
""")

st.info("""
**成药性筛选规则说明**

本模块不是连续型预测模型，而是统计候选分子满足 6 条成药性经验规则的数量：

1. MolWt < 500
2. LogP < 5
3. TPSA < 140
4. HBD ≤ 5
5. HBA ≤ 10
6. RotatableBonds < 10

其中 MolWt、LogP、HBD 和 HBA 主要对应 Lipinski Rule of Five；
TPSA 和 RotatableBonds 主要参考 Veber 规则中与口服吸收和生物利用度相关的经验判断。

页面同时保留一个“成药性筛选分数”，计算方式为：

成药性筛选分数 = 通过规则数 / 6

规定：

- 6/6 条：分数 = 1.00，等级 = 优秀
- 5/6 条：分数 ≈ 0.83，等级 = 较好
- 4/6 条：分数 ≈ 0.67，等级 = 中等
- ≤3/6 条：分数 ≤ 0.50，等级 = 较弱

""")

# ====================== 单分子预测 ======================

st.header("单分子成药性筛选")

default_smiles = "CCOc1ccc(N)cc1"

smiles = st.text_input(
    "输入分子 SMILES",
    value=default_smiles
)

if st.button("开始成药性筛选"):

    result = calculate_admet_properties(smiles)

    if result is None:
        st.error("未能识别当前 SMILES，请检查输入格式。")
        st.stop()

    # ================= 分子结构图 =================

    st.subheader("分子结构")

    mol = Chem.MolFromSmiles(smiles)

    img = Draw.MolToImage(mol, size=(400, 400))

    st.image(img)

    result_df = pd.DataFrame([result])
    result_df = add_druglikeness_rule_summary(result_df)
    result = result_df.iloc[0].to_dict()

    # ================= 指标展示 =================

    st.subheader("核心指标")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("MolWt", result["MolWt"])

    with col2:
        st.metric("LogP", result["LogP"])

    with col3:
        st.metric("TPSA", result["TPSA"])

    with col4:
        st.metric("通过规则数", f"{int(result['passed_rule_count'])} / 6")

    with col5:
        st.metric("筛选等级", result["druglikeness_level"])

    # ================= 全部性质 =================

    st.subheader("完整属性结果")

    st.dataframe(
        format_display_df(result_df),
        use_container_width=True
    )

    # ================= 雷达图 =================

    st.subheader("理化性质雷达图")

    radar_categories = [
        "MolWt",
        "LogP",
        "TPSA",
        "HBD",
        "HBA",
        "RotatableBonds"
    ]

    radar_values = [
        result["MolWt"] / 100,
        result["LogP"] * 10,
        result["TPSA"] / 20,
        result["HBD"] * 10,
        result["HBA"] * 5,
        result["RotatableBonds"] * 5
    ]

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=radar_values,
        theta=radar_categories,
        fill='toself',
        name='理化性质雷达图'
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            )
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig_radar,
        use_container_width=True
    )

    # ================= Lipinski =================

    st.subheader("Lipinski Rule")

    lipinski_df = pd.DataFrame({
        "Rule": [
            "MolWt ≤ 500",
            "LogP ≤ 5",
            "HBD ≤ 5",
            "HBA ≤ 10"
        ],
        "Status": [
            result["MolWt"] <= 500,
            result["LogP"] <= 5,
            result["HBD"] <= 5,
            result["HBA"] <= 10
        ]
    })

    st.dataframe(
        lipinski_df,
        use_container_width=True
    )

    # ================= 药物化学解释 =================

    st.subheader("指标说明")

    st.info(f"""
MolWt = {result["MolWt"]}

分子量通常与吸收效率、组织分布及口服利用度相关。

LogP = {result["LogP"]}

LogP 反映分子的脂溶性，对膜通透性和溶解行为有较大影响。

TPSA = {result["TPSA"]}

TPSA 常用于评估分子的极性及跨膜能力。

CNS Permeability = {result["CNS_Permeability"]}

用于反映分子穿越血脑屏障的潜在能力。

PAINS Alert = {result["PAINS_Alert"]}

用于识别可能导致假阳性的结构片段。

通过规则数 = {int(result["passed_rule_count"])} / 6

成药性筛选等级 = {result["druglikeness_level"]}

成药性筛选分数 = {result["ADMET_Score"]}

""")

    # ================= 成药性评价 =================

    st.subheader("综合评价")

    passed_count = int(result["passed_rule_count"])

    if passed_count >= 5:
        st.success("该分子满足 5 条及以上成药性经验规则，规则型成药性筛选表现较好，可进一步开展 QSAR、分子对接或后续优化分析。")

    elif passed_count >= 4:
        st.warning("该分子满足 4/6 条成药性经验规则，达到基础推荐标准，但仍建议结合 QSAR、分子对接和具体结构特征进一步判断。")

    else:
        st.error("该分子满足的成药性经验规则少于 4 条，建议优先进行结构优化或谨慎推进。")

# ====================== 批量预测 ======================

st.header("批量成药性筛选")

source_mode = st.radio(
    "批量成药性筛选数据来源",
    ["使用已生成的候选分子集 results/generated_molecules.csv", "手动上传候选分子文件"],
    horizontal=True
)

df = None

if source_mode == "使用已生成的候选分子集 results/generated_molecules.csv":
    if GENERATED_MOLECULES_PATH.exists():
        df = pd.read_csv(GENERATED_MOLECULES_PATH)
        st.success(f"已读取已生成候选分子集：{GENERATED_MOLECULES_PATH}")
    else:
        st.warning("未找到 results/generated_molecules.csv。请先在“分子生成”页面生成候选分子，或改为手动上传文件。")
else:
    uploaded_file = st.file_uploader(
        "上传包含 smiles 列的 CSV 文件",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("请上传候选分子文件后进行批量成药性筛选。")

if df is not None:

    if "smiles" not in df.columns:
        st.error("CSV 文件中未检测到 smiles 列。")
        st.stop()

    st.success(f"已读取 {len(df)} 条分子记录。")

    if st.button("开始批量分析"):

        progress_bar = st.progress(0)

        status_text = st.empty()

        status_text.info("正在进行成药性批量筛选，请稍候...")

        result_list = []

        total = len(df)

        for idx, row in df.reset_index(drop=True).iterrows():

            smi = row["smiles"]

            result = calculate_admet_properties(smi)

            if result is not None:
                result_row = row.to_dict()
                result_row.update(result)
                result_list.append(result_row)

            progress = (idx + 1) / total

            progress_bar.progress(progress)

            status_text.info(
                f"正在分析第 {idx + 1} / {total} 个分子"
            )

        result_df = pd.DataFrame(result_list)
        result_df = clean_druglikeness_output(result_df)
        result_df = add_druglikeness_rule_summary(result_df)

        status_text.success("批量分析完成。")

        st.subheader("批量预测结果")

        st.dataframe(
            format_display_df(result_df),
            use_container_width=True
        )

        st.caption(
            "说明：批量结果中的成药性筛选分数 = 通过规则数 / 6。"
            "它是离散规则分数，用于早期筛选和排序，不代表真实实验值。"
        )

        st.subheader("通过规则数分布")

        score_df = result_df.dropna(subset=["passed_rule_count"]).copy()

        if not score_df.empty:
            count_df = (
                score_df["passed_rule_count"]
                .value_counts()
                .sort_index()
                .reset_index()
            )
            count_df.columns = ["通过规则数", "分子数量"]
            count_df["通过规则数标签"] = count_df["通过规则数"].astype(str) + " / 6"

            fig = px.bar(
                count_df,
                x="通过规则数标签",
                y="分子数量",
                text="分子数量",
                title="Distribution of Passed Drug-likeness Rules",
                labels={
                    "通过规则数标签": "通过规则数",
                    "分子数量": "分子数量"
                }
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(
                height=430,
                margin=dict(l=10, r=10, t=55, b=20)
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "说明：该图展示候选分子满足 6 条成药性经验规则中的几条。"
                "推荐标准为至少满足 4/6 条规则，而不是设置连续型分数阈值。"
            )
        else:
            st.warning("当前批量结果中没有可用于可视化的通过规则数。")

        if source_mode == "使用已生成的候选分子集 results/generated_molecules.csv":
            result_df.to_csv(
                GENERATED_ADMET_PATH,
                index=False,
                encoding="utf-8-sig"
            )
            result_df.to_csv(
                ADMET_BATCH_PATH,
                index=False,
                encoding="utf-8-sig"
            )
            st.success("成药性筛选结果已保存至 results/generated_druglikeness_predictions.csv")
            merged_path = update_generated_screening_results()
            if merged_path is not None:
                st.success("已同步生成综合筛选结果：results/generated_screening_results.csv")
                top10_path, top10_df = create_top10_docking_candidates()
                if top10_path is not None:
                    st.success("已生成推荐用于分子对接的 Top 10 候选分子：results/top10_docking_candidates.csv")
                    show_top10_docking_candidates("download_top10_docking_after_admet")
                else:
                    st.warning("综合筛选结果已生成，但暂时无法筛选 Top 10 分子对接候选分子。请检查是否包含 compound_id、smiles、qsar_probability 和成药性筛选分数等列。")
        else:
            result_df.to_csv(
                ADMET_BATCH_PATH,
                index=False,
                encoding="utf-8-sig"
            )
            st.success("成药性筛选结果已保存至 results/druglikeness_screening_results.csv")
