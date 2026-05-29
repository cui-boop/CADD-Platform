import os
import sys
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

st.set_page_config(
    page_title="ADMET预测",
    layout="wide"
)

st.title("💊 ADMET 预测")

st.markdown("""
本模块用于评估候选分子的基础 ADMET 特征，帮助快速判断化合物的成药潜力。

当前支持的分析内容包括：

分子理化性质 ｜ Lipinski Rule ｜ CNS 渗透趋势 ｜ Bioavailability ｜ PAINS Alert ｜ 综合 ADMET Score

适用于早期化合物筛选、先导优化以及成药性初步评估。
""")

# ====================== 单分子预测 ======================

st.header("单分子 ADMET 分析")

default_smiles = "CCOc1ccc(N)cc1"

smiles = st.text_input(
    "输入分子 SMILES",
    value=default_smiles
)

if st.button("开始 ADMET 分析"):

    result = calculate_admet_properties(smiles)

    if result is None:
        st.error("未能识别当前 SMILES，请检查输入格式。")
        st.stop()

    # ================= 分子结构图 =================

    st.subheader("分子结构")

    mol = Chem.MolFromSmiles(smiles)

    img = Draw.MolToImage(mol, size=(400, 400))

    st.image(img)

    # ================= 指标展示 =================

    st.subheader("核心指标")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("MolWt", result["MolWt"])

    with col2:
        st.metric("LogP", result["LogP"])

    with col3:
        st.metric("TPSA", result["TPSA"])

    with col4:
        st.metric("ADMET Score", result["ADMET_Score"])

    # ================= 全部性质 =================

    st.subheader("完整属性结果")

    result_df = pd.DataFrame([result])

    st.dataframe(
        result_df,
        use_container_width=True
    )

    # ================= 雷达图 =================

    st.subheader("ADMET Profile")

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
        name='ADMET Profile'
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

ADMET Score = {result["ADMET_Score"]}

综合反映当前分子的整体成药趋势。
""")

    # ================= 成药性评价 =================

    st.subheader("综合评价")

    score = result["ADMET_Score"]

    if score >= 0.85:
        st.success("该分子整体表现较好，可进一步开展对接或后续优化分析。")

    elif score >= 0.65:
        st.warning("该分子具备一定成药潜力，部分性质仍有优化空间。")

    else:
        st.error("当前分子的综合成药性相对有限。")

# ====================== 批量预测 ======================

st.header("批量 ADMET 分析")

uploaded_file = st.file_uploader(
    "上传包含 smiles 列的 CSV 文件",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "smiles" not in df.columns:
        st.error("CSV 文件中未检测到 smiles 列。")
        st.stop()

    st.success(f"已读取 {len(df)} 条分子记录。")

    if st.button("开始批量分析"):

        progress_bar = st.progress(0)

        status_text = st.empty()

        status_text.info("正在进行 ADMET 批量计算，请稍候...")

        smiles_list = df["smiles"].tolist()

        result_list = []

        total = len(smiles_list)

        for idx, smi in enumerate(smiles_list):

            result = calculate_admet_properties(smi)

            if result is not None:
                result_list.append(result)

            progress = (idx + 1) / total

            progress_bar.progress(progress)

            status_text.info(
                f"正在分析第 {idx + 1} / {total} 个分子"
            )

        result_df = pd.DataFrame(result_list)

        status_text.success("批量分析完成。")

        st.subheader("批量预测结果")

        st.dataframe(
            result_df,
            use_container_width=True
        )

        st.subheader("ADMET Score 分布")

        fig = px.histogram(
            result_df,
            x="ADMET_Score",
            nbins=20,
            title="ADMET Score Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Top Candidate Molecules")

        top_df = result_df.sort_values(
            "ADMET_Score",
            ascending=False
        ).head(10)

        st.dataframe(
            top_df,
            use_container_width=True
        )

        os.makedirs("results", exist_ok=True)

        result_df.to_csv(
            "results/admet_prediction_results.csv",
            index=False
        )

        st.success("结果已保存至 results/admet_prediction_results.csv")