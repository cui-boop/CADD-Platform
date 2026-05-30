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

st.title("ADMET 预测")

st.markdown("""
本模块用于计算候选药物分子的 ADMET 相关性质，
包括：

- 分子理化性质
- Lipinski Rule of Five
- 药物相似性分析
- CNS 渗透预测
- Bioavailability Score
- PAINS Alert
- 综合 ADMET Score

用于辅助药物筛选与成药性评价。
""")

# ====================== 单分子预测 ======================

st.header("1. 单分子 ADMET 分析")

default_smiles = "CCOc1ccc(N)cc1"

smiles = st.text_input(
    "输入分子 SMILES",
    value=default_smiles
)

if st.button("开始 ADMET 分析"):

    result = calculate_admet_properties(smiles)

    if result is None:
        st.error("SMILES 无效")
        st.stop()

    # ================= 分子结构图 =================

    st.subheader("二维分子结构")

    mol = Chem.MolFromSmiles(smiles)

    img = Draw.MolToImage(mol, size=(400, 400))

    st.image(img)

    # ================= 指标展示 =================

    st.subheader("关键性质指标")

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

    st.subheader("完整 ADMET 属性")

    result_df = pd.DataFrame([result])

    st.dataframe(
        result_df,
        use_container_width=True
    )

    # ================= 雷达图 =================

    st.subheader("ADMET Radar")

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

    st.subheader("Lipinski Rule Analysis")

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

    st.subheader("药物化学解释")

    st.info(f"""
    • MolWt = {result["MolWt"]}：
    分子量影响吸收、分布和口服利用度。

    • LogP = {result["LogP"]}：
    LogP 反映疏水性，影响膜通透性和溶解性。

    • TPSA = {result["TPSA"]}：
    TPSA 与细胞膜通透能力相关。

    • CNS Permeability：
    当前预测结果为 {result["CNS_Permeability"]}

    • PAINS Alert：
    当前结果为 {result["PAINS_Alert"]}

    • ADMET Score：
    综合评分为 {result["ADMET_Score"]}
    """)

    # ================= 成药性评价 =================

    st.subheader("成药性综合评价")

    score = result["ADMET_Score"]

    if score >= 0.85:
        st.success("该分子具有优秀成药性，建议进入后续分子对接分析。")

    elif score >= 0.65:
        st.warning("该分子具有中等成药性，可进一步优化。")

    else:
        st.error("该分子成药性较弱。")

# ====================== 批量预测 ======================

st.header("2. 批量 ADMET 分析")

uploaded_file = st.file_uploader(
    "上传包含 smiles 列的 CSV 文件",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "smiles" not in df.columns:
        st.error("CSV 必须包含 smiles 列")
        st.stop()

    result_df = batch_calculate_admet(
        df["smiles"].tolist()
    )

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

    st.success("批量 ADMET 分析完成")