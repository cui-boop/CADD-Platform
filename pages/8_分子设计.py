
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from utils.molecular_design import (
    rdkit_available,
    make_demo_design_candidates,
    load_candidate_pool_from_results,
    analyze_parent_profile,
    generate_rule_based_designs,
    save_designed_molecules,
    get_rdkit_mol_image,
)

st.set_page_config(page_title="分子设计与结构优化", page_icon="💡", layout="wide")
st.title("💡 规则驱动的分子设计与结构优化")

st.markdown("""
本页面面向候选分子的后续优化阶段，结合 QSAR 预测、ADMET 评价、分子对接结果及综合评分信息， 对具有进一步研究价值的分子开展结构修饰与类似物设计。

它不是深度学习生成模型，而是以已有候选分子为基础，通过预设的药物化学优化策略生成一系列结构相关的衍生分子， 包括官能团替换、侧链修饰及简单骨架扩展等常见设计方式。 生成的新结构可继续进入 QSAR、ADMET 与分子对接流程进行重新评估， 用于比较不同结构变化对活性、成药性及结合能力的潜在影响。
""")

if rdkit_available():
    st.success("已加载 RDKit 环境：可进行分子结构解析、SMILES 校验、理化性质计算及结构可视化。")
else:
    st.warning("未检测到 RDKit环境：模块仍可运行，但不能进行严格的 SMILES 校验和分子绘图。建议后续安装 rdkit。")

st.subheader("1. 数据来源")
mode = st.radio("请选择候选分子数据", ["从 results 文件夹读取", "手动上传候选分子 CSV", "使用演示数据"], horizontal=True)

candidate_df = None
if mode == "从 results 文件夹读取":
    st.info( "系统将依次检索 results/final_ranking.csv、results/docking_results.csv、" "results/admet_results.csv 及 results/qsar_predictions.csv 中的候选分子信息。" )
    candidate_df = load_candidate_pool_from_results()
    if candidate_df.empty:
        st.error("未能从 results 文件夹中发现可用于设计分析的候选分子数据。")
    else:
        st.success("候选分子数据加载完成。")
elif mode == "手动上传候选分子 CSV":
    st.markdown("上传文件至少需要包含：`compound_id, smiles`。如包含以下字段，可获得更完整的优化建议：`target, qsar_probability, admet_score, docking_score, docking_confidence, recommendation, key_residues, binding_interpretation`。")
    uploaded = st.file_uploader("上传候选分子文件", type=["csv"])
    if uploaded is not None:
        candidate_df = pd.read_csv(uploaded)
        if not {"compound_id", "smiles"}.issubset(candidate_df.columns):
            st.error("文件中至少需要包含 compound_id 与 smiles 两列。")
            candidate_df = None
        else:
            st.success("候选分子文件上传成功。")
    else:
        st.warning("请上传候选分子 文件。")
else:
    candidate_df = make_demo_design_candidates()
    st.success("已加载演示数据。")

if candidate_df is not None and not candidate_df.empty:
    st.subheader("2. 候选分子列表")
    st.dataframe(candidate_df, use_container_width=True)
    candidate_df["compound_id"] = candidate_df["compound_id"].astype(str)

    selected = st.selectbox("选择一个候选分子作为设计母体", candidate_df["compound_id"].dropna().astype(str).unique().tolist())
    selected_row = candidate_df[candidate_df["compound_id"].astype(str) == selected].iloc[0].to_dict()

    st.subheader("3. 母体分子信息")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**Compound ID：** {selected_row.get('compound_id', '')}")
        st.write(f"**Target：** {selected_row.get('target', 'Unknown')}")
        st.write(f"**SMILES：** `{selected_row.get('smiles', '')}`")
        for c in ["qsar_probability", "admet_score", "docking_score", "docking_confidence", "recommendation", "key_residues"]:
            if c in selected_row and pd.notna(selected_row.get(c)):
                st.write(f"**{c}：** {selected_row.get(c)}")
        if "binding_interpretation" in selected_row and pd.notna(selected_row.get("binding_interpretation")):
            st.write(f"**对接解释：** {selected_row.get('binding_interpretation')}")
    with col2:
        img = get_rdkit_mol_image(selected_row.get("smiles", ""))
        st.image(img, caption="母体分子结构") if img is not None else st.info("当前环境无法绘制分子结构。")

    st.subheader("4. 自动识别的问题与优化方向")
    for issue in analyze_parent_profile(selected_row):
        st.markdown(f"- {issue}")

    st.subheader("5. 设置分子设计参数")
    ca, cb = st.columns(2)
    with ca:
        max_designs = st.slider("最多生成几个设计分子", 1, 12, 6)
    with cb:
        emphasis = st.selectbox("设计侧重点", ["均衡优化", "增强结合能力", "改善 ADMET", "降低分子量 / LogP", "增强氢键作用"])

    st.markdown(""" 当前设计策略主要包括卤素替换、芳香环修饰、F/OH/CH3 基团引入以及基于 QSAR、ADMET 和对接结果的定向优化建议。 安装 RDKit 后，系统将自动完成结构有效性检查、理化性质计算及分子结构可视化。 """)

    if st.button("生成设计分子", type="primary"):
        try:
            design_df = generate_rule_based_designs(selected_row, max_designs=max_designs, emphasis=emphasis)
            save_path = save_designed_molecules(design_df)
            st.success(f"分子设计完成，结果已保存到：{save_path}")
            st.subheader("6. 设计结果")
            st.dataframe(design_df, use_container_width=True)

            if rdkit_available():
                st.subheader("7. 结构预览")
                valid = design_df[(design_df["designed_smiles"].notna()) & (design_df["designed_smiles"] != "")]
                if not valid.empty:
                    cols = st.columns(3)
                    for i, (_, row) in enumerate(valid.iterrows()):
                        with cols[i % 3]:
                            img = get_rdkit_mol_image(row["designed_smiles"])
                            if img is not None:
                                st.image(img, caption=f"{row['design_id']}：{row['design_strategy']}")
                else:
                    st.info("没有可绘制的设计分子。")

            st.subheader("8. 下载 designed_molecules.csv")
            st.download_button(
                label="📥 下载 designed_molecules.csv",
                data=design_df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="designed_molecules.csv",
                mime="text/csv",
            )
            st.info( "当前结果为基于规则策略生成的结构优化建议。" "正式进入研究流程前，建议进一步开展 QSAR 预测、ADMET 评价、" "分子对接分析以及人工化学合理性审核。" )
        except Exception as e:
            st.error(f"分子设计失败：{e}")
