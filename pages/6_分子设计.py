
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

st.set_page_config(page_title="分子设计与结构优化", page_icon="🧩", layout="wide")
st.title("🧩 规则驱动的分子设计与结构优化")

st.markdown("""
本模块用于在 QSAR、ADMET、分子对接和综合评分结果的基础上，对高潜力候选分子进行轻量级结构优化。

它不是深度学习生成模型，而是一个更稳定、易解释的规则驱动分子设计模块。核心思想是：先分析候选分子的活性、类药性和对接表现，再围绕其核心骨架进行小幅结构修改，生成一批可用于后续 QSAR / ADMET / docking 再评价的类似物。
""")

if rdkit_available():
    st.success("已检测到 RDKit：可以进行 SMILES 有效性检查、描述符计算和分子结构绘图。")
else:
    st.warning("未检测到 RDKit：模块仍可运行，但不能进行严格的 SMILES 校验和分子绘图。建议后续安装 rdkit。")

st.subheader("1. 选择候选分子来源")
mode = st.radio("请选择数据来源", ["从 results 文件夹读取", "手动上传候选分子 CSV", "使用演示数据"], horizontal=True)

candidate_df = None
if mode == "从 results 文件夹读取":
    st.info("系统会尝试读取 results/final_ranking.csv、results/docking_results.csv、results/admet_results.csv、results/qsar_predictions.csv。")
    candidate_df = load_candidate_pool_from_results()
    if candidate_df.empty:
        st.error("没有从 results 文件夹中读取到可用候选分子。")
    else:
        st.success("已读取候选分子数据。")
elif mode == "手动上传候选分子 CSV":
    st.markdown("上传文件至少需要包含：`compound_id, smiles`。推荐额外包含：`target, qsar_probability, admet_score, docking_score, docking_confidence, recommendation, key_residues, binding_interpretation`。")
    uploaded = st.file_uploader("上传候选分子 CSV", type=["csv"])
    if uploaded is not None:
        candidate_df = pd.read_csv(uploaded)
        if not {"compound_id", "smiles"}.issubset(candidate_df.columns):
            st.error("上传文件必须包含 compound_id 和 smiles 两列。")
            candidate_df = None
        else:
            st.success("候选分子文件上传成功。")
    else:
        st.warning("请上传候选分子 CSV。")
else:
    candidate_df = make_demo_design_candidates()
    st.success("已加载演示候选分子。")

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

    st.markdown("""
当前轻量规则包括：卤素替换、芳香环引入 F / OH / CH3、根据 QSAR / ADMET / docking 结果生成优化建议。安装 RDKit 后会自动进行 SMILES 有效性检查和描述符计算。
""")

    if st.button("生成设计分子", type="primary"):
        try:
            design_df = generate_rule_based_designs(selected_row, max_designs=max_designs, emphasis=emphasis)
            save_path = save_designed_molecules(design_df)
            st.success(f"分子设计完成，结果已保存到：{save_path}")
            st.subheader("6. 设计分子结果")
            st.dataframe(design_df, use_container_width=True)

            if rdkit_available():
                st.subheader("7. 设计分子结构预览")
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
            st.info("这些设计分子属于规则驱动的结构优化建议。正式使用前，需要继续进行 QSAR、ADMET、分子对接和人工化学合理性检查。")
        except Exception as e:
            st.error(f"分子设计失败：{e}")
