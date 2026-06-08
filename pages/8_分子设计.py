
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:
    px = None

from utils.top10_molecular_design import (
    rdkit_available,
    load_top10_candidates,
    make_demo_top10_candidates,
    diagnose_parent_candidate,
    generate_rule_based_designs,
    save_designed_molecules,
    get_mol_image,
    TOP10_DOCKING_PATH,
    DOCKING_RESULTS_PATH,
)

st.set_page_config(page_title="分子设计与结构优化", page_icon="💡", layout="wide")

st.title("💡 基于 Top 10 候选分子的规则驱动分子设计")

st.markdown(
    """
    本模块用于对推荐进入分子对接的候选分子进行规则驱动结构优化。
    分子对接网站：https://vina.scripps.edu/
    """
)

st.header("一、读取 Top 10 候选分子docking结果文件")

with st.expander("候选分子输入格式说明", expanded=True):
    format_df = pd.DataFrame(
        [
            {"列名": "compound_id", "是否必需": "必需", "含义": "分子编号，必须和生成、QSAR、成药性筛选、docking 结果中的编号一致"},
            {"列名": "smiles", "是否必需": "建议必需", "含义": "小分子 SMILES，用于展示、结构绘图和规则优化"},
            {"列名": "target", "是否必需": "必需", "含义": "对接靶点，例如 EGFR"},
            {"列名": "docking_score", "是否必需": "必需", "含义": "分子对接得分，通常越低越好"},
        ]
    )
    st.dataframe(format_df, use_container_width=True, hide_index=True)

source_mode = st.radio(
    "请选择候选分子读取方式",
    ["读取默认 Top 10 docking结果文件", "手动输入候选分子docking结果", "上传候选分子docking结果 CSV"],
    horizontal=True,
)

manual_data = None
uploaded_file = None

if source_mode == "读取默认 Top 10 docking结果文件":
    st.info(f"默认读取Top 10 docking结果文件")
    mode = "default"

elif source_mode == "手动输入候选分子docking结果":
    mode = "manual"
    with st.form("manual_candidate_form"):
        c1, c2 = st.columns(2)
        with c1:
            compound_id = st.text_input("compound_id")
            target = st.text_input("target", value="EGFR")
        with c2:
            docking_score = st.number_input("docking_score", value=-8.0, step=0.1, format="%.3f")
        smiles = st.text_area("smiles", height=90, placeholder="请输入候选分子的 SMILES")
        submitted_manual = st.form_submit_button("读取手动输入候选分子", type="primary")
        if submitted_manual:
            if not compound_id.strip() or not smiles.strip() or not target.strip():
                st.error("compound_id、smiles、target 和 docking_score 都需要填写。")
                st.stop()
            manual_data = {
                "compound_id": compound_id.strip(),
                "smiles": smiles.strip(),
                "target": target.strip(),
                "docking_score": docking_score,
            }
else:
    mode = "upload"
    uploaded_file = st.file_uploader("上传候选分子 CSV", type=["csv"])

load_clicked = False
if mode == "default":
    load_clicked = st.button("读取默认 Top 10 docking文件", type="primary")
elif mode == "upload":
    load_clicked = st.button("读取上传文件", type="primary")
elif mode == "manual" and manual_data is not None:
    load_clicked = True

if load_clicked or ("candidate_df_design_v3" not in st.session_state and mode == "default"):
    try:
        if mode == "default":
            candidate_df, source_name = load_top10_candidates(source_mode="default")
        elif mode == "upload":
            candidate_df, source_name = load_top10_candidates(source_mode="upload", uploaded_file=uploaded_file)
        else:
            candidate_df, source_name = load_top10_candidates(source_mode="manual", manual_data=manual_data)

        if candidate_df.empty:
            st.warning("没有读取到候选分子。请检查文件是否存在、上传文件是否为空，或字段名是否正确。")
            use_demo = st.checkbox("仅用于测试页面：使用内置演示数据", value=False)
            if use_demo:
                candidate_df = make_demo_top10_candidates()
                source_name = "内置演示数据"
            else:
                st.stop()

        st.session_state["candidate_df_design_v3"] = candidate_df
        st.session_state["source_name_design_v3"] = source_name

    except Exception as e:
        st.error(f"读取候选分子失败：{e}")
        st.stop()

if "candidate_df_design_v3" not in st.session_state:
    st.stop()

candidate_df = st.session_state["candidate_df_design_v3"]
source_name = st.session_state.get("source_name_design_v3", "")

st.success(f"已读取候选分子来源")

if DOCKING_RESULTS_PATH.exists():
    st.success(f"已检测到 docking 结果文件")
else:
    st.warning(f"未检测到默认 docking 结果文件。如果候选输入中已经包含 docking_score，则不影响使用。")

if rdkit_available():
    st.success("已检测到 RDKit：可以进行 SMILES 校验、理化性质计算和结构图绘制。")
else:
    st.warning("未检测到 RDKit：页面仍可运行，但结构图绘制和严格 SMILES 校验不可用。")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("候选分子数", len(candidate_df))
m2.metric("预测 Active 数", int(candidate_df["qsar_prediction"].astype(str).str.lower().eq("active").sum()))
max_qsar = candidate_df["qsar_probability"].max()
max_admet = candidate_df["admet_score"].max()
best_docking = candidate_df["docking_score"].min()
m3.metric("最高 QSAR 概率", "NA" if pd.isna(max_qsar) else f"{max_qsar:.3f}")
m4.metric("最高成药性分数", "NA" if pd.isna(max_admet) else f"{max_admet:.3f}")
m5.metric("最佳 docking score", "NA" if pd.isna(best_docking) else f"{best_docking:.2f}")

st.subheader("Top 10 候选分子总览")

display_cols = [
    "rank_for_docking",
    "compound_id",
    "smiles",
    "docking_score",
    "qsar_probability",
    "qsar_prediction",
    "admet_score",
    "design_priority_score",
    "MolWt",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "Rule_Score",
    "Lipinski_Passed",
]
display_cols = [c for c in display_cols if c in candidate_df.columns]
shown = candidate_df[display_cols].rename(
    columns={
        "rank_for_docking": "排名",
        "compound_id": "化合物编号",
        "smiles": "SMILES",
        "docking_score": "docking_score",
        "qsar_probability": "QSAR活性概率",
        "qsar_prediction": "QSAR预测类别",
        "admet_score": "成药性筛选分数",
        "design_priority_score": "设计优先级分数",
        "Rule_Score": "规则类药性分数",
    }
)
st.dataframe(shown, use_container_width=True, hide_index=True)

st.header("二、选择一个候选分子作为设计母体")

option_labels = []
for _, row in candidate_df.iterrows():
    rank_value = row.get("rank_for_docking")
    rank_text = "NA" if pd.isna(rank_value) else str(int(rank_value))
    qsar_text = "NA" if pd.isna(row.get("qsar_probability")) else f"{row.get('qsar_probability'):.3f}"
    admet_text = "NA" if pd.isna(row.get("admet_score")) else f"{row.get('admet_score'):.3f}"
    docking_text = "NA" if pd.isna(row.get("docking_score")) else f"{row.get('docking_score'):.2f}"
    option_labels.append(f"Rank {rank_text} | {row.get('compound_id')} | QSAR={qsar_text} | ADMET={admet_text} | Docking={docking_text}")

selected_label = st.selectbox("选择母体分子", option_labels)
selected_rank_text = selected_label.split("|")[0].replace("Rank", "").strip()

if selected_rank_text == "NA":
    selected_compound_id = selected_label.split("|")[1].strip()
    selected_row = candidate_df[candidate_df["compound_id"].astype(str) == selected_compound_id].iloc[0].to_dict()
else:
    selected_rank = int(selected_rank_text)
    selected_row = candidate_df[candidate_df["rank_for_docking"].astype(int) == selected_rank].iloc[0].to_dict()

left, right = st.columns([2, 1])

with left:
    st.subheader("母体分子信息")
    st.write(f"**Rank：** {selected_row.get('rank_for_docking')}")
    st.write(f"**compound_id：** {selected_row.get('compound_id')}")
    st.write(f"**SMILES：** `{selected_row.get('smiles')}`")
    st.write(f"**docking score：** {selected_row.get('docking_score')}")
    st.write(f"**QSAR Active 概率：** {selected_row.get('qsar_probability')}")
    st.write(f"**QSAR 预测类别：** {selected_row.get('qsar_prediction')}")
    st.write(f"**成药性筛选分数：** {selected_row.get('admet_score')}")
    st.write(f"**设计优先级分数：** {selected_row.get('design_priority_score')}")
    st.markdown("**母体分子理化性质：**")
    property_cols = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds"]
    st.dataframe(pd.DataFrame([{col: selected_row.get(col) for col in property_cols}]), use_container_width=True, hide_index=True)

with right:
    st.subheader("母体分子结构")
    parent_img = get_mol_image(selected_row.get("smiles", ""))
    if parent_img is not None:
        st.image(parent_img, caption=str(selected_row.get("compound_id")))
    else:
        st.info("当前环境无法绘制该分子结构。")

st.subheader("母体分子诊断")
for item in diagnose_parent_candidate(selected_row):
    st.markdown(f"- {item}")

st.header("三、规则驱动结构优化")

p1, p2 = st.columns(2)
with p1:
    max_designs = st.slider("最多生成设计分子数量", min_value=1, max_value=12, value=6)
with p2:
    emphasis = st.selectbox("设计侧重点", ["均衡优化", "增强活性", "改善成药性", "为对接优化", "降低 LogP / 分子量"])

st.markdown(
    """
    当前规则主要包括：卤素替换、芳香环小取代基引入、降低强疏水基团、缩短部分侧链等。
    生成的结构需要重新进行 QSAR、成药性筛选以及 docking 结果分析。
    """
)

if st.button("生成结构优化分子", type="primary"):
    try:
        design_df = generate_rule_based_designs(selected_row, max_designs=max_designs, emphasis=emphasis)
        full_path, simple_path = save_designed_molecules(design_df)

        st.success(f"完整设计结果已保存")
        st.success(f"后续 QSAR / ADMET 再预测输入文件已保存")

        tab_result, tab_structure, tab_download = st.tabs(["设计结果表", "结构预览", "下载结果"])

        with tab_result:
            result_cols = [
                "designed_compound_id",
                "designed_smiles",
                "design_strategy",
                "design_reason",
                "property_change_comment",
                "valid_smiles",
                "MolWt",
                "LogP",
                "TPSA",
                "HBD",
                "HBA",
                "RotatableBonds",
                "Rule_Score",
                "Lipinski_Passed",
                "MolWt_delta",
                "LogP_delta",
                "TPSA_delta",
                "RotatableBonds_delta",
            ]
            result_cols = [c for c in result_cols if c in design_df.columns]
            st.dataframe(design_df[result_cols], use_container_width=True, hide_index=True)

            st.markdown("### 性质变化解释")
            comment_df = design_df[design_df["designed_smiles"].astype(str).str.len() > 0].copy()

            if comment_df.empty:
                st.info("暂无可展示的性质变化解释。")
            else:
                for _, row in comment_df.iterrows():
                    compound_id = row.get("designed_compound_id", "")
                    strategy = row.get("design_strategy", "")
                    comment = row.get("property_change_comment", "")
                    if not comment:
                        comment = "该设计分子的核心性质变化较小，建议进入 QSAR、ADMET 和 docking 模块进行二次评价。"
                    st.markdown(
                        f"""
                        <div style="border:1px solid #e5e7eb;border-radius:12px;padding:14px 18px;margin-bottom:12px;background-color:#f8fafc;">
                            <div style="font-weight:700;font-size:16px;margin-bottom:6px;">{compound_id}｜{strategy}</div>
                            <div style="font-size:15px;line-height:1.7;">{comment}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with tab_structure:
            if rdkit_available():
                valid_designs = design_df[
                    design_df["designed_smiles"].notna()
                    & (design_df["designed_smiles"].astype(str).str.len() > 0)
                ].head(9)
                if valid_designs.empty:
                    st.info("没有可绘制的设计分子。")
                else:
                    cols = st.columns(3)
                    for i, (_, row) in enumerate(valid_designs.iterrows()):
                        with cols[i % 3]:
                            img = get_mol_image(row["designed_smiles"])
                            if img is not None:
                                st.image(img, caption=f"{row['designed_compound_id']} | {row['design_strategy']}")
                            else:
                                st.info(f"{row['designed_compound_id']} 无法绘图")
            else:
                st.info("未检测到 RDKit，无法绘制设计分子结构。")

        with tab_download:
            st.download_button(
                "📥 下载完整 designed_molecules.csv",
                data=design_df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="designed_molecules.csv",
                mime="text/csv",
                use_container_width=True,
            )
            simple_df = design_df[
                design_df["designed_smiles"].notna()
                & (design_df["designed_smiles"].astype(str).str.len() > 0)
            ][["designed_compound_id", "designed_smiles"]].rename(
                columns={"designed_compound_id": "compound_id", "designed_smiles": "smiles"}
            )
            st.download_button(
                "📥 下载 designed_molecules_for_prediction.csv",
                data=simple_df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="designed_molecules_for_prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.info(
                "下一步：将结果文件重新送入活性预测和成药性筛选模块，"
                "比较设计前后的 QSAR 概率、成药性筛选分数、docking score 和关键理化性质。"
            )

    except Exception as e:
        st.error(f"分子设计失败：{e}")
