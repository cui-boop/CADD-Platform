import os
from datetime import datetime

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="项目管理",
    page_icon="📁",
    layout="wide"
)

st.title("📁 项目管理")

st.markdown(
    """
    本页面用于创建和查看 CADD 分析项目。用户可以记录项目名称、疾病方向、
    靶点名称、数据来源和项目备注，方便后续进行活性数据整理、QSAR 建模和结果报告生成。
    """
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PROJECT_FILE = os.path.join(RESULTS_DIR, "projects.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

st.header("一、新建项目")

with st.form("create_project_form"):
    project_name = st.text_input(
        "项目名称",
        value="EGFR_inhibitor_screening"
    )

    disease_area = st.selectbox(
        "疾病方向",
        [
            "肿瘤",
            "神经退行性疾病",
            "炎症与免疫疾病",
            "心血管疾病",
            "代谢性疾病",
            "感染性疾病",
            "其他"
        ]
    )

    target_name = st.text_input(
        "靶点名称",
        value="EGFR"
    )

    target_id = st.text_input(
        "靶点数据库编号",
        value="CHEMBL203"
    )

    data_source = st.selectbox(
        "数据来源",
        [
            "ChEMBL",
            "PubChem BioAssay",
            "MoleculeNet",
            "自定义数据",
            "其他"
        ]
    )

    activity_type = st.selectbox(
        "主要活性指标",
        [
            "IC50",
            "Ki",
            "EC50",
            "二分类标签",
            "其他"
        ]
    )

    description = st.text_area(
        "项目备注",
        value="本项目以内置 EGFR ChEMBL IC50 数据为示例，用于构建 QSAR 活性预测模型。"
    )

    submitted = st.form_submit_button("保存项目")

    if submitted:
        if project_name.strip() == "":
            st.error("项目名称不能为空。")
        else:
            new_project = pd.DataFrame(
                [
                    {
                        "project_name": project_name,
                        "disease_area": disease_area,
                        "target_name": target_name,
                        "target_id": target_id,
                        "data_source": data_source,
                        "activity_type": activity_type,
                        "description": description,
                        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                ]
            )

            if os.path.exists(PROJECT_FILE):
                old_projects = pd.read_csv(PROJECT_FILE)
                all_projects = pd.concat(
                    [old_projects, new_project],
                    ignore_index=True
                )
            else:
                all_projects = new_project

            all_projects.to_csv(
                PROJECT_FILE,
                index=False,
                encoding="utf-8-sig"
            )

            st.success("项目已保存！")

st.divider()

st.header("二、已有项目")

if os.path.exists(PROJECT_FILE):
    projects = pd.read_csv(PROJECT_FILE)

    if len(projects) > 0:
        st.dataframe(projects, use_container_width=True)

        st.subheader("项目概览")

        col1, col2, col3 = st.columns(3)
        col1.metric("项目总数", len(projects))
        col2.metric("涉及靶点数", projects["target_name"].nunique())
        col3.metric("数据来源数", projects["data_source"].nunique())

        st.subheader("选择项目查看详情")

        selected_project = st.selectbox(
            "请选择项目",
            projects["project_name"].tolist()
        )

        project_info = projects[projects["project_name"] == selected_project].iloc[0]

        st.markdown(f"**项目名称：** {project_info['project_name']}")
        st.markdown(f"**疾病方向：** {project_info['disease_area']}")
        st.markdown(f"**靶点名称：** {project_info['target_name']}")
        st.markdown(f"**靶点编号：** {project_info['target_id']}")
        st.markdown(f"**数据来源：** {project_info['data_source']}")
        st.markdown(f"**主要活性指标：** {project_info['activity_type']}")
        st.markdown(f"**创建时间：** {project_info['created_time']}")
        st.markdown(f"**项目备注：** {project_info['description']}")

    else:
        st.info("目前还没有项目，请先新建一个项目。")
else:
    st.info("目前还没有项目，请先新建一个项目。")

st.divider()

st.header("三、推荐项目流程")

st.success(
    """
    推荐流程：
    新建项目 → 活性数据整理 → QSAR 模型训练 → ADMET 预测 → 可解释性分析 → 
    分子对接结果分析 → 综合评分 → 结果报告生成
    """
)