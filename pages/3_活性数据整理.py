import os
import sys
import pandas as pd
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

from utils.data_cleaning import (
    read_activity_csv,
    standardize_chembl_data,
    clean_standard_activity_data
)


st.set_page_config(
    page_title="活性数据整理",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 活性数据整理")

st.markdown(
    """
    本模块用于整理 ChEMBL 或标准格式的化合物活性数据。
    系统会自动计算 pActivity，并根据阈值划分 Active / Inactive 标签，
    为后续 QSAR 模型训练提供标准输入文件。
    """
)

st.subheader("1. 数据格式说明")

st.markdown(
    """
    本页面支持两类数据：

    **类型 1：ChEMBL 原始导出数据**

    需要包含以下列：

    - Molecule ChEMBL ID
    - Smiles
    - Target Name
    - Standard Type
    - Standard Value
    - Standard Units
    - Standard Relation

    **类型 2：项目标准格式数据**

    需要包含以下列：

    - compound_id
    - smiles
    - target
    - activity_type
    - activity_value
    - unit
    """
)

st.subheader("2. 选择数据来源")

data_source = st.radio(
    "请选择数据来源",
    ["使用内置 EGFR 示例数据", "上传自己的 CSV 文件"]
)

df = None

if data_source == "使用内置 EGFR 示例数据":
    example_path = os.path.join(ROOT_DIR, "data", "chembl_egfr_ic50_raw.csv")

    if os.path.exists(example_path):
        df = read_activity_csv(example_path)
        st.success("已读取内置 EGFR ChEMBL 示例数据。")
    else:
        st.error("未找到 data/chembl_egfr_ic50_raw.csv，请检查数据文件是否存在。")

else:
    uploaded_file = st.file_uploader("请上传 CSV 文件", type=["csv"])

    if uploaded_file is not None:
        df = read_activity_csv(uploaded_file)
        st.success("文件上传成功。")

if df is not None:
    st.subheader("3. 原始数据预览")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("4. 数据清洗参数")

    data_type = st.selectbox(
        "请选择数据类型",
        ["ChEMBL 原始导出数据", "项目标准格式数据"]
    )

    threshold = st.slider(
        "活性分类阈值：pActivity ≥ 阈值 判定为 Active",
        min_value=4.0,
        max_value=9.0,
        value=6.0,
        step=0.1
    )

    if st.button("开始整理数据"):
        try:
            if data_type == "ChEMBL 原始导出数据":
                cleaned_df, summary = standardize_chembl_data(
                    df,
                    threshold=threshold
                )
            else:
                cleaned_df, summary = clean_standard_activity_data(
                    df,
                    threshold=threshold
                )

            st.success("数据整理完成！")

            st.subheader("5. 数据整理统计")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("原始记录数", summary.get("原始记录数", 0))

            clean_count = summary.get(
                "去重后分子数",
                summary.get("清洗后分子数", 0)
            )
            col2.metric("清洗后分子数", clean_count)

            col3.metric("Active 数量", summary.get("Active 数量", 0))
            col4.metric("Inactive 数量", summary.get("Inactive 数量", 0))

            st.subheader("6. 清洗后数据预览")
            st.dataframe(cleaned_df.head(50), use_container_width=True)

            output_dir = os.path.join(ROOT_DIR, "data")
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, "cleaned_activity.csv")
            cleaned_df.to_csv(output_path, index=False, encoding="utf-8-sig")

            st.info(f"清洗后的数据已保存到：{output_path}")

            csv_data = cleaned_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label="下载 cleaned_activity.csv",
                data=csv_data,
                file_name="cleaned_activity.csv",
                mime="text/csv"
            )

            st.subheader("7. 标签分布")
            label_counts = cleaned_df["label"].value_counts()
            st.bar_chart(label_counts)

            st.subheader("8. pActivity 分布")
            st.bar_chart(cleaned_df["pactivity"])

        except Exception as e:
            st.error(f"数据整理失败：{e}")

else:
    st.info("请先选择内置数据或上传 CSV 文件。")