import os
import sys

import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

from utils.data_cleaning import (
    read_activity_csv,
    clean_activity_data_auto
)


st.set_page_config(
    page_title="活性数据整理",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 活性数据整理")

st.markdown(
    """
    本模块用于整理 ChEMBL 或项目标准格式的化合物活性数据。
    系统会自动识别数据格式，提取化合物 ID、SMILES、靶点、活性类型、
    活性值和单位，并计算 pActivity，生成 Active / Inactive 标签。
    """
)

st.divider()

st.header("一、数据格式说明")

st.markdown(
    """
    本页面支持两类数据格式：

    **1. ChEMBL 原始导出数据**

    例如从 ChEMBL Activities 页面直接下载的 CSV 文件。常见列包括：

    - Molecule ChEMBL ID
    - Smiles
    - Target Name
    - Standard Type
    - Standard Value
    - Standard Units
    - Standard Relation

    **2. 项目标准格式数据**

    也就是已经整理过的数据，至少需要包含：

    - compound_id
    - smiles
    - target
    - activity_type
    - activity_value
    - unit

    系统会自动识别上传文件属于哪一种格式。
    """
)

st.info(
    """
    推荐 ChEMBL 下载筛选条件：
    Target Type = SINGLE PROTEIN；
    Organism = Homo sapiens；
    Standard Type = IC50 或 Ki；
    Standard Units = nM；
    Standard Relation = =；
    Standard Value > 0；
    Smiles 非空。
    """
)

st.divider()

st.header("二、选择数据来源")

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
    uploaded_file = st.file_uploader(
        "请上传 CSV 文件",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = read_activity_csv(uploaded_file)
        st.success("文件上传成功。")

if df is not None:
    st.header("三、原始数据预览")

    st.write(f"数据行数：{df.shape[0]}")
    st.write(f"数据列数：{df.shape[1]}")

    with st.expander("查看原始数据列名"):
        st.write(list(df.columns))

    st.dataframe(df.head(20), use_container_width=True)

    st.divider()

    st.header("四、数据整理参数")

    st.info(
        "系统会自动识别上传文件是 ChEMBL 原始导出数据，还是项目标准格式数据。"
    )

    threshold = st.slider(
        "活性分类阈值：pActivity ≥ 阈值 判定为 Active",
        min_value=4.0,
        max_value=9.0,
        value=6.0,
        step=0.1
    )

    st.markdown(
        """
        默认情况下：

        - pActivity ≥ 6：Active
        - pActivity < 6：Inactive

        如果活性单位为 nM，则计算公式为：

        pActivity = 9 - log10(activity_value)
        """
    )

    if st.button("开始整理数据"):
        try:
            cleaned_df, summary = clean_activity_data_auto(
                df,
                threshold=threshold
            )

            st.success("数据整理完成！")

            st.divider()

            st.header("五、数据整理统计")

            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric(
                "识别数据类型",
                summary.get("识别数据类型", "未知")
            )

            col2.metric(
                "原始记录数",
                summary.get("原始记录数", 0)
            )

            col3.metric(
                "清洗后分子数",
                summary.get("清洗后分子数", 0)
            )

            col4.metric(
                "Active 数量",
                summary.get("Active 数量", 0)
            )

            col5.metric(
                "Inactive 数量",
                summary.get("Inactive 数量", 0)
            )

            st.header("六、清洗后数据预览")

            st.dataframe(
                cleaned_df.head(50),
                use_container_width=True
            )

            st.markdown(
                """
                清洗后的数据包含以下核心列：

                - compound_id
                - smiles
                - target
                - activity_type
                - activity_value
                - unit
                - pactivity
                - label
                - record_count
                """
            )

            output_dir = os.path.join(ROOT_DIR, "data")
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, "cleaned_activity.csv")
            cleaned_df.to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig"
            )

            st.info(f"清洗后的数据已保存到：{output_path}")

            csv_data = cleaned_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label="下载 cleaned_activity.csv",
                data=csv_data,
                file_name="cleaned_activity.csv",
                mime="text/csv"
            )

            st.divider()

            st.header("七、标签分布")

            if "label" in cleaned_df.columns:
                label_counts = cleaned_df["label"].value_counts()
                st.bar_chart(label_counts)
            else:
                st.warning("清洗后数据中未找到 label 列。")

            st.header("八、pActivity 分布")

            if "pactivity" in cleaned_df.columns:
                st.bar_chart(cleaned_df["pactivity"])
            else:
                st.warning("清洗后数据中未找到 pactivity 列。")

        except Exception as e:
            st.error(f"数据整理失败：{e}")

else:
    st.info("请先选择内置数据或上传 CSV 文件。")