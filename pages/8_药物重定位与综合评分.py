import streamlit as st
import pandas as pd
from pathlib import Path

from utils.scoring import (
    merge_and_score,
    save_final_ranking,
    make_demo_qsar_df,
    make_demo_admet_df,
    make_demo_docking_df
)


st.set_page_config(
    page_title="药物重定位与综合评分",
    page_icon="💊",
    layout="wide"
)

st.title("💊 药物重定位与综合评分")

st.markdown("""
本模块用于整合 **QSAR 活性概率**、**ADMET score** 和 **Docking score**，
计算候选分子的综合评分，并输出推荐等级。

综合评分公式为：

```text
综合评分 = 0.4 × QSAR 活性概率 + 0.3 × ADMET score + 0.3 × Docking score 归一化值
```

其中：

- QSAR 活性概率越高，说明模型预测该分子具有活性的可能性越大；
- ADMET score 越高，说明该分子类药性和成药性越好；
- docking score 通常越低越好，因此需要先归一化为 0 到 1 的分数。
""")

st.subheader("1. 输入文件格式要求")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **QSAR 结果文件**

    文件名建议：

    ```text
    results/qsar_predictions.csv
    ```

    必须包含：

    ```text
    compound_id
    smiles
    target
    qsar_probability
    qsar_prediction
    ```
    """)

with col2:
    st.markdown("""
    **ADMET 结果文件**

    文件名建议：

    ```text
    results/admet_results.csv
    ```

    必须包含：

    ```text
    compound_id
    admet_score
    ```

    推荐额外包含：

    ```text
    MolWt
    LogP
    TPSA
    HBD
    HBA
    ```
    """)

with col3:
    st.markdown("""
    **Docking 结果文件**

    文件名建议：

    ```text
    results/docking_results.csv
    ```

    必须包含：

    ```text
    compound_id
    docking_score
    ```

    可额外包含：

    ```text
    smiles
    target
    interaction
    ```
    """)

st.subheader("2. 选择数据来源")

mode = st.radio(
    "请选择综合评分所使用的数据来源",
    [
        "从 results 文件夹读取",
        "手动上传 CSV 文件",
        "使用演示数据"
    ]
)


def read_csv_if_exists(path: str):
    file_path = Path(path)

    if file_path.exists():
        return pd.read_csv(file_path)

    return None


def show_input_data(qsar_df, admet_df, docking_df):
    with st.expander("查看 QSAR 预测结果"):
        st.dataframe(qsar_df, use_container_width=True)

    with st.expander("查看 ADMET 结果"):
        st.dataframe(admet_df, use_container_width=True)

    with st.expander("查看 Docking 结果"):
        st.dataframe(docking_df, use_container_width=True)


def run_scoring(qsar_df, admet_df, docking_df):
    final_df = merge_and_score(qsar_df, admet_df, docking_df)

    save_path = save_final_ranking(
        final_df,
        output_path="results/final_ranking.csv"
    )

    st.success(f"综合评分计算完成，结果已保存到：{save_path}")

    st.subheader("综合评分结果")

    st.dataframe(final_df, use_container_width=True)

    if not final_df.empty:
        best = final_df.iloc[0]

        st.info(
            f"综合评分最高的候选分子是 **{best['compound_id']}**，"
            f"综合评分为 **{best['total_score']}**，"
            f"推荐等级为 **{best['recommendation']}**。"
        )

    csv_data = final_df.to_csv(index=False, encoding="utf-8-sig")

    st.download_button(
        label="📥 下载 final_ranking.csv",
        data=csv_data,
        file_name="final_ranking.csv",
        mime="text/csv"
    )

    return final_df


if mode == "从 results 文件夹读取":

    st.info("""
    系统将尝试自动读取以下三个文件：

    ```text
    results/qsar_predictions.csv
    results/admet_results.csv
    results/docking_results.csv
    ```
    """)

    qsar_df = read_csv_if_exists("results/qsar_predictions.csv")
    admet_df = read_csv_if_exists("results/admet_results.csv")
    docking_df = read_csv_if_exists("results/docking_results.csv")

    missing_files = []

    if qsar_df is None:
        missing_files.append("results/qsar_predictions.csv")

    if admet_df is None:
        missing_files.append("results/admet_results.csv")

    if docking_df is None:
        missing_files.append("results/docking_results.csv")

    if missing_files:
        st.error("缺少以下文件：")

        for file in missing_files:
            st.write(f"- {file}")

        st.warning("你可以先使用“手动上传 CSV 文件”或“使用演示数据”测试本页面。")

    else:
        st.success("三个输入文件读取成功。")

        show_input_data(qsar_df, admet_df, docking_df)

        if st.button("计算综合评分", type="primary"):
            try:
                run_scoring(qsar_df, admet_df, docking_df)

            except Exception as e:
                st.error(f"综合评分计算失败：{e}")


elif mode == "手动上传 CSV 文件":

    st.markdown("""
    请分别上传 QSAR、ADMET 和 docking 三个结果文件。三个文件必须通过 `compound_id` 对应同一批候选分子。
    """)

    qsar_file = st.file_uploader(
        "上传 QSAR 预测结果 CSV",
        type=["csv"],
        key="qsar_file"
    )

    admet_file = st.file_uploader(
        "上传 ADMET 结果 CSV",
        type=["csv"],
        key="admet_file"
    )

    docking_file = st.file_uploader(
        "上传 Docking 结果 CSV",
        type=["csv"],
        key="docking_file"
    )

    if qsar_file is not None and admet_file is not None and docking_file is not None:
        try:
            qsar_df = pd.read_csv(qsar_file)
            admet_df = pd.read_csv(admet_file)
            docking_df = pd.read_csv(docking_file)

            st.success("三个文件上传成功。")

            show_input_data(qsar_df, admet_df, docking_df)

            if st.button("计算综合评分", type="primary"):
                run_scoring(qsar_df, admet_df, docking_df)

        except Exception as e:
            st.error(f"文件读取或综合评分失败：{e}")

    else:
        st.warning("请上传 QSAR、ADMET 和 Docking 三个 CSV 文件。")


elif mode == "使用演示数据":

    st.info("当前使用系统内置的演示数据，适合在 B、C 模块尚未完成时测试 D 模块。")

    qsar_df = make_demo_qsar_df()
    admet_df = make_demo_admet_df()
    docking_df = make_demo_docking_df()

    show_input_data(qsar_df, admet_df, docking_df)

    if st.button("使用演示数据计算综合评分", type="primary"):
        try:
            run_scoring(qsar_df, admet_df, docking_df)

        except Exception as e:
            st.error(f"演示数据综合评分失败：{e}")


st.subheader("评分规则说明")

st.markdown("""
本模块采用加权综合评分方法：

```text
total_score = 0.4 × qsar_probability
            + 0.3 × admet_score
            + 0.3 × docking_score_norm
```

推荐等级划分：

```text
total_score ≥ 0.75：高推荐
0.50 ≤ total_score < 0.75：中等推荐
total_score < 0.50：低推荐
```

其中 docking score 归一化规则为：

```text
docking_score ≤ -10：1.0
docking_score ≥ -5 ：0.0
-10 到 -5 之间线性归一化
```

因此，docking score 越低，归一化后的 docking 得分越高。
""")
