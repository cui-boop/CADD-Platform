import os
import zipfile

import pandas as pd
import streamlit as st
import plotly.express as px

from utils.run_manager import load_run_history, get_run_dir


st.set_page_config(
    page_title="历史项目与结果汇总",
    layout="wide"
)

st.title("历史项目与结果汇总")

st.markdown(
    """
    本页面用于查看每一次模型训练生成的历史记录。
    每次训练完成后，系统会自动保存所使用的数据集、模型参数、评价指标、
    特征重要性和预测结果，方便后续回顾与汇总。
    """
)

history_df = load_run_history()

if history_df.empty:
    st.info("目前还没有历史项目。请先进入“模型训练”页面完成一次模型训练。")
    st.stop()

st.header("一、查看已有项目")

selected_run = st.selectbox(
    "选择一个历史项目查看",
    history_df["run_id"].tolist()
)

run_info = history_df[history_df["run_id"] == selected_run].iloc[0]
run_dir = get_run_dir(selected_run)

st.subheader("项目基本信息")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Run ID", selected_run)
col2.metric("靶点", run_info.get("target", "Unknown"))
col3.metric("样本数", int(run_info.get("sample_count", 0)))
col4.metric("模型", run_info.get("model_type", "Unknown"))

st.dataframe(
    pd.DataFrame([run_info]),
    use_container_width=True
)

st.header("二、模型性能结果")

metrics_path = os.path.join(run_dir, "qsar_metrics.csv")

if os.path.exists(metrics_path):
    metrics_df = pd.read_csv(metrics_path)
    st.dataframe(metrics_df, use_container_width=True)

    if "Metric" in metrics_df.columns and "Value" in metrics_df.columns:
        metric_cols = st.columns(min(4, len(metrics_df)))

        for col, (_, row) in zip(metric_cols, metrics_df.iterrows()):
            col.metric(row["Metric"], round(row["Value"], 4) if pd.notna(row["Value"]) else "NA")
else:
    st.warning("未找到本次项目的 qsar_metrics.csv。")

cm_path = os.path.join(run_dir, "confusion_matrix.csv")

if os.path.exists(cm_path):
    st.subheader("混淆矩阵")

    cm_df = pd.read_csv(cm_path, index_col=0)
    st.dataframe(cm_df, use_container_width=True)

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        title="Confusion Matrix"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

st.header("三、模型参数")

params_path = os.path.join(run_dir, "qsar_model_parameters.csv")

if os.path.exists(params_path):
    params_df = pd.read_csv(params_path)
    st.dataframe(params_df, use_container_width=True)
else:
    st.warning("未找到本次项目的模型参数文件。")

st.header("四、特征重要性")

importance_path = os.path.join(run_dir, "feature_importance.csv")

if os.path.exists(importance_path):
    importance_df = pd.read_csv(importance_path)
    st.dataframe(importance_df, use_container_width=True)

    if "feature" in importance_df.columns and "importance" in importance_df.columns:
        fig_imp = px.bar(
            importance_df.head(10),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 10 Feature Importance"
        )
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.warning("未找到本次项目的特征重要性文件。")

st.header("五、预测结果")

prediction_path = os.path.join(run_dir, "qsar_predictions.csv")

if os.path.exists(prediction_path):
    prediction_df = pd.read_csv(prediction_path)
    st.dataframe(prediction_df.head(100), use_container_width=True)

    st.download_button(
        label="下载本次 QSAR 预测结果",
        data=prediction_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{selected_run}_qsar_predictions.csv",
        mime="text/csv"
    )
else:
    st.warning("未找到本次项目的预测结果文件。")

st.header("六、导出本次项目结果")

zip_path = os.path.join("results", f"{selected_run}.zip")

if st.button("打包本次项目结果"):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(run_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, run_dir)
                zipf.write(file_path, arcname)

    st.success("项目结果已打包完成。")

if os.path.exists(zip_path):
    with open(zip_path, "rb") as f:
        st.download_button(
            label="下载本次项目 ZIP 文件",
            data=f,
            file_name=f"{selected_run}.zip",
            mime="application/zip"
        )