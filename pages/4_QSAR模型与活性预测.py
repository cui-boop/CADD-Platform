import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

# 让 pages 里的文件可以导入 utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from utils.descriptors import build_descriptor_dataframe, calculate_basic_descriptors
from utils.qsar_model import (
    train_random_forest_classifier,
    save_model,
    get_feature_importance,
    predict_single_smiles,
)


st.set_page_config(
    page_title="QSAR模型与活性预测",
    layout="wide"
)

st.title("QSAR 模型与活性预测")
st.markdown("""
本模块用于读取清洗后的活性数据，计算分子描述符，训练随机森林 QSAR 分类模型，
并对新的候选分子进行 Active / Inactive 活性预测。
""")

DATA_PATH = "data/cleaned_activity.csv"
MODEL_PATH = "models/qsar_random_forest.pkl"
PREDICTION_PATH = "results/qsar_predictions.csv"
FEATURE_IMPORTANCE_PATH = "results/feature_importance.csv"


st.header("1. 读取清洗后的活性数据")

if not os.path.exists(DATA_PATH):
    st.warning("未找到 data/cleaned_activity.csv。请先让 A 完成活性数据整理模块。")
    st.stop()

df = pd.read_csv(DATA_PATH)

st.subheader("数据预览")
st.dataframe(df.head())

required_cols = ["compound_id", "smiles", "target", "pactivity", "label"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"数据缺少必要列：{missing_cols}")
    st.stop()

st.success(f"成功读取数据，共 {df.shape[0]} 条记录。")


st.header("2. 计算分子描述符")

X, valid_df = build_descriptor_dataframe(df, smiles_col="smiles")

st.write(f"有效分子数量：{X.shape[0]}")
st.write(f"描述符数量：{X.shape[1]}")

st.dataframe(X.head())

if X.shape[0] < 10:
    st.warning("有效样本数较少，模型结果可能不稳定。")


st.header("3. QSAR 随机森林模型训练")

col1, col2, col3 = st.columns(3)

with col1:
    test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)

with col2:
    n_estimators = st.slider("随机森林树数量", 50, 500, 200, 50)

with col3:
    random_state = st.number_input("随机种子", value=42, step=1)

if st.button("开始训练模型"):
    y = valid_df["label"]

    if y.nunique() < 2:
        st.error("label 中只有一个类别，无法训练分类模型。请检查 Active / Inactive 标签。")
        st.stop()

    model, metrics, cm, report, X_train, X_test, y_train, y_test, y_pred, y_prob = train_random_forest_classifier(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators
    )

    save_model(model, list(X.columns), MODEL_PATH)

    st.success("模型训练完成")

    st.subheader("模型性能指标")

    metrics_df = pd.DataFrame(
        list(metrics.items()),
        columns=["Metric", "Value"]
    )

    st.dataframe(metrics_df)

    os.makedirs("results", exist_ok=True)
    metrics_df.to_csv("results/qsar_metrics.csv", index=False)

    st.subheader("混淆矩阵")

    cm_df = pd.DataFrame(
        cm,
        index=["真实 Inactive", "真实 Active"],
        columns=["预测 Inactive", "预测 Active"]
    )

    st.dataframe(cm_df)

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        title="Confusion Matrix"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("特征重要性")

    importance_df = get_feature_importance(model, list(X.columns))
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    st.dataframe(importance_df)

    fig_imp = px.bar(
        importance_df.head(10),
        x="importance",
        y="feature",
        orientation="h",
        title="Top 10 Feature Importance"
    )

    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("训练集内全部分子的预测结果")

    all_prob = model.predict_proba(X)[:, 1]
    all_pred = model.predict(X)

    prediction_df = valid_df[["compound_id", "smiles", "target"]].copy()
    prediction_df["qsar_probability"] = all_prob
    prediction_df["qsar_prediction"] = [
        "Active" if p == 1 else "Inactive" for p in all_pred
    ]

    prediction_df.to_csv(PREDICTION_PATH, index=False)

    st.dataframe(prediction_df)

    st.success(f"QSAR 预测完成")


st.header("4. 新分子活性预测")

st.markdown("""
输入一个新的候选分子 SMILES，模型会计算相同描述符，并预测该分子属于 Active 的概率。
""")

new_smiles = st.text_input("请输入候选分子 SMILES", value="CCOc1ccc(N)cc1")

if st.button("预测新分子活性"):
    if not os.path.exists(MODEL_PATH):
        st.error("尚未找到已训练模型。请先训练模型。")
        st.stop()

    import joblib

    model_package = joblib.load(MODEL_PATH)
    model = model_package["model"]
    feature_names = model_package["feature_names"]

    desc = calculate_basic_descriptors(new_smiles)

    if desc is None:
        st.error("输入的 SMILES 无效，请检查分子结构。")
        st.stop()

    desc_df = pd.DataFrame([desc])
    desc_df = desc_df[feature_names]

    prediction, probability = predict_single_smiles(model, desc_df)

    st.subheader("预测结果")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("预测类别", prediction)

    with col2:
        st.metric("Active 概率", f"{probability:.3f}")

    st.dataframe(desc_df)