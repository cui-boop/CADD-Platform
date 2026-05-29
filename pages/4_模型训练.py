import os
import sys

import pandas as pd
import streamlit as st
import plotly.express as px

# 让 pages 里的文件可以导入 utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from utils.descriptors import build_descriptor_dataframe
from utils.qsar_model import (
    train_random_forest_classifier,
    save_model,
    get_feature_importance,
)

from utils.run_manager import (
    create_run_id,
    create_run_dir,
    copy_file_if_exists,
    save_run_history,
)


st.set_page_config(
    page_title="🧠 模型训练",
    layout="wide"
)

st.title("模型训练")
st.markdown("""
本模块用于读取清洗后的活性数据，计算分子描述符，并训练随机森林 QSAR 分类模型。
训练完成后，模型会保存到 `models/qsar_random_forest.pkl`，供“活性预测”页面调用；同时本次训练结果会自动保存到历史项目中。
""")


# ============================== 路径设置 ==============================

DATA_PATH = "data/cleaned_activity.csv"
MODEL_PATH = "models/qsar_random_forest.pkl"
PREDICTION_PATH = "results/qsar_predictions.csv"
FEATURE_IMPORTANCE_PATH = "results/feature_importance.csv"
PARAMETERS_PATH = "results/qsar_model_parameters.csv"


# ============================== 辅助函数 ==============================

def explain_feature_meaning(feature_name):
    """
    根据常用分子描述符名称，返回对应的药物设计意义解释。
    如果没有预设解释，则返回通用说明。
    """
    explanation_map = {
        "MolWt": "分子量，反映分子大小，可能影响结合口袋匹配、吸收和成药性。",
        "LogP": "脂水分配系数，反映分子疏水性，可能影响靶点结合、膜通透性和溶解性。",
        "TPSA": "拓扑极性表面积，反映分子极性，常与吸收、通透性和氢键能力有关。",
        "HBD": "氢键供体数量，可能影响蛋白-配体氢键相互作用和膜通透性。",
        "HBA": "氢键受体数量，可能影响蛋白结合、溶解性和吸收过程。",
        "RotatableBonds": "可旋转键数量，反映分子柔性，柔性过高可能影响结合构象稳定性。",
        "HeavyAtomCount": "重原子数量，反映分子规模和结构复杂度。",
        "RingCount": "环数量，反映分子刚性和骨架特征，可能影响结合构象。",
        "FractionCSP3": "sp3 碳比例，反映分子三维性，可能影响选择性和成药性。",
        "NumAromaticRings": "芳香环数量，可能影响疏水相互作用、π-π 相互作用和结合稳定性。"
    }

    return explanation_map.get(
        feature_name,
        "该特征为模型使用的分子描述符，其重要性较高说明它对 Active / Inactive 分类贡献较大。"
    )


# ============================== 1. 读取清洗后的活性数据 ==============================

st.header("1. 读取清洗后的活性数据")

if not os.path.exists(DATA_PATH):
    st.warning("未找到 data/cleaned_activity.csv。请先完成活性数据整理模块。")
    st.stop()

df = pd.read_csv(DATA_PATH)

st.subheader("数据预览")
st.dataframe(df.head(), use_container_width=True)

required_cols = ["compound_id", "smiles", "target", "pactivity", "label"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"数据缺少必要列：{missing_cols}")
    st.stop()

st.success(f"成功读取数据，共 {df.shape[0]} 条记录。")


# ============================== 2. 计算分子描述符 ==============================

st.header("2. 计算分子描述符")

X, valid_df = build_descriptor_dataframe(df, smiles_col="smiles")
X = X.fillna(0)

st.write(f"有效分子数量：{X.shape[0]}")
st.write(f"描述符数量：{X.shape[1]}")

st.dataframe(X.head(), use_container_width=True)

if X.shape[0] < 10:
    st.warning("有效样本数较少，模型结果可能不稳定。建议正式展示时使用更多样本。")


# ============================== 3. QSAR 随机森林模型训练 ==============================

st.header("3. QSAR 随机森林模型训练")

st.subheader("模型参数设置")

col1, col2, col3 = st.columns(3)

with col1:
    test_size = st.slider(
        "测试集比例 test_size",
        min_value=0.10,
        max_value=0.50,
        value=0.20,
        step=0.05
    )

with col2:
    n_estimators = st.slider(
        "随机森林树数量 n_estimators",
        min_value=50,
        max_value=500,
        value=250,
        step=50
    )

with col3:
    max_depth_option = st.selectbox(
        "最大树深度 max_depth",
        ["不限制", 3, 5, 10, 20, 30],
        index=0
    )

col4, col5, col6 = st.columns(3)

with col4:
    max_features = st.slider(
        "最大特征比例 max_features",
        min_value=0.10,
        max_value=1.00,
        value=0.30,
        step=0.05
    )

with col5:
    min_samples_split = st.slider(
        "节点最小分裂样本数 min_samples_split",
        min_value=2,
        max_value=20,
        value=2,
        step=1
    )

with col6:
    min_samples_leaf = st.slider(
        "叶节点最小样本数 min_samples_leaf",
        min_value=1,
        max_value=10,
        value=1,
        step=1
    )

random_state = st.number_input(
    "随机种子 random_state",
    min_value=0,
    max_value=9999,
    value=42,
    step=1
)

max_depth = None if max_depth_option == "不限制" else int(max_depth_option)

st.info("""
本模块采用 Random Forest 分类模型构建 QSAR 活性预测模型。
用户可以调节测试集比例、随机森林树数量、最大树深度、最大特征比例等参数。
其中 n_estimators 控制决策树数量，max_depth 控制单棵树复杂度，
max_features 控制每次分裂时参与选择的特征比例。
这些参数会影响模型的拟合能力和泛化能力。
""")

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
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    save_model(model, list(X.columns), MODEL_PATH)

    st.success("模型训练完成")

    params_df = pd.DataFrame({
        "Parameter": [
            "test_size",
            "n_estimators",
            "max_depth",
            "max_features",
            "min_samples_split",
            "min_samples_leaf",
            "random_state",
            "class_weight"
        ],
        "Value": [
            test_size,
            n_estimators,
            "None" if max_depth is None else max_depth,
            max_features,
            min_samples_split,
            min_samples_leaf,
            random_state,
            "balanced"
        ]
    })

    params_df.to_csv(PARAMETERS_PATH, index=False)

    st.subheader("本次模型参数")
    st.dataframe(params_df, use_container_width=True)

    st.subheader("模型性能指标")

    metrics_df = pd.DataFrame(
        list(metrics.items()),
        columns=["Metric", "Value"]
    )

    st.dataframe(metrics_df, use_container_width=True)
    metrics_df.to_csv("results/qsar_metrics.csv", index=False)

    st.subheader("混淆矩阵")

    cm_df = pd.DataFrame(
        cm,
        index=["真实 Inactive", "真实 Active"],
        columns=["预测 Inactive", "预测 Active"]
    )

    st.dataframe(cm_df, use_container_width=True)

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        title="Confusion Matrix"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("模型可解释性分析")

    st.markdown("""
    本部分基于随机森林模型的全局特征重要性进行解释。
    重要性越高，说明该分子描述符在模型区分 Active 和 Inactive 分子时贡献越大。
    """)

    importance_df = get_feature_importance(model, list(X.columns))
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    st.markdown("**全部特征重要性：**")
    st.dataframe(importance_df, use_container_width=True)

    top_importance_df = importance_df.head(10).copy()
    top_importance_df["药物设计意义"] = top_importance_df["feature"].apply(
        explain_feature_meaning
    )

    fig_imp = px.bar(
        top_importance_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 10 Feature Importance"
    )

    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("**Top 10 重要特征解释：**")
    st.dataframe(
        top_importance_df[["feature", "importance", "药物设计意义"]],
        use_container_width=True
    )

    if len(top_importance_df) > 0:
        top_features = "、".join(top_importance_df["feature"].head(3).tolist())
        st.info(
            f"本次 QSAR 模型中，{top_features} 等特征的重要性较高，"
            "说明模型在进行活性分类时较多依赖这些分子性质。"
            "这些结果可以为后续候选分子的结构优化提供参考。"
        )

    st.subheader("训练集内全部分子的预测结果")

    all_prob = model.predict_proba(X)[:, 1]
    all_pred = model.predict(X)

    prediction_df = valid_df[["compound_id", "smiles", "target"]].copy()
    prediction_df["qsar_probability"] = all_prob
    prediction_df["qsar_prediction"] = [
        "Active" if p == 1 else "Inactive" for p in all_pred
    ]

    prediction_df.to_csv(PREDICTION_PATH, index=False)

    st.dataframe(prediction_df, use_container_width=True)

    # ================= 保存本次训练为历史项目 =================

    run_id = create_run_id()
    run_dir = create_run_dir(run_id)

    dataset_info = pd.DataFrame([
        {
            "dataset_path": DATA_PATH,
            "sample_count": df.shape[0],
            "valid_molecule_count": X.shape[0],
            "descriptor_count": X.shape[1],
            "target": df["target"].iloc[0] if "target" in df.columns else "Unknown"
        }
    ])

    dataset_info_path = os.path.join(run_dir, "dataset_info.csv")
    dataset_info.to_csv(
        dataset_info_path,
        index=False,
        encoding="utf-8-sig"
    )

    # 保存混淆矩阵到本次 run 文件夹
    cm_path = os.path.join(run_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_path, encoding="utf-8-sig")

    # 复制本次训练生成的模型和结果文件
    copy_file_if_exists(MODEL_PATH, run_dir)
    copy_file_if_exists(PARAMETERS_PATH, run_dir)
    copy_file_if_exists("results/qsar_metrics.csv", run_dir)
    copy_file_if_exists(FEATURE_IMPORTANCE_PATH, run_dir)
    copy_file_if_exists(PREDICTION_PATH, run_dir)

    history_record = {
        "run_id": run_id,
        "created_time": run_id[:16],
        "dataset": DATA_PATH,
        "target": df["target"].iloc[0] if "target" in df.columns else "Unknown",
        "sample_count": df.shape[0],
        "valid_molecule_count": X.shape[0],
        "descriptor_count": X.shape[1],
        "model_type": "Random Forest",
        "accuracy": metrics.get("Accuracy"),
        "precision": metrics.get("Precision"),
        "recall": metrics.get("Recall"),
        "f1": metrics.get("F1"),
        "roc_auc": metrics.get("ROC_AUC")
    }

    save_run_history(history_record)

    st.success("QSAR 模型训练与训练集预测完成。请进入“活性预测”页面，对候选分子进行分析。")
    st.success(f"本次模型训练结果已保存为历史项目：{run_id}")
