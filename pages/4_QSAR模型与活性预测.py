import os
import sys
import hashlib
import pandas as pd
import streamlit as st
import plotly.express as px
from rdkit import Chem
from streamlit_ketcher import st_ketcher

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


# ============================== 路径设置 ==============================

DATA_PATH = "data/cleaned_activity.csv"
MODEL_PATH = "models/qsar_random_forest.pkl"
PREDICTION_PATH = "results/qsar_predictions.csv"
FEATURE_IMPORTANCE_PATH = "results/feature_importance.csv"
PARAMETERS_PATH = "results/qsar_model_parameters.csv"
NEW_MOLECULE_PREDICTION_PATH = "results/new_molecule_qsar_prediction.csv"


# ============================== 辅助函数 ==============================

def canonicalize_smiles(smiles):
    """
    检查 SMILES 是否有效，并返回标准化 SMILES。
    如果无效，则返回 None。
    """
    if smiles is None or str(smiles).strip() == "":
        return None

    mol = Chem.MolFromSmiles(str(smiles).strip())

    if mol is None:
        return None

    return Chem.MolToSmiles(mol)


def make_ketcher_key(smiles):
    """
    根据输入 SMILES 生成稳定的组件 key。
    当输入 SMILES 改变时，Ketcher 会重新加载结构。
    """
    smiles = str(smiles)
    return hashlib.md5(smiles.encode("utf-8")).hexdigest()


def ensure_feature_columns(desc_df, feature_names):
    """
    保证新分子的描述符列与训练模型时的特征列完全一致。
    缺失列补 0，多余列删除，并按训练时特征顺序排列。
    """
    desc_df = desc_df.copy()

    for col in feature_names:
        if col not in desc_df.columns:
            desc_df[col] = 0

    desc_df = desc_df[feature_names]
    desc_df = desc_df.fillna(0)

    return desc_df


# ============================== 1. 读取清洗后的活性数据 ==============================

st.header("1. 读取清洗后的活性数据")

if not os.path.exists(DATA_PATH):
    st.warning("未找到 data/cleaned_activity.csv。请先让 A 完成活性数据整理模块。")
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

    st.subheader("特征重要性")

    importance_df = get_feature_importance(model, list(X.columns))
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    st.dataframe(importance_df, use_container_width=True)

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

    st.dataframe(prediction_df, use_container_width=True)

    st.success(f"QSAR 预测完成")


# ============================== 4. 新分子活性预测 ==============================

st.header("4. 新分子活性预测")

st.markdown("""
在下方输入候选分子的 SMILES，系统会自动在结构编辑器中显示二维结构。
用户可以在编辑器中修改分子结构，修改完成后点击编辑器右下角的 **Apply**，
再点击“预测当前编辑器中的结构”进行活性预测。
""")

default_smiles = "CCOc1ccc(N)cc1"

input_smiles = st.text_input(
    "输入分子 SMILES",
    value=default_smiles,
    help="SMILES 是分子结构字符串，例如 CCO 表示乙醇。"
)

canonical_input_smiles = canonicalize_smiles(input_smiles)

if canonical_input_smiles is None:
    st.error("当前输入的 SMILES 无效，无法加载到结构编辑器。请检查后重新输入。")
    st.stop()

st.subheader("可交互分子结构编辑器")

ketcher_key = "ketcher_" + make_ketcher_key(canonical_input_smiles)

edited_smiles = st_ketcher(
    canonical_input_smiles,
    key=ketcher_key
)

if edited_smiles is None or str(edited_smiles).strip() == "":
    current_smiles = canonical_input_smiles
else:
    current_smiles = str(edited_smiles).strip()

canonical_current_smiles = canonicalize_smiles(current_smiles)

if canonical_current_smiles is None:
    st.warning("编辑器中的结构暂时无法转换为有效 SMILES，请检查结构后点击 Apply。")
    st.stop()

st.session_state["qsar_current_editor_smiles"] = canonical_current_smiles

st.markdown("**当前用于预测的 SMILES：**")
st.code(canonical_current_smiles)

if st.button("预测当前编辑器中的结构"):
    if not os.path.exists(MODEL_PATH):
        st.error("尚未找到已训练模型。请先在上方完成 QSAR 模型训练。")
        st.stop()

    import joblib

    model_package = joblib.load(MODEL_PATH)
    model = model_package["model"]
    feature_names = model_package["feature_names"]

    predict_smiles = st.session_state.get(
        "qsar_current_editor_smiles",
        canonical_current_smiles
    )

    desc = calculate_basic_descriptors(predict_smiles)

    if desc is None:
        st.error("无法计算该分子的描述符，请检查 SMILES。")
        st.stop()

    desc_df = pd.DataFrame([desc])
    desc_df = ensure_feature_columns(desc_df, feature_names)

    prediction, probability = predict_single_smiles(model, desc_df)

    st.subheader("预测结果")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("预测类别", prediction)

    with col2:
        st.metric("Active 概率", f"{probability:.3f}")

    if prediction == "Active":
        st.success("该分子被模型预测为潜在活性分子，可进入后续 ADMET 和分子对接分析。")
    else:
        st.warning("该分子被模型预测为低活性分子，后续可谨慎考虑。")

    st.subheader("当前分子的描述符")
    st.dataframe(desc_df, use_container_width=True)

    os.makedirs("results", exist_ok=True)

    new_prediction_df = pd.DataFrame([{
        "compound_id": "New_Molecule",
        "smiles": predict_smiles,
        "target": "User_Input",
        "qsar_probability": probability,
        "qsar_prediction": prediction
    }])

    new_prediction_df.to_csv(
        NEW_MOLECULE_PREDICTION_PATH,
        index=False
    )

    st.success("新分子预测完成")
