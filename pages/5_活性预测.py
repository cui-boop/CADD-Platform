import os
import sys
import hashlib

import joblib
import pandas as pd
import streamlit as st

from rdkit import Chem
from streamlit_ketcher import st_ketcher

# 让 pages 里的文件可以导入 utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from utils.descriptors import calculate_basic_descriptors
from utils.qsar_model import predict_single_smiles


st.set_page_config(
    page_title="活性预测",
    layout="wide"
)

st.title("活性预测")

st.markdown(
    """
    本页面用于对新的候选分子进行 QSAR 活性预测。
    用户输入或编辑一个候选分子的 SMILES 后，系统会读取“模型训练”页面保存的随机森林 QSAR 模型，
    并预测该候选分子属于 Active 或 Inactive 的概率。
    """
)


# ============================== 路径设置 ==============================

MODEL_PATH = "models/qsar_random_forest.pkl"
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


# ============================== 1. 候选分子输入 ==============================

st.header("一、候选分子输入")

st.markdown(
    """
    在下方输入候选分子的 SMILES，系统会自动在结构编辑器中显示二维结构。
    用户可以在编辑器中修改分子结构，修改完成后点击编辑器右下角的 **Apply**，
    再点击“开始活性预测”进行 QSAR 预测。
    """
)

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

st.session_state["activity_prediction_current_smiles"] = canonical_current_smiles

st.markdown("**当前用于预测的 SMILES：**")
st.code(canonical_current_smiles)


# ============================== 2. QSAR 活性预测 ==============================

st.header("二、QSAR 活性预测结果")

if st.button("开始活性预测", use_container_width=True):
    if not os.path.exists(MODEL_PATH):
        st.error("尚未找到已训练模型。请先进入“模型训练”页面完成 QSAR 模型训练。")
        st.stop()

    model_package = joblib.load(MODEL_PATH)
    model = model_package["model"]
    feature_names = model_package["feature_names"]

    predict_smiles = st.session_state.get(
        "activity_prediction_current_smiles",
        canonical_current_smiles
    )

    desc = calculate_basic_descriptors(predict_smiles)

    if desc is None:
        st.error("无法计算该分子的描述符，请检查 SMILES。")
        st.stop()

    desc_df = pd.DataFrame([desc])
    desc_df = ensure_feature_columns(desc_df, feature_names)

    prediction, probability = predict_single_smiles(model, desc_df)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("预测类别", prediction)

    with col2:
        st.metric("Active 概率", f"{probability:.3f}")

    if prediction == "Active":
        st.success("该分子被模型预测为潜在活性分子，可进入后续 ADMET、分子对接或动力学模拟分析。")
    else:
        st.warning("该分子被模型预测为低活性分子，后续可谨慎考虑或进行结构优化。")

    st.subheader("当前分子的描述符")
    st.dataframe(desc_df, use_container_width=True)

    os.makedirs("results", exist_ok=True)

    new_prediction_df = pd.DataFrame(
        [
            {
                "compound_id": "New_Molecule",
                "smiles": predict_smiles,
                "target": "User_Input",
                "qsar_probability": probability,
                "qsar_prediction": prediction
            }
        ]
    )

    new_prediction_df.to_csv(
        NEW_MOLECULE_PREDICTION_PATH,
        index=False
    )

    st.success("新分子活性预测完成。")

else:
    st.info("点击“开始活性预测”后，将显示预测类别、Active 概率和当前分子的描述符。")
