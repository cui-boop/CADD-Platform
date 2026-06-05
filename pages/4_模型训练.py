from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)


# ========================= 页面配置 =========================
st.set_page_config(
    page_title="模型训练", 
    page_icon="🤖", 
    layout="wide"
)
st.title("🤖 模型训练")

st.markdown("""
<style>
.main .block-container {padding-top: 1.3rem; max-width: 1250px;}
h1 {font-weight: 850;}
.card {
    background: #ffffff;
    border: 1px solid #e7edf5;
    border-radius: 16px;
    padding: 1.05rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(15,23,42,0.04);
}
.note {color:#667085; font-size:0.92rem;}
div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e7edf5;
    border-radius: 14px;
    padding: .85rem;
    box-shadow: 0 1px 8px rgba(15,23,42,0.04);
}
</style>
""", unsafe_allow_html=True)


st.markdown(
    "本模块使用 **10 个 RDKit 理化参数和Morgan 指纹** 构建随机森林 QSAR 二分类模型。"
)


# ========================= 固定目录 =========================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "qsar_random_forest.pkl"
METRICS_PATH = RESULT_DIR / "qsar_metrics.csv"
PRED_PATH = RESULT_DIR / "qsar_predictions.csv"
FI_PATH = RESULT_DIR / "feature_importance.csv"
PARAM_PATH = RESULT_DIR / "qsar_model_parameters.csv"


DESC_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
]


# ========================= 工具函数 =========================
def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def list_data_files():
    if not DATA_DIR.exists():
        return []
    return sorted([
        p.name for p in DATA_DIR.iterdir()
        if p.suffix.lower() in [".csv", ".tsv", ".txt"]
    ])


def guess_smiles_col(df: pd.DataFrame) -> str:
    for c in ["smiles", "SMILES", "canonical_smiles", "Canonical_SMILES", "smile", "Smile"]:
        if c in df.columns:
            return c
    return df.columns[0]


def label_to_binary(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip().lower()

    if s in ["1", "1.0", "active", "act", "yes", "true", "positive"]:
        return 1

    if s in ["0", "0.0", "inactive", "inact", "no", "false", "negative"]:
        return 0

    try:
        v = float(s)
        if v == 1:
            return 1
        if v == 0:
            return 0
    except Exception:
        pass

    return np.nan


def binary_label_columns(df: pd.DataFrame, smiles_col: str):
    cols = []

    for c in df.columns:
        if c == smiles_col:
            continue

        s = df[c].dropna().map(label_to_binary).dropna()
        if len(s) > 0 and set(s.unique()).issubset({0, 1}) and len(set(s.unique())) == 2:
            cols.append(c)

    preferred = ["label", "Label", "activity", "Activity", "class", "Class", "target", "Target"]
    return sorted(cols, key=lambda x: (x not in preferred, x))


def canonicalize_smiles(smi):
    if smi is None or pd.isna(smi) or str(smi).strip() == "":
        return None

    mol = Chem.MolFromSmiles(str(smi).strip())
    if mol is None:
        return None

    return Chem.MolToSmiles(mol)


def calc_10_rdkit_desc(mol):
    return [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        mol.GetNumHeavyAtoms(),
    ]


@st.cache_data(show_spinner=False)
def build_feature_matrix(smiles_values, radius, n_bits):
    generator = GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))

    desc_rows = []
    fp_rows = []
    valid_index = []
    valid_smiles = []

    for i, smi in enumerate(smiles_values):
        can_smi = canonicalize_smiles(smi)
        if can_smi is None:
            continue

        mol = Chem.MolFromSmiles(can_smi)

        fp_arr = np.zeros((int(n_bits),), dtype=np.int8)
        fp = generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, fp_arr)

        desc_rows.append(calc_10_rdkit_desc(mol))
        fp_rows.append(fp_arr)
        valid_index.append(i)
        valid_smiles.append(can_smi)

    if len(valid_index) == 0:
        return np.empty((0, len(DESC_NAMES) + int(n_bits))), [], []

    x_desc = np.asarray(desc_rows, dtype=np.float32)
    x_fp = np.asarray(fp_rows, dtype=np.float32)
    x = np.hstack([x_desc, x_fp]).astype(np.float32)

    return x, valid_index, valid_smiles


def feature_names(n_bits):
    return DESC_NAMES + [f"Morgan_{i:04d}" for i in range(int(n_bits))]


def best_threshold_by_f1(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return 0.5

    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[np.argmax(f1)])


def evaluate_model(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, y_prob),
        "PR AUC": average_precision_score(y_true, y_prob),
        "Threshold": threshold,
    }

    return metrics, y_pred


def plot_label_distribution(y):
    tmp = pd.Series(y).map({0: "Inactive", 1: "Active"}).value_counts().reset_index()
    tmp.columns = ["Label", "Count"]

    fig = px.bar(tmp, x="Label", y="Count", text="Count", title="标签分布")
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=350,
        yaxis_title="Molecule Count",
        margin=dict(l=10, r=10, t=55, b=20),
    )
    return fig


def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"ROC AUC = {auc_value:.3f}",
        line=dict(width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(dash="dash"),
    ))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=420,
        margin=dict(l=10, r=10, t=55, b=20),
    )
    return fig


def plot_pr(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode="lines",
        name=f"PR AUC = {ap:.3f}",
        line=dict(width=3),
    ))
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=420,
        margin=dict(l=10, r=10, t=55, b=20),
    )
    return fig


def plot_confusion_matrix(cm):
    cm_df = pd.DataFrame(
        cm,
        index=["True Inactive", "True Active"],
        columns=["Pred Inactive", "Pred Active"],
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=55, b=20),
    )
    return fig


def plot_top_importance(fi_df, top_n=20):
    top = fi_df.head(top_n).sort_values("importance", ascending=True)

    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        text="importance",
        title=f"Top {top_n} Feature Importance",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=560,
        xaxis_title="Importance",
        yaxis_title="",
        margin=dict(l=10, r=50, t=55, b=20),
    )
    return fig


def plot_descriptor_importance(fi_df):
    desc_imp = fi_df[fi_df["feature"].isin(DESC_NAMES)].copy()
    desc_imp = desc_imp.sort_values("importance", ascending=True)

    fig = px.bar(
        desc_imp,
        x="importance",
        y="feature",
        orientation="h",
        text="importance",
        title="10 个 RDKit 理化参数的重要性",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=430,
        xaxis_title="Importance",
        yaxis_title="",
        margin=dict(l=10, r=45, t=55, b=20),
    )
    return fig


# ========================= 数据选择 =========================
data_files = list_data_files()

if not data_files:
    st.error("没有在 data/ 目录中找到 CSV、TSV 或 TXT 数据文件。")
    st.stop()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("1. 选择训练数据")

data_file = st.selectbox("数据文件", data_files)
df = read_table(DATA_DIR / data_file)

smiles_col = guess_smiles_col(df)
label_candidates = binary_label_columns(df, smiles_col)

if not label_candidates:
    st.error("没有识别到有效的二分类标签列。标签需要是 0/1、Active/Inactive、True/False 等格式。")
    st.stop()

target_col = label_candidates[0]

st.caption(
    f"当前数据：{df.shape[0]:,} 行 × {df.shape[1]:,} 列；"
    f"自动识别 SMILES 列：{smiles_col}；自动识别标签列：{target_col}"
)
st.markdown('</div>', unsafe_allow_html=True)


# ========================= 参数设置 =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("2. 模型参数设置")

p1, p2, p3, p4 = st.columns(4)

with p1:
    morgan_bits = st.selectbox("Morgan 指纹位数", [512, 1024, 2048], index=1)

with p2:
    morgan_radius = st.selectbox("Morgan 半径", [2, 3], index=0)

with p3:
    max_features = st.selectbox("max_features", ["sqrt", "log2"], index=0)

with p4:
    threshold_mode = st.selectbox("分类阈值", ["固定 0.5", "自动选择 F1 最佳阈值"], index=1)

p5, p6, p7, p8, p9 = st.columns(5)

with p5:
    n_estimators = st.slider("随机森林树数", 100, 500, 220, step=20)

with p6:
    max_depth = st.slider("max_depth", 4, 30, 18, step=2)

with p7:
    min_samples_leaf = st.slider("min_samples_leaf", 1, 8, 2, step=1)

with p8:
    test_size = st.slider("测试集比例", 0.10, 0.30, 0.20, step=0.05)

with p9:
    class_weight = st.selectbox("类别权重", ["balanced_subsample", "balanced", "不使用"], index=0)

class_weight_value = None if class_weight == "不使用" else class_weight

st.markdown(
    f"""
    <div class="note">
    当前特征：10 个 RDKit 理化参数 + {morgan_bits} bit Morgan 指纹；
    当前模型：RandomForestClassifier。
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)


# ========================= 数据预览 =========================
work_df = df[[smiles_col, target_col]].copy()
work_df["label"] = work_df[target_col].map(label_to_binary)
work_df = work_df.dropna(subset=[smiles_col, "label"])
work_df["label"] = work_df["label"].astype(int)

left, right = st.columns([1.0, 1.1])

with left:
    st.plotly_chart(plot_label_distribution(work_df["label"]), use_container_width=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("10 个 RDKit 理化参数")
    desc_df = pd.DataFrame({
        "Descriptor": DESC_NAMES,
        "Meaning": [
            "分子量",
            "脂水分配系数",
            "拓扑极性表面积",
            "氢键供体数",
            "氢键受体数",
            "可旋转键数",
            "环数量",
            "芳香环数量",
            "sp3 碳比例",
            "重原子数",
        ],
    })
    st.dataframe(desc_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ========================= 训练模型 =========================
if st.button("开始训练随机森林 QSAR 模型", type="primary", use_container_width=True):

    if work_df["label"].nunique() != 2:
        st.error("当前标签列不是有效的二分类标签。")
        st.stop()

    if work_df["label"].value_counts().min() < 5:
        st.error("至少有一个类别样本过少，无法稳定训练模型。")
        st.stop()

    with st.spinner("正在计算 RDKit 理化参数和 Morgan 指纹..."):
        x, valid_index, valid_smiles = build_feature_matrix(
            tuple(work_df[smiles_col].astype(str).tolist()),
            int(morgan_radius),
            int(morgan_bits),
        )

    if len(valid_index) < 20:
        st.error("有效 SMILES 数量太少，无法训练模型。")
        st.stop()

    y = work_df["label"].iloc[valid_index].to_numpy(dtype=int)

    x_train, x_test, y_train, y_test, smi_train, smi_test = train_test_split(
        x,
        y,
        valid_smiles,
        test_size=float(test_size),
        random_state=42,
        stratify=y,
    )

    rf_params = {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "min_samples_leaf": int(min_samples_leaf),
        "max_features": max_features,
        "class_weight": class_weight_value,
        "n_jobs": -1,
        "random_state": 42,
    }

    with st.spinner("正在训练随机森林模型..."):
        model = RandomForestClassifier(**rf_params)
        model.fit(x_train, y_train)

    train_prob = model.predict_proba(x_train)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    if threshold_mode == "固定 0.5":
        threshold = 0.5
    else:
        threshold = best_threshold_by_f1(y_train, train_prob)

    metrics, y_pred = evaluate_model(y_test, test_prob, threshold)
    cm = confusion_matrix(y_test, y_pred)

    names = feature_names(morgan_bits)
    fi_df = pd.DataFrame({
        "feature": names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    pred_df = pd.DataFrame({
        "smiles": smi_test,
        "true_label": y_test,
        "pred_label": y_pred,
        "prob_active": test_prob,
    }).sort_values("prob_active", ascending=False)

    metrics_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value": list(metrics.values()),
    })

    param_df = pd.DataFrame({
        "Parameter": [
            "Model",
            "Dataset",
            "自动识别的 SMILES 列",
            "自动识别的标签列",
            "RDKit descriptors",
            "Morgan radius",
            "Morgan bits",
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "class_weight",
            "test_size",
            "threshold_mode",
            "threshold",
            "train_samples",
            "test_samples",
            "created_at",
        ],
        "Value": [
            "RandomForestClassifier",
            data_file,
            smiles_col,
            target_col,
            ", ".join(DESC_NAMES),
            morgan_radius,
            morgan_bits,
            n_estimators,
            max_depth,
            min_samples_leaf,
            max_features,
            class_weight,
            test_size,
            threshold_mode,
            threshold,
            len(y_train),
            len(y_test),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ],
    })

    joblib.dump(
        {
            "model": model,
            "descriptor_names": DESC_NAMES,
            "feature_names": names,
            "morgan_radius": int(morgan_radius),
            "morgan_bits": int(morgan_bits),
            "threshold": float(threshold),
            "target_col": target_col,
            "smiles_col": smiles_col,
            "rf_params": rf_params,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        MODEL_PATH,
    )

    metrics_df.to_csv(METRICS_PATH, index=False)
    pred_df.to_csv(PRED_PATH, index=False)
    fi_df.to_csv(FI_PATH, index=False)
    param_df.to_csv(PARAM_PATH, index=False)

    st.success(f"模型训练完成")

    st.subheader("3. 模型评价结果")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC AUC", f"{metrics['ROC AUC']:.3f}")
    m2.metric("PR AUC", f"{metrics['PR AUC']:.3f}")
    m3.metric("F1 Score", f"{metrics['F1 Score']:.3f}")
    m4.metric("Recall", f"{metrics['Recall']:.3f}")
    m5.metric("Threshold", f"{metrics['Threshold']:.3f}")

    c1, c2 = st.columns(2)
    c1.plotly_chart(plot_roc(y_test, test_prob), use_container_width=True)
    c2.plotly_chart(plot_pr(y_test, test_prob), use_container_width=True)

    c1, c2 = st.columns([0.9, 1.1])
    c1.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)
    c2.plotly_chart(plot_top_importance(fi_df, 20), use_container_width=True)

    st.plotly_chart(plot_descriptor_importance(fi_df), use_container_width=True)

    st.subheader("4. 结果文件")
    st.dataframe(param_df, use_container_width=True, hide_index=True)



else:
    st.info("选择数据文件和模型参数后，点击按钮开始训练。")
