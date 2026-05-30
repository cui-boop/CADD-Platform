from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, DataStructs, Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

try:
    from streamlit_ketcher import st_ketcher
except Exception:
    st_ketcher = None


# ========================= 页面配置 =========================
st.set_page_config(page_title="活性分子预测", page_icon="🧪", layout="wide")

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

st.title("活性分子预测")
st.markdown(
    "加载已经训练好的 **随机森林 QSAR 模型**，支持单分子 SMILES 预测、交互式分子绘制预测和批量候选分子预测。"
)


# ========================= 固定目录 =========================
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "qsar_random_forest.pkl"
RESULT_DIR = ROOT / "results"

RESULT_DIR.mkdir(exist_ok=True)

SINGLE_PATH = RESULT_DIR / "single_molecule_prediction.csv"
BATCH_PATH = RESULT_DIR / "new_molecule_qsar_prediction.csv"
GENERATED_MOLECULES_PATH = RESULT_DIR / "generated_molecules.csv"
GENERATED_QSAR_PATH = RESULT_DIR / "generated_qsar_predictions.csv"


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


def calculate_feature_row(smiles, package):
    can_smi = canonicalize_smiles(smiles)

    if can_smi is None:
        return None, None, None

    mol = Chem.MolFromSmiles(can_smi)

    morgan_radius = int(package.get("morgan_radius", 2))
    morgan_bits = int(package.get("morgan_bits", 1024))
    desc_names = package.get("descriptor_names", DESC_NAMES)

    generator = GetMorganGenerator(radius=morgan_radius, fpSize=morgan_bits)

    fp_arr = np.zeros((morgan_bits,), dtype=np.int8)
    fp = generator.GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    desc = calc_10_rdkit_desc(mol)
    x = np.hstack([
        np.asarray(desc, dtype=np.float32),
        fp_arr.astype(np.float32),
    ]).reshape(1, -1)

    desc_df = pd.DataFrame([desc], columns=desc_names)

    return x, can_smi, desc_df


def predict_one(smiles, package):
    x, can_smi, desc_df = calculate_feature_row(smiles, package)

    if x is None:
        return None

    model = package["model"]
    threshold = float(package.get("threshold", 0.5))

    prob_active = float(model.predict_proba(x)[0, 1])
    pred_label = int(prob_active >= threshold)

    return {
        "canonical_smiles": can_smi,
        "prob_active": prob_active,
        "pred_label": pred_label,
        "prediction": "Active" if pred_label == 1 else "Inactive",
        "threshold": threshold,
        "descriptor_table": desc_df,
    }


def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=(520, 340))


def plot_probability_gauge(prob, threshold):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={"valueformat": ".3f"},
        title={"text": "Active Probability"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"thickness": 0.28},
            "threshold": {
                "line": {"width": 4},
                "thickness": 0.75,
                "value": threshold,
            },
        },
    ))

    fig.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=55, b=15),
    )

    return fig


def read_uploaded_table(file):
    name = file.name.lower()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)

    if name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(file, sep="\t")

    return pd.read_csv(file)


def guess_smiles_col(df):
    for c in ["smiles", "SMILES", "canonical_smiles", "Canonical_SMILES", "smile", "Smile"]:
        if c in df.columns:
            return c
    return df.columns[0]


def get_auto_smiles_col(df):
    """
    自动识别 SMILES 列。
    """
    for c in ["smiles", "SMILES", "canonical_smiles", "Canonical_SMILES", "smile", "Smile"]:
        if c in df.columns:
            return c
    return None


def clean_prediction_output(out_df):
    """
    清理批量活性预测结果，避免同时出现 smiles 和 canonical_smiles 两列。
    后续模块统一使用 smiles 列。
    """
    out_df = out_df.copy()

    if "canonical_smiles" in out_df.columns:
        if "smiles" in out_df.columns:
            out_df["smiles"] = out_df["canonical_smiles"].combine_first(out_df["smiles"])
            out_df = out_df.drop(columns=["canonical_smiles"])
        else:
            out_df = out_df.rename(columns={"canonical_smiles": "smiles"})

    return out_df


def batch_predict(df, smiles_col, package):
    rows = []

    for i, smi in enumerate(df[smiles_col].tolist()):
        result = predict_one(smi, package)

        if result is None:
            rows.append({
                "input_index": i,
                "input_smiles": smi,
                "canonical_smiles": None,
                "prob_active": np.nan,
                "pred_label": np.nan,
                "prediction": "Invalid SMILES",
                "threshold": package.get("threshold", 0.5),
            })
        else:
            rows.append({
                "input_index": i,
                "input_smiles": smi,
                "canonical_smiles": result["canonical_smiles"],
                "prob_active": result["prob_active"],
                "pred_label": result["pred_label"],
                "prediction": result["prediction"],
                "threshold": result["threshold"],
            })

    return pd.DataFrame(rows)



def run_ketcher(default_smiles):
    if st_ketcher is None:
        st.warning("没有检测到 streamlit-ketcher。请先运行：pip install streamlit-ketcher")
        return None

    try:
        return st_ketcher(value=default_smiles, height=520, key="ketcher_editor")
    except TypeError:
        try:
            return st_ketcher(default_smiles, key="ketcher_editor")
        except TypeError:
            return st_ketcher(default_smiles)


# ========================= 加载模型 =========================
if not MODEL_PATH.exists():
    st.error("没有找到 models/qsar_random_forest.pkl。请先在 QSAR 模型训练页面训练模型。")
    st.stop()

package = joblib.load(MODEL_PATH)

threshold = float(package.get("threshold", 0.5))
desc_names = package.get("descriptor_names", DESC_NAMES)
morgan_radius = int(package.get("morgan_radius", 2))
morgan_bits = int(package.get("morgan_bits", 1024))


# ========================= 模型信息 =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("模型信息")

c1, c2, c3, c4 = st.columns(4)
c1.metric("模型类型", "Random Forest")
c2.metric("RDKit 参数", f"{len(desc_names)} 个")
c3.metric("Morgan 指纹", f"{morgan_bits} bit")
c4.metric("分类阈值", f"{threshold:.3f}")

st.caption("模型文件：models/qsar_random_forest.pkl")
st.markdown('</div>', unsafe_allow_html=True)


tab1, tab2 = st.tabs(["单分子交互预测", "批量候选分子预测"])


# ========================= 单分子预测 =========================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1. 输入或绘制分子结构")

    default_smiles = "CCOc1ccc(N)cc1"

    smiles_text = st.text_input(
        "输入 SMILES",
        value=default_smiles,
        help="可以直接输入 SMILES，也可以在下方结构编辑器中画分子。",
    )

    st.markdown("#### 交互式分子结构编辑器")
    drawn_smiles = run_ketcher(smiles_text)

    use_drawer = st.checkbox("优先使用结构编辑器中的分子进行预测", value=True)

    if use_drawer and drawn_smiles and canonicalize_smiles(drawn_smiles) is not None:
        final_smiles = drawn_smiles
    else:
        final_smiles = smiles_text

    st.caption(f"当前用于预测的 SMILES：{final_smiles}")

    run_pred = st.button("预测该分子活性", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if run_pred:
        result = predict_one(final_smiles, package)

        if result is None:
            st.error("SMILES 无效，无法预测。")
            st.stop()

        save_df = pd.DataFrame([{
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_smiles": final_smiles,
            "canonical_smiles": result["canonical_smiles"],
            "prob_active": result["prob_active"],
            "pred_label": result["pred_label"],
            "prediction": result["prediction"],
            "threshold": result["threshold"],
        }])

        save_df.to_csv(SINGLE_PATH, index=False)

        st.subheader("2. 预测结果")

        r1, r2, r3 = st.columns([0.85, 1.15, 1.0])

        with r1:
            st.metric("预测类别", result["prediction"])
            st.metric("Active 概率", f"{result['prob_active']:.4f}")
            st.metric("分类阈值", f"{result['threshold']:.3f}")

            if result["prediction"] == "Active":
                st.success("该分子被预测为潜在活性分子。")
            else:
                st.warning("该分子被预测为低活性或非活性分子。")

        with r2:
            img = draw_molecule(result["canonical_smiles"])
            if img is not None:
                st.image(img, caption="Molecular Structure")

        with r3:
            st.plotly_chart(
                plot_probability_gauge(result["prob_active"], result["threshold"]),
                use_container_width=True,
            )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("3. 10 个 RDKit 理化参数")

        desc_show = result["descriptor_table"].T.reset_index()
        desc_show.columns = ["Descriptor", "Value"]
        st.dataframe(desc_show, use_container_width=True, hide_index=True)

        st.caption("结果已保存：results/single_molecule_prediction.csv")
        st.markdown('</div>', unsafe_allow_html=True)


# ========================= 批量预测 =========================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1. 选择候选分子来源")

    source_mode = st.radio(
        "批量预测数据来源",
        ["使用已生成的候选分子集 results/generated_molecules.csv", "手动上传候选分子文件"],
        horizontal=True,
    )

    df = None

    if source_mode == "使用已生成的候选分子集 results/generated_molecules.csv":
        if GENERATED_MOLECULES_PATH.exists():
            df = pd.read_csv(GENERATED_MOLECULES_PATH)
            st.success(f"已读取已生成候选分子集：{GENERATED_MOLECULES_PATH}")
        else:
            st.warning("未找到 results/generated_molecules.csv。请先在“分子生成”页面生成候选分子，或改为手动上传文件。")
    else:
        uploaded = st.file_uploader(
            "支持 CSV、TSV、TXT、XLSX，至少包含一列 SMILES",
            type=["csv", "tsv", "txt", "xlsx", "xls"],
        )

        if uploaded is not None:
            df = read_uploaded_table(uploaded)
        else:
            st.info("请上传候选分子文件后进行批量预测。")

    st.markdown('</div>', unsafe_allow_html=True)

    if df is not None:
        smiles_col = get_auto_smiles_col(df)

        if smiles_col is None:
            st.error("当前候选分子文件中未检测到 smiles 列，请确认文件至少包含 smiles 或 canonical_smiles 列。")
            st.stop()

        st.metric("候选分子数量", f"{len(df):,}")

        if st.button("开始批量预测", type="primary", use_container_width=True):
            with st.spinner("正在计算分子特征并预测活性..."):
                pred_df = batch_predict(df, smiles_col, package)

            out_df = pd.concat(
                [
                    df.reset_index(drop=True),
                    pred_df.drop(columns=["input_smiles"]).reset_index(drop=True),
                ],
                axis=1,
            )

            out_df = clean_prediction_output(out_df)

            out_df = out_df.sort_values(
                "prob_active",
                ascending=False,
                na_position="last",
            )

            # 统一列名，便于后续和 ADMET 结果合并，并作为分子设计阶段输入
            out_df["qsar_probability"] = out_df["prob_active"]
            out_df["qsar_prediction"] = out_df["prediction"]

            if source_mode == "使用已生成的候选分子集 results/generated_molecules.csv":
                out_df.to_csv(GENERATED_QSAR_PATH, index=False, encoding="utf-8-sig")
                out_df.to_csv(BATCH_PATH, index=False, encoding="utf-8-sig")
                st.success("批量活性预测完成，结果已保存：results/generated_qsar_predictions.csv")
            else:
                out_df.to_csv(BATCH_PATH, index=False, encoding="utf-8-sig")
                st.success("批量预测完成，结果已保存：results/new_molecule_qsar_prediction.csv")

            valid = out_df.dropna(subset=["prob_active"]).copy()

            m1, m2, m3 = st.columns(3)
            m1.metric("有效分子", f"{len(valid):,}")
            m2.metric("预测 Active", f"{int((valid['pred_label'] == 1).sum()):,}")
            m3.metric("最高 Active 概率", f"{valid['prob_active'].max():.4f}" if len(valid) else "NA")

            st.subheader("Active 概率排序图")

            scatter_df = valid.sort_values(
                "prob_active",
                ascending=False
            ).reset_index(drop=True)

            scatter_df["rank_by_active_probability"] = range(1, len(scatter_df) + 1)

            hover_cols = [
                col for col in [
                    "compound_id",
                    "smiles",
                    "prob_active",
                    "prediction",
                    "threshold"
                ]
                if col in scatter_df.columns
            ]

            fig = px.scatter(
                scatter_df,
                x="rank_by_active_probability",
                y="prob_active",
                color="prediction",
                hover_data=hover_cols,
                title="Candidate Molecules Ranked by Active Probability",
                labels={
                    "rank_by_active_probability": "候选分子排序",
                    "prob_active": "Active 概率",
                    "prediction": "预测类别"
                }
            )

            fig.add_hline(
                y=threshold,
                line_dash="dash",
                annotation_text=f"分类阈值 {threshold:.3f}",
                annotation_position="bottom right"
            )

            fig.update_traces(
                marker=dict(size=10, opacity=0.85)
            )

            fig.update_layout(
                height=430,
                margin=dict(l=10, r=10, t=55, b=20),
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Top 20 高分候选分子")

            top = valid.head(20).copy()
            top["show_name"] = top["smiles"].astype(str).str.slice(0, 45)

            fig = px.bar(
                top.sort_values("prob_active"),
                x="prob_active",
                y="show_name",
                orientation="h",
                text="prob_active",
                title="Top 20 Candidate Molecules",
            )
            fig.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside",
                cliponaxis=False,
            )
            fig.update_layout(
                height=620,
                xaxis_title="Active Probability",
                yaxis_title="",
                margin=dict(l=10, r=45, t=55, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("完整预测结果")
            st.dataframe(out_df, use_container_width=True, hide_index=True)

            csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
            download_name = (
                "generated_qsar_predictions.csv"
                if source_mode == "使用已生成的候选分子集 results/generated_molecules.csv"
                else "new_molecule_qsar_prediction.csv"
            )

            st.download_button(
                "下载预测结果 CSV",
                data=csv_bytes,
                file_name=download_name,
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("请选择已生成候选分子集，或上传候选分子文件后进行批量活性预测。")
