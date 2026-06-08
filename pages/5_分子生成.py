import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# 让 pages 里的文件可以导入 utils
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.smiles_gru_generation import (
    torch_available,
    rdkit_available,
    load_training_smiles,
    train_smiles_gru,
    generate_smiles_table,
    save_generated_molecules,
    get_mol_image,
)


st.set_page_config(
    page_title="分子生成",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 SMILES-GRU 分子生成")

st.markdown(
    """
    本模块基于 SMILES 序列生成模型开展新分子结构设计。
    用户可以选择当前平台内置的 EGFR ChEMBL 单靶点活性数据或自行上传包含smiles列的文件，
    模型会学习 SMILES 字符串的排列规律，
    然后逐字符采样生成具有相似结构特征的新候选分子。

    生成结果将自动进行 RDKit 有效性检查、去重和基础理化性质计算。
    """
)


# ============================== 路径设置 ==============================

DEFAULT_DATA_PATH = "data/cleaned_activity.csv"
GENERATOR_MODEL_PATH = "models/smiles_gru_generator.pt"
FULL_OUTPUT_PATH = "results/gru_generated_molecules.csv"
COMPATIBLE_OUTPUT_PATH = "results/generated_molecules.csv"
TRAINING_LOSS_PATH = "results/smiles_gru_training_loss.csv"


def prepare_generation_output(df):
    """
    统一分子生成结果表格格式。

    SMILES-GRU 原始输出中：
    - smiles：模型直接生成的原始字符串；
    - canonical_smiles：RDKit 标准化后的 SMILES。

    为避免页面和结果文件出现两列 SMILES，这里只保留标准化后的 SMILES，
    并统一命名为 smiles。后续 QSAR、成药性筛选和分子设计页面都使用这一列。
    """
    if df is None or df.empty:
        return df

    out_df = df.copy()

    if "canonical_smiles" in out_df.columns:
        if "smiles" in out_df.columns:
            out_df = out_df.drop(columns=["smiles"])
        out_df = out_df.rename(columns={"canonical_smiles": "smiles"})

    return out_df


def get_smiles_column(df):
    """
    返回当前结果表中可用于结构预览的 SMILES 列名。
    """
    if "smiles" in df.columns:
        return "smiles"
    if "canonical_smiles" in df.columns:
        return "canonical_smiles"
    return None


# ============================== 一、环境检查 ==============================

st.header("一、运行环境检测")

col1, col2 = st.columns(2)

with col1:
    if torch_available():
        st.success("已检测到 PyTorch：可以训练和运行 SMILES-GRU 生成模型。")
    else:
        st.error("未检测到 PyTorch：无法执行模型训练任务，请先安装 torch。")
        st.code("python -m pip install torch", language="powershell")

with col2:
    if rdkit_available():
        st.success("已检测到 RDKit：可进行分子结构解析、有效性验证及性质计算。")
    else:
        st.warning("未检测到 RDKit：模型仍可完成字符串生成，但无法进行严格的化学结构有效性校验。")

if not torch_available():
    st.stop()

# ============================== 二、训练数据设置 ==============================

st.header("二、训练数据配置")

data_source = st.radio(
    "选择训练数据来源",
    ["使用内置数据", "上传包含 SMILES 列的 CSV 文件"],
    horizontal=True
)

data_path = DEFAULT_DATA_PATH

if data_source == "使用内置数据":
    if not os.path.exists(DEFAULT_DATA_PATH):
        st.error("未找到内置数据，请先完成活性数据整理。")
        st.stop()

    preview_df = pd.read_csv(DEFAULT_DATA_PATH)
    st.success("数据读取成功")

else:
    uploaded_file = st.file_uploader(
        "上传训练用 CSV 文件",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info("请先上传包含 SMILES 列的 CSV 文件。")
        st.stop()

    preview_df = pd.read_csv(uploaded_file)

    if "smiles" not in preview_df.columns:
        st.error("上传文件必须包含 SMILES 列。")
        st.stop()

    os.makedirs("data", exist_ok=True)
    data_path = "data/uploaded_smiles_for_generation.csv"

    preview_df.to_csv(
        data_path,
        index=False,
        encoding="utf-8-sig"
    )

    st.success("文件解析完成")

st.subheader("训练数据集预览")
st.dataframe(preview_df.head(), use_container_width=True)

if "label" in preview_df.columns:
    active_only = st.checkbox(
        "仅使用 Active 分子训练生成模型",
        value=True,
        help="建议开启，仅使用活性分子可使模型更聚焦于潜在有效结构空间布。"
    )
else:
    active_only = False
    st.info("当前数据集中未检测到活性标签，将使用全部分子结构参与训练。")



# ============================== 三、训练模型 ==============================

st.header("三、训练 SMILES-GRU 生成模型")

col_a, col_b, col_c = st.columns(3)

with col_a:
    epochs = st.slider(
        "训练轮数",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="轮数越多，学习越充分，但也更耗时，过高可能过拟合。"
    )

with col_b:
    batch_size = st.selectbox(
        "batch_size",
        [16, 32, 64, 128],
        index=2,
        help="每次送入模型训练的 SMILES 数量。越大训练越稳定，但占用内存更多。"
    )

with col_c:
    max_len = st.slider(
        "最大 SMILES 长度",
        min_value=40,
        max_value=200,
        value=120,
        step=10,
        help="训练时保留的 SMILES 最大字符长度，超过这个长度的分子会被截断或过滤。"
    )

col_d, col_e, col_f = st.columns(3)

with col_d:
    embedding_dim = st.selectbox(
        "embedding_dim",
        [32, 64, 128],
        index=1,
        help="每个 SMILES 字符被转换成向量后的维度。可以理解为字符表示的复杂度。"
    )

with col_e:
    hidden_dim = st.selectbox(
        "hidden_dim",
        [64, 128, 256],
        index=1,
        help="GRU 隐藏层维度，决定模型记忆和学习序列规律的能力。越大模型表达能力越强，但也更容易过拟合。"
    )

with col_f:
    num_layers = st.selectbox(
        "GRU 层数",
        [1, 2, 3],
        index=1,
        help="GRU 网络堆叠的层数。层数越多，模型可以学习更复杂的 SMILES 序列模式。"
    )

learning_rate = st.number_input(
    "learning_rate",
    min_value=0.0001,
    max_value=0.01,
    value=0.001,
    step=0.0001,
    format="%.4f",
    help="学习率，控制模型每次参数更新的步长。太大可能不稳定，太小训练会很慢。"
)

device = "cpu"

if os.path.exists(GENERATOR_MODEL_PATH):
    st.success(f"已检测到训练好的生成模型")
else:
    st.info("当前未发现已训练模型，首次使用请先完成模型训练。")

if st.button("开始训练生成模型", type="primary"):
    try:
        with st.spinner("正在读取 SMILES 并训练模型，可能需要几十秒到几分钟..."):
            smiles_list = load_training_smiles(
                data_path=data_path,
                smiles_col="smiles",
                label_col="label",
                active_only=active_only,
                max_len=max_len
            )

            model_path, loss_df, train_info = train_smiles_gru(
                smiles_list=smiles_list,
                model_path=GENERATOR_MODEL_PATH,
                max_len=max_len,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=0.1,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                random_state=42
            )

        st.success(f"模型训练完成，结果已保存")

        st.subheader("训练参数与统计信息")
        st.dataframe(
            pd.DataFrame([train_info]),
            use_container_width=True
        )

        st.subheader("训练损失")
        st.line_chart(
            loss_df.set_index("epoch")["loss"]
        )

        os.makedirs("results", exist_ok=True)

        loss_df.to_csv(
            TRAINING_LOSS_PATH,
            index=False,
            encoding="utf-8-sig"
        )

        st.success("训练损失已保存。")

    except Exception as e:
        st.error(f"训练失败：{e}")


# ============================== 四、生成候选分子 ==============================

st.header("四、候选分子生成")

st.markdown(
    """
    若有已训练完成的生成模型，可以直接点击“生成候选分子”。
    生成结果会保存为 `results/gru_generated_molecules.csv` 和 `results/generated_molecules.csv`。
    """
)

if not os.path.exists(GENERATOR_MODEL_PATH):
    st.warning("当前未发现已训练模型。请先训练生成模型。")
else:
    st.success("已检测到训练好的 SMILES-GRU 模型，可以进行分子生成。")

col_g, col_h, col_i = st.columns(3)

with col_g:
    n_molecules = st.slider(
        "目标生成分子数",
        min_value=5,
        max_value=200,
        value=50,
        step=5
    )

with col_h:
    temperature = st.slider(
        "采样温度 temperature",
        min_value=0.3,
        max_value=1.5,
        value=0.8,
        step=0.1,
        help="温度越低越保守，温度越高越随机。"
    )

with col_i:
    generation_max_len = st.slider(
        "生成最大长度",
        min_value=40,
        max_value=200,
        value=120,
        step=10
    )

filter_valid = st.checkbox(
    "只保留 RDKit 判定为有效的 SMILES",
    value=True
)

filter_novel = st.checkbox(
    "过滤训练集中已经存在的分子",
    value=True
)

if st.button("生成候选分子"):
    if not os.path.exists(GENERATOR_MODEL_PATH):
        st.error("当前未发现已训练模型，请先训练模型。")
        st.stop()

    try:
        with st.spinner("正在生成候选分子并进行有效性检查..."):
            generated_df, summary = generate_smiles_table(
                model_path=GENERATOR_MODEL_PATH,
                n_molecules=n_molecules,
                temperature=temperature,
                max_len=generation_max_len,
                max_attempts=max(n_molecules * 100, 1500),
                filter_valid=filter_valid,
                filter_novel=filter_novel,
                device=device
            )

        if generated_df.empty:
            st.error("未生成有效分子。可以尝试提高 temperature、增加生成数量或减少过滤条件。")
            st.stop()

        output_df = prepare_generation_output(generated_df)

        full_path, compatible_path = save_generated_molecules(
            output_df,
            full_output_path=FULL_OUTPUT_PATH,
            compatible_output_path=COMPATIBLE_OUTPUT_PATH
        )

        st.success("候选分子生成完成。")

        st.subheader("生成统计")
        st.dataframe(
            pd.DataFrame([summary]),
            use_container_width=True
        )

        st.subheader("本次生成分子结果")
        st.dataframe(
            output_df,
            use_container_width=True
        )

        st.download_button(
            key="download_current_gru_generated_molecules",
            label="下载本次生成结果 gru_generated_molecules.csv",
            data=output_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="gru_generated_molecules.csv",
            mime="text/csv"
        )

        compatible_df = output_df[["compound_id", "smiles"]].copy()

        st.download_button(
            key="download_current_generated_molecules",
            label="下载后续模块输入 generated_molecules.csv",
            data=compatible_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="generated_molecules.csv",
            mime="text/csv"
        )

        if rdkit_available():
            st.subheader("本次生成分子结构预览")

            smiles_col = get_smiles_column(output_df)
            show_df = output_df.head(12)
            cols = st.columns(3)

            for i, (_, row) in enumerate(show_df.iterrows()):
                with cols[i % 3]:
                    if smiles_col is not None:
                        img = get_mol_image(row[smiles_col])
                        if img is not None:
                            st.image(
                                img,
                                caption=f"{row['compound_id']} | LogP={row.get('LogP', 'NA')}",
                                width=250
                        )
                        else:
                            st.caption("该 SMILES 无法绘制结构图")
                            st.code(str(row[smiles_col]))

    except Exception as e:
        st.error(f"生成失败：{e}")

