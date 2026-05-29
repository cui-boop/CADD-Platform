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
    本模块采用轻量级 **SMILES-GRU 字符级语言模型** 进行分子生成。
    模型会读取当前项目中的 `cleaned_activity.csv`，学习其中 SMILES 字符串的排列规律，
    然后逐字符采样生成新的候选分子。

    生成结果会经过 RDKit 有效性检查、去重和基础理化性质计算，
    并保存为后续“分子设计与结构优化”、QSAR 活性预测和 ADMET 评价可使用的候选分子文件。
    """
)


# ============================== 路径设置 ==============================

DEFAULT_DATA_PATH = "data/cleaned_activity.csv"
GENERATOR_MODEL_PATH = "models/smiles_gru_generator.pt"
FULL_OUTPUT_PATH = "results/gru_generated_molecules.csv"
COMPATIBLE_OUTPUT_PATH = "results/generated_molecules.csv"
TRAINING_LOSS_PATH = "results/smiles_gru_training_loss.csv"


# ============================== 一、环境检查 ==============================

st.header("一、环境检查")

col1, col2 = st.columns(2)

with col1:
    if torch_available():
        st.success("已检测到 PyTorch：可以训练和运行 SMILES-GRU 生成模型。")
    else:
        st.error("未检测到 PyTorch。请先安装 torch。")
        st.code("python -m pip install torch", language="powershell")

with col2:
    if rdkit_available():
        st.success("已检测到 RDKit：可以进行 SMILES 有效性检查和分子性质计算。")
    else:
        st.warning("未检测到 RDKit：可以训练字符串模型，但无法严格检查生成分子是否合法。")

if not torch_available():
    st.stop()


# ============================== 二、训练数据设置 ==============================

st.header("二、训练数据设置")

data_source = st.radio(
    "选择训练数据来源",
    ["使用 data/cleaned_activity.csv", "上传包含 smiles 列的 CSV 文件"],
    horizontal=True
)

data_path = DEFAULT_DATA_PATH

if data_source == "使用 data/cleaned_activity.csv":
    if not os.path.exists(DEFAULT_DATA_PATH):
        st.error("未找到 data/cleaned_activity.csv，请先完成活性数据整理。")
        st.stop()

    preview_df = pd.read_csv(DEFAULT_DATA_PATH)
    st.success("已读取 data/cleaned_activity.csv。")

else:
    uploaded_file = st.file_uploader(
        "上传训练用 CSV 文件",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info("请先上传包含 smiles 列的 CSV 文件。")
        st.stop()

    preview_df = pd.read_csv(uploaded_file)

    if "smiles" not in preview_df.columns:
        st.error("上传文件必须包含 smiles 列。")
        st.stop()

    os.makedirs("data", exist_ok=True)
    data_path = "data/uploaded_smiles_for_generation.csv"

    preview_df.to_csv(
        data_path,
        index=False,
        encoding="utf-8-sig"
    )

    st.success("上传数据已读取。")

st.subheader("训练数据预览")
st.dataframe(preview_df.head(), use_container_width=True)

if "label" in preview_df.columns:
    active_only = st.checkbox(
        "仅使用 Active 分子训练生成模型",
        value=True,
        help="推荐开启：模型会更倾向于学习活性分子的结构分布。"
    )
else:
    active_only = False
    st.info("当前数据没有 label 列，将使用全部 SMILES 训练。")

st.info(
    """
    注意：SMILES-GRU 的作用是学习训练集中 SMILES 的结构分布并生成新分子，
    生成结果仍需要继续进行 QSAR 活性预测、ADMET 评价和人工化学合理性检查。
    """
)


# ============================== 三、训练模型 ==============================

st.header("三、训练 SMILES-GRU 生成模型")

st.markdown(
    """
    训练任务：给定前面的 SMILES 字符，预测下一个字符。
    训练完成后，模型会保存到 `models/smiles_gru_generator.pt`。

    如果已经提前训练过模型，汇报时可以不再点击训练按钮，
    直接查看下方“使用已有缓存结果”部分。
    """
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    epochs = st.slider(
        "训练轮数 epochs",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )

with col_b:
    batch_size = st.selectbox(
        "batch_size",
        [16, 32, 64, 128],
        index=2
    )

with col_c:
    max_len = st.slider(
        "最大 SMILES 长度 max_len",
        min_value=40,
        max_value=200,
        value=120,
        step=10
    )

col_d, col_e, col_f = st.columns(3)

with col_d:
    embedding_dim = st.selectbox(
        "embedding_dim",
        [32, 64, 128],
        index=1
    )

with col_e:
    hidden_dim = st.selectbox(
        "hidden_dim",
        [64, 128, 256],
        index=1
    )

with col_f:
    num_layers = st.selectbox(
        "GRU 层数",
        [1, 2, 3],
        index=1
    )

learning_rate = st.number_input(
    "learning_rate",
    min_value=0.0001,
    max_value=0.01,
    value=0.001,
    step=0.0001,
    format="%.4f"
)

device = "cpu"

if os.path.exists(GENERATOR_MODEL_PATH):
    st.success(f"已检测到训练好的生成模型：{GENERATOR_MODEL_PATH}")
else:
    st.info("尚未检测到训练好的生成模型。第一次使用需要先训练一次。")

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

        st.success(f"生成模型训练完成，模型已保存到：{model_path}")

        st.subheader("训练信息")
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

        st.success("训练损失已保存到 results/smiles_gru_training_loss.csv。")

    except Exception as e:
        st.error(f"训练失败：{e}")


# ============================== 四、生成候选分子 ==============================

st.header("四、生成候选分子")

st.markdown(
    """
    如果已经存在训练好的模型，可以直接点击“生成候选分子”。
    生成结果会保存为 `results/gru_generated_molecules.csv` 和 `results/generated_molecules.csv`。
    """
)

if not os.path.exists(GENERATOR_MODEL_PATH):
    st.warning("尚未找到训练好的模型。请先训练生成模型，或确认 models/smiles_gru_generator.pt 是否存在。")
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
        st.error("尚未找到训练好的生成模型，请先训练模型。")
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
            st.error("没有生成有效分子。可以尝试提高 temperature、增加生成数量或减少过滤条件。")
            st.stop()

        full_path, compatible_path = save_generated_molecules(
            generated_df,
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
            generated_df,
            use_container_width=True
        )

        st.info(
            f"完整结果已保存到：{full_path}；"
            f"供后续分子设计模块上传使用的简化文件已保存到：{compatible_path}。"
        )

        st.download_button(
            key="download_current_gru_generated_molecules",
            label="下载本次生成结果 gru_generated_molecules.csv",
            data=generated_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="gru_generated_molecules.csv",
            mime="text/csv"
        )

        compatible_df = generated_df[["compound_id", "canonical_smiles"]].rename(
            columns={"canonical_smiles": "smiles"}
        )

        st.download_button(
            key="download_current_generated_molecules",
            label="下载后续模块输入 generated_molecules.csv",
            data=compatible_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="generated_molecules.csv",
            mime="text/csv"
        )

        if rdkit_available():
            st.subheader("本次生成分子结构预览")

            show_df = generated_df.head(12)
            cols = st.columns(3)

            for i, (_, row) in enumerate(show_df.iterrows()):
                with cols[i % 3]:
                    img = get_mol_image(row["canonical_smiles"])
                    if img is not None:
                        st.image(
                            img,
                            caption=f"{row['compound_id']} | LogP={row.get('LogP', 'NA')}"
                        )
                    else:
                        st.write(row["canonical_smiles"])

    except Exception as e:
        st.error(f"生成失败：{e}")


# ============================== 五、使用已有缓存结果 ==============================

st.header("五、使用已有缓存结果")

st.markdown(
    """
    如果已经提前完成过一次训练和分子生成，可以直接读取缓存结果。
    """
)

if os.path.exists(FULL_OUTPUT_PATH):
    st.success(f"已检测到缓存生成结果：{FULL_OUTPUT_PATH}")

    cached_df = pd.read_csv(FULL_OUTPUT_PATH)

    st.subheader("缓存生成分子结果")
    st.dataframe(cached_df, use_container_width=True)

    st.download_button(
        key="download_cached_gru_generated_molecules",
        label="下载缓存完整结果 gru_generated_molecules.csv",
        data=cached_df.to_csv(index=False, encoding="utf-8-sig"),
        file_name="gru_generated_molecules.csv",
        mime="text/csv"
    )

    if os.path.exists(COMPATIBLE_OUTPUT_PATH):
        simple_df = pd.read_csv(COMPATIBLE_OUTPUT_PATH)

        st.download_button(
            key="download_cached_generated_molecules",
            label="下载后续模块输入 generated_molecules.csv",
            data=simple_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="generated_molecules.csv",
            mime="text/csv"
        )

        st.info(
            "后续可以将 `results/generated_molecules.csv` 上传到“分子设计与结构优化”页面，"
            "也可以继续进行 QSAR 活性预测和 ADMET 评价。"
        )
    else:
        st.warning("未找到 results/generated_molecules.csv。可以重新点击“生成候选分子”生成简化文件。")

    if rdkit_available() and "canonical_smiles" in cached_df.columns:
        st.subheader("缓存分子结构预览")

        show_df = cached_df.head(12)
        cols = st.columns(3)

        for i, (_, row) in enumerate(show_df.iterrows()):
            with cols[i % 3]:
                img = get_mol_image(row["canonical_smiles"])
                if img is not None:
                    st.image(
                        img,
                        caption=f"{row['compound_id']} | LogP={row.get('LogP', 'NA')}"
                    )
                else:
                    st.write(row["canonical_smiles"])

else:
    st.info("目前还没有缓存生成结果。请先训练模型并生成一次候选分子。")


# ============================== 六、下一步建议 ==============================

st.header("六、下一步建议")

st.markdown(
    """
    生成分子不能直接视为候选药物。建议继续进行：

    1. 将 `results/generated_molecules.csv` 上传到“分子设计与结构优化”页面；
    2. 对生成分子进行 QSAR 活性预测；
    3. 对生成分子进行 ADMET 评价；
    4. 必要时再进行分子对接和人工化学合理性检查。
    """
)
