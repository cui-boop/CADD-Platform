import streamlit as st
import pandas as pd

from utils.docking import (
    load_docking_csv,
    save_docking_results,
    make_docking_template,
    plot_docking_scores
)


st.set_page_config(
    page_title="分子对接结果分析",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 分子对接结果分析")

st.markdown("""
本模块用于分析分子对接结果。用户上传 `docking_results.csv` 后，平台会自动读取 docking score，
按照结合能从低到高进行排序，并绘制 docking score 图。

一般来说，**docking score 越低，表示配体与靶点蛋白的结合可能越稳定**。

本模块不在网页端实时运行 AutoDock Vina，而是对已经得到的分子对接结果进行整理、排序和可视化分析。
""")

st.subheader("1. docking_results.csv 文件格式要求")

st.markdown("""
上传的 `docking_results.csv` 至少需要包含以下两列：

```text
compound_id,docking_score
```

推荐完整格式为：

```text
compound_id,smiles,target,docking_score,pose_file,interaction
```

其中：

- `compound_id`：化合物编号，必须和 QSAR、ADMET 结果中的编号一致；
- `smiles`：分子 SMILES，可选但推荐保留；
- `target`：靶点名称；
- `docking_score`：分子对接结合能，通常单位为 kcal/mol，数值越低越好；
- `pose_file`：对接构象图片或文件名，可选；
- `interaction`：关键相互作用，例如氢键、疏水作用、π-π 作用等，可选。
""")

template_df = make_docking_template()

with st.expander("查看 docking_results.csv 模板"):
    st.dataframe(template_df, use_container_width=True)

template_csv = template_df.to_csv(index=False, encoding="utf-8-sig")

st.download_button(
    label="📥 下载 docking_results.csv 模板",
    data=template_csv,
    file_name="docking_results_template.csv",
    mime="text/csv"
)

st.subheader("2. 上传分子对接结果文件")

uploaded_file = st.file_uploader(
    "请上传 docking_results.csv",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        docking_df = load_docking_csv(uploaded_file)

        save_path = save_docking_results(
            docking_df,
            output_path="results/docking_results.csv"
        )

        st.success(f"对接结果读取成功，已保存到：{save_path}")

        st.subheader("3. 分子对接结合能排名")

        st.dataframe(
            docking_df,
            use_container_width=True
        )

        if not docking_df.empty:
            best = docking_df.iloc[0]

            st.info(
                f"当前 docking score 最优的分子是 **{best['compound_id']}**，"
                f"docking score = **{best['docking_score']} kcal/mol**。"
            )

        st.subheader("4. Docking score 可视化")

        fig, fig_path = plot_docking_scores(
            docking_df,
            output_path="results/docking_score_plot.png"
        )

        st.pyplot(fig)

        st.caption(f"图像已保存到：{fig_path}")

        st.subheader("5. 下载标准化后的 docking 结果")

        output_csv = docking_df.to_csv(index=False, encoding="utf-8-sig")

        st.download_button(
            label="📥 下载 docking_results_standardized.csv",
            data=output_csv,
            file_name="docking_results_standardized.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"文件处理失败：{e}")

else:
    st.warning("请先上传 docking_results.csv 文件。")

    st.markdown("""
    如果你现在只是测试页面功能，可以点击下面按钮生成一份演示 docking 数据。
    """)

    if st.button("使用演示数据生成 docking_results.csv"):
        try:
            demo_df = make_docking_template()

            save_path = save_docking_results(
                demo_df,
                output_path="results/docking_results.csv"
            )

            st.success(f"演示 docking 结果已保存到：{save_path}")

            st.dataframe(demo_df, use_container_width=True)

            fig, fig_path = plot_docking_scores(
                demo_df,
                output_path="results/docking_score_plot.png"
            )

            st.pyplot(fig)

            st.caption(f"图像已保存到：{fig_path}")

        except Exception as e:
            st.error(f"演示数据生成失败：{e}")

st.subheader("模块说明")

st.markdown("""
本页面的主要作用是完成分子对接结果的整理和展示。

正式项目中，`docking_results.csv` 可以来自 AutoDock Vina、PyRx、CB-Dock2、SwissDock 等工具的预计算结果。

后续“药物重定位与综合评分”页面会读取：

```text
results/docking_results.csv
```

并将其与 QSAR 预测结果和 ADMET 结果进行整合，得到最终候选分子的综合评分和推荐等级。
""")
