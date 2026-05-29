import streamlit as st
import pandas as pd

from utils.docking import (
    load_docking_csv,
    save_docking_results,
    make_smart_docking_template,
    analyze_docking_results,
    summarize_multi_target_results,
    plot_docking_scores,
    plot_multi_target_heatmap,
    plot_confidence_by_target
)


st.set_page_config(
    page_title="分子对接结果分析",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 分子对接结果分析")

st.markdown("""
本模块用于整理和分析分子对接结果，帮助快速查看候选分子的结合表现、多靶点趋势以及潜在优化方向。

除了基础 docking score 外，当前页面还会结合氢键、疏水作用、π-π 相互作用、盐桥及关键残基等信息， 
进一步生成 docking confidence，并给出对应的结合模式解释与结构优化建议。

适用于： 分子筛选、多靶点比较、药物重定位分析、对接结果整理、结合模式解释。
""")

st.subheader("1. 输入数据格式")

st.markdown("""
最低需要包含以下字段：

```text
compound_id,target,docking_score
```

推荐使用完整字段：

```text
compound_id,smiles,target,docking_score,hbond_count,hydrophobic_count,pi_pi_count,salt_bridge_count,key_residues,interaction
```

字段说明：

- `compound_id`：化合物编号，必须和 QSAR、ADMET 结果中的编号一致；
- `smiles`：分子 SMILES，可选但推荐保留；
- `target`：靶点名称，例如 EGFR、HER2、BACE1；
- `docking_score`：分子对接结合能，通常单位为 kcal/mol，数值越低越好；
- `hbond_count`：氢键数量，可选；
- `hydrophobic_count`：疏水相互作用数量，可选；
- `pi_pi_count`：π-π 堆积数量，可选；
- `salt_bridge_count`：盐桥数量，可选；
- `key_residues`：关键结合残基，可选；
- `interaction`：人工记录的相互作用描述，可选。
""")

template_df = make_smart_docking_template()

with st.expander("查看 docking_results.csv 示例模板"):
    st.dataframe(template_df, use_container_width=True)

template_csv = template_df.to_csv(index=False, encoding="utf-8-sig")
st.download_button(
    label="下载 docking_results.csv 模板",
    data=template_csv,
    file_name="smart_docking_results_template.csv",
    mime="text/csv"
)

st.subheader("2. 数据来源")

mode = st.radio(
    "请选择数据来源",
    ["上传 docking_results.csv", "使用演示数据"],
    horizontal=True
)

docking_df = None

if mode == "上传 docking_results.csv":
    uploaded_file = st.file_uploader("上传 docking_results.csv", type=["csv"])

    if uploaded_file is not None:
        try:
            docking_df = load_docking_csv(uploaded_file)
            docking_df = analyze_docking_results(docking_df)
            save_path = save_docking_results(docking_df, output_path="results/docking_results.csv")
            st.success(f"数据处理完成，结果已保存到：{save_path}")
        except Exception as e:
            st.error(f"文件处理失败：{e}")
    else:
        st.warning("上传 docking_results.csv，或选择“使用演示数据”后即可开始分析。")

else:
    docking_df = make_smart_docking_template()
    docking_df = analyze_docking_results(docking_df)
    save_path = save_docking_results(docking_df, output_path="results/docking_results.csv")
    st.success(f"演示数据加载完成，演示结果已保存到：{save_path}")

if docking_df is not None and not docking_df.empty:

    tab1, tab2, tab3, tab4 = st.tabs([
        "对接结果总览",
        "智能解释与优化建议",
        "多靶点比较",
        "导出结果"
    ])

    with tab1:
        st.subheader("3. 标准化后的 docking 结果")
        st.dataframe(docking_df, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("候选分子数量", docking_df["compound_id"].nunique())

        with col2:
            st.metric("靶点数量", docking_df["target"].nunique())

        with col3:
            best_row = docking_df.sort_values("docking_score", ascending=True).iloc[0]
            st.metric("最佳 docking score", f"{best_row['docking_score']:.2f}")

        with col4:
            best_conf = docking_df.sort_values("docking_confidence", ascending=False).iloc[0]
            st.metric("最高 docking confidence", f"{best_conf['docking_confidence']:.3f}")

        st.subheader("4. Docking score 排名")
        targets = ["全部靶点"] + sorted(docking_df["target"].dropna().astype(str).unique().tolist())
        selected_target = st.selectbox("选择展示靶点", targets)
        plot_target = None if selected_target == "全部靶点" else selected_target

        try:
            fig, fig_path = plot_docking_scores(
                docking_df,
                selected_target=plot_target,
                output_path="results/docking_score_plot.png"
            )
            st.pyplot(fig)
            st.caption(f"图像已保存至：{fig_path}")
        except Exception as e:
            st.error(f"绘图失败：{e}")

        st.info("""
        Docking score 越低通常表示结合能越有利，但单独依赖 docking score 往往不足以完整反映结合质量。
        因此本模块进一步引入结合氢键、疏水作用及关键残基等信息，对整体结合可信度进行综合评估。
        """)

    with tab2:
        st.subheader("5. Docking智能解释")

        view_cols = [
            "compound_id",
            "target",
            "docking_score",
            "hbond_count",
            "hydrophobic_count",
            "pi_pi_count",
            "salt_bridge_count",
            "key_residues",
            "docking_score_norm",
            "interaction_score",
            "docking_confidence",
            "binding_interpretation",
            "optimization_suggestion"
        ]

        available_cols = [col for col in view_cols if col in docking_df.columns]

        st.dataframe(
            docking_df[available_cols].sort_values(
                ["docking_confidence", "docking_score"],
                ascending=[False, True]
            ),
            use_container_width=True
        )

        st.subheader("6. 单分子解释详情")

        selected_compound = st.selectbox(
            "选择 compound_id 查看解释",
            sorted(docking_df["compound_id"].dropna().astype(str).unique().tolist())
        )

        selected_rows = docking_df[docking_df["compound_id"].astype(str) == selected_compound].copy()
        selected_rows = selected_rows.sort_values("docking_confidence", ascending=False)

        for _, row in selected_rows.iterrows():
            st.markdown(f"### {row['compound_id']} - {row['target']}")
            st.write(f"**Docking score：** {row['docking_score']}")
            st.write(f"**Docking confidence：** {row['docking_confidence']}")
            st.write(f"**关键残基：** {row.get('key_residues', '未提供')}")
            st.write(f"**结合模式解释：** {row['binding_interpretation']}")
            st.write(f"**结构优化建议：** {row['optimization_suggestion']}")

    with tab3:
        st.subheader("7. 同一候选分子对多个靶点的 docking score 比较")

        multi_df = summarize_multi_target_results(docking_df)
        st.dataframe(multi_df, use_container_width=True)

        st.markdown("""
        多靶点比较可以帮助判断：

        - 某个分子是否对多个靶点都保持良好的结合趋势；
        - 是否对特定靶点具有更强结合偏好；
        - 是否具备潜在的多靶点抑制或药物重定位价值。
        """)

        st.subheader("8. 多靶点 docking score 热图")
        try:
            heatmap_fig, heatmap_path = plot_multi_target_heatmap(
                docking_df,
                output_path="results/multi_target_docking_heatmap.png"
            )
            st.pyplot(heatmap_fig)
            st.caption(f"图像已保存至：{heatmap_path}")
        except Exception as e:
            st.error(f"热图绘制失败：{e}")

        st.subheader("9. 不同靶点 docking confidence 比较")
        try:
            conf_fig, conf_path = plot_confidence_by_target(
                docking_df,
                output_path="results/docking_confidence_by_target.png"
            )
            st.pyplot(conf_fig)
            st.caption(f"图像已保存至：{conf_path}")
        except Exception as e:
            st.error(f"confidence 图绘制失败：{e}")

        st.subheader("10. 药物重定位解释")

        if not multi_df.empty:
            for _, row in multi_df.iterrows():
                st.markdown(
                    f"- **{row['compound_id']}**：最佳结合靶点为 **{row['best_target']}**，"
                    f"最佳 docking score 为 **{row['best_docking_score']}**。"
                    f"该分子共覆盖 **{row['target_count']}** 个靶点，"
                    f"平均 docking confidence 为 **{row['mean_docking_confidence']}**。"
                    f"{row['multi_target_interpretation']}"
                )

    with tab4:
        st.subheader("11. 保存与下载")

        save_path = save_docking_results(docking_df, output_path="results/docking_results.csv")
        st.success(f"标准化与智能解释后的 docking 结果已保存到：{save_path}")

        output_csv = docking_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 下载智能解释后的 docking_results.csv",
            data=output_csv,
            file_name="smart_docking_results.csv",
            mime="text/csv"
        )

        multi_df = summarize_multi_target_results(docking_df)
        multi_csv = multi_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 下载多靶点比较结果 multi_target_summary.csv",
            data=multi_csv,
            file_name="multi_target_summary.csv",
            mime="text/csv"
        )
