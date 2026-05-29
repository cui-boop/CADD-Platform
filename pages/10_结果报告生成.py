import streamlit as st
import pandas as pd
from pathlib import Path

from utils.report import (
    generate_markdown_report,
    save_markdown_report,
    generate_html_report
)


st.set_page_config(
    page_title="结果报告生成",
    page_icon="📄",
    layout="wide"
)

st.title("📄 结果报告生成")

st.markdown("""
本模块用于根据候选分子的综合评分结果生成项目报告。  
报告内容包括项目基本信息、分析流程、综合评分结果、最优候选分子、排序表和结论建议。
""")

st.subheader("1. 输入项目基本信息")

project_name = st.text_input("项目名称", value="CADD 候选分子综合筛选项目")
target_name = st.text_input("靶点名称", value="EGFR")
description = st.text_area(
    "项目简介",
    value="本项目整合 QSAR 活性预测、ADMET 评价和分子对接结果，对候选分子进行综合评分和推荐。"
)

st.subheader("2. 选择综合评分结果文件")

mode = st.radio(
    "请选择 final_ranking.csv 来源",
    ["读取 results/final_ranking.csv", "手动上传 final_ranking.csv"]
)

final_df = None

if mode == "读取 results/final_ranking.csv":
    path = Path("results/final_ranking.csv")

    if path.exists():
        final_df = pd.read_csv(path)
        st.success(f"已读取：{path}")
        st.dataframe(final_df, use_container_width=True)
    else:
        st.error("没有找到 results/final_ranking.csv。请先在“药物重定位与综合评分”页面生成该文件。")

else:
    uploaded_file = st.file_uploader("上传 final_ranking.csv", type=["csv"])

    if uploaded_file is not None:
        final_df = pd.read_csv(uploaded_file)
        st.success("文件上传成功。")
        st.dataframe(final_df, use_container_width=True)
    else:
        st.warning("请上传 final_ranking.csv。")


if final_df is not None:
    st.subheader("3. 生成报告")

    if st.button("生成 Markdown / HTML 报告", type="primary"):
        try:
            report_text = generate_markdown_report(
                final_df=final_df,
                project_name=project_name,
                target_name=target_name,
                description=description
            )

            md_path = save_markdown_report(report_text)
            html_path = generate_html_report(report_text)

            st.success(f"Markdown 报告生成成功：{md_path}")
            st.success(f"HTML 报告生成成功：{html_path}")

            st.subheader("报告预览")
            st.markdown(report_text)

            st.download_button(
                label="📥 下载 Markdown 报告",
                data=report_text,
                file_name="cadd_report.md",
                mime="text/markdown"
            )

            with open(html_path, "r", encoding="utf-8") as f:
                html_data = f.read()

            st.download_button(
                label="📥 下载 HTML 报告",
                data=html_data,
                file_name="cadd_report.html",
                mime="text/html"
            )

        except Exception as e:
            st.error(f"报告生成失败：{e}")
