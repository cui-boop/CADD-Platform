import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from utils.knowledge_search_extraction import (
    make_demo_text,
    search_and_fetch_literature,
    rule_based_extract,
    call_openai_extractor,
    result_to_tables,
    generate_markdown_summary,
    save_extraction_outputs,
    save_literature_outputs
)


st.set_page_config(
    page_title="文献检索与 LLM 知识提取",
    page_icon="📚",
    layout="wide"
)

st.title("📚 文献检索与 LLM 知识提取")

st.markdown("""
本模块用于从关键词检索 PubMed 文献，并从标题、摘要或 PMC 全文中提取 CADD 相关知识。
""")

tab_search, tab_extract = st.tabs(["关键词检索文献", "知识提取与解析"])

with tab_search:
    st.subheader("1. 输入关键词检索 PubMed / PMC")

    col1, col2, col3 = st.columns(3)

    with col1:
        keyword_1 = st.text_input("关键词 1", value="EGFR inhibitor")

    with col2:
        keyword_2 = st.text_input("关键词 2", value="docking")

    with col3:
        keyword_3 = st.text_input("关键词 3", value="ADMET")

    col4, col5, col6 = st.columns(3)

    with col4:
        search_field = st.selectbox(
            "检索字段",
            ["Title/Abstract", "All Fields", "MeSH Terms"]
        )

    with col5:
        retmax = st.slider("返回文献数量", min_value=1, max_value=20, value=5)

    with col6:
        sort = st.selectbox("排序方式", ["relevance", "pub date"])

    fetch_full_text = st.checkbox(
        "尝试获取 PMC 全文",
        value=True,
        help="只有与 PubMed 记录关联且可由 PMC 获取的开放全文才能成功获取。没有全文时仍会保留摘要。"
    )

    with st.expander("高级设置"):
        ncbi_api_key = st.text_input("NCBI API Key，可选", type="password")
        email = st.text_input("邮箱，可选，用于 NCBI 请求标识", value="")

    if st.button("检索文献", type="primary"):
        try:
            with st.spinner("正在检索 PubMed 并获取文献信息..."):
                search_result = search_and_fetch_literature(
                    keyword_1=keyword_1,
                    keyword_2=keyword_2,
                    keyword_3=keyword_3,
                    search_field=search_field,
                    retmax=retmax,
                    sort=sort,
                    fetch_full_text=fetch_full_text,
                    api_key=ncbi_api_key,
                    email=email
                )

            st.session_state["literature_search_result"] = search_result
            save_paths = save_literature_outputs(search_result)


        except Exception as e:
            st.error(f"文献检索失败：{e}")

    if "literature_search_result" in st.session_state:
        search_result = st.session_state["literature_search_result"]
        articles = search_result.get("articles", [])

        st.subheader("2. 检索结果")

        if not articles:
            st.warning("没有检索到文献。请更换关键词。")
        else:
            display_df = pd.DataFrame([
                {
                    "PMID": a.get("pmid", ""),
                    "PMC_ID": a.get("pmc_id", ""),
                    "年份": a.get("year", ""),
                    "标题": a.get("title", ""),
                    "期刊": a.get("journal", ""),
                    "摘要长度": len(a.get("abstract", "")),
                    "全文长度": len(a.get("full_text", ""))
                }
                for a in articles
            ])

            st.dataframe(display_df, use_container_width=True)

            article_labels = [
                f"{a.get('pmid', '')} | {a.get('title', '')[:90]}"
                for a in articles
            ]

            selected_label = st.selectbox("选择一篇文献用于预览或知识提取", article_labels)
            selected_index = article_labels.index(selected_label)
            selected_article = articles[selected_index]

            st.session_state["selected_article"] = selected_article

            st.markdown("### 文献预览")
            st.write(f"**PMID：** {selected_article.get('pmid', '')}")
            st.write(f"**PMCID：** {selected_article.get('pmc_id', '') or '未获取到'}")
            st.write(f"**标题：** {selected_article.get('title', '')}")
            st.write(f"**期刊：** {selected_article.get('journal', '')}")
            st.write(f"**年份：** {selected_article.get('year', '')}")
            st.write(f"**DOI：** {selected_article.get('doi', '')}")

            st.markdown("#### 摘要")
            st.info(selected_article.get("abstract", "") or "未获取到摘要。")

            full_text = selected_article.get("full_text", "")
            if full_text:
                st.markdown("#### PMC 全文片段")
                st.text_area("全文预览", value=full_text[:6000], height=260)
            else:
                st.warning("这篇文献没有获取到 PMC 全文。后续知识提取将使用标题和摘要。")

            csv_data = pd.DataFrame(articles).to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "📥 下载检索结果 CSV",
                data=csv_data,
                file_name="literature_search_articles.csv",
                mime="text/csv"
            )

with tab_extract:
    st.subheader("3. 选择知识提取文本来源")

    source = st.radio(
        "文本来源",
        ["使用检索选中的文献", "使用演示文本", "手动粘贴文本", "上传 TXT 文件"],
        horizontal=True
    )

    title = ""
    text = ""

    if source == "使用检索选中的文献":
        selected_article = st.session_state.get("selected_article")
        if selected_article is None:
            st.warning("请先在“关键词检索文献”页检索并选择一篇文献。")
        else:
            title = selected_article.get("title", "")
            abstract = selected_article.get("abstract", "")
            full_text = selected_article.get("full_text", "")
            text = full_text if full_text else abstract

            st.text_input("文献标题", value=title, disabled=True)
            st.text_area("用于知识提取的文本", value=text[:12000], height=260)

    elif source == "使用演示文本":
        demo = make_demo_text()
        title = st.text_input("文献标题", value=demo["title"])
        text = st.text_area("文献摘要 / 正文片段", value=demo["text"], height=260)

    elif source == "手动粘贴文本":
        title = st.text_input("文献标题", value="")
        text = st.text_area("文献摘要 / 正文片段", value="", height=300)

    else:
        title = st.text_input("文献标题", value="")
        uploaded_file = st.file_uploader("上传 TXT 文件", type=["txt"])
        if uploaded_file is not None:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            text = st.text_area("文件内容预览与编辑", value=text, height=300)
        else:
            st.warning("请上传 TXT 文件")

    st.subheader("4. 选择提取模式")

    extract_mode = st.radio(
        "请选择知识提取模式",
        ["规则提取模式", "LLM 智能提取模式"],
        horizontal=True
    )

    api_key = ""
    model = "gpt-4.1-mini"

    if extract_mode == "LLM 智能提取模式":
        st.info("LLM 模式需要 OpenAI API Key。")
        api_key = st.text_input("请输入 OpenAI API Key", type="password")
        model = st.text_input("模型名称", value="gpt-4.1-mini")
    else:
        st.info("规则提取模式无需 API Key")

    if st.button("开始知识提取", type="primary"):
        if not text.strip():
            st.error("请先输入、上传或选择文献内容。")
        else:
            try:
                with st.spinner("正在提取知识..."):
                    if extract_mode == "LLM 智能提取模式":
                        if not api_key.strip():
                            st.error("LLM 模式需要输入 OpenAI API Key")
                            st.stop()

                        result = call_openai_extractor(
                            text=text,
                            title=title,
                            api_key=api_key,
                            model=model
                        )
                    else:
                        result = rule_based_extract(text=text, title=title)

                st.session_state["knowledge_result"] = result
                output_paths = save_extraction_outputs(result)

                st.success("知识提取完成")

            except Exception as e:
                st.error(f"知识提取失败：{e}")

    if "knowledge_result" in st.session_state:
        result = st.session_state["knowledge_result"]
        tables = result_to_tables(result)
        md_text = generate_markdown_summary(result)

        tab1, tab2, tab3, tab4 = st.tabs([
            "结构化总览",
            "活性 / ADMET / Docking",
            "机制信息",
            "导出结果"
        ])

        with tab1:
            st.subheader("结构化总览")

            overview_df = tables["overview"].copy()
            if "字段" in overview_df.columns:
                overview_df = overview_df[
                    ~overview_df["字段"].isin(["提取模式", "总结"])
                ].reset_index(drop=True)

            st.dataframe(overview_df, use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("靶点数量", len(result.get("targets", [])))
            with col2:
                st.metric("化合物数量", len(result.get("compounds", [])))
            with col3:
                st.metric("活性数据条数", len(result.get("activity_data", [])))
            with col4:
                st.metric("关键残基数量", len(result.get("key_residues", [])))

        with tab2:
            st.subheader("活性数据")
            if tables["activity"].empty:
                _ = st.info("未识别到明确活性数据。")
            else:
                st.dataframe(tables["activity"], use_container_width=True)

            st.subheader("ADMET 信息")
            if tables["admet"].empty:
                _ = st.info("未识别到明确 ADMET 信息。")
            else:
                st.dataframe(tables["admet"], use_container_width=True)

            st.subheader("关键残基信息")
            if tables["docking"].empty:
                _ = st.info("未识别到明确 docking 信息。")
            else:
                st.dataframe(tables["docking"], use_container_width=True)

            residues = result.get("key_residues", [])
        

        with tab3:
            st.subheader("机制信息")
            if tables["mechanism"].empty:
                _ = st.info("未识别到明确机制信息。")
            else:
                st.dataframe(tables["mechanism"], use_container_width=True)

        with tab4:
            st.subheader("导出结果")

            st.download_button(
                "📥 下载 Markdown 摘要",
                data=md_text,
                file_name="knowledge_summary.md",
                mime="text/markdown"
            )

            export_overview_df = tables["overview"].copy()
            if "字段" in export_overview_df.columns:
                export_overview_df = export_overview_df[
                    ~export_overview_df["字段"].isin(["提取模式", "总结"])
                ].reset_index(drop=True)

            csv_data = export_overview_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "📥 下载 CSV 总览",
                data=csv_data,
                file_name="knowledge_overview.csv",
                mime="text/csv"
            )
