import sys, json
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import pandas as pd
import streamlit as st
from utils.knowledge_search_extraction import search_and_fetch, rule_extract, call_openai_extractor, tables, markdown, save_outputs, save_search, demo_text

st.set_page_config(page_title='文献检索与 LLM 知识提取', page_icon='📚', layout='wide')
st.title('📚 文献检索与 LLM 知识提取')
st.markdown('''本模块支持两种使用方式：先用关键词从 PubMed 检索文献并尝试获取 PMC 全文，再对选中文献进行知识提取；也可以直接粘贴摘要或全文片段进行解析。规则提取模式不需要 API Key，LLM 模式需要 OpenAI API Key。''')

tab_search, tab_extract = st.tabs(['关键词检索文献', '知识提取与解析'])

with tab_search:
    st.subheader('1. 关键词检索 PubMed / PMC')
    c1,c2,c3=st.columns(3)
    with c1: k1=st.text_input('关键词 1', value='EGFR inhibitor')
    with c2: k2=st.text_input('关键词 2', value='docking')
    with c3: k3=st.text_input('关键词 3', value='ADMET')
    c4,c5,c6=st.columns(3)
    with c4: field=st.selectbox('检索字段', ['Title/Abstract','All Fields','MeSH Terms'])
    with c5: retmax=st.slider('返回文献数量',1,20,5)
    with c6: sort=st.selectbox('排序方式',['relevance','pub date'])
    fulltext=st.checkbox('尝试获取 PMC 全文', value=True, help='只有关联到 PMC 且开放可获取的文献才能获取全文。')
    with st.expander('高级设置'):
        ncbi_key=st.text_input('NCBI API Key，可选', type='password')
        email=st.text_input('邮箱，可选，用于 NCBI 请求标识', value='')
    if st.button('检索文献', type='primary'):
        try:
            with st.spinner('正在检索 PubMed 并获取文献信息...'):
                res=search_and_fetch(k1,k2,k3,field,retmax,sort,fulltext,ncbi_key,email)
            st.session_state['lit_res']=res
            paths=save_search(res)
            st.success(f"检索完成。查询式：{res['query']}")
            st.success(f"结果已保存：{paths['csv']}")
        except Exception as e:
            st.error(f'文献检索失败：{e}')
    if 'lit_res' in st.session_state:
        arts=st.session_state['lit_res'].get('articles',[])
        if not arts:
            st.warning('没有检索到文献。')
        else:
            df=pd.DataFrame([{'pmid':a.get('pmid',''),'pmc_id':a.get('pmc_id',''),'year':a.get('year',''),'title':a.get('title',''),'journal':a.get('journal',''),'abstract_length':len(a.get('abstract','')),'full_text_length':len(a.get('full_text',''))} for a in arts])
            st.dataframe(df,use_container_width=True)
            labels=[f"{a.get('pmid','')} | {a.get('title','')[:90]}" for a in arts]
            lab=st.selectbox('选择一篇文献用于预览或知识提取', labels)
            a=arts[labels.index(lab)]
            st.session_state['selected_article']=a
            st.write(f"**PMID：** {a.get('pmid','')}  **PMCID：** {a.get('pmc_id','') or '未获取到'}")
            st.write(f"**标题：** {a.get('title','')}")
            st.info(a.get('abstract','') or '未获取到摘要。')
            if a.get('full_text',''):
                st.text_area('PMC 全文片段', value=a['full_text'][:6000], height=260)
            else:
                st.warning('未获取到 PMC 全文，后续可使用标题和摘要提取。')
            st.download_button('📥 下载检索结果 CSV', data=pd.DataFrame(arts).to_csv(index=False,encoding='utf-8-sig'), file_name='literature_search_articles.csv', mime='text/csv')

with tab_extract:
    st.subheader('2. 选择知识提取文本来源')
    src=st.radio('文本来源',['使用检索选中的文献','使用演示文本','手动粘贴文本','上传 TXT 文件'],horizontal=True)
    title=''; text=''
    if src=='使用检索选中的文献':
        a=st.session_state.get('selected_article')
        if a is None: st.warning('请先检索并选择一篇文献。')
        else:
            title=a.get('title',''); text=a.get('full_text','') or a.get('abstract','')
            st.text_input('文献标题', value=title, disabled=True)
            st.text_area('用于知识提取的文本', value=text[:12000], height=260)
    elif src=='使用演示文本':
        d=demo_text(); title=st.text_input('文献标题', value=d['title']); text=st.text_area('文献摘要 / 正文片段', value=d['text'], height=260)
    elif src=='手动粘贴文本':
        title=st.text_input('文献标题', value=''); text=st.text_area('文献摘要 / 正文片段', value='', height=300)
    else:
        title=st.text_input('文献标题', value=''); f=st.file_uploader('上传 TXT 文件', type=['txt'])
        if f is not None: text=st.text_area('文件内容预览与编辑', value=f.read().decode('utf-8',errors='ignore'), height=300)
    st.subheader('3. 选择提取模式')
    mode=st.radio('提取模式',['规则提取模式','LLM 智能提取模式'],horizontal=True)
    api_key=''; model='gpt-4.1-mini'
    if mode=='LLM 智能提取模式':
        st.info('LLM 模式需要 OpenAI API Key。不要把 Key 写入代码或提交到 GitHub。')
        api_key=st.text_input('请输入 OpenAI API Key', type='password')
        model=st.text_input('模型名称', value='gpt-4.1-mini')
    else:
        st.info('规则提取模式无需 API Key，适合课堂展示和离线测试。')
    if st.button('开始知识提取', type='primary'):
        if not text.strip(): st.error('请先输入、上传或选择文献内容。')
        else:
            try:
                with st.spinner('正在提取知识...'):
                    if mode=='LLM 智能提取模式':
                        if not api_key.strip(): st.stop()
                        result=call_openai_extractor(text,title,api_key,model)
                    else:
                        result=rule_extract(text,title)
                st.session_state['knowledge_result']=result
                paths=save_outputs(result)
                st.success(f"知识提取完成，已保存：{paths['json']}")
            except Exception as e:
                st.error(f'知识提取失败：{e}')
    if 'knowledge_result' in st.session_state:
        result=st.session_state['knowledge_result']; tb=tables(result); md=markdown(result)
        t1,t2,t3,t4=st.tabs(['结构化总览','活性 / ADMET / Docking','分子设计启发','导出结果'])
        with t1:
            st.dataframe(tb['overview'], use_container_width=True)
            c1,c2,c3,c4=st.columns(4)
            c1.metric('靶点数量', len(result.get('targets',[]))); c2.metric('化合物数量',len(result.get('compounds',[]))); c3.metric('活性数据条数',len(result.get('activity_data',[]))); c4.metric('关键残基数量',len(result.get('key_residues',[])))
            st.write(result.get('summary',''))
        with t2:
            st.subheader('活性数据'); st.dataframe(tb['activity'], use_container_width=True) if not tb['activity'].empty else st.info('未识别到明确活性数据。')
            st.subheader('ADMET 信息'); st.dataframe(tb['admet'], use_container_width=True) if not tb['admet'].empty else st.info('未识别到明确 ADMET 信息。')
            st.subheader('Docking 信息'); st.dataframe(tb['docking'], use_container_width=True) if not tb['docking'].empty else st.info('未识别到明确 docking 信息。')
        with t3:
            st.subheader('机制信息'); st.dataframe(tb['mechanism'], use_container_width=True) if not tb['mechanism'].empty else st.info('未识别到明确机制信息。')
            st.subheader('分子设计启发'); st.dataframe(tb['design'], use_container_width=True) if not tb['design'].empty else st.info('未生成分子设计建议。')
        with t4:
            st.download_button('📥 下载 JSON 结果', data=json.dumps(result,ensure_ascii=False,indent=2), file_name='knowledge_extraction_result.json', mime='application/json')
            st.download_button('📥 下载 Markdown 摘要', data=md, file_name='knowledge_summary.md', mime='text/markdown')
            st.download_button('📥 下载 CSV 总览', data=tb['overview'].to_csv(index=False,encoding='utf-8-sig'), file_name='knowledge_overview.csv', mime='text/csv')
