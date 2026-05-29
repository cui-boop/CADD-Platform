import streamlit as st

st.set_page_config(
    page_title="CADD 综合实践平台",
    page_icon="💊",
    layout="wide"
)

st.title("💊 CADD 综合实践平台")

st.markdown("""
本平台是一个面向计算机辅助药物设计课程实践的综合网站，
支持活性数据整理、QSAR 建模、ADMET 预测、分子对接结果分析、
候选分子综合评分和结果报告生成。
""")

st.subheader("平台主流程")

st.markdown("""
1. 项目管理  
2. 活性数据整理  
3. QSAR 模型与活性预测  
4. 分子属性与 ADMET 预测  
5. 可解释性分析  
6. 分子对接结果分析  
7. 药物重定位与综合评分  
8. 结果报告生成  
""")

st.info("请从左侧页面菜单选择功能模块。")