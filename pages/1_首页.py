import streamlit as st

st.set_page_config(
    page_title="首页",
    page_icon="💊",
    layout="wide"
)

st.title("💊 CADD 综合实践平台")

st.markdown(
    """
    本平台是一个面向计算机辅助药物设计课程实践的综合网站。
    平台支持用户上传 ChEMBL 活性数据，并完成数据整理、QSAR 建模、
    ADMET 属性预测、模型解释、分子对接结果分析和候选分子综合评分。
    """
)

st.divider()

st.header("一、平台定位")

st.markdown(
    """
    本网站是一个通用型 CADD 数据分析平台。
    
    用户可以上传符合要求的 ChEMBL 单靶点活性数据，平台会自动完成数据清洗、
    pActivity 计算、活性标签划分，并为后续 QSAR 模型训练提供标准数据。
    
    本项目目前以内置的 EGFR 活性数据作为示例案例。
    """
)

st.header("二、平台主流程")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. 数据输入")
    st.markdown(
        """
        - 项目管理  
        - 上传 ChEMBL 活性数据  
        - 读取内置 EGFR 示例数据  
        - 检查数据格式  
        """
    )

with col2:
    st.subheader("2. 模型分析")
    st.markdown(
        """
        - 活性数据整理  
        - QSAR 模型训练  
        - 活性预测  
        - ADMET 属性计算  
        - 可解释性分析  
        """
    )

with col3:
    st.subheader("3. 结果输出")
    st.markdown(
        """
        - 分子对接结果分析  
        - 候选分子综合评分  
        - 药物重定位辅助分析  
        - 结果报告生成  
        """
    )

st.header("三、网站功能模块")

st.markdown(
    """
    1. **项目管理**：创建和查看 CADD 分析项目。  
    2. **活性数据整理**：清洗 ChEMBL 活性数据，计算 pActivity，并划分 Active / Inactive 标签。  
    3. **QSAR 模型与活性预测**：基于分子结构描述符训练机器学习模型，预测候选分子活性。  
    4. **分子属性与 ADMET 预测**：计算分子量、LogP、TPSA、氢键供体、氢键受体等性质。  
    5. **可解释性分析**：展示模型特征重要性，辅助理解结构-活性关系。  
    6. **分子对接结果分析**：展示候选分子与靶点蛋白的 docking score。  
    7. **药物重定位与综合评分**：整合 QSAR、ADMET 和 docking 结果，对候选分子排序。  
    8. **资源与知识库**：整理 ChEMBL、PubChem、PDB、RDKit、AutoDock Vina 等资源。  
    9. **结果报告生成**：输出项目数据整理、模型训练和候选分子筛选结果。  
    """
)

st.header("四、内置示例数据")

st.markdown(
    """
    本项目内置了从 ChEMBL 数据库下载的人源 EGFR 靶点活性数据作为示例。
    """
)

st.info(
    """
    示例数据筛选条件：
    
    Target：EGFR / CHEMBL203  
    Organism：Homo sapiens  
    Target Type：Single Protein  
    Activity Type：IC50  
    Standard Units：nM  
    Standard Relation：=  
    """
)

st.markdown(
    """
    该数据可用于构建 EGFR 抑制剂活性预测模型。平台也支持用户上传其他 ChEMBL 单靶点活性数据，
    例如 BACE1、ACHE、COX2、JAK2、VEGFR2 等靶点。
    """
)

st.header("五、推荐使用流程")

st.success(
    """
    推荐流程：
    项目管理 → 活性数据整理 → QSAR 模型与活性预测 → ADMET 预测 → 可解释性分析 → 
    分子对接结果分析 → 综合评分 → 结果报告生成
    """
)