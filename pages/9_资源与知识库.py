import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="资源与知识库",
    page_icon="📚",
    layout="wide"
)

st.title("📚 资源与知识库")

st.markdown("""
本模块用于整理计算机辅助药物设计过程中常用的数据库、软件工具和理论知识。
这些资源可以为活性数据获取、分子结构处理、蛋白结构获取、分子对接和结果解释提供支持。
""")

st.subheader("1. 常用 CADD 数据库与工具")

resources = pd.DataFrame([
    {
        "类型": "化合物数据库",
        "数据库 / 工具": "PubChem",
        "用途": "查询化合物结构、CID、SMILES、基本理化性质",
        "适用模块": "活性数据整理、候选分子查询"
    },
    {
        "类型": "活性数据库",
        "数据库 / 工具": "ChEMBL",
        "用途": "获取化合物-靶点活性数据，如 IC50、Ki、EC50",
        "适用模块": "活性数据整理、QSAR 建模"
    },
    {
        "类型": "蛋白结构数据库",
        "数据库 / 工具": "PDB",
        "用途": "获取蛋白质三维结构，用于分子对接和结构分析",
        "适用模块": "分子对接结果分析"
    },
    {
        "类型": "蛋白信息数据库",
        "数据库 / 工具": "UniProt",
        "用途": "查询靶点蛋白序列、功能、物种来源和注释信息",
        "适用模块": "项目管理、靶点信息整理"
    },
    {
        "类型": "分子处理工具",
        "数据库 / 工具": "RDKit",
        "用途": "SMILES 解析、分子描述符计算、分子指纹计算、结构处理",
        "适用模块": "QSAR、ADMET、可解释性分析"
    },
    {
        "类型": "机器学习工具",
        "数据库 / 工具": "scikit-learn",
        "用途": "构建随机森林、SVM、逻辑回归等机器学习模型",
        "适用模块": "QSAR 模型与活性预测"
    },
    {
        "类型": "分子对接工具",
        "数据库 / 工具": "AutoDock Vina",
        "用途": "计算配体与靶点蛋白的结合构象和结合能",
        "适用模块": "分子对接结果分析"
    },
    {
        "类型": "分子对接工具",
        "数据库 / 工具": "PyRx",
        "用途": "图形化完成配体准备、受体准备和 AutoDock Vina 对接",
        "适用模块": "分子对接结果分析"
    },
    {
        "类型": "结构可视化工具",
        "数据库 / 工具": "PyMOL",
        "用途": "展示蛋白-配体结合模式、氢键、疏水作用和结合口袋",
        "适用模块": "分子对接结果分析、报告生成"
    },
    {
        "类型": "ADMET 预测工具",
        "数据库 / 工具": "SwissADME",
        "用途": "预测类药性、药代性质和部分 ADME 指标",
        "适用模块": "ADMET 预测"
    },
    {
        "类型": "ADMET 预测工具",
        "数据库 / 工具": "pkCSM",
        "用途": "预测吸收、分布、代谢、排泄和毒性相关指标",
        "适用模块": "ADMET 预测"
    },
    {
        "类型": "可视化工具",
        "数据库 / 工具": "Plotly / Matplotlib",
        "用途": "绘制活性分布、ROC 曲线、特征重要性和 docking score 图",
        "适用模块": "数据展示、QSAR、对接分析、报告生成"
    }
])

st.dataframe(resources, use_container_width=True)

csv_data = resources.to_csv(index=False, encoding="utf-8-sig")

st.download_button(
    label="📥 下载资源库 CSV",
    data=csv_data,
    file_name="cadd_resources.csv",
    mime="text/csv"
)

st.subheader("2. 本平台中的资源使用方式")

st.markdown("""
本平台按照药物发现早期阶段的计算流程进行设计，各类数据库和工具在流程中的作用如下：

```text
活性数据来源：
ChEMBL / 用户上传 CSV

分子结构处理：
RDKit 解析 SMILES，计算分子描述符和类药性指标

QSAR 建模：
scikit-learn 训练 Random Forest 模型

ADMET 评价：
RDKit 描述符 + Lipinski 规则 + ADMET score

分子对接：
AutoDock Vina / PyRx / CB-Dock2 等工具得到 docking score

结构可视化：
PyMOL 展示蛋白-配体结合模式

综合评分：
整合 QSAR probability、ADMET score 和 docking score

报告输出：
生成候选分子排序结果和项目分析报告
```
""")

st.subheader("3. CADD 基础概念说明")

concepts = pd.DataFrame([
    {
        "概念": "SMILES",
        "解释": "一种用字符串表示分子结构的方法，常用于化合物数据存储和计算处理。"
    },
    {
        "概念": "QSAR",
        "解释": "定量构效关系，通过分子结构特征预测化合物活性或性质。"
    },
    {
        "概念": "分子描述符",
        "解释": "用数值表示分子性质的特征，如分子量、LogP、TPSA、氢键供体和受体数量。"
    },
    {
        "概念": "ADMET",
        "解释": "吸收、分布、代谢、排泄和毒性，是评价候选药物成药性的重要指标。"
    },
    {
        "概念": "Lipinski 五规则",
        "解释": "用于初步判断口服小分子类药性的经验规则，包括分子量、LogP、氢键供体和受体等。"
    },
    {
        "概念": "分子对接",
        "解释": "模拟小分子配体与靶点蛋白结合构象，并估计结合能。"
    },
    {
        "概念": "Docking score",
        "解释": "分子对接得到的结合能评分，通常越低表示结合越稳定。"
    },
    {
        "概念": "综合评分",
        "解释": "整合活性预测、ADMET 评价和分子对接结果，对候选分子进行排序和推荐。"
    }
])

st.dataframe(concepts, use_container_width=True)

st.subheader("4. 本项目模块对应关系")

module_map = pd.DataFrame([
    {
        "平台模块": "活性数据整理",
        "主要输入": "原始活性 CSV",
        "主要输出": "cleaned_activity.csv",
        "支撑工具": "Pandas、RDKit"
    },
    {
        "平台模块": "QSAR 模型与活性预测",
        "主要输入": "cleaned_activity.csv",
        "主要输出": "qsar_predictions.csv",
        "支撑工具": "RDKit、scikit-learn"
    },
    {
        "平台模块": "ADMET 预测",
        "主要输入": "SMILES / 候选分子列表",
        "主要输出": "admet_results.csv",
        "支撑工具": "RDKit、Lipinski 规则"
    },
    {
        "平台模块": "分子对接结果分析",
        "主要输入": "docking_results.csv",
        "主要输出": "标准化 docking 结果和 docking score 图",
        "支撑工具": "AutoDock Vina、PyRx、PyMOL"
    },
    {
        "平台模块": "药物重定位与综合评分",
        "主要输入": "qsar_predictions.csv、admet_results.csv、docking_results.csv",
        "主要输出": "final_ranking.csv",
        "支撑工具": "Pandas、综合评分函数"
    },
    {
        "平台模块": "结果报告生成",
        "主要输入": "final_ranking.csv",
        "主要输出": "Markdown / HTML 报告",
        "支撑工具": "Streamlit、Pandas"
    }
])

st.dataframe(module_map, use_container_width=True)