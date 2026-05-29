# CADD-Platform

本项目是一个基于 Streamlit 的计算机辅助药物设计综合实践平台，面向药物发现早期阶段的数据整理、活性预测、成药性评价和候选分子筛选流程。

## 1. 项目简介

平台支持用户上传 ChEMBL 单靶点活性数据，并完成以下分析流程：

- 项目管理
- 活性数据整理
- pActivity / pIC50 计算
- Active / Inactive 标签划分
- QSAR 模型训练
- 分子属性与 ADMET 预测
- 可解释性分析
- 分子对接结果分析
- 药物重定位与综合评分
- 结果报告生成

本项目目前以内置的 EGFR 靶点活性数据作为示例数据。

## 2. 示例数据

内置示例数据来源于 ChEMBL 数据库。

数据筛选条件：

- Target: EGFR
- Target ChEMBL ID: CHEMBL203
- Organism: Homo sapiens
- Target Type: Single Protein
- Activity Type: IC50
- Standard Units: nM
- Standard Relation: =

数据文件位于：

    data/chembl_egfr_ic50_raw.csv
    data/chembl_egfr_ic50_model_ready.csv
    data/cleaned_activity.csv

其中：

- `chembl_egfr_ic50_raw.csv` 是从 ChEMBL 下载的原始活性数据。
- `chembl_egfr_ic50_model_ready.csv` 是整理后的 EGFR 示例建模数据。
- `cleaned_activity.csv` 是后续 QSAR 模型读取的标准输入文件。

## 3. 数据格式

平台支持 ChEMBL 原始导出数据，也支持项目标准格式数据。

项目标准格式至少应包含以下列：

    compound_id
    smiles
    target
    activity_type
    activity_value
    unit

数据整理后会生成以下列：

    compound_id
    smiles
    target
    activity_type
    activity_value
    unit
    pactivity
    label

其中：

- `pactivity = 9 - log10(IC50_nM)`
- 默认 `pactivity >= 6` 判定为 Active
- 默认 `pactivity < 6` 判定为 Inactive

## 4. 项目结构

    CADD-Platform/
    │
    ├── app.py
    ├── README.md
    ├── requirements.txt
    │
    ├── pages/
    │   ├── 1_首页.py
    │   ├── 2_项目管理.py
    │   ├── 3_活性数据整理.py
    │   ├── 4_QSAR模型与活性预测.py
    │   ├── 5_ADMET预测.py
    │   ├── 6_可解释性分析.py
    │   ├── 7_分子对接结果分析.py
    │   ├── 8_药物重定位与综合评分.py
    │   ├── 9_资源与知识库.py
    │   └── 10_结果报告生成.py
    │
    ├── utils/
    │   ├── data_cleaning.py
    │   ├── descriptors.py
    │   ├── qsar_model.py
    │   ├── admet.py
    │   ├── docking.py
    │   ├── scoring.py
    │   └── report.py
    │
    ├── data/
    ├── models/
    └── results/

## 5. 环境依赖

主要依赖包括：

- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- plotly
- joblib
- rdkit


## 6. 运行网站

在项目根目录下运行：

    streamlit run app.py

运行后，浏览器会自动打开本地网站。

## 7. 推荐使用流程

推荐按照以下顺序使用平台：

1. 项目管理
2. 活性数据整理
3. QSAR 模型与活性预测
4. 分子属性与 ADMET 预测
5. 可解释性分析
6. 分子对接结果分析
7. 药物重定位与综合评分
8. 结果报告生成
