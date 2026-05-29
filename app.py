import streamlit as st


st.set_page_config(
    page_title="首页",
    page_icon="💊",
    layout="wide"
)


# =========================
# 页面样式
# =========================
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #e0f2fe 0%, #ecfdf5 55%, #fef9c3 100%);
        padding: 36px 42px;
        border-radius: 24px;
        border: 1px solid #e5e7eb;
        margin-bottom: 28px;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 850;
        color: #111827;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #374151;
        line-height: 1.8;
        max-width: 1050px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 750;
        color: #111827;
        margin-top: 28px;
        margin-bottom: 18px;
    }

    .intro-box {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 24px 28px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
        color: #374151;
        line-height: 1.8;
        font-size: 16px;
    }

    .module-card {
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 22px;
        min-height: 170px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.035);
    }

    .module-number {
        display: inline-block;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background-color: #0f766e;
        color: white;
        text-align: center;
        line-height: 32px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .module-title {
        font-size: 19px;
        font-weight: 750;
        color: #1e3a8a;
        margin-bottom: 10px;
    }

    .module-text {
        color: #4b5563;
        font-size: 15px;
        line-height: 1.7;
    }

    .note {
        background-color: #f9fafb;
        border-left: 5px solid #14b8a6;
        padding: 20px 24px;
        border-radius: 12px;
        color: #374151;
        line-height: 1.8;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# 顶部介绍
# =========================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">💊 CADD 综合实践平台</div>
        <div class="hero-subtitle">
            本平台面向计算机辅助药物设计课程实践，围绕药物发现早期阶段的核心流程进行设计。
            用户可以从项目创建开始，依次完成活性数据整理、QSAR 模型训练、分子属性评价、
            模型解释、分子对接结果分析、候选分子综合评分以及结果报告生成。
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# 平台定位
# =========================
st.markdown('<div class="section-title">一、平台定位</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="intro-box">
        本网站是一个面向单靶点活性数据分析的 CADD 实践平台。
        平台以内置的 EGFR ChEMBL 活性数据作为示例，同时支持用户上传其他符合格式要求的
        ChEMBL 单靶点活性数据，例如 BACE1、ACHE、COX2、JAK2、VEGFR2 等。
        <br><br>
        核心目标是将“数据整理—模型训练—活性预测—成药性评价—结构验证—综合评分—报告输出”
        连接成一条完整的分析流程，帮助完成候选分子的早期筛选。
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# 平台功能与分析流程
# =========================
st.markdown('<div class="section-title">二、平台功能与分析流程</div>', unsafe_allow_html=True)

modules = [
    (
        "项目管理",
        "创建 CADD 分析项目，记录疾病方向、靶点名称、数据来源和主要活性指标，为后续分析建立项目背景。"
    ),
    (
        "活性数据整理",
        "读取 ChEMBL 原始数据或项目标准格式数据，提取 SMILES、活性值和单位，计算 pActivity 并生成 Active / Inactive 标签。"
    ),
    (
        "QSAR 模型与活性预测",
        "基于分子描述符和随机森林模型建立结构-活性关系，用于预测候选分子是否具有潜在靶点活性。"
    ),
    (
        "ADMET 预测",
        "计算分子量、LogP、TPSA、氢键供体、氢键受体等性质，并结合 Lipinski 规则进行初步类药性评价。"
    ),
    (
        "可解释性分析",
        "展示 QSAR 模型的特征重要性，分析哪些分子性质对活性预测贡献较大，为结构优化提供参考。"
    ),
    (
        "分子对接结果分析",
        "导入候选分子的 docking score，展示结合能排序，从蛋白-配体结合角度辅助验证候选分子。"
    ),
    (
        "药物重定位与综合评分",
        "整合 QSAR 活性概率、ADMET 评价和分子对接结果，对候选分子进行综合评分和优先级排序。"
    ),
    (
        "资源与知识库",
        "整理常用数据库、工具和药物设计相关知识，为数据获取、靶点理解和结果解释提供支持。"
    ),
    (
        "结果报告生成",
        "汇总项目基本信息、数据整理结果、模型训练结果、ADMET 评价、对接分析和综合评分结果，形成报告。"
    )
]

for i in range(0, len(modules), 3):
    cols = st.columns(3)
    for j, col in enumerate(cols):
        index = i + j
        if index < len(modules):
            title, text = modules[index]
            with col:
                st.markdown(
                    f"""
                    <div class="module-card">
                        <div class="module-number">{index + 1}</div>
                        <div class="module-title">{title}</div>
                        <div class="module-text">{text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    st.write("")


# =========================
# 示例数据说明
# =========================
st.markdown('<div class="section-title">三、内置示例数据</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="note">
        本项目内置了从 ChEMBL 数据库下载的人源 EGFR 靶点 IC50 活性数据作为示例。
        <br><br>
        <strong>数据筛选条件包括：</strong>
        <ul>
            <li>Target = EGFR / CHEMBL203</li>
            <li>Organism = Homo sapiens</li>
            <li>Target Type = Single Protein</li>
            <li>Activity Type = IC50</li>
            <li>Standard Units = nM</li>
            <li>Standard Relation = =</li>
        </ul>
        该示例数据主要用于演示从真实 ChEMBL 活性数据到 QSAR 模型训练的完整流程。
        平台并不局限于 EGFR，用户可以上传其他单靶点活性数据进行同样的分析。
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# 使用建议
# =========================
st.markdown('<div class="section-title">四、推荐使用顺序</div>', unsafe_allow_html=True)

st.success(
    "推荐顺序：项目管理 → 活性数据整理 → QSAR 模型与活性预测 → ADMET 预测 → "
    "可解释性分析 → 分子对接结果分析 → 药物重定位与综合评分 → 结果报告生成"
)