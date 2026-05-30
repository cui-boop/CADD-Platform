import re
from pathlib import Path

import streamlit as st


# =========================
# 页面基础设置
# =========================
st.set_page_config(
    page_title="首页",
    page_icon="💊",
    layout="wide"
)


# =========================
# 自动读取 pages 页面
# =========================
BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "pages"


def clean_page_title(page_file: Path) -> str:
    """将页面文件名转换为展示标题。"""
    title = page_file.stem

    # 去掉开头序号，例如：1_、01_、4.、4-、4、
    title = re.sub(r"^\d+[\s_\-\.、]*", "", title)

    # 替换分隔符
    title = title.replace("_", " ").replace("-", " ").strip()

    return title or page_file.stem


def get_page_order(page_file: Path):
    """按文件名前面的数字排序，没有数字的页面排在后面。"""
    match = re.match(r"^(\d+)", page_file.stem)
    number = int(match.group(1)) if match else 999
    return number, clean_page_title(page_file)


def load_pages():
    """读取 pages 文件夹下的功能页面。"""
    if not PAGES_DIR.exists():
        return []

    page_files = [
        file for file in PAGES_DIR.glob("*.py")
        if not file.name.startswith("_")
    ]

    return sorted(page_files, key=get_page_order)


def get_page_icon(title: str) -> str:
    """根据页面标题自动匹配图标。"""
    title_lower = title.lower()

    rules = [
    (["活性数据整理", "数据", "清洗", "整理"], "🧹"),
    (["模型训练", "qsar", "模型", "训练"], "🤖"),
    (["分子生成", "生成"], "🧬"),
    (["活性预测", "预测", "活性"], "🎯"),
    (["成药性筛选", "成药", "admet", "性质", "lipinski"], "🧪"),
    (["分子设计", "设计"], "💡"),
    (["文献检索", "llm", "知识提取", "文献", "检索", "知识"], "📚"),
    ] 

    for keywords, icon in rules:
        if any(keyword.lower() in title_lower for keyword in keywords):
            return icon

    return "🔹"


pages = load_pages()


# =========================
# 页面样式
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.12), transparent 32%),
            radial-gradient(circle at top right, rgba(37, 99, 235, 0.12), transparent 30%),
            linear-gradient(180deg, #f8fafc 0%, #ffffff 45%, #f8fafc 100%);
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* 顶部介绍 */
    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #0f766e 0%, #2563eb 58%, #1e3a8a 100%);
        padding: 52px 58px;
        border-radius: 30px;
        color: white;
        margin-bottom: 34px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 270px;
        height: 270px;
        right: -70px;
        top: -80px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.15);
    }

    .hero::before {
        content: "";
        position: absolute;
        width: 160px;
        height: 160px;
        right: 210px;
        bottom: -80px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.10);
    }

    .hero-title {
        position: relative;
        z-index: 1;
        font-size: 56px;
        font-weight: 900;
        line-height: 1.15;
        margin-bottom: 22px;
        letter-spacing: -0.8px;
    }

    .hero-subtitle {
        position: relative;
        z-index: 1;
        max-width: 1050px;
        font-size: 20px;
        line-height: 1.9;
        color: rgba(255, 255, 255, 0.92);
    }

    /* 分区标题 */
    .section-title {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 30px;
        font-weight: 900;
        color: #111827;
        margin-top: 22px;
        margin-bottom: 24px;
    }

    .section-title::before {
        content: "";
        width: 9px;
        height: 34px;
        border-radius: 999px;
        background: linear-gradient(180deg, #14b8a6, #2563eb);
        flex-shrink: 0;
    }

    /* 功能入口外层间距 */
    [data-testid="column"] {
        padding-left: 0.35rem !important;
        padding-right: 0.35rem !important;
    }

    /* 功能入口卡片 */
    [data-testid="stPageLink"] {
        width: 100%;
        height: 96px;
        margin-bottom: 18px;
    }

    [data-testid="stPageLink"] a {
        width: 100% !important;
        height: 96px !important;
        min-height: 96px !important;
        max-height: 96px !important;
        box-sizing: border-box !important;

        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 12px !important;

        padding: 14px 22px !important;
        border-radius: 22px !important;
        border: 1px solid #e5e7eb !important;
        background: rgba(255, 255, 255, 0.94) !important;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.055) !important;
        transition: all 0.18s ease !important;
        text-decoration: none !important;
    }

    [data-testid="stPageLink"] a:hover {
        transform: translateY(-3px);
        border-color: #99f6e4 !important;
        box-shadow: 0 16px 34px rgba(15, 23, 42, 0.10) !important;
        background: #ffffff !important;
        text-decoration: none !important;
    }

    /* 功能入口文字 */
    [data-testid="stPageLink"] p {
        margin: 0 !important;
        font-size: 22px !important;
        font-weight: 850 !important;
        line-height: 1.25 !important;
        color: #111827 !important;
        text-align: center !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        word-break: keep-all !important;
    }

    /* 放大 page_icon */
    [data-testid="stPageLink"] a span {
        font-size: 34px !important;
        line-height: 1 !important;
    }

    [data-testid="stPageLink"] a svg {
        width: 34px !important;
        height: 34px !important;
    }

    /* 示例数据模块 */
    .data-box {
        background: linear-gradient(135deg, #ecfeff 0%, #f0fdf4 100%);
        border: 1px solid #ccfbf1;
        border-radius: 22px;
        padding: 28px 32px;
        color: #374151;
        line-height: 1.85;
        font-size: 17px;
        box-shadow: 0 10px 28px rgba(15, 118, 110, 0.08);
    }

    /* 小屏幕适配 */
    @media (max-width: 900px) {
        .hero {
            padding: 36px 30px;
        }

        .hero-title {
            font-size: 40px;
        }

        .hero-subtitle {
            font-size: 17px;
        }

        [data-testid="stPageLink"] {
            height: 92px;
        }

        [data-testid="stPageLink"] a {
            height: 92px !important;
            min-height: 92px !important;
            max-height: 92px !important;
        }

        [data-testid="stPageLink"] p {
            font-size: 19px !important;
        }

        [data-testid="stPageLink"] a span {
            font-size: 30px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# 顶部首页介绍
# =========================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">💊 CADD 综合实践平台</div>
        <div class="hero-subtitle">
            本平台面向计算机辅助药物设计中的小分子候选药物早期筛选流程，
            内置 EGFR ChEMBL 单靶点活性数据作为示例数据，
            集成活性数据整理、模型训练、分子生成、活性预测、成药性筛选、
            分子设计以及文献检索与 LLM 知识提取功能。
            平台旨在通过数据驱动建模与多维度分子评价，
            辅助用户完成从活性数据处理到候选分子优化分析的完整实践流程。
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# 功能入口
# =========================
st.markdown('<div class="section-title">功能入口</div>', unsafe_allow_html=True)

if not pages:
    st.warning("未检测到 pages 文件夹下的功能页面。请确认项目目录中存在 pages 文件夹，并且其中包含 .py 页面文件。")
else:
    for row_start in range(0, len(pages), 3):
        cols = st.columns(3, gap="medium")

        for i, page_file in enumerate(pages[row_start: row_start + 3]):
            index = row_start + i + 1
            title = clean_page_title(page_file)
            icon = get_page_icon(title)

            with cols[i]:
                st.page_link(
                    f"pages/{page_file.name}",
                    label=f"{index:02d} {title}",
                    icon=icon
                )


# =========================
# 示例数据说明
# =========================
st.markdown('<div class="section-title">内置示例数据</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="data-box">
        平台内置一套 EGFR ChEMBL 单靶点活性数据，可直接用于演示完整分析流程。
        也可以根据页面要求上传其他单靶点活性数据，完成类似的建模、预测与筛选分析。
    </div>
    """,
    unsafe_allow_html=True
)