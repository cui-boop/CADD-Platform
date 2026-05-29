# utils/report.py

from pathlib import Path
from datetime import datetime
import pandas as pd


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def generate_markdown_report(
    final_df: pd.DataFrame,
    project_name: str = "CADD 综合实践平台",
    target_name: str = "未指定靶点",
    description: str = ""
) -> str:
    """
    根据综合评分结果生成 Markdown 报告。
    """

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if final_df is None or final_df.empty:
        raise ValueError("final_df 为空，无法生成报告。")

    required_cols = ["total_score", "recommendation"]
    for col in required_cols:
        if col not in final_df.columns:
            raise ValueError(f"final_df 中缺少必要列：{col}")

    total_molecules = len(final_df)
    high_count = (final_df["recommendation"] == "高推荐").sum()
    medium_count = (final_df["recommendation"] == "中等推荐").sum()
    low_count = (final_df["recommendation"] == "低推荐").sum()

    best = final_df.sort_values("total_score", ascending=False).iloc[0]

    top_df = final_df.sort_values("total_score", ascending=False).head(10)

    report = f"""# 计算机辅助药物设计综合分析报告

## 一、项目基本信息

- 项目名称：{project_name}
- 靶点名称：{target_name}
- 报告生成时间：{now}

{description}

## 二、分析流程概述

本平台按照计算机辅助药物设计早期筛选流程进行分析，主要包括：

1. QSAR 模型预测候选分子的潜在活性；
2. ADMET / 类药性评分评估候选分子的成药性；
3. 分子对接结果用于评估候选分子与靶点蛋白的结合能力；
4. 综合 QSAR 活性概率、ADMET score 和 docking score，得到候选分子的综合评分和推荐等级。

综合评分公式为：
综合评分 = 0.4 × QSAR 活性概率 + 0.3 × ADMET score + 0.3 × Docking score 归一化值
"""