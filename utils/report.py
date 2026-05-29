from pathlib import Path
from datetime import datetime
import html
import pandas as pd


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "无数据"

    display_df = df.copy()
    display_df = display_df.astype(str)

    headers = list(display_df.columns)
    rows = display_df.values.tolist()

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_markdown_report(
    final_df: pd.DataFrame,
    project_name: str = "CADD 综合实践平台",
    target_name: str = "未指定靶点",
    description: str = ""
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if final_df is None or final_df.empty:
        raise ValueError("final_df 为空，无法生成报告。")

    required_cols = [
        "compound_id",
        "smiles",
        "target",
        "qsar_probability",
        "admet_score",
        "docking_score",
        "total_score",
        "recommendation"
    ]

    missing = [c for c in required_cols if c not in final_df.columns]

    if missing:
        raise ValueError(f"final_ranking.csv 缺少必要列：{missing}")

    total_molecules = len(final_df)
    high_count = (final_df["recommendation"] == "高推荐").sum()
    medium_count = (final_df["recommendation"] == "中等推荐").sum()
    low_count = (final_df["recommendation"] == "低推荐").sum()

    best = final_df.sort_values("total_score", ascending=False).iloc[0]

    table_text = dataframe_to_markdown_table(final_df)

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

```text
综合评分 = 0.4 × QSAR 活性概率 + 0.3 × ADMET score + 0.3 × Docking score 归一化值
```

## 三、候选分子总体结果

- 候选分子总数：{total_molecules}
- 高推荐分子数：{high_count}
- 中等推荐分子数：{medium_count}
- 低推荐分子数：{low_count}

## 四、最优候选分子

- Compound ID：{best["compound_id"]}
- SMILES：{best["smiles"]}
- Target：{best["target"]}
- QSAR 活性概率：{best["qsar_probability"]}
- ADMET score：{best["admet_score"]}
- Docking score：{best["docking_score"]}
- 综合评分：{best["total_score"]}
- 推荐等级：{best["recommendation"]}

## 五、综合评分排序表

{table_text}

## 六、结果解释

综合评分较高的分子通常同时具有较高的 QSAR 活性预测概率、较好的 ADMET score，以及较低的 docking score。  
其中 docking score 越低，通常表示配体与靶点蛋白的结合越稳定；ADMET score 越高，说明分子在类药性方面表现越好；QSAR 活性概率越高，说明模型预测该分子具有活性的可能性越大。

## 七、结论与建议

根据本次综合评分结果，推荐优先关注“高推荐”等级的候选分子。  
这些分子可以作为后续分子优化、分子对接精细分析或实验验证的优先对象。  
对于“中等推荐”分子，可结合具体结构特征进一步优化；对于“低推荐”分子，建议暂不作为优先候选对象。
"""

    return report


def save_markdown_report(report_text: str, output_path="results/cadd_report.md") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return output_path


def generate_html_report(markdown_text: str, output_path="results/cadd_report.html") -> Path:
    escaped_text = html.escape(markdown_text)

    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>CADD 综合分析报告</title>
    <style>
        body {{
            font-family: Arial, "Microsoft YaHei", sans-serif;
            line-height: 1.7;
            margin: 40px;
            color: #222;
        }}
        h1, h2, h3 {{
            color: #2E7D32;
        }}
        pre {{
            background: #f5f5f5;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }}
    </style>
</head>
<body>
<pre>{escaped_text}</pre>
</body>
</html>
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_text)

    return output_path
