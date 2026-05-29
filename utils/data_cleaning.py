import pandas as pd
import numpy as np


def read_activity_csv(file):
    """
    读取活性数据。
    支持普通逗号分隔 CSV，也支持 ChEMBL 下载的分号分隔 CSV。
    """
    try:
        df = pd.read_csv(file)
        if df.shape[1] == 1:
            df = pd.read_csv(file, sep=";")
    except Exception:
        df = pd.read_csv(file, sep=";")

    return df


def convert_to_pactivity(value, unit):
    """
    将 IC50 / Ki 等活性值转换为 pActivity。
    如果单位是 nM:
        pActivity = 9 - log10(value)
    """
    try:
        value = float(value)
    except Exception:
        return np.nan

    if value <= 0:
        return np.nan

    unit = str(unit).strip().lower()

    if unit == "nm":
        return 9 - np.log10(value)
    elif unit in ["um", "μm"]:
        return 6 - np.log10(value)
    elif unit == "mm":
        return 3 - np.log10(value)
    elif unit == "m":
        return -np.log10(value)
    else:
        return np.nan


def standardize_chembl_data(df, threshold=6.0):
    """
    将 ChEMBL 导出的原始活性数据整理成项目统一格式。

    输出列：
    compound_id, smiles, target, activity_type,
    activity_value, unit, pactivity, label
    """

    df = df.copy()

    column_map = {
        "Molecule ChEMBL ID": "compound_id",
        "Smiles": "smiles",
        "Target Name": "target",
        "Standard Type": "activity_type",
        "Standard Value": "activity_value",
        "Standard Units": "unit",
        "Standard Relation": "standard_relation"
    }

    missing_cols = [col for col in column_map if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少 ChEMBL 必要列：{missing_cols}")

    df = df.rename(columns=column_map)

    keep_cols = [
        "compound_id",
        "smiles",
        "target",
        "activity_type",
        "activity_value",
        "unit",
        "standard_relation"
    ]

    df = df[keep_cols]

    df = df.dropna(subset=[
        "compound_id",
        "smiles",
        "activity_type",
        "activity_value",
        "unit"
    ])

    df = df[df["activity_value"].astype(str).str.strip() != ""]

    df["activity_value"] = pd.to_numeric(df["activity_value"], errors="coerce")
    df = df.dropna(subset=["activity_value"])
    df = df[df["activity_value"] > 0]

    df["pactivity"] = df.apply(
        lambda row: convert_to_pactivity(row["activity_value"], row["unit"]),
        axis=1
    )

    df = df.dropna(subset=["pactivity"])

    df["label"] = df["pactivity"].apply(
        lambda x: "Active" if x >= threshold else "Inactive"
    )

    # 对重复化合物进行处理：同一 compound_id 取 pactivity 中位数
    grouped = []

    for compound_id, group in df.groupby("compound_id"):
        first = group.iloc[0]
        median_pactivity = group["pactivity"].median()
        median_activity_value = 10 ** (9 - median_pactivity)

        grouped.append({
            "compound_id": compound_id,
            "smiles": first["smiles"],
            "target": first["target"],
            "activity_type": first["activity_type"],
            "activity_value": median_activity_value,
            "unit": first["unit"],
            "pactivity": median_pactivity,
            "label": "Active" if median_pactivity >= threshold else "Inactive",
            "record_count": len(group)
        })

    cleaned_df = pd.DataFrame(grouped)

    summary = {
        "原始记录数": len(df),
        "去重后分子数": len(cleaned_df),
        "Active 数量": int((cleaned_df["label"] == "Active").sum()),
        "Inactive 数量": int((cleaned_df["label"] == "Inactive").sum())
    }

    return cleaned_df, summary


def clean_standard_activity_data(df, threshold=6.0):
    """
    处理已经是项目统一格式的数据。
    需要至少包含：
    compound_id, smiles, target, activity_type, activity_value, unit
    """

    required_cols = [
        "compound_id",
        "smiles",
        "target",
        "activity_type",
        "activity_value",
        "unit"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列：{missing_cols}")

    df = df.copy()

    df = df.dropna(subset=[
        "compound_id",
        "smiles",
        "activity_value",
        "unit"
    ])

    df["activity_value"] = pd.to_numeric(df["activity_value"], errors="coerce")
    df = df.dropna(subset=["activity_value"])
    df = df[df["activity_value"] > 0]

    df["pactivity"] = df.apply(
        lambda row: convert_to_pactivity(row["activity_value"], row["unit"]),
        axis=1
    )

    df = df.dropna(subset=["pactivity"])

    df["label"] = df["pactivity"].apply(
        lambda x: "Active" if x >= threshold else "Inactive"
    )

    df = df.drop_duplicates(subset=["compound_id"])

    summary = {
        "原始记录数": len(df),
        "清洗后分子数": len(df),
        "Active 数量": int((df["label"] == "Active").sum()),
        "Inactive 数量": int((df["label"] == "Inactive").sum())
    }

    return df, summary