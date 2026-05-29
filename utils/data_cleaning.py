import pandas as pd
import numpy as np


def read_activity_csv(file):
    """
    读取活性数据。
    自动识别逗号、分号、制表符等常见分隔符。
    兼容 ChEMBL 导出的 CSV 文件。
    """
    try:
        df = pd.read_csv(file, sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(file, sep=";")
        except Exception:
            df = pd.read_csv(file)

    df.columns = [str(c).strip() for c in df.columns]
    return df


def convert_to_pactivity(value, unit):
    """
    将 IC50 / Ki / EC50 等活性值转换为 pActivity。
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
    elif unit in ["um", "μm", "µm"]:
        return 6 - np.log10(value)
    elif unit == "mm":
        return 3 - np.log10(value)
    elif unit == "m":
        return -np.log10(value)
    else:
        return np.nan


def normalize_colname(col):
    """
    统一列名格式，便于模糊匹配。
    """
    return (
        str(col)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
    )


def find_column(df, possible_names):
    """
    根据多个可能列名自动寻找实际列名。
    忽略大小写、空格、下划线、横线等差异。
    """
    normalized_columns = {}

    for col in df.columns:
        normalized_columns[normalize_colname(col)] = col

    for name in possible_names:
        key = normalize_colname(name)
        if key in normalized_columns:
            return normalized_columns[key]

    return None


def detect_data_format(df):
    """
    自动识别数据格式：
    1. ChEMBL 原始导出数据
    2. 项目标准格式数据
    """

    chembl_columns = {
        "compound_id": [
            "Molecule ChEMBL ID",
            "Molecule ChEMBL Id",
            "molecule_chembl_id",
            "Parent Molecule ChEMBL ID",
            "Parent Molecule ChEMBL Id"
        ],
        "smiles": [
            "Smiles",
            "Canonical Smiles",
            "canonical_smiles",
            "Molecule Smiles",
            "Molecular Formula"
        ],
        "target": [
            "Target Name",
            "target_name",
            "Target ChEMBL ID",
            "Target ChEMBL Id",
            "target_chembl_id"
        ],
        "activity_type": [
            "Standard Type",
            "standard_type",
            "Activity Type",
            "activity_type"
        ],
        "activity_value": [
            "Standard Value",
            "standard_value",
            "Activity Value",
            "activity_value"
        ],
        "unit": [
            "Standard Units",
            "standard_units",
            "Units",
            "unit"
        ]
    }

    standard_columns = {
        "compound_id": [
            "compound_id",
            "compound id",
            "id"
        ],
        "smiles": [
            "smiles",
            "canonical_smiles",
            "canonical smiles"
        ],
        "target": [
            "target",
            "target_name",
            "target name"
        ],
        "activity_type": [
            "activity_type",
            "activity type",
            "standard_type",
            "standard type"
        ],
        "activity_value": [
            "activity_value",
            "activity value",
            "standard_value",
            "standard value"
        ],
        "unit": [
            "unit",
            "units",
            "standard_units",
            "standard units"
        ]
    }

    chembl_found = {
        key: find_column(df, names)
        for key, names in chembl_columns.items()
    }

    standard_found = {
        key: find_column(df, names)
        for key, names in standard_columns.items()
    }

    if all(v is not None for v in chembl_found.values()):
        return "chembl", chembl_found

    if all(v is not None for v in standard_found.values()):
        return "standard", standard_found

    raise ValueError(
        "无法识别数据格式。请确认文件至少包含 ChEMBL 原始列，"
        "或包含 compound_id, smiles, target, activity_type, activity_value, unit。"
        f" 当前文件列名为：{list(df.columns)}"
    )


def clean_activity_data_auto(df, threshold=6.0):
    """
    自动识别并清洗活性数据。
    支持：
    1. ChEMBL 原始导出数据
    2. 项目标准格式数据
    """

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    data_format, col_map = detect_data_format(df)

    cleaned = pd.DataFrame()

    cleaned["compound_id"] = df[col_map["compound_id"]]
    cleaned["smiles"] = df[col_map["smiles"]]
    cleaned["target"] = df[col_map["target"]]
    cleaned["activity_type"] = df[col_map["activity_type"]]
    cleaned["activity_value"] = df[col_map["activity_value"]]
    cleaned["unit"] = df[col_map["unit"]]

    relation_col = find_column(
        df,
        [
            "Standard Relation",
            "standard_relation",
            "Relation",
            "relation"
        ]
    )

    if relation_col is not None:
        cleaned["standard_relation"] = df[relation_col]

        # 保留等号关系的数据；如果上传数据里没有等号关系列，则不做这个筛选
        cleaned = cleaned[
            cleaned["standard_relation"]
            .astype(str)
            .str.strip()
            .isin(["=", "'='", '"="'])
        ]

    before_rows = len(cleaned)

    cleaned = cleaned.dropna(
        subset=[
            "compound_id",
            "smiles",
            "target",
            "activity_type",
            "activity_value",
            "unit"
        ]
    )

    cleaned["compound_id"] = cleaned["compound_id"].astype(str).str.strip()
    cleaned["smiles"] = cleaned["smiles"].astype(str).str.strip()
    cleaned["target"] = cleaned["target"].astype(str).str.strip()
    cleaned["activity_type"] = cleaned["activity_type"].astype(str).str.strip()
    cleaned["unit"] = cleaned["unit"].astype(str).str.strip()

    cleaned = cleaned[cleaned["smiles"] != ""]
    cleaned = cleaned[cleaned["compound_id"] != ""]

    cleaned["activity_value"] = pd.to_numeric(
        cleaned["activity_value"],
        errors="coerce"
    )

    cleaned = cleaned.dropna(subset=["activity_value"])
    cleaned = cleaned[cleaned["activity_value"] > 0]

    cleaned["pactivity"] = cleaned.apply(
        lambda row: convert_to_pactivity(
            row["activity_value"],
            row["unit"]
        ),
        axis=1
    )

    cleaned = cleaned.dropna(subset=["pactivity"])

    if len(cleaned) == 0:
        raise ValueError(
            "清洗后没有可用数据。请检查 Standard Units 是否为 nM、uM、mM 或 M，"
            "并确认 Standard Value 为有效正数。"
        )

    # 同一个 compound_id 可能有多条实验记录，取 pActivity 中位数
    grouped_rows = []

    for compound_id, group in cleaned.groupby("compound_id"):
        first = group.iloc[0]
        median_pactivity = group["pactivity"].median()

        unit = str(first["unit"]).strip().lower()

        if unit == "nm":
            median_activity_value = 10 ** (9 - median_pactivity)
        elif unit in ["um", "μm", "µm"]:
            median_activity_value = 10 ** (6 - median_pactivity)
        elif unit == "mm":
            median_activity_value = 10 ** (3 - median_pactivity)
        elif unit == "m":
            median_activity_value = 10 ** (-median_pactivity)
        else:
            median_activity_value = first["activity_value"]

        grouped_rows.append(
            {
                "compound_id": compound_id,
                "smiles": first["smiles"],
                "target": first["target"],
                "activity_type": first["activity_type"],
                "activity_value": median_activity_value,
                "unit": first["unit"],
                "pactivity": median_pactivity,
                "label": "Active" if median_pactivity >= threshold else "Inactive",
                "record_count": len(group)
            }
        )

    cleaned_df = pd.DataFrame(grouped_rows)

    summary = {
        "识别数据类型": "ChEMBL 原始数据" if data_format == "chembl" else "项目标准格式数据",
        "原始记录数": before_rows,
        "清洗后分子数": len(cleaned_df),
        "Active 数量": int((cleaned_df["label"] == "Active").sum()),
        "Inactive 数量": int((cleaned_df["label"] == "Inactive").sum())
    }

    return cleaned_df, summary