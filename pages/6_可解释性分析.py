import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("QSAR Model Explainability")

st.markdown("### Feature Importance Visualization")

# 模拟 QSAR 特征重要性数据
feature_data = {
    "Feature": [
        "MolWt",
        "LogP",
        "TPSA",
        "HBA",
        "HBD"
    ],
    "Importance": [
        0.32,
        0.25,
        0.18,
        0.15,
        0.10
    ]
}

df = pd.DataFrame(feature_data)

# 显示数据表
st.dataframe(df)

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    df["Feature"],
    df["Importance"]
)

# 设置标题
ax.set_title(
    "Random Forest Feature Importance",
    fontsize=16
)

# 坐标轴
ax.set_xlabel(
    "Descriptors",
    fontsize=12
)

ax.set_ylabel(
    "Importance Score",
    fontsize=12
)

# 显示数值
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.01,
        f"{height:.2f}",
        ha='center',
        fontsize=10
    )

# Streamlit显示
st.pyplot(fig)

# 解释部分
st.markdown("## Interpretation")

st.markdown("""
### Descriptor Meaning

- **MolWt**  
  Molecular weight influences molecular size and bioavailability.

- **LogP**  
  LogP reflects hydrophobicity and membrane permeability.

- **TPSA**  
  TPSA affects drug absorption and transport.

- **HBA / HBD**  
  Hydrogen bond acceptors/donors influence intermolecular interactions.
""")