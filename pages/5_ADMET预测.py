import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import numpy as np

from utils.admet import *

# =========================
# 页面设置
# =========================
st.set_page_config(page_title="ADMET Prediction System", layout="wide")

st.title("🧬 ADMET Prediction & Drug-likeness Analysis Platform")
st.markdown("### Computational Drug Discovery Module (CADD Platform)")

# =========================
# 输入
# =========================
smiles = st.text_input("Enter SMILES", "CCOc1ccc(N)cc1")

if smiles:

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        st.error("Invalid SMILES")
        st.stop()

    # =========================
    # 分子结构图
    # =========================
    st.subheader("Molecular Structure")

    img = Draw.MolToImage(mol, size=(300, 300))
    st.image(img)

    # =========================
    # 计算属性
    # =========================
    props = calculate_basic_properties(smiles)

    lipinski = lipinski_rule(props)
    drug_score = drug_likeness_score(props)
    risk_score = admet_risk_score(props)
    final_score = admet_final_score(drug_score, risk_score)

    # =========================
    # 三列展示（科研UI）
    # =========================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Drug-likeness Score", drug_score)
        st.metric("ADMET Risk", risk_score)

    with col2:
        st.metric("Final ADMET Score", final_score)
        st.metric("Lipinski Violations", lipinski["violations"])

    with col3:
        st.metric("Lipinski Pass", lipinski["pass"])

    # =========================
    # 详细属性表
    # =========================
    st.subheader("Physicochemical Properties")

    st.dataframe(props)

    # =========================
    # 雷达图（科研感提升关键）
    # =========================
    st.subheader("Molecular Property Profile (Radar View)")

    labels = list(props.keys())
    values = list(props.values())

    values_norm = np.array(values) / np.max(values)

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values_norm = np.concatenate((values_norm, [values_norm[0]]))
    angles += angles[:1]

    fig = plt.figure()
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, values_norm, linewidth=2)
    ax.fill(angles, values_norm, alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)

    st.pyplot(fig)

    # =========================
    # 科研解释
    # =========================
    st.subheader("Interpretation (Scientific Insight)")

    st.markdown("""
- **MolWt**: influences absorption and distribution  
- **LogP**: affects membrane permeability  
- **TPSA**: determines polarity and oral bioavailability  
- **HBD/HBA**: hydrogen bonding capability affects binding affinity  
- **Rotatable Bonds**: molecular flexibility affects docking stability  

---

### Conclusion:
This compound is evaluated based on multi-dimensional ADMET space,
balancing **drug-likeness** and **toxicity risk**.
""")