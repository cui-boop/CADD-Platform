from pathlib import Path
import json
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


DEFAULT_TARGETS = [
    "EGFR", "HER2", "BACE1", "JAK2", "ALK", "VEGFR", "VEGFR2", "PI3K", "AKT",
    "mTOR", "CDK2", "CDK4", "CDK6", "AChE", "COX-2", "PTGS2", "DPP4",
    "ACE2", "Mpro", "3CLpro", "PD-1", "PD-L1", "PARP1", "BRAF", "MEK",
    "CYP450", "CYP3A4", "CYP2D6", "hERG"
]

DEFAULT_DISEASES = [
    "cancer", "tumor", "carcinoma", "lung cancer", "breast cancer",
    "non-small cell lung cancer", "NSCLC", "Alzheimer", "diabetes",
    "inflammation", "infection", "neurotoxicity", "toxicity"
]

DEFAULT_ADMET_TERMS = [
    "ADMET", "absorption", "distribution", "metabolism", "excretion",
    "toxicity", "hepatotoxicity", "cardiotoxicity", "hERG", "CYP450",
    "CYP3A4", "CYP2D6", "solubility", "permeability", "bioavailability",
    "clearance", "half-life", "BBB", "plasma protein binding"
]

DEFAULT_DOCKING_TERMS = [
    "docking", "binding affinity", "binding energy", "hydrogen bond",
    "hydrophobic interaction", "pi-pi", "π-π", "salt bridge",
    "active site", "binding pocket", "key residue", "molecular dynamics",
    "RMSD", "RMSF"
]

ACTIVITY_PATTERN = re.compile(
    r"\b(IC50|EC50|Ki|Kd|MIC|GI50|CC50)\b\s*(?:=|:|of|was|is|约|为)?\s*([<>≤≥~≈]?\s*\d+(?:\.\d+)?)\s*(nM|uM|µM|μM|mM|M|ng/mL|ug/mL|µg/mL|μg/mL)?",
    flags=re.IGNORECASE
)

RESIDUE_PATTERN = re.compile(
    r"\b([A-Z][a-z]{2}\s?\d{1,4}|[A-Z]{3}\d{1,4})\b"
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[。！？.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def find_terms(text: str, terms: List[str]) -> List[str]:
    text_lower = text.lower()
    found = []
    for term in terms:
        if term.lower() in text_lower:
            found.append(term)
    return sorted(set(found), key=lambda x: x.lower())


def extract_activity_data(text: str) -> List[Dict[str, str]]:
    records = []
    for match in ACTIVITY_PATTERN.finditer(text):
        records.append({
            "activity_type": match.group(1),
            "value": match.group(2).replace(" ", ""),
            "unit": match.group(3) or "",
            "matched_text": match.group(0)
        })
    return records


def extract_candidate_compounds(text: str) -> List[str]:
    patterns = [
        r"\bCHEMBL\d+\b",
        r"\bCID\s?\d+\b",
        r"\b[A-Z][a-z]+tinib\b",
        r"\bgefitinib\b",
        r"\berlotinib\b",
        r"\bosimertinib\b",
        r"\blapatinib\b",
        r"\bafatinib\b",
        r"\bcompound\s+\d+[a-zA-Z]?\b",
        r"\bcompound\s+[A-Z]\d*\b"
    ]
    found = []
    for p in patterns:
        found.extend(re.findall(p, text, flags=re.IGNORECASE))

    clean = []
    seen = set()
    for item in found:
        key = item.lower()
        if key not in seen:
            clean.append(item)
            seen.add(key)
    return clean


def extract_residues(text: str) -> List[str]:
    residues = []
    for item in RESIDUE_PATTERN.findall(text):
        token = item.replace(" ", "").upper()
        if token[:3] in {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
            "TYR", "VAL"
        }:
            residues.append(token)
    return sorted(set(residues))


def extract_relevant_sentences(text: str, keywords: List[str], max_sentences: int = 6) -> List[str]:
    sentences = split_sentences(text)
    scored = []
    for sent in sentences:
        score = sum(1 for kw in keywords if str(kw).lower() in sent.lower())
        if score > 0:
            scored.append((score, sent))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:max_sentences]]


def rule_based_extract(text: str, title: str = "") -> Dict[str, Any]:
    full_text = normalize_text((title or "") + "\n" + (text or ""))
    activity_data = extract_activity_data(full_text)

    targets = find_terms(full_text, DEFAULT_TARGETS)
    diseases = find_terms(full_text, DEFAULT_DISEASES)
    compounds = extract_candidate_compounds(full_text)
    admet_terms = find_terms(full_text, DEFAULT_ADMET_TERMS)
    docking_terms = find_terms(full_text, DEFAULT_DOCKING_TERMS)
    residues = extract_residues(full_text)

    activity_keywords = [r["activity_type"] for r in activity_data] + ["IC50", "Ki", "activity", "inhibitory"]
    admet_sentences = extract_relevant_sentences(full_text, DEFAULT_ADMET_TERMS, max_sentences=5)
    docking_sentences = extract_relevant_sentences(full_text, DEFAULT_DOCKING_TERMS + residues, max_sentences=5)
    mechanism_sentences = extract_relevant_sentences(full_text, targets + diseases + ["mechanism", "inhibit", "pathway"], max_sentences=5)

    design_suggestions = []
    if residues:
        design_suggestions.append("可围绕关键残基 " + "、".join(residues[:8]) + " 分析氢键、疏水作用或 π-π 作用。")
    if any(t.lower() == "hydrogen bond" for t in docking_terms) or residues:
        design_suggestions.append("设计类似物时应优先保留能够形成关键氢键的基团。")
    if any(t.lower() == "solubility" for t in admet_terms) or "logp" in full_text.lower():
        design_suggestions.append("若文献提示溶解性或 LogP 问题，应避免继续增加强疏水取代基。")
    if not design_suggestions:
        design_suggestions.append("建议结合 QSAR、ADMET 和 docking 结果进一步确定结构优化方向。")

    summary_parts = []
    if targets:
        summary_parts.append("识别到靶点：" + "、".join(targets))
    if compounds:
        summary_parts.append("识别到候选化合物：" + "、".join(compounds[:8]))
    if activity_data:
        summary_parts.append(f"识别到 {len(activity_data)} 条活性指标。")
    if admet_terms:
        summary_parts.append("文本包含 ADMET 相关信息。")
    if docking_terms:
        summary_parts.append("文本包含 docking 或结合模式信息。")

    return {
        "mode": "规则提取",
        "title": title,
        "research_topic": title or "未提供标题",
        "disease_context": diseases,
        "targets": targets,
        "compounds": compounds,
        "activity_data": activity_data,
        "admet_information": admet_sentences,
        "docking_information": docking_sentences,
        "key_residues": residues,
        "mechanism_information": mechanism_sentences,
        "design_suggestions": design_suggestions,
        "summary": " ".join(summary_parts) if summary_parts else "未在文本中识别到足够明确的 CADD 关键信息。"
    }


def ncbi_get(endpoint: str, params: Dict[str, Any], timeout: int = 20) -> requests.Response:
    url = f"{NCBI_BASE}/{endpoint}"
    clean_params = {k: v for k, v in params.items() if v not in [None, ""]}
    response = requests.get(url, params=clean_params, timeout=timeout)
    response.raise_for_status()
    return response


def build_query(keyword_1: str, keyword_2: str = "", keyword_3: str = "", search_field: str = "Title/Abstract") -> str:
    keywords = [k.strip() for k in [keyword_1, keyword_2, keyword_3] if k and k.strip()]
    if not keywords:
        return ""

    if search_field == "Title/Abstract":
        return " AND ".join([f'("{kw}"[Title/Abstract])' for kw in keywords])
    if search_field == "MeSH Terms":
        return " AND ".join([f'("{kw}"[MeSH Terms])' for kw in keywords])
    if search_field == "All Fields":
        return " AND ".join([f'("{kw}"[All Fields])' for kw in keywords])

    return " AND ".join(keywords)


def search_pubmed(
    query: str,
    retmax: int = 5,
    sort: str = "relevance",
    api_key: str = "",
    email: str = "",
    tool: str = "CADD_Streamlit_Knowledge_Extraction"
) -> List[str]:
    if not query.strip():
        return []

    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": int(retmax),
        "sort": sort,
        "api_key": api_key,
        "email": email,
        "tool": tool
    }

    response = ncbi_get("esearch.fcgi", params)
    data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])


def _get_text(element: Optional[ET.Element]) -> str:
    if element is None:
        return ""
    return normalize_text(" ".join(element.itertext()))


def fetch_pubmed_articles(
    pmids: List[str],
    api_key: str = "",
    email: str = "",
    tool: str = "CADD_Streamlit_Knowledge_Extraction"
) -> List[Dict[str, Any]]:
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "api_key": api_key,
        "email": email,
        "tool": tool
    }

    response = ncbi_get("efetch.fcgi", params)
    root = ET.fromstring(response.text)

    articles = []

    for article in root.findall(".//PubmedArticle"):
        pmid = _get_text(article.find(".//PMID"))
        title = _get_text(article.find(".//ArticleTitle"))

        abstract_parts = []
        for abs_text in article.findall(".//Abstract/AbstractText"):
            label = abs_text.attrib.get("Label", "")
            text = _get_text(abs_text)
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)

        abstract = normalize_text(" ".join(abstract_parts))
        journal = _get_text(article.find(".//Journal/Title"))
        year = _get_text(article.find(".//PubDate/Year"))

        authors = []
        for author in article.findall(".//AuthorList/Author")[:8]:
            last = _get_text(author.find("LastName"))
            fore = _get_text(author.find("ForeName"))
            collective = _get_text(author.find("CollectiveName"))
            if collective:
                authors.append(collective)
            elif last or fore:
                authors.append((fore + " " + last).strip())

        doi = ""
        for aid in article.findall(".//ArticleIdList/ArticleId"):
            if aid.attrib.get("IdType") == "doi":
                doi = _get_text(aid)
                break

        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "authors": "; ".join(authors),
            "doi": doi,
            "pmc_id": ""
        })

    return articles


def link_pubmed_to_pmc(
    pmid: str,
    api_key: str = "",
    email: str = "",
    tool: str = "CADD_Streamlit_Knowledge_Extraction"
) -> str:
    if not pmid:
        return ""

    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": pmid,
        "retmode": "xml",
        "api_key": api_key,
        "email": email,
        "tool": tool
    }

    try:
        response = ncbi_get("elink.fcgi", params)
        root = ET.fromstring(response.text)
        pmc_numeric = _get_text(root.find(".//LinkSetDb/Link/Id"))
        if pmc_numeric:
            return "PMC" + pmc_numeric
    except Exception:
        return ""

    return ""


def fetch_pmc_full_text(
    pmc_id: str,
    api_key: str = "",
    email: str = "",
    tool: str = "CADD_Streamlit_Knowledge_Extraction"
) -> str:
    if not pmc_id:
        return ""

    numeric_id = str(pmc_id).replace("PMC", "").strip()

    params = {
        "db": "pmc",
        "id": numeric_id,
        "retmode": "xml",
        "api_key": api_key,
        "email": email,
        "tool": tool
    }

    try:
        response = ncbi_get("efetch.fcgi", params, timeout=30)
        root = ET.fromstring(response.text)

        sections = []
        for elem in root.findall(".//body//p"):
            text = _get_text(elem)
            if text:
                sections.append(text)

        if not sections:
            for elem in root.findall(".//abstract//p"):
                text = _get_text(elem)
                if text:
                    sections.append(text)

        return normalize_text(" ".join(sections))
    except Exception:
        return ""


def search_and_fetch_literature(
    keyword_1: str,
    keyword_2: str = "",
    keyword_3: str = "",
    search_field: str = "Title/Abstract",
    retmax: int = 5,
    sort: str = "relevance",
    fetch_full_text: bool = True,
    api_key: str = "",
    email: str = ""
) -> Dict[str, Any]:
    query = build_query(keyword_1, keyword_2, keyword_3, search_field)
    pmids = search_pubmed(
        query=query,
        retmax=retmax,
        sort=sort,
        api_key=api_key,
        email=email
    )

    articles = fetch_pubmed_articles(pmids, api_key=api_key, email=email)

    if fetch_full_text:
        for article in articles:
            pmc_id = link_pubmed_to_pmc(article.get("pmid", ""), api_key=api_key, email=email)
            article["pmc_id"] = pmc_id

            if pmc_id:
                time.sleep(0.12)
                full_text = fetch_pmc_full_text(pmc_id, api_key=api_key, email=email)
                article["full_text"] = full_text
            else:
                article["full_text"] = ""
            time.sleep(0.12)
    else:
        for article in articles:
            article["full_text"] = ""

    return {
        "query": query,
        "pmids": pmids,
        "articles": articles
    }


def result_to_tables(result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    overview = pd.DataFrame([
        {"字段": "研究主题", "内容": result.get("research_topic", "")},
        {"字段": "靶点", "内容": "; ".join(result.get("targets", []))},
        {"字段": "疾病背景", "内容": "; ".join(result.get("disease_context", []))},
        {"字段": "候选化合物", "内容": "; ".join(result.get("compounds", []))},
        {"字段": "关键残基", "内容": "; ".join(result.get("key_residues", []))},
    ])

    activity = pd.DataFrame(result.get("activity_data", []))
    design = pd.DataFrame({"design_suggestions": result.get("design_suggestions", [])})
    admet = pd.DataFrame({"admet_information": result.get("admet_information", [])})
    docking = pd.DataFrame({"docking_information": result.get("docking_information", [])})
    mechanism = pd.DataFrame({"mechanism_information": result.get("mechanism_information", [])})

    return {
        "overview": overview,
        "activity": activity,
        "design": design,
        "admet": admet,
        "docking": docking,
        "mechanism": mechanism
    }


def generate_markdown_summary(result: Dict[str, Any]) -> str:
    def bullet_list(items):
        if not items:
            return "- 未识别到明确内容"
        return "\n".join([f"- {item}" for item in items])

    activity_lines = []
    for item in result.get("activity_data", []):
        activity_lines.append(
            f"- {item.get('activity_type', '')}: {item.get('value', '')} {item.get('unit', '')}，匹配文本：{item.get('matched_text', item.get('evidence', ''))}"
        )

    md = f"""# 文献知识提取结果

## 一、研究主题

{result.get("research_topic", "")}

## 二、识别到的靶点

{bullet_list(result.get("targets", []))}

## 三、疾病或应用背景

{bullet_list(result.get("disease_context", []))}

## 四、候选化合物

{bullet_list(result.get("compounds", []))}

## 五、活性数据

{chr(10).join(activity_lines) if activity_lines else "- 未识别到明确活性数据"}

## 六、ADMET 信息

{bullet_list(result.get("admet_information", []))}

## 七、分子对接与关键残基

### 对接相关句子
{bullet_list(result.get("docking_information", []))}

### 关键残基
{bullet_list(result.get("key_residues", []))}

## 八、机制信息

{bullet_list(result.get("mechanism_information", []))}

## 九、分子设计启发

{bullet_list(result.get("design_suggestions", []))}
"""
    return md


def save_extraction_outputs(result: Dict[str, Any], prefix: str = "knowledge_extraction") -> Dict[str, str]:
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / f"{prefix}_result.json"
    md_path = RESULTS_DIR / f"{prefix}_summary.md"
    csv_path = RESULTS_DIR / f"{prefix}_overview.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    md_text = generate_markdown_summary(result)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    tables = result_to_tables(result)
    tables["overview"].to_csv(csv_path, index=False, encoding="utf-8-sig")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "csv": str(csv_path)
    }


def save_literature_outputs(search_result: Dict[str, Any], prefix: str = "literature_search") -> Dict[str, str]:
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / f"{prefix}_result.json"
    csv_path = RESULTS_DIR / f"{prefix}_articles.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(search_result, f, ensure_ascii=False, indent=2)

    articles_df = pd.DataFrame(search_result.get("articles", []))
    articles_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return {
        "json": str(json_path),
        "csv": str(csv_path)
    }


def call_openai_extractor(text: str, title: str, api_key: str, model: str = "gpt-4.1-mini") -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as e:
        raise ImportError("未安装 openai 包。请先运行：pip install openai") from e

    client = OpenAI(api_key=api_key)

    prompt = f"""
你是一个计算机辅助药物设计 CADD 文献知识提取助手。
请从给定文献标题和正文中提取结构化信息。重点服务于 QSAR、ADMET、分子对接、分子设计和报告生成。

请只输出 JSON，不要输出 Markdown。

JSON 字段要求：
{{
  "mode": "LLM提取",
  "title": "...",
  "research_topic": "...",
  "disease_context": ["..."],
  "targets": ["..."],
  "compounds": ["..."],
  "activity_data": [
    {{"compound": "", "target": "", "activity_type": "", "value": "", "unit": "", "evidence": ""}}
  ],
  "admet_information": ["..."],
  "docking_information": ["..."],
  "key_residues": ["..."],
  "mechanism_information": ["..."],
  "design_suggestions": ["..."],
  "summary": "..."
}}

标题：
{title}

正文：
{text[:18000]}
"""

    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2
    )

    output_text = getattr(response, "output_text", None)
    if not output_text:
        try:
            output_text = response.output[0].content[0].text
        except Exception:
            output_text = str(response)

    cleaned = output_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        result = json.loads(cleaned)
    except Exception:
        result = {
            "mode": "LLM提取",
            "title": title,
            "research_topic": title,
            "disease_context": [],
            "targets": [],
            "compounds": [],
            "activity_data": [],
            "admet_information": [],
            "docking_information": [],
            "key_residues": [],
            "mechanism_information": [],
            "design_suggestions": [],
            "summary": cleaned
        }

    required = [
        "mode", "title", "research_topic", "disease_context", "targets",
        "compounds", "activity_data", "admet_information", "docking_information",
        "key_residues", "mechanism_information", "design_suggestions", "summary"
    ]
    for key in required:
        result.setdefault(key, [] if key not in ["mode", "title", "research_topic", "summary"] else "")

    result["mode"] = "LLM提取"
    result["title"] = title

    return result


def make_demo_text() -> Dict[str, str]:
    title = "Discovery of EGFR kinase inhibitors with improved docking interactions and ADMET profiles"
    text = """
Epidermal growth factor receptor EGFR is an important therapeutic target in non-small cell lung cancer.
Several quinazoline derivatives were designed as EGFR tyrosine kinase inhibitors.
Compound A showed potent inhibitory activity with an IC50 of 18 nM against EGFR, while compound B showed an IC50 of 95 nM.
Molecular docking suggested that compound A formed a hydrogen bond with MET793 and hydrophobic interactions with LEU718 and VAL726 in the ATP-binding pocket.
The binding affinity of compound A was estimated as -9.6 kcal/mol.
ADMET evaluation indicated acceptable solubility and moderate CYP3A4 inhibition risk.
The quinazoline core may be retained, while polar substituents can be optimized to improve solubility and reduce toxicity.
"""
    return {"title": title, "text": normalize_text(text)}
