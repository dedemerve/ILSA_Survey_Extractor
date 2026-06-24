"""
Controlled vocabulary and taxonomic coding for ILSA meta-analysis Excel export.

Rule-based only (no API, no JSON mutation). Aligns with Borenstein/Cooper-style
meta-analytic stratification and ILSA assessment domains.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

# ── Hierarchical document / study filters (Excel dropdown) ──
STUDY_EMPIRICAL_ML = "Empirical Study - Machine Learning"
STUDY_EMPIRICAL_TRADITIONAL = "Empirical Study - Traditional Statistics"
STUDY_TECHNICAL_FRAMEWORK = "Technical/Assessment Framework"
STUDY_DESCRIPTIVE_NATIONAL = "Descriptive National Report"
STUDY_UNCLASSIFIED = "Unclassified"

STUDY_FILTER_OPTIONS: tuple[str, ...] = (
    STUDY_EMPIRICAL_ML,
    STUDY_EMPIRICAL_TRADITIONAL,
    STUDY_TECHNICAL_FRAMEWORK,
    STUDY_DESCRIPTIVE_NATIONAL,
    STUDY_UNCLASSIFIED,
)

TARGET_DOMAIN_OPTIONS: tuple[str, ...] = (
    "Mathematics",
    "Science",
    "Reading",
    "Digital/Computer Literacy",
    "Civic Education",
    "Problem Solving",
    "Non-Cognitive / Process Output",
    "Composite / Multi-Domain",
    "Other / Unspecified",
    "N/A: Technical Report",
)

TARGET_DIMENSION_OPTIONS: tuple[str, ...] = (
    "Cognitive Achievement",
    "Process Data / Log Metrics",
    "Attitudinal / Affective",
    "Policy / System Outcome",
    "Methodological (no DV)",
    "Other",
)

ML_FAMILY_OPTIONS: tuple[str, ...] = (
    "Tree-Based / Ensemble Learning",
    "Generalized Linear Models (GLM)",
    "Deep Learning",
    "Clustering / Unsupervised Learning",
    "Traditional Psychometrics / Multilevel Modeling",
    "Other ML / Not Classified",
    "N/A: Technical Report",
    "Not Reported: Likely Traditional Methods",
)

PREDICTOR_LEVEL_OPTIONS: tuple[str, ...] = (
    "Student Level",
    "School/Teacher Level",
    "System/Country Level",
    "Unspecified",
    "N/A: Technical Report",
)

PREDICTOR_CATEGORY_OPTIONS: tuple[str, ...] = (
    "Student: Demographic",
    "Student: SES",
    "Student: Attitudinal/Behavioral",
    "Student: Prior Achievement",
    "Student: Process Data",
    "School/Teacher: Context",
    "School/Teacher: Practice",
    "System: Policy/Context",
    "Other",
    "N/A: Technical Report",
)

PV_FILTER_OPTIONS: tuple[str, ...] = (
    "Pooled PVs (Rubin Rules)",
    "Single PV Draw",
    "Average PVs",
    "All PVs Analyzed Separately",
    "WLE / IRT Theta",
    "Not Applicable (Framework)",
    "Not Reported",
    "Other",
)

MD_FILTER_OPTIONS: tuple[str, ...] = (
    "Listwise Deletion",
    "Pairwise Deletion",
    "Mean Imputation",
    "Single Imputation",
    "KNN Imputation",
    "Multiple Imputation",
    "Not Reported",
    "Other",
)

WEIGHTS_FILTER_OPTIONS: tuple[str, ...] = ("True", "False", "Unknown")

_SENTINEL_PREFIXES = ("n/a:", "not reported", "missing")

_NATIONAL_REPORT_MARKERS = (
    "national report",
    "international report",
    "initial findings",
    "highlights",
    "volume i",
    "volume ii",
    "volume iii",
    "compass brief",
    "policy brief",
)

# Journal text traps: must NOT trigger technical_report / Framework (Soru D).
_FALSE_POSITIVE_PHRASES: tuple[str, ...] = (
    "self-reported",
    "self reported",
    "technical efficiency",
    "technical (stem) track",
    "technical engineering problem",
)

# Whole-phrase signals for real IEA/OECD manuals (word-boundary safe).
_REPORT_SIGNAL_RE = re.compile(
    r"\b(?:technical\s+report|national\s+report|international\s+report|"
    r"assessment\s+framework|user\s+guide|methods\s+and\s+procedures|"
    r"encyclopedia|idb\s+user)\b",
    re.IGNORECASE,
)

_TECHNICAL_WORD_RE = re.compile(r"\btechnical\b", re.IGNORECASE)
_REPORT_WORD_RE = re.compile(r"\breport\b", re.IGNORECASE)

# Schema confounder category → (level, filter category)
_SCHEMA_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    "socioeconomic": ("Student Level", "Student: SES"),
    "demographic": ("Student Level", "Student: Demographic"),
    "student_attitude": ("Student Level", "Student: Attitudinal/Behavioral"),
    "student_behavior": ("Student Level", "Student: Attitudinal/Behavioral"),
    "prior_achievement": ("Student Level", "Student: Prior Achievement"),
    "process_data": ("Student Level", "Student: Process Data"),
    "teacher": ("School/Teacher Level", "School/Teacher: Practice"),
    "school": ("School/Teacher Level", "School/Teacher: Context"),
    "ict": ("Student Level", "Student: Attitudinal/Behavioral"),
    "curriculum": ("School/Teacher Level", "School/Teacher: Context"),
    "parent_home": ("Student Level", "Student: SES"),
    "peer_effects": ("School/Teacher Level", "School/Teacher: Context"),
    "system_level": ("System/Country Level", "System: Policy/Context"),
}

# (regex, target_domain, target_dimension) — first match wins
_TARGET_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"pv\d*\s*math|pv\d*math|mathem|numeracy|quantitative literacy", re.I), "Mathematics", "Cognitive Achievement"),
    (re.compile(r"pv\d*\s*scie|pv\d*scie|science achievement|scientific literacy", re.I), "Science", "Cognitive Achievement"),
    (re.compile(r"pv\d*\s*read|pv\d*read|reading literacy|reading achievement", re.I), "Reading", "Cognitive Achievement"),
    (re.compile(r"computer.?based|digital literacy|ict literacy|computational", re.I), "Digital/Computer Literacy", "Cognitive Achievement"),
    (re.compile(r"civic|citizenship|global competenc", re.I), "Civic Education", "Cognitive Achievement"),
    (re.compile(r"problem.?solv|ps task|creative thinking", re.I), "Problem Solving", "Cognitive Achievement"),
    (re.compile(r"response time|log.?file|process data|votat|time on task|click", re.I), "Non-Cognitive / Process Output", "Process Data / Log Metrics"),
    (re.compile(r"dropout|well.?being|anxiety|motivation|self.?efficacy|belonging|climate", re.I), "Non-Cognitive / Process Output", "Attitudinal / Affective"),
    (re.compile(r"resilien|policy|leadership|instructional time", re.I), "Non-Cognitive / Process Output", "Policy / System Outcome"),
    (re.compile(r"achievement|proficiency|literacy|score|performance", re.I), "Composite / Multi-Domain", "Cognitive Achievement"),
]

_ML_FAMILY_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"random forest|xgboost|gradient boost|lightgbm|catboost|decision tree|adaboost|bagging|ensemble|extra trees", re.I), "Tree-Based / Ensemble Learning"),
    (re.compile(r"logistic regression|linear regression|lasso|ridge|elastic net|glm|probit", re.I), "Generalized Linear Models (GLM)"),
    (re.compile(r"neural|deep learning|cnn|lstm|rnn|mlp|transformer", re.I), "Deep Learning"),
    (re.compile(r"k-means|kmeans|dbscan|cluster|latent class|latent profile|topic model", re.I), "Clustering / Unsupervised Learning"),
    (re.compile(r"\birt\b|mplus|hlm|sem|structural equation|multilevel|mixed effect|conquest|winsteps|tam\b|lavaan|plausible value|rubin", re.I), "Traditional Psychometrics / Multilevel Modeling"),
    (re.compile(r"svm|support vector|naive bayes|knn|k-nearest", re.I), "Other ML / Not Classified"),
]

_PV_LABEL_MAP: dict[str, str] = {
    "rubin_rules": "Pooled PVs (Rubin Rules)",
    "single_pv": "Single PV Draw",
    "average_pv": "Average PVs",
    "all_pv": "All PVs Analyzed Separately",
    "mitml": "Multiple Imputation",
    "wle": "WLE / IRT Theta",
    "irt_theta": "WLE / IRT Theta",
    "not_applicable": "Not Applicable (Framework)",
    "not_reported": "Not Reported",
}

_MD_LABEL_MAP: dict[str, str] = {
    "listwise_deletion": "Listwise Deletion",
    "pairwise_deletion": "Pairwise Deletion",
    "mean_imputation": "Mean Imputation",
    "single_imputation": "Single Imputation",
    "knn_imputation": "KNN Imputation",
    "multiple_imputation": "Multiple Imputation",
    "not_reported": "Not Reported",
}


def _norm_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _is_sentinel(value: Any) -> bool:
    text = _norm_text(value).lower()
    if not text:
        return True
    return any(text.startswith(p) for p in _SENTINEL_PREFIXES)


def _normalize_publication_type(value: Any) -> str:
    return _norm_text(value).lower().replace("-", "_").replace(" ", "_")


def _mask_false_positive_phrases(text: str) -> str:
    """Remove journal econometric phrases before report/technical matching (Soru D)."""
    masked = text.lower()
    for phrase in _FALSE_POSITIVE_PHRASES:
        masked = masked.replace(phrase, " ")
    return masked


def _is_journal_publication(row: pd.Series) -> bool:
    """Rule 1: journal articles are never Technical/Assessment Framework."""
    return _normalize_publication_type(row.get("publication_type")) == "journal"


def _is_conference_publication(row: pd.Series) -> bool:
    """Conference papers follow the same empirical-only branch as journals."""
    return _normalize_publication_type(row.get("publication_type")) == "conference"


def _is_empirical_channel(row: pd.Series) -> bool:
    """Publication channels that cannot be Framework (journal/conference)."""
    return _is_journal_publication(row) or _is_conference_publication(row)


def _classify_empirical_study_filter(row: pd.Series) -> str:
    """
    Soru B/C: journal/conference/methodology-paper → ML vs Traditional only.
    DEA/SFA 'technical efficiency' stays here (Traditional), never Framework.
    """
    if _has_substantive_ml(row):
        return STUDY_EMPIRICAL_ML
    return STUDY_EMPIRICAL_TRADITIONAL


def is_technical_report_row(row: pd.Series) -> bool:
    """
    document_class gate: true only for real report/framework documents.

    Soru A/D: journals never classified via 'report'/'technical' substring traps.
    """
    if _is_empirical_channel(row):
        return False

    pub = _normalize_publication_type(row.get("publication_type"))
    if pub == "report":
        return True

    sc = _norm_text(row.get("source_category")).lower()
    if sc == "technical_report":
        return True

    # Methodology papers in report/book form (meta-research / manuals), not journal tests.
    if sc == "methodology_paper" and pub in ("report", "book_chapter"):
        return True

    title_masked = _mask_false_positive_phrases(_norm_text(row.get("title")))
    sc_masked = _mask_false_positive_phrases(sc)
    combined = f"{title_masked} {sc_masked} {pub}"

    if _REPORT_SIGNAL_RE.search(combined):
        return True

    # Residual word-boundary checks (after masking self-reported / technical efficiency).
    if _TECHNICAL_WORD_RE.search(title_masked) and pub in ("report", "book_chapter", ""):
        return True
    if _REPORT_WORD_RE.search(title_masked) and "self" not in title_masked:
        return True

    return False


def assign_document_class(row: pd.Series) -> str:
    """Top-level document_class aligned with publication_type hierarchy (Soru A/D)."""
    if _is_empirical_channel(row):
        return "empirical_article"
    if is_technical_report_row(row):
        return "technical_report"
    text = _publication_text_for_class(row)
    if "peer_reviewed" in text or "article" in text:
        return "empirical_article"
    return "unclassified"


def _publication_text_for_class(row: pd.Series) -> str:
    return " ".join(
        _norm_text(row.get(k)).lower()
        for k in ("publication_type", "source_category", "title")
    )


def _has_substantive_ml(row: pd.Series) -> bool:
    for col in ("ml_techniques", "ml_primary", "ml_all_techniques"):
        text = _norm_text(row.get(col)).lower()
        if not text or _is_sentinel(text):
            continue
        if text in ("present",):
            continue
        # Traditional psychometrics-only → traditional stats branch
        if re.search(
            r"\b(irt|mplus|hlm|sem|structural equation|multilevel model)\b",
            text,
            re.I,
        ) and not re.search(
            r"random forest|xgboost|neural|svm|deep learning|classif",
            text,
            re.I,
        ):
            continue
        return True
    return False


def classify_study_filter(row: pd.Series) -> str:
    """
    Zero-hallucination study_filter_type (Soru A–C).

    Hierarchy:
      1) publication_type journal/conference → never Framework
      2) methodology_paper + journal → ML or Traditional (by ml_techniques)
      3) report / technical_report document_class → Framework or National Report
    """
    pub = _normalize_publication_type(row.get("publication_type"))
    sc = _norm_text(row.get("source_category")).lower()

    # Rule 1 & 3: all journal/conference articles are empirical branches only.
    if _is_empirical_channel(row):
        return _classify_empirical_study_filter(row)

    # Methodology paper in non-journal form without empirical channel → framework family.
    if sc == "methodology_paper":
        return STUDY_TECHNICAL_FRAMEWORK

    doc_class = assign_document_class(row)

    if doc_class == "empirical_article":
        return _classify_empirical_study_filter(row)

    if doc_class == "technical_report" or pub == "report":
        blob = " ".join(
            _norm_text(row.get(k))
            for k in ("title", "source_category", "outcome_summary")
        ).lower()
        if any(m in blob for m in _NATIONAL_REPORT_MARKERS):
            return STUDY_DESCRIPTIVE_NATIONAL
        return STUDY_TECHNICAL_FRAMEWORK

    if pub in ("thesis", "preprint", "book_chapter"):
        if is_technical_report_row(row):
            blob = _norm_text(row.get("title")).lower()
            if any(m in blob for m in _NATIONAL_REPORT_MARKERS):
                return STUDY_DESCRIPTIVE_NATIONAL
            return STUDY_TECHNICAL_FRAMEWORK
        return _classify_empirical_study_filter(row)

    return STUDY_UNCLASSIFIED


def map_target_domain_dimension(
    target_variable: Any,
    *,
    document_class: str = "",
    study_filter: str = "",
) -> tuple[str, str]:
    if study_filter in (STUDY_TECHNICAL_FRAMEWORK, STUDY_DESCRIPTIVE_NATIONAL):
        return "N/A: Technical Report", "Methodological (no DV)"
    if _is_sentinel(target_variable):
        return "Other / Unspecified", "Other"

    text = _norm_text(target_variable)
    for pattern, domain, dimension in _TARGET_RULES:
        if pattern.search(text):
            return domain, dimension
    return "Other / Unspecified", "Other"


def map_ml_family(ml_text: Any, *, study_filter: str = "") -> str:
    if study_filter in (STUDY_TECHNICAL_FRAMEWORK, STUDY_DESCRIPTIVE_NATIONAL):
        return "N/A: Technical Report"
    text = _norm_text(ml_text)
    if _is_sentinel(text):
        if study_filter == STUDY_EMPIRICAL_TRADITIONAL:
            return "Not Reported: Likely Traditional Methods"
        return "Not Reported: Likely Traditional Methods"
    for pattern, family in _ML_FAMILY_RULES:
        if pattern.search(text):
            return family
    return "Other ML / Not Classified"


def map_pv_filter(value: Any) -> str:
    key = _norm_text(value).lower().replace(" ", "_")
    if _is_sentinel(value):
        return "Not Reported"
    return _PV_LABEL_MAP.get(key, "Other")


def map_md_filter(value: Any) -> str:
    key = _norm_text(value).lower().replace(" ", "_")
    if _is_sentinel(value):
        return "Not Reported"
    return _MD_LABEL_MAP.get(key, "Other")


def map_weights_filter(value: Any) -> str:
    if value is True or _norm_text(value).upper() == "TRUE":
        return "True"
    if value is False or _norm_text(value).upper() == "FALSE":
        return "False"
    return "Unknown"


def categorize_predictor(
    variable_name: Any,
    schema_category: Any = None,
    *,
    study_filter: str = "",
) -> tuple[str, str]:
    if study_filter in (STUDY_TECHNICAL_FRAMEWORK, STUDY_DESCRIPTIVE_NATIONAL):
        return "N/A: Technical Report", "N/A: Technical Report"

    sc = _norm_text(schema_category).lower()
    if sc in _SCHEMA_CATEGORY_MAP:
        return _SCHEMA_CATEGORY_MAP[sc]

    text = _norm_text(variable_name).lower()
    if _is_sentinel(text):
        return "Unspecified", "Other"

    if any(w in text for w in ("escs", "hisei", "pared", "ses", "homepos", "wealth", "income", "books")):
        return "Student Level", "Student: SES"
    if any(w in text for w in ("gender", "sex", "immig", "age", "grade", "birth")):
        return "Student Level", "Student: Demographic"
    if any(w in text for w in ("prior", "achievement", "wle", "pv1", "pv2", "score")):
        return "Student Level", "Student: Prior Achievement"
    if any(w in text for w in ("time on task", "log", "process", "votat", "ict use", "click")):
        return "Student Level", "Student: Process Data"
    if any(w in text for w in ("teacher", "instruction", "pedagog")):
        return "School/Teacher Level", "School/Teacher: Practice"
    if any(w in text for w in ("school", "class size", "climate", "urban", "rural", "private", "public")):
        return "School/Teacher Level", "School/Teacher: Context"
    if any(w in text for w in ("gdp", "gini", "country", "policy", "system", "tracking")):
        return "System/Country Level", "System: Policy/Context"
    if any(w in text for w in ("motiv", "anxiety", "efficacy", "interest", "belong", "enjoy")):
        return "Student Level", "Student: Attitudinal/Behavioral"

    return "Student Level", "Student: Attitudinal/Behavioral"


def _map_predictor_string(predictors: Any, study_filter: str) -> str:
    if _is_sentinel(predictors):
        return ""
    parts = re.split(r"[,;|]", _norm_text(predictors))
    cats: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        _, cat = categorize_predictor(p, study_filter=study_filter)
        if cat not in cats:
            cats.append(cat)
    return "; ".join(cats)


def apply_academic_taxonomy(
    df_master: pd.DataFrame,
    df_findings: pd.DataFrame,
    df_confounders: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add controlled-vocabulary filter columns (no imputation of substantive values)."""
    master = df_master.copy()
    findings = df_findings.copy()
    confounders = df_confounders.copy()

    if not master.empty:
        # Recompute document_class with publication-first rules before study_filter (Soru A/D).
        master["document_class"] = master.apply(assign_document_class, axis=1)
        master["study_filter_type"] = master.apply(classify_study_filter, axis=1)
        ml_src = master["ml_techniques"] if "ml_techniques" in master.columns else master.get("ml_primary")
        master["ml_family"] = [
            map_ml_family(ml_src.iat[i] if ml_src is not None else "", study_filter=master["study_filter_type"].iat[i])
            for i in range(len(master))
        ]
        if "plausible_values_handling" in master.columns:
            master["pv_filter_label"] = master["plausible_values_handling"].map(map_pv_filter)
        if "missing_data_handling" in master.columns:
            master["md_filter_label"] = master["missing_data_handling"].map(map_md_filter)
        if "student_weights_used" in master.columns:
            master["weights_filter"] = master["student_weights_used"].map(map_weights_filter)

    lookup_cols = ["file_name", "study_filter_type", "document_class"]
    lookup = (
        master[lookup_cols].drop_duplicates(subset=["file_name"])
        if not master.empty and "file_name" in master.columns
        else pd.DataFrame(columns=lookup_cols)
    )

    if not findings.empty and "file_name" in findings.columns and not lookup.empty:
        for col in ("study_filter_type", "document_class"):
            if col in findings.columns:
                findings = findings.drop(columns=[col])
        findings = findings.merge(lookup, on="file_name", how="left")

        sf = findings["study_filter_type"] if "study_filter_type" in findings.columns else pd.Series([""] * len(findings))
        dc = findings["document_class"] if "document_class" in findings.columns else pd.Series([""] * len(findings))

        domains: list[str] = []
        dimensions: list[str] = []
        pred_cats: list[str] = []
        for i in range(len(findings)):
            dom, dim = map_target_domain_dimension(
                findings["target_variable"].iat[i] if "target_variable" in findings.columns else "",
                document_class=_norm_text(dc.iat[i] if i < len(dc) else ""),
                study_filter=_norm_text(sf.iat[i] if i < len(sf) else ""),
            )
            domains.append(dom)
            dimensions.append(dim)
            if "top_predictors" in findings.columns:
                pred_cats.append(
                    _map_predictor_string(
                        findings["top_predictors"].iat[i],
                        _norm_text(sf.iat[i] if i < len(sf) else ""),
                    )
                )
            else:
                pred_cats.append("")

        findings["target_domain"] = domains
        findings["target_dimension"] = dimensions
        findings["predictor_filter_categories"] = pred_cats

    if not confounders.empty and "file_name" in confounders.columns and not lookup.empty:
        for col in ("study_filter_type",):
            if col in confounders.columns:
                confounders = confounders.drop(columns=[col])
        confounders = confounders.merge(
            lookup[["file_name", "study_filter_type"]],
            on="file_name",
            how="left",
        )
        levels: list[str] = []
        categories: list[str] = []
        for i in range(len(confounders)):
            sf = _norm_text(
                confounders["study_filter_type"].iat[i]
                if "study_filter_type" in confounders.columns
                else ""
            )
            vn = confounders["variable_name"].iat[i] if "variable_name" in confounders.columns else ""
            sc = confounders["category"].iat[i] if "category" in confounders.columns else None
            lvl, cat = categorize_predictor(vn, sc, study_filter=sf)
            levels.append(lvl)
            categories.append(cat)
        confounders["predictor_level"] = levels
        confounders["predictor_category"] = categories

    return master, findings, confounders


def build_analysis_dashboard(
    df_master: pd.DataFrame,
    df_findings: pd.DataFrame,
    df_confounders: pd.DataFrame,
) -> pd.DataFrame:
    """Pivot-friendly summary for Excel sheet 0_Dashboard_Analysis_Control."""
    rows: list[dict[str, Any]] = [
        {"Section": "Overview", "Dimension": "Metric", "Value": "Count", "Notes": ""},
        {"Section": "Overview", "Dimension": "Unique articles (Master)", "Value": len(df_master), "Notes": ""},
        {"Section": "Overview", "Dimension": "Main findings rows", "Value": len(df_findings), "Notes": ""},
        {"Section": "Overview", "Dimension": "Confounder rows", "Value": len(df_confounders), "Notes": ""},
        {
            "Section": "Stage 2-3",
            "Dimension": "Pipeline status",
            "Value": "Not yet applied",
            "Notes": (
                "Terminology alignment (Stage 2) and RAG analytic agent (Stage 3) "
                "will be proposed in future work; this dashboard reflects Stage-1 "
                "controlled vocabulary only."
            ),
        },
    ]

    def _add_counts(section: str, series: pd.Series) -> None:
        if series is None or series.empty:
            return
        for label, count in series.value_counts().items():
            rows.append(
                {
                    "Section": section,
                    "Dimension": str(label),
                    "Value": int(count),
                    "Notes": "Use as Excel column filter / slicer",
                }
            )

    if not df_master.empty and "study_filter_type" in df_master.columns:
        _add_counts("Study filter (Master)", df_master["study_filter_type"])
    if not df_master.empty and "ml_family" in df_master.columns:
        _add_counts("ML family (Master)", df_master["ml_family"])
    if not df_findings.empty and "target_domain" in df_findings.columns:
        _add_counts("Target domain (Findings)", df_findings["target_domain"])
    if not df_confounders.empty and "predictor_category" in df_confounders.columns:
        _add_counts("Predictor category (Confounders)", df_confounders["predictor_category"])

    rows.append(
        {
            "Section": "Filter recipe",
            "Dimension": "Low-noise empirical ML slice",
            "Value": "Example",
            "Notes": (
                "study_filter_type = Empirical Study - Machine Learning; "
                "target_domain = Mathematics; ml_family = Tree-Based / Ensemble Learning"
            ),
        }
    )
    return pd.DataFrame(rows)


def audit_journal_technical_conflicts(df_master: pd.DataFrame) -> list[str]:
    """
    Rule 4: log journal + technical_report conflicts for manual review.
    Returns lines like '[CONFLICT FOUND]: file_name -> manual_check_required'.
    """
    if df_master.empty or "publication_type" not in df_master.columns:
        return []
    lines: list[str] = []
    for _, row in df_master.iterrows():
        pub = _normalize_publication_type(row.get("publication_type"))
        doc = _norm_text(row.get("document_class")).lower()
        if pub == "journal" and doc == "technical_report":
            fn = _norm_text(row.get("file_name")) or "(unknown)"
            lines.append(f"[CONFLICT FOUND]: {fn} -> manual_check_required")
    return lines


def write_classification_audit_log(
    df_master: pd.DataFrame,
    log_path: Path,
) -> int:
    """Append classification conflicts to audit_log.txt (Rule 4)."""
    lines = audit_journal_technical_conflicts(df_master)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Classification audit (journal vs technical_report)\n"
        "# Auto-generated by build_tabular_dataset.py — review any CONFLICT FOUND lines.\n"
    )
    body = "\n".join(lines) + ("\n" if lines else "")
    log_path.write_text(header + body, encoding="utf-8")
    return len(lines)


def taxonomy_validation_lists() -> dict[str, tuple[str, ...]]:
    """Fixed dropdown vocabularies for openpyxl data validation."""
    return {
        "study_filter_type": STUDY_FILTER_OPTIONS,
        "ml_family": ML_FAMILY_OPTIONS,
        "pv_filter_label": PV_FILTER_OPTIONS,
        "md_filter_label": MD_FILTER_OPTIONS,
        "weights_filter": WEIGHTS_FILTER_OPTIONS,
        "target_domain": TARGET_DOMAIN_OPTIONS,
        "target_dimension": TARGET_DIMENSION_OPTIONS,
        "predictor_level": PREDICTOR_LEVEL_OPTIONS,
        "predictor_category": PREDICTOR_CATEGORY_OPTIONS,
    }
