"""Shared rules for when main_findings must be non-empty (no OpenAI dependency)."""

from __future__ import annotations

import re

_NON_EMPIRICAL_SOURCE_CATEGORIES = frozenset({
    "review_article", "methodology_paper", "technical_report",
})

_REPORT_SCOPE_KEYWORDS = re.compile(
    r"\b(?:framework|user\s+guide|technical\s+report|implementation\s+manual|"
    r"encyclopedia|assessment\s+design|sampling\s+manual|international\s+database|"
    r"codebook|instrument\s+design|scaling\s+methodology|quality\s+assurance|"
    r"assessment\s+framework|technical\s+standards)\b",
    re.IGNORECASE,
)

_REPORT_DOCUMENT_KEYWORDS = re.compile(
    r"\b(?:national\s+report|international\s+report|regional\s+report|"
    r"european\s+report|asian\s+report|latin\s+american|initial\s+findings|"
    r"highlights|compass\s+brief|policy\s+brief|methods\s+and\s+procedures|"
    r"volume\s+\d+|encyclopedia|user\s+guide|idb\s+user)\b",
    re.IGNORECASE,
)

_REPORT_FILENAME_TOKENS = (
    "national report",
    "national_report",
    "international report",
    "international_report",
    "framework",
    "user guide",
    "user_guide",
    "technical report",
    "technical_report",
    "encyclopedia",
    "methods and procedures",
    "methods_and_procedures",
    "idb_user",
    "compass brief",
    "policy brief",
    "initial findings",
    "highlights",
)


def _coerce_outcome_text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


_PIPELINE_FAILURE_PREFIXES = ("EXTRACTION_FAILED", "__EXTRACTION_FAILED__:")


def substantive_outcome_summary(data: dict) -> bool:
    text = _coerce_outcome_text(data.get("outcome_summary"))
    if not text or any(text.startswith(p) for p in _PIPELINE_FAILURE_PREFIXES):
        return False
    if text.startswith("No narrative outcome summary"):
        return False
    return len(text) >= 20


def has_empirical_ml_signals(data: dict) -> bool:
    ml = data.get("ml_techniques") if isinstance(data.get("ml_techniques"), dict) else {}
    if ml.get("primary"):
        return True
    techniques = ml.get("all_techniques")
    if isinstance(techniques, list) and techniques:
        return True
    if data.get("confounders_identified"):
        return True
    sd = data.get("sample_details") if isinstance(data.get("sample_details"), dict) else {}
    if sd.get("total_students"):
        return True
    if sd.get("countries"):
        return True
    return False


def _report_context_blob(data: dict, metadata: dict | None) -> str:
    parts: list[str] = []
    meta = metadata or {}
    for key in ("title", "file_name"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    for key in ("outcome_summary", "null_fields_interpretation"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    survey = data.get("survey_design")
    if isinstance(survey, dict):
        wfi = survey.get("weight_fields_interpretation")
        if isinstance(wfi, str) and wfi.strip():
            parts.append(wfi.strip())
    return "\n".join(parts)


def _filename_signals_official_report(meta: dict) -> bool:
    fn = str(meta.get("file_name") or "").lower()
    title = str(meta.get("title") or "").lower()
    combined = f"{fn} {title}"
    return any(tok in combined for tok in _REPORT_FILENAME_TOKENS)


def _finding_has_ml_metrics(finding: dict) -> bool:
    """True when a main_findings row reports trained-model performance (not descriptive tables)."""
    metrics = str(finding.get("performance_metrics") or "").lower()
    target = str(finding.get("target_variable") or "").lower()
    if any(tok in metrics for tok in ("accuracy", "f1", "auc", "rmse", "r²", "r2", "mae", "kappa")):
        return True
    if any(tok in target for tok in ("random forest", "xgboost", "neural", "logistic", "svm")):
        return True
    return False


def is_official_report_document(data: dict, metadata: dict | None = None) -> bool:
    """IEA/OECD manuals, frameworks, and national reports: empty main_findings."""
    meta = metadata or {}
    ml = data.get("ml_techniques") if isinstance(data.get("ml_techniques"), dict) else {}
    empty_ml = not ml.get("primary") and not (
        isinstance(ml.get("all_techniques"), list) and ml.get("all_techniques")
    )

    pub = str(meta.get("publication_type") or "").lower()
    sc = str(meta.get("source_category") or "").lower()

    # Empirical exception: report-type docs that actually train/evaluate ML models.
    if not empty_ml and pub == "report":
        mf = data.get("main_findings") if isinstance(data.get("main_findings"), list) else []
        if mf and any(isinstance(f, dict) and _finding_has_ml_metrics(f) for f in mf):
            return False
        if data.get("research_design_type") == "predictive" and ml.get("primary"):
            return False

    if not empty_ml:
        return False

    sd = data.get("sample_details") if isinstance(data.get("sample_details"), dict) else {}
    no_empirical_sample = not sd.get("total_students") and not sd.get("countries")
    blob = _report_context_blob(data, meta)
    has_scope_keywords = bool(
        _REPORT_SCOPE_KEYWORDS.search(blob) or _REPORT_DOCUMENT_KEYWORDS.search(blob)
    )

    is_report_meta = (
        pub == "report"
        or "report" in pub
        or sc in ("technical_report", "methodology_paper")
    )
    fn_report = _filename_signals_official_report(meta)

    # National/technical IEA/OECD reports: no ML primary → knowledge extraction only.
    # Journal/conference methodology papers are empirical reviews, not official manuals.
    if pub == "report" or sc == "technical_report":
        return True
    if sc == "methodology_paper" and pub in ("report", "book_chapter", None, ""):
        return True
    if sc == "methodology_paper" and pub in ("journal", "conference", "preprint", "thesis"):
        return False
    if pub == "book_chapter" and (has_scope_keywords or fn_report):
        return True
    if empty_ml and is_report_meta and (has_scope_keywords or fn_report):
        return True
    if empty_ml and pub == "report" and fn_report:
        return True
    if (
        data.get("research_design_type") == "exploratory"
        and no_empirical_sample
        and (has_scope_keywords or is_report_meta or fn_report)
    ):
        return True
    return False


def article_requires_main_findings(data: dict, metadata: dict | None = None) -> bool:
    """True when outcome_summary or empirical signals imply ≥1 StructuredFinding."""
    if is_official_report_document(data, metadata):
        return False
    if substantive_outcome_summary(data):
        return True
    if has_empirical_ml_signals(data):
        return True
    rd = data.get("research_design_type")
    if rd in ("predictive", "causal_observational", "causal_experimental"):
        sc = (metadata or {}).get("source_category")
        if sc not in _NON_EMPIRICAL_SOURCE_CATEGORIES:
            return True
    return False


_VALID_PUBLICATION_TYPES = frozenset({
    "journal", "conference", "book_chapter", "preprint", "report", "thesis",
})
_VALID_SOURCE_CATEGORIES = frozenset({
    "technical_report", "review_article", "methodology_paper", "peer_reviewed_research",
})
_VALID_RESEARCH_DESIGN_TYPES = frozenset({
    "predictive", "causal_observational", "causal_experimental", "exploratory",
})
_VALID_PV_HANDLING = frozenset({
    "rubin_rules", "single_pv", "average_pv", "all_pv",
    "mitml", "wle", "irt_theta", "not_applicable", "not_reported",
})
_VALID_MD_HANDLING = frozenset({
    "listwise_deletion", "pairwise_deletion", "mean_imputation", "single_imputation",
    "knn_imputation", "multiple_imputation", "not_reported",
})

_DEFAULT_REPORT_HNRE = (
    "This document is an official IEA/OECD technical report or assessment framework. "
    "It describes sampling, scaling, and instrument design rather than student-level "
    "predictive modeling. Plausible values are therefore not applicable to the "
    "document's primary analytic purpose; any missing-data discussion refers to "
    "fieldwork or scaling procedures, not a single ML imputation strategy."
)

_DEFAULT_REPORT_NFI = (
    "This extraction targets an official assessment framework, user guide, or "
    "technical manual without an empirical ML study. Sample sizes, ML algorithms, "
    "and predictive findings are intentionally null or empty per schema rules for "
    "non-empirical IEA/OECD documentation."
)


def _empty_ml_block(data: dict) -> bool:
    ml = data.get("ml_techniques") if isinstance(data.get("ml_techniques"), dict) else {}
    return not ml.get("primary") and not (
        isinstance(ml.get("all_techniques"), list) and ml.get("all_techniques")
    )


def should_apply_report_literal_coercion(
    data: dict, metadata: dict | None = None,
) -> bool:
    """True when sanitize should force official-report literals (incl. LLM typos)."""
    if is_official_report_document(data, metadata):
        return True
    if not _empty_ml_block(data):
        return False
    meta = metadata or {}
    blob = _report_context_blob(data, meta)
    if not _REPORT_SCOPE_KEYWORDS.search(blob):
        return False
    pt = meta.get("publication_type")
    if isinstance(pt, str) and pt not in _VALID_PUBLICATION_TYPES:
        if "report" in pt.lower() or "framework" in pt.lower() or "manual" in pt.lower():
            return True
    sc = meta.get("source_category")
    if isinstance(sc, str) and sc not in _VALID_SOURCE_CATEGORIES:
        lowered = sc.lower()
        if any(tok in lowered for tok in ("report", "framework", "manual", "guide", "technical")):
            return True
    pv = data.get("plausible_values_handling")
    if isinstance(pv, str) and pv not in _VALID_PV_HANDLING:
        return True
    md = data.get("missing_data_handling")
    if isinstance(md, str) and md not in _VALID_MD_HANDLING:
        return True
    return False


def _coerce_report_literals(data: dict, metadata: dict | None = None) -> None:
    """Force schema-safe literals for IEA/OECD manuals; mutates data and metadata in place."""
    meta = metadata if isinstance(metadata, dict) else {}

    pt = meta.get("publication_type")
    if pt not in _VALID_PUBLICATION_TYPES:
        meta["publication_type"] = "report"

    sc = meta.get("source_category")
    if sc not in _VALID_SOURCE_CATEGORIES:
        normed = str(sc or "").lower().replace("-", "_").replace(" ", "_")
        if "method" in normed or "framework" in normed:
            meta["source_category"] = "methodology_paper"
        else:
            meta["source_category"] = "technical_report"

    data["plausible_values_handling"] = "not_applicable"
    data["missing_data_handling"] = "not_reported"

    rdt = data.get("research_design_type")
    if rdt is None:
        data["research_design_type"] = "exploratory"
    elif rdt not in _VALID_RESEARCH_DESIGN_TYPES:
        data["research_design_type"] = "exploratory"
    elif rdt in ("predictive", "causal_observational", "causal_experimental"):
        data["research_design_type"] = "exploratory"

    data["main_findings"] = []
    data["confounders_identified"] = []

    ml = data.get("ml_techniques")
    if not isinstance(ml, dict):
        ml = {}
        data["ml_techniques"] = ml
    ml["primary"] = None
    ml["all_techniques"] = []

    hnre = data.get("handling_not_reported_explanation")
    if not isinstance(hnre, str) or not hnre.strip():
        data["handling_not_reported_explanation"] = _DEFAULT_REPORT_HNRE

    nfi = data.get("null_fields_interpretation")
    if not isinstance(nfi, str) or not nfi.strip():
        data["null_fields_interpretation"] = _DEFAULT_REPORT_NFI

    survey = data.get("survey_design")
    if isinstance(survey, dict):
        for bool_key in ("student_weights_used", "replicate_weights_used"):
            if bool_key in survey and not isinstance(survey[bool_key], (bool, type(None))):
                survey[bool_key] = None
