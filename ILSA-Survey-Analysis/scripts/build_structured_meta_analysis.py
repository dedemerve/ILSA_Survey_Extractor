#!/usr/bin/env python3
"""
Build structured meta-analysis workbook from on-disk ILSA article JSON.

Pipeline:
  I.   Resanitize (GPTExtractor._sanitize) + tabular rules (no hallucination)
  II.  Three relational sheets + Data_Quality_Report
  III. Validation metrics per article

Output: outputs/ILSA_Structured_Meta_Analysis.xlsx
Failures: outputs/failed_to_process.log
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.extractors.gpt_extractor import GPTExtractor
from src.schemas.findings_validation import (
    article_requires_main_findings,
    is_official_report_document,
    substantive_outcome_summary,
)
from src.schemas.models import validate_public_article_json

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_tabular_dataset import (  # noqa: E402
    _ILLEGAL_XLSX_CHARS,
    _format_countries,
    _join_list,
    _is_blank,
    discover_json_paths,
    load_and_dedupe_articles,
)

DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_XLSX = DEFAULT_OUTPUTS_DIR / "ILSA_Structured_Meta_Analysis.xlsx"
DEFAULT_FAIL_LOG = DEFAULT_OUTPUTS_DIR / "failed_to_process.log"

NOT_REPORTED = "Not Reported"
NOT_APPLICABLE_TR = "Not Applicable (Technical Report)"

# Canonical ML taxonomy (lowercase key → canonical label). No inference beyond string match.
_CANONICAL_ML: dict[str, str] = {
    "random forest": "Random Forest",
    "random forests": "Random Forest",
    "rf": "Random Forest",
    "xgboost": "XGBoost",
    "xgb": "XGBoost",
    "gradient boosting": "Gradient Boosting",
    "gbm": "Gradient Boosting",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "support vector machine": "Support Vector Machine",
    "svm": "Support Vector Machine",
    "logistic regression": "Logistic Regression",
    "linear regression": "Linear Regression",
    "ridge regression": "Ridge Regression",
    "lasso": "Lasso Regression",
    "elastic net": "Elastic Net",
    "neural network": "Neural Network",
    "deep neural network": "Neural Network",
    "multilayer perceptron": "Neural Network",
    "mlp": "Neural Network",
    "cnn": "Convolutional Neural Network",
    "rnn": "Recurrent Neural Network",
    "lstm": "LSTM",
    "decision tree": "Decision Tree",
    "decision trees": "Decision Tree",
    "k-nearest neighbors": "K-Nearest Neighbors",
    "knn": "K-Nearest Neighbors",
    "naive bayes": "Naive Bayes",
    "multilevel modeling": "Multilevel Modeling",
    "multilevel model": "Multilevel Modeling",
    "multilevel modelling": "Multilevel Modeling",
    "hierarchical linear modeling": "Multilevel Modeling",
    "hierarchical linear model": "Multilevel Modeling",
    "hlm": "Multilevel Modeling",
    "mixed effects": "Mixed Effects Model",
    "mixed-effects": "Mixed Effects Model",
    "latent class": "Latent Class Analysis",
    "latent profile": "Latent Profile Analysis",
    "cluster analysis": "Cluster Analysis",
    "k-means": "K-Means Clustering",
    "kmeans": "K-Means Clustering",
    "principal component": "Principal Component Analysis",
    "pca": "Principal Component Analysis",
    "structural equation": "Structural Equation Modeling",
    "sem": "Structural Equation Modeling",
    "boosting": "Boosting",
    "adaboost": "AdaBoost",
    "ensemble": "Ensemble Learning",
    "stacking": "Stacking Ensemble",
    "bagging": "Bagging",
}

_METRIC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:accuracy|acc)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?%?)", re.I), "Accuracy"),
    (re.compile(r"\bAUC\s*[:=]\s*([0-9.]+)", re.I), "AUC"),
    (re.compile(r"(?:R²|R2|r-squared)\s*[:=]\s*([0-9.]+)", re.I), "R²"),
    (re.compile(r"\bRMSE\s*[:=]\s*([0-9.]+)", re.I), "RMSE"),
    (re.compile(r"\bMAE\s*[:=]\s*([0-9.]+)", re.I), "MAE"),
    (re.compile(r"\bMSE\s*[:=]\s*([0-9.]+)", re.I), "MSE"),
    (re.compile(r"\bF1\s*[:=]\s*([0-9.]+)", re.I), "F1"),
    (re.compile(r"\bprecision\s*[:=]\s*([0-9.]+)", re.I), "Precision"),
    (re.compile(r"\brecall\s*[:=]\s*([0-9.]+)", re.I), "Recall"),
    (re.compile(r"\bkappa\s*[:=]\s*([0-9.]+)", re.I), "Kappa"),
]


def _excel_str(value: Any) -> Any:
    if isinstance(value, str):
        return _ILLEGAL_XLSX_CHARS.sub("", value)
    return value


def _is_technical_report(meta: dict[str, Any], data: dict[str, Any]) -> bool:
    pub = str(meta.get("publication_type") or "").lower()
    sc = str(meta.get("source_category") or "").lower()
    if pub == "report" or sc == "technical_report":
        return True
    return is_official_report_document(data, meta)


def _canonical_ml(name: str | None) -> str:
    if not name or not str(name).strip():
        return NOT_REPORTED
    raw = str(name).strip()
    key = raw.lower()
    if key in _CANONICAL_ML:
        return _CANONICAL_ML[key]
    for fragment, canonical in sorted(_CANONICAL_ML.items(), key=lambda x: -len(x[0])):
        if fragment in key:
            return canonical
    return raw


def _canonicalize_techniques(techniques: Any) -> tuple[str | None, str | None]:
    if not techniques:
        return None, None
    if isinstance(techniques, str):
        techniques = [techniques]
    if not isinstance(techniques, list):
        return NOT_REPORTED, None
    canon = []
    seen: set[str] = set()
    for t in techniques:
        if not t:
            continue
        c = _canonical_ml(str(t))
        if c not in seen:
            seen.add(c)
            canon.append(c)
    if not canon:
        return NOT_REPORTED, None
    return canon[0], ", ".join(canon)


def _extract_metric_scores(text: str | None) -> str:
    if not text or not str(text).strip():
        return NOT_REPORTED
    if str(text).strip().lower() in ("not reported", "n/a", "na"):
        return NOT_REPORTED
    found: list[str] = []
    seen: set[str] = set()
    for pattern, label in _METRIC_PATTERNS:
        for m in pattern.finditer(text):
            token = f"{label}={m.group(1)}"
            if token not in seen:
                seen.add(token)
                found.append(token)
    if found:
        return "; ".join(found)
    # No parseable score — keep short literal only if it looks like a bare number
    stripped = text.strip()[:120]
    if re.fullmatch(r"[0-9.]+%?", stripped):
        return stripped
    return NOT_REPORTED


def _resanitize_record(raw: dict[str, Any]) -> dict[str, Any]:
    sanitized = GPTExtractor._sanitize(raw)
    model = validate_public_article_json(sanitized)
    GPTExtractor._post_process_model(model)
    return model.model_dump(mode="json")


def _not_reported_if_blank(value: Any) -> str:
    """Master/metadata fields: never use Not Applicable; only Not Reported when missing."""
    if _is_blank(value):
        return NOT_REPORTED
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def _completeness_fields(meta: dict, data: dict, *, technical: bool) -> list[bool]:
    survey = data.get("survey_design") or {}
    sample = data.get("sample_details") or {}
    ml = data.get("ml_techniques") or {}
    fields: list[Any] = [
        meta.get("title"),
        meta.get("doi"),
        meta.get("year"),
        meta.get("publication_type"),
        data.get("research_design_type"),
        data.get("plausible_values_handling"),
        data.get("missing_data_handling"),
        survey.get("weight_fields_interpretation"),
        sample.get("sample_filtering_criteria"),
        ml.get("primary") or ml.get("all_techniques"),
    ]
    if not technical:
        fields.extend([
            data.get("main_findings"),
            data.get("confounders_identified"),
            data.get("outcome_summary"),
        ])
    return [not _is_blank(f) for f in fields]


def _is_empirical(meta: dict, data: dict) -> bool:
    if _is_technical_report(meta, data):
        return False
    sc = str(meta.get("source_category") or "")
    if sc in ("review_article", "methodology_paper") and not substantive_outcome_summary(data):
        ml = data.get("ml_techniques") or {}
        if not ml.get("primary") and not ml.get("all_techniques"):
            return False
    if article_requires_main_findings(data, meta):
        return True
    ml = data.get("ml_techniques") or {}
    if ml.get("primary") or ml.get("all_techniques"):
        return True
    if substantive_outcome_summary(data):
        return True
    return False


def process_articles(
    articles: list[dict[str, Any]],
    fail_log: Path,
) -> list[dict[str, Any]]:
    processed: list[dict[str, Any]] = []
    fail_lines: list[str] = []

    for article in articles:
        path = article.get("_json_path", "")
        raw = {"metadata": article.get("metadata") or {}, "data": article.get("data") or {}}
        try:
            record = _resanitize_record(raw)
            meta = record.get("metadata") or {}
            data = record.get("data") or {}
            technical = _is_technical_report(meta, data)
            processed.append({
                **article,
                "metadata": meta,
                "data": data,
                "_technical_report": technical,
            })
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            fail_lines.append(f"{path}\t{exc.__class__.__name__}: {exc}")
        except Exception as exc:
            fail_lines.append(f"{path}\t{exc.__class__.__name__}: {exc}\n{traceback.format_exc()}")

    fail_log.parent.mkdir(parents=True, exist_ok=True)
    fail_log.write_text("\n".join(fail_lines) + ("\n" if fail_lines else ""), encoding="utf-8")
    print(f"Processed {len(processed)}/{len(articles)} articles; failures logged: {len(fail_lines)}", flush=True)
    return processed


def build_tables(
    articles: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master_rows: list[dict[str, Any]] = []
    finding_rows: list[dict[str, Any]] = []
    confounder_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    for article in articles:
        meta = article.get("metadata") or {}
        data = article.get("data") or {}
        technical = article.get("_technical_report", False)
        file_name = meta.get("file_name")
        doi = meta.get("doi")
        survey = data.get("survey_design") or {}
        sample = data.get("sample_details") or {}
        ml = data.get("ml_techniques") or {}

        primary, all_ml = _canonicalize_techniques(ml.get("all_techniques"))
        if ml.get("primary"):
            primary = _canonical_ml(str(ml.get("primary")))

        master_rows.append({
            "file_name": file_name,
            "doi": doi if not _is_blank(doi) else NOT_REPORTED,
            "title": _not_reported_if_blank(meta.get("title")),
            "authors": _not_reported_if_blank(_join_list(meta.get("authors"))),
            "year": meta.get("year") if meta.get("year") is not None else NOT_REPORTED,
            "publication_type": _not_reported_if_blank(meta.get("publication_type")),
            "source_category": _not_reported_if_blank(meta.get("source_category")),
            "research_design_type": _not_reported_if_blank(data.get("research_design_type")),
            "student_weights_used": _not_reported_if_blank(survey.get("student_weights_used")),
            "replicate_weights_used": _not_reported_if_blank(survey.get("replicate_weights_used")),
            "weight_fields_interpretation": _not_reported_if_blank(
                survey.get("weight_fields_interpretation")
            ),
            "missing_data_handling": _not_reported_if_blank(data.get("missing_data_handling")),
            "plausible_values_handling": _not_reported_if_blank(data.get("plausible_values_handling")),
            "ml_techniques_primary": primary if primary else NOT_REPORTED,
            "all_ml_techniques": all_ml if all_ml else NOT_REPORTED,
            "total_students": sample.get("total_students")
            if sample.get("total_students") is not None
            else NOT_REPORTED,
            "countries_formatted": _format_countries(sample.get("countries")) or NOT_REPORTED,
            "sample_filtering_criteria": _not_reported_if_blank(
                sample.get("sample_filtering_criteria")
            ),
            "outcome_summary": _not_reported_if_blank(data.get("outcome_summary")),
            "corpus_source": article.get("_corpus"),
            "is_technical_report": technical,
        })

        filled = _completeness_fields(meta, data, technical=technical)
        completeness = round(sum(filled) / len(filled), 3) if filled else 0.0
        empirical = _is_empirical(meta, data)
        findings = data.get("main_findings") or []
        has_missing_findings = (
            empirical
            and not technical
            and (not isinstance(findings, list) or len(findings) == 0)
        )
        findings_applicability = (
            NOT_APPLICABLE_TR if technical else "Applicable"
        )

        quality_rows.append({
            "file_name": file_name,
            "doi": doi or NOT_REPORTED,
            "is_empirical": empirical,
            "is_technical_report": technical,
            "data_completeness_score": completeness,
            "has_missing_findings": has_missing_findings,
            "n_main_findings": 0 if technical else (len(findings) if isinstance(findings, list) else 0),
            "n_confounders": 0 if technical else len(data.get("confounders_identified") or []),
            "findings_applicability": findings_applicability,
            "confounders_applicability": NOT_APPLICABLE_TR if technical else "Applicable",
        })

        if technical:
            finding_rows.append({
                "file_name": file_name,
                "doi": doi or NOT_REPORTED,
                "dataset_used": NOT_APPLICABLE_TR,
                "target_variable": NOT_APPLICABLE_TR,
                "top_predictors": NOT_APPLICABLE_TR,
                "performance_metrics": NOT_APPLICABLE_TR,
                "standardized_conclusion": NOT_APPLICABLE_TR,
            })
            confounder_rows.append({
                "file_name": file_name,
                "doi": doi or NOT_REPORTED,
                "variable_name": NOT_APPLICABLE_TR,
                "variable_code": NOT_APPLICABLE_TR,
                "category": NOT_APPLICABLE_TR,
                "description": NOT_APPLICABLE_TR,
            })
            continue

        if not isinstance(findings, list) or not findings:
            finding_rows.append({
                "file_name": file_name,
                "doi": doi or NOT_REPORTED,
                "dataset_used": NOT_REPORTED,
                "target_variable": NOT_REPORTED,
                "top_predictors": NOT_REPORTED,
                "performance_metrics": NOT_REPORTED,
                "standardized_conclusion": NOT_REPORTED,
            })
        else:
            for f in findings:
                if not isinstance(f, dict):
                    continue
                finding_rows.append({
                    "file_name": file_name,
                    "doi": doi or NOT_REPORTED,
                    "dataset_used": f.get("dataset_used") or NOT_REPORTED,
                    "target_variable": f.get("target_variable") or NOT_REPORTED,
                    "top_predictors": _join_list(f.get("top_predictors")) or NOT_REPORTED,
                    "performance_metrics": _extract_metric_scores(f.get("performance_metrics")),
                    "standardized_conclusion": f.get("standardized_conclusion") or NOT_REPORTED,
                })

        confounders = data.get("confounders_identified") or []
        if not isinstance(confounders, list) or not confounders:
            confounder_rows.append({
                "file_name": file_name,
                "doi": doi or NOT_REPORTED,
                "variable_name": NOT_REPORTED,
                "variable_code": NOT_REPORTED,
                "category": NOT_REPORTED,
                "description": NOT_REPORTED,
            })
        else:
            for c in confounders:
                if not isinstance(c, dict):
                    continue
                confounder_rows.append({
                    "file_name": file_name,
                    "doi": doi or NOT_REPORTED,
                    "variable_name": c.get("variable_name") or NOT_REPORTED,
                    "variable_code": c.get("variable_code") or NOT_REPORTED,
                    "category": c.get("category") or NOT_REPORTED,
                    "description": c.get("description") or NOT_REPORTED,
                })

    def _df(rows: list[dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        cleaned = [{k: _excel_str(v) for k, v in r.items()} for r in rows]
        return pd.DataFrame(cleaned)

    return (
        _df(master_rows),
        _df(finding_rows),
        _df(confounder_rows),
        _df(quality_rows),
    )


def write_workbook(
    master: pd.DataFrame,
    findings: pd.DataFrame,
    confounders: pd.DataFrame,
    quality: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        master.to_excel(writer, sheet_name="Articles_Master", index=False)
        findings.to_excel(writer, sheet_name="Main_Findings", index=False)
        confounders.to_excel(writer, sheet_name="Confounders", index=False)
        quality.to_excel(writer, sheet_name="Data_Quality_Report", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--output-xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--fail-log", type=Path, default=DEFAULT_FAIL_LOG)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("-v", action="store_true")
    args = parser.parse_args()

    paths = discover_json_paths(args.outputs_dir.resolve())
    print(f"Discovered {len(paths)} JSON files", flush=True)
    articles, warnings = load_and_dedupe_articles(
        paths, args.outputs_dir.resolve(), dedupe=True, verbose=args.v, workers=args.workers,
    )
    for w in warnings:
        print(w, flush=True)

    processed = process_articles(articles, args.fail_log.resolve())
    master, findings, confounders, quality = build_tables(processed)
    write_workbook(master, findings, confounders, quality, args.output_xlsx.resolve())

    print(
        f"Wrote {args.output_xlsx}\n"
        f"  Articles_Master: {len(master)}\n"
        f"  Main_Findings: {len(findings)}\n"
        f"  Confounders: {len(confounders)}\n"
        f"  Data_Quality_Report: {len(quality)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
