"""
Deterministic, anti-hallucination gap-fill for ILSA article JSON (Excel-ready).

Uses only text already present in the record (metadata + data fields). No PDF
re-read and no external knowledge. Builds on GPTExtractor._sanitize and
_post_process_model, then applies Tier A/B rules from the gap-fill spec.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.extractors.gpt_extractor import GPTExtractor, _title_from_file_name
from src.schemas.models import ILSAArticleMetadata, validate_public_article_json

_ILLEGAL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_YEAR_IN_TEXT = re.compile(r"\((\d{4})\)|\b(19\d{2}|20\d{2})\b")
_WEIGHT_VAR_RE = re.compile(
    r"\b(W_FSTUWT|TOTWGT|WGT|STUWT|FINWT|HOUWT|SENWT)\b",
    re.IGNORECASE,
)
_N_STUDENTS_RE = re.compile(
    r"(?:\bn\s*[=:]\s*|sample of\s+|total of\s+)(\d{1,3}(?:[,\s]\d{3})*)\s*(?:students|participants|learners|pupils)?",
    re.IGNORECASE,
)

_INVALID_STR = frozenset({
    "not_reported", "not_applicable", "n/a", "na", "unknown", "", "null", "none",
})


def _strip_control_chars(value: Any) -> Any:
    if isinstance(value, str):
        return _ILLEGAL_CHARS.sub("", value)
    if isinstance(value, list):
        return [_strip_control_chars(v) for v in value]
    if isinstance(value, dict):
        return {k: _strip_control_chars(v) for k, v in value.items()}
    return value


def _record_text_blob(meta: dict[str, Any], data: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("title", "file_name", "doi", "venue"):
        v = meta.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    if isinstance(meta.get("authors"), list):
        parts.extend(str(a) for a in meta["authors"] if a)
    for key in (
        "outcome_summary",
        "null_fields_interpretation",
        "handling_not_reported_explanation",
    ):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    sd = data.get("survey_design") or {}
    if isinstance(sd, dict):
        wfi = sd.get("weight_fields_interpretation")
        if isinstance(wfi, str):
            parts.append(wfi)
    sample = data.get("sample_details") or {}
    if isinstance(sample, dict):
        sfc = sample.get("sample_filtering_criteria")
        if isinstance(sfc, str):
            parts.append(sfc)
    return " ".join(parts).lower()


def _year_from_strings(*texts: str | None) -> int | None:
    for text in texts:
        if not text:
            continue
        for m in _YEAR_IN_TEXT.finditer(text):
            y = m.group(1) or m.group(2)
            if y:
                year = int(y)
                if 1990 <= year <= 2035:
                    return year
    return None


def _infer_weights_bools(blob: str) -> tuple[bool | None, bool | None, str | None]:
    student: bool | None = None
    replicate: bool | None = None
    weight_name: str | None = None

    m = _WEIGHT_VAR_RE.search(blob)
    if m:
        weight_name = m.group(1).upper()

    if any(
        p in blob
        for p in (
            "w_fstuwt",
            "totwgt",
            "survey weight",
            "sampling weight",
            "weighted estimate",
            "used weights",
            "applied weights",
            "weight variable",
        )
    ):
        student = True
    if any(
        p in blob
        for p in (
            "weights ignored",
            "without weights",
            "unweighted",
            "did not use weight",
            "no weight",
            "weights were not used",
        )
    ):
        student = False

    if any(p in blob for p in ("brr", "fay", "jackknife", "replicate weight")):
        replicate = True
    if "replicate weight" in blob and "not" in blob and "use" in blob:
        replicate = False

    return student, replicate, weight_name


def _parse_total_students(blob: str) -> int | None:
    best: int | None = None
    for m in _N_STUDENTS_RE.finditer(blob):
        raw = m.group(1).replace(",", "").replace(" ", "")
        try:
            n = int(raw)
        except ValueError:
            continue
        if n > 0 and (best is None or n > best):
            best = n
    return best


def _ensure_null_fields_interpretation(meta: dict[str, Any], data: dict[str, Any]) -> None:
    critical_nulls = 0
    if not meta.get("title"):
        critical_nulls += 1
    if meta.get("year") is None:
        critical_nulls += 1
    if not meta.get("doi"):
        critical_nulls += 1
    sample = data.get("sample_details") or {}
    if isinstance(sample, dict) and sample.get("total_students") is None:
        critical_nulls += 1
    ml = data.get("ml_techniques") or {}
    if isinstance(ml, dict) and not ml.get("primary") and not ml.get("all_techniques"):
        critical_nulls += 1

    existing = data.get("null_fields_interpretation")
    if critical_nulls <= 3:
        return
    if isinstance(existing, str) and len(existing.strip()) >= 40:
        return

    gaps: list[str] = []
    if not meta.get("title"):
        gaps.append("title was not present in the extraction record")
    if meta.get("year") is None:
        gaps.append("publication year could not be inferred from file_name or title")
    if not meta.get("doi"):
        gaps.append("DOI was not found in any field of this record")
    sample = data.get("sample_details") or {}
    if isinstance(sample, dict) and sample.get("total_students") is None:
        gaps.append("total sample size was not explicitly stated in outcome_summary or sample text")
    ml = data.get("ml_techniques") or {}
    if isinstance(ml, dict) and not ml.get("primary"):
        gaps.append("primary ML model was not identified in ml_techniques or outcome_summary")

    data["null_fields_interpretation"] = (
        "Several bibliographic or analytic fields remain null after gap-fill because "
        + "; ".join(gaps)
        + ". Values were not invented beyond what appears in this JSON record."
    )


def _patch_main_findings_metrics(data: dict[str, Any]) -> None:
    findings = data.get("main_findings")
    if not isinstance(findings, list):
        data["main_findings"] = []
        return
    for item in findings:
        if not isinstance(item, dict):
            continue
        pm = item.get("performance_metrics")
        if not isinstance(pm, str) or not pm.strip() or pm.strip().lower() in _INVALID_STR:
            item["performance_metrics"] = "Not reported"
        for key in ("dataset_used", "target_variable", "standardized_conclusion"):
            v = item.get(key)
            if isinstance(v, str):
                item[key] = _ILLEGAL_CHARS.sub("", v).strip()


def _expand_short_weight_interpretation(data: dict[str, Any], blob: str) -> None:
    sd = data.get("survey_design")
    if not isinstance(sd, dict):
        return
    wfi = sd.get("weight_fields_interpretation")
    if isinstance(wfi, str) and len(wfi.strip()) >= 80:
        return
    outcome = data.get("outcome_summary")
    if not isinstance(outcome, str) or len(outcome.strip()) < 40:
        return
    extra = outcome.strip()[:400]
    base = (wfi or "").strip()
    if base:
        sd["weight_fields_interpretation"] = f"{base} Additional context from outcome_summary: {extra}"
    else:
        sd["weight_fields_interpretation"] = (
            f"Weighting details were sparse in survey_design; outcome_summary states: {extra}"
        )


def apply_excel_gap_fill_dict(parsed: dict[str, Any]) -> dict[str, Any]:
    """Run sanitize → validate → post-process → deterministic gap-fill."""
    parsed = _strip_control_chars(parsed)
    sanitized = GPTExtractor._sanitize(parsed)
    model = validate_public_article_json(sanitized)
    GPTExtractor._post_process_model(model)

    meta = model.metadata.model_dump(mode="python")
    data = model.data.model_dump(mode="python")

    blob = _record_text_blob(meta, data)

    if meta.get("year") is None:
        y = _year_from_strings(meta.get("file_name"), meta.get("title"))
        if y is not None:
            meta["year"] = y

    if not meta.get("title") and meta.get("file_name"):
        backfill = _title_from_file_name(meta["file_name"])
        if backfill:
            meta["title"] = backfill

    sd = data.get("survey_design") or {}
    if isinstance(sd, dict):
        sw, rw, wn = _infer_weights_bools(blob)
        if sd.get("student_weights_used") is None and sw is not None:
            sd["student_weights_used"] = sw
        if sd.get("replicate_weights_used") is None and rw is not None:
            sd["replicate_weights_used"] = rw
        if not sd.get("weight_variable_name") and wn:
            sd["weight_variable_name"] = wn
        data["survey_design"] = sd

    sample = data.get("sample_details") or {}
    if isinstance(sample, dict) and sample.get("total_students") is None:
        n = _parse_total_students(blob)
        if n is not None:
            sample["total_students"] = n
        data["sample_details"] = sample

    _expand_short_weight_interpretation(data, blob)
    _patch_main_findings_metrics(data)
    _ensure_null_fields_interpretation(meta, data)

    out = {"metadata": meta, "data": data}
    out = _strip_control_chars(out)
    final = validate_public_article_json(out)
    GPTExtractor._post_process_model(final)
    return final.model_dump(mode="json")


def enrich_article_dict(raw: dict[str, Any]) -> dict[str, Any]:
    return apply_excel_gap_fill_dict(raw)


def enrich_article_json_file(path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = enrich_article_dict(raw)
    if not dry_run:
        path.write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return out
