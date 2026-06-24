import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from pydantic import ValidationError

from src.schemas import ILSAArticleMetadata
from src.schemas.findings_validation import (
    _coerce_report_literals,
    article_requires_main_findings,
    is_official_report_document,
    should_apply_report_literal_coercion,
    substantive_outcome_summary,
)
from src.schemas.models import Confounder, StructuredFinding, coerce_optional_bool

if TYPE_CHECKING:
    from src.extractors.pdf_processor import ProcessedPDF

logger = logging.getLogger(__name__)

CONFOUNDER_CATEGORIES = frozenset({
    "socioeconomic", "demographic", "student_attitude", "student_behavior",
    "teacher", "school", "ict", "curriculum", "parent_home", "process_data",
    "prior_achievement", "peer_effects", "system_level",
})

_INVALID_VARIABLE_CODES = frozenset({"n/a", "na", "null", "none", ""})
_INVALID_FIELD_STRINGS = frozenset({
    "not_reported", "not_applicable", "n/a", "na", "unknown", "",
})
_DEFAULT_SAMPLE_FILTERING = (
    "Used the full available sample for the specified countries. "
    "No additional inclusion or exclusion criteria were reported in the manuscript."
)

_ACTION_SEQUENCE_CODE = re.compile(r"^-?\d+_-?\d+_-?\d+$")
_MICRO_ACTION_CODES = frozenset({"start", "end", "reset", "click"})
_MICRO_CONFOUNDER_KEYWORDS = (
    "slider", "tf-idf", "tfidf", "word2vec", "word2vec", "embedding",
    "action triple", "action frequency", "n-gram", "ngram", "keystroke",
    "button click", "state transition", "cosine similarity", "token weight",
    "behavior weight", "feature vector", "action embedding", "tf-idf feature",
)

# Well-known ILSA construct codes → category (PISA, TIMSS, PIRLS, TALIS, ICILS, ICCS, PIAAC)
ILSA_CODE_TO_CATEGORY: dict[str, str] = {
    # 1. SOCIOECONOMIC & PARENT/HOME
    "ESCS": "socioeconomic",
    "HOMEPOS": "socioeconomic",
    "WEALTH": "socioeconomic",
    "HISEI": "socioeconomic",
    "HISCED": "socioeconomic",
    "MISCED": "socioeconomic",
    "FISCED": "socioeconomic",
    "HEDRES": "socioeconomic",
    "CULTPOSS": "socioeconomic",
    "PARED": "socioeconomic",
    "BSBGHER": "socioeconomic",
    "ASBGHER": "socioeconomic",
    "ASBGHRL": "socioeconomic",
    "EMOSUPS": "parent_home",
    "FAMSUPSL": "parent_home",
    "ASBGERL": "parent_home",
    # 2. DEMOGRAPHIC
    "ST004Q01TA": "demographic",
    "ST004D01T": "demographic",
    "ITSEX": "demographic",
    "IMMIG": "demographic",
    "LANGN": "demographic",
    "AGE": "demographic",
    "REPEAT": "demographic",
    "TQ-01": "demographic",
    "TQ-02": "demographic",
    # 3. STUDENT ATTITUDE
    "MATHEFF": "student_attitude",
    "SCIEEFF": "student_attitude",
    "ANXMAT": "student_attitude",
    "BELONG": "student_attitude",
    "BSBM16A": "student_attitude",
    "JOYREAD": "student_attitude",
    "MOTIV": "student_attitude",
    "PISADIFF": "student_attitude",
    "BSMJ": "student_attitude",
    "EUDMO": "student_attitude",
    "BSBG11A": "student_attitude",
    "BSBGSLS": "student_attitude",
    "BSBGSLB": "student_attitude",
    "BSBGSLC": "student_attitude",
    "BSBGSLP": "student_attitude",
    "BSBGSLE": "student_attitude",
    "BSBGSCS": "student_attitude",
    "BSBGSCB": "student_attitude",
    "BSBGSCC": "student_attitude",
    "BSBGSCP": "student_attitude",
    "BSBGSCE": "student_attitude",
    "BSBS22H": "student_attitude",
    "BSBB23H": "student_attitude",
    "BSBC33H": "student_attitude",
    "BSBP38H": "student_attitude",
    "BSBE28H": "student_attitude",
    "ASBGSMR": "student_attitude",
    "ASBGCRD": "student_attitude",
    "S_CIV": "student_attitude",
    # 4. STUDENT BEHAVIOR & COGNITION
    "METASPAM": "student_behavior",
    "METASUM": "student_behavior",
    "UNDREM": "student_behavior",
    "FAMCON": "student_behavior",
    "ABSENT": "student_behavior",
    "TRUANT": "student_behavior",
    "ST097Q01TA": "student_behavior",
    "BSBG09C": "student_behavior",
    "BSBS21": "student_behavior",
    "BSBB22": "student_behavior",
    "BSBC32": "student_behavior",
    "BSBP37": "student_behavior",
    "BSBE27": "student_behavior",
    "BTBS18B": "student_behavior",
    "LMINS": "student_behavior",
    "CILUSE": "student_behavior",
    # 5. TEACHER
    "ADINST": "teacher",
    "DIRINS": "teacher",
    "PERFEED": "teacher",
    "FEEDBACK": "teacher",
    "TEACHSUP": "teacher",
    "BSBGICS": "teacher",
    "BTBS18A": "teacher",
    "BTBS18CA": "teacher",
    "BTBS18CB": "teacher",
    "BTBS18CC": "teacher",
    "BTBS18CD": "teacher",
    "BTBS18CE": "teacher",
    "BTBG08A": "teacher",
    "BTBG09D": "teacher",
    "BTBG09E": "teacher",
    "BTBG12A": "teacher",
    "BTBG12B": "teacher",
    "BTBG12C": "teacher",
    "BTBG12D": "teacher",
    "BTBG12E": "teacher",
    "BTBG12F": "teacher",
    "BTBG12G": "teacher",
    "BTBSESI": "teacher",
    "TQ-03": "teacher",
    "TQ-04": "teacher",
    "TQ-06": "teacher",
    "TQ-08": "teacher",
    "TQ-11": "teacher",
    "TQ-16": "teacher",
    "TQ-51": "teacher",
    "TQ-53": "teacher",
    "TT3G01": "teacher",
    "TTPD": "teacher",
    "SEFF": "teacher",
    "TCHYRS": "teacher",
    # 6. SCHOOL
    "SCHLTYPE": "school",
    "STRATIO": "school",
    "SCHSIZE": "school",
    "PROPCAT": "school",
    "TOTAT": "school",
    "STUBEHA": "school",
    "TEACHBEHA": "school",
    "BCBG10A": "school",
    "BCBG07": "school",
    "BCBG13BA": "school",
    "BCBG13BB": "school",
    "BCBG13CA": "school",
    "BCBG13CC": "school",
    "BCBG14H": "school",
    "BCBG16B": "school",
    "BCBG16J": "school",
    "BCBG16K": "school",
    "TQ-50": "school",
    # 7. PEER EFFECTS / CLASSROOM CLIMATE
    "DISCLIMA": "peer_effects",
    "TCDISCLIMA": "peer_effects",
    "PERCOMP": "peer_effects",
    "BULLY": "peer_effects",
    # 8. ICT
    "ICTRES": "ict",
    "ST011Q04TA": "ict",
    "ENTUSE": "ict",
    "HOMESCH": "ict",
    "USESCH": "ict",
    "SOIAICT": "ict",
    "AUTICT": "ict",
    "BTBM20C": "ict",
    "TQ-52": "ict",
    "COMPEFF": "ict",
    "S_CIL": "ict",
    # 9. CURRICULUM & TIME ON TASK
    "SMINS": "curriculum",
    "MMINS": "curriculum",
    "TMINS": "curriculum",
    "ITCOURSE": "curriculum",
    "BTBS14": "curriculum",
    "BCBG06B": "curriculum",
    "TQ-13": "curriculum",
    "TQ-14": "curriculum",
    # 10. SYSTEM LEVEL
    "IDCNTRY": "system_level",
    "CNT": "system_level",
    "GDP": "system_level",
    "TRACKING": "system_level",
    # 11. PROCESS DATA & LOGS
    "VOTAT": "process_data",
    "sequence_length": "process_data",
    "time_on_task": "process_data",
    "n_actions": "process_data",
    # 12. PRIOR ACHIEVEMENT
    "PV1READ": "prior_achievement",
    "PV1MATH": "prior_achievement",
}


def _is_micro_process_confounder(name: str, code: str) -> bool:
    """Drop ML inputs / raw log micro-actions mistakenly listed as confounders."""
    name_lower = name.lower()
    code_str = code.strip()
    code_lower = code_str.lower()

    if code_lower == "votat" or "votat" in name_lower:
        return False
    if any(
        phrase in name_lower
        for phrase in (
            "total time", "time on task", "response time", "total action",
            "number of visits", "visits per", "sequence length",
        )
    ):
        return False
    if code_lower in ("sequence_length", "seq_length"):
        return False

    if _ACTION_SEQUENCE_CODE.match(code_str):
        return True
    if code_lower in _MICRO_ACTION_CODES:
        return True
    if code_lower.startswith(("tfidf_", "word2vec_", "w2v_")):
        return True
    if re.search(r"\d+_\d+_", code_str):
        return True
    if any(kw in name_lower or kw in code_lower for kw in _MICRO_CONFOUNDER_KEYWORDS):
        return True
    if re.search(r"slider\s*[+-]?\s*\d", name_lower):
        return True
    if "slider" in name_lower and re.search(r"\(\s*-?\d+_\d+", name_lower):
        return True
    if re.search(r"\b0_0_0\b", name_lower):
        return True
    if re.search(r"\d+_\d+_-?\d+", name_lower):
        return True
    return False


def _is_survey_weight_confounder(name: str, code: str) -> bool:
    """Drop sampling-weight variables mistakenly listed as predictors/confounders."""
    code_key = (code or "").strip().upper()
    if not code_key:
        return False
    if code_key.endswith("WGT") or code_key.startswith("W_"):
        return True
    if code_key in (
        "TOTWGT", "FSTUWT", "W_FSTUWT", "SCHWGT", "HOUWGT", "SENWGT",
        "MATWGT", "SCIWGT", "REAWGT",
    ):
        return True
    name_lower = name.lower()
    if any(
        phrase in name_lower
        for phrase in (
            "survey weight", "sampling weight", "student weight",
            "senate weight", "house weight", "replicate weight",
        )
    ):
        return True
    return False


def _title_from_file_name(file_name: str) -> Optional[str]:
    """Derive a catalog title from pipeline file_name when metadata.title is missing."""
    if not isinstance(file_name, str) or not file_name.strip():
        return None
    stem = file_name.strip()
    if stem.lower().endswith(".pdf"):
        stem = stem[:-4]
    stem = re.sub(r"^\d+\.\s*", "", stem).strip()
    match = re.match(r"^.+?\(\d{4}\)\.\s*(.+)$", stem)
    if match:
        title = match.group(1).strip()
        return title if len(title) >= 10 else None
    if re.match(r"^978\d{10,13}(?:-en)?$", stem, re.I):
        return f"OECD publication {stem}"
    if re.match(r"^[\da-f]{6,12}-en$", stem, re.I):
        return f"OECD document {stem}"
    return stem if len(stem) >= 15 else None


def _doi_backfill_blob(data: dict, meta: dict | None) -> str:
    """Concatenate JSON text fields that may contain a DOI (no PDF/API)."""
    parts: list[str] = []
    meta = meta or {}
    for key in ("title", "file_name", "venue"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    for key in (
        "outcome_summary",
        "null_fields_interpretation",
        "handling_not_reported_explanation",
    ):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    survey = data.get("survey_design")
    if isinstance(survey, dict):
        wfi = survey.get("weight_fields_interpretation")
        if isinstance(wfi, str) and wfi.strip():
            parts.append(wfi.strip())
    return "\n".join(parts)


def _slug_variable_code(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower()).strip("_")
    return slug[:48] if slug else "unnamed_variable"


_SENTENCE_NAME_STARTERS = (
    r"^the\s+",
    r"^teacher[- ]related\s+variables\s+such\s+as\s+(?:the\s+)?",
    r"^students?['']?\s+",
    r"^student['']?s?\s+",
)
_UNDREM_NAME_PATTERN = re.compile(
    r"\b(meta[- ]?cognit\w*|understanding|remembering)\b", re.I,
)
_INVENTED_METACOG_SLUGS = frozenset({
    "meta_cognition_understanding",
    "meta_cognition",
    "metacognition",
    "metacognition_understanding",
    "remembering",
    "understanding",
})


def _condense_variable_name(name: str) -> str:
    """Cap labels at 8 words; strip pasted sentence starters."""
    if not isinstance(name, str) or not name.strip():
        return name
    text = name.strip()
    if text.count("(") < text.count(")"):
        text = re.sub(r"\)+$", "", text).strip()
    if text.count("(") > text.count(")"):
        text += ")" * (text.count("(") - text.count(")"))
    for pat in _SENTENCE_NAME_STARTERS:
        text = re.sub(pat, "", text, flags=re.I).strip()
    words = text.split()
    if len(words) > 8:
        text = " ".join(words[:8])
    return text.strip() or name.strip()


def _looks_like_ilsa_code(label: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]{1,}", label.strip()))


def _is_invented_confounder_code(code: str) -> bool:
    """True for pseudo-ILSA / prose codes; false for official ILSA acronyms."""
    if not isinstance(code, str) or not code.strip():
        return False
    c = code.strip()
    key = c.upper()
    if key in ILSA_CODE_TO_CATEGORY:
        return False
    if _looks_like_ilsa_code(c):
        return False
    if re.search(r"[:;]|(?:\band\b)", c, re.I):
        return True
    if " " in c:
        return True
    if "/" in c:
        return True
    lower = c.lower()
    if lower in _INVENTED_METACOG_SLUGS:
        return True
    if re.search(r"meta[-_]?cognit", lower):
        return True
    return False


def _fix_invented_variable_code(code: str, name: str) -> str:
    """Map invented codes to ILSA slug or official construct."""
    condensed = _condense_variable_name(name)
    combined = f"{code} {condensed}"
    if _UNDREM_NAME_PATTERN.search(combined):
        return "UNDREM"
    paren = re.search(r"\(([A-Z][A-Z0-9_]{2,})\)", condensed) or re.search(
        r"\(([A-Z][A-Z0-9_]{2,})\)", code,
    )
    if paren:
        return paren.group(1)
    return _slug_variable_code(condensed)


def _collapse_undrem_duplicates(items: list[dict]) -> list[dict]:
    """Merge split UNDREM sub-parts into one row."""
    undrem = [c for c in items if c["variable_code"].upper() == "UNDREM"]
    rest = [c for c in items if c["variable_code"].upper() != "UNDREM"]
    if len(undrem) <= 1:
        return items
    return rest + [{
        "variable_code": "UNDREM",
        "variable_name": "Metacognition (understanding, remembering)",
        "category": "student_behavior",
    }]


def _clean_confounder_part(part: str) -> str:
    part = part.strip()
    part = re.sub(r"^(the|a|an)\s+", "", part, flags=re.I)
    part = re.sub(
        r"\s+(variable|variables|factor|factors|indicator|indices|index|score|scores)$",
        "",
        part,
        flags=re.I,
    )
    return part.strip()


def _split_confounder_label(text: str) -> list[str]:
    """Split combined labels ('Gender and Age', 'ESCS, HOMEPOS, and WEALTH')."""
    cleaned = re.sub(
        r"\s+(combined|together|respectively|etc\.?)\s*$",
        "",
        text.strip(),
        flags=re.I,
    )
    # Keep unified constructs as one row (e.g. "Economic and education indicators")
    if re.search(
        r"\b(indicators?|variables?|features|predictors|covariates|controls)\b",
        cleaned,
        flags=re.I,
    ) and "," not in cleaned:
        return [cleaned] if cleaned else []

    if not re.search(r"[,;]|\band\b|&", cleaned, flags=re.I):
        return [cleaned] if cleaned else []

    parts = re.split(r"\s*(?:,|;|\band\b|&)\s*", cleaned, flags=re.I)
    parts = [_clean_confounder_part(p) for p in parts if p and _clean_confounder_part(p)]
    return parts if len(parts) > 1 else ([cleaned] if cleaned else [])


def _coerce_confounder_category(
    cat: Optional[str], name: str, code: str,
) -> str:
    code_key = code.strip().upper()
    if code_key in ILSA_CODE_TO_CATEGORY:
        return ILSA_CODE_TO_CATEGORY[code_key]
    if cat in CONFOUNDER_CATEGORIES:
        return cat
    text = f"{name} {code}".lower()
    keyword_map = (
        ("process_data", (
            "tf-idf", "tfidf", "word2vec", "votat", "process data", "log file",
            "action sequence", "response time", "time-to-first", "click", "reset",
            "n-gram", "markov", "0_0_0", "sequence mining",
        )),
        ("system_level", (
            "gdp", "gini", "country-level", "national policy", "education expenditure",
            "oecd average", "tracking age", "macro", "economic indicator",
        )),
        ("peer_effects", (
            "peer", "bullying", "classroom climate", "disciplinary climate",
            "class-average", "class average", "class discipline",
        )),
        ("prior_achievement", (
            "prior", "previous score", "prior-year", "wle", "plausible value",
            "pv1", "pv2", "prior achievement", "prior math", "prior reading",
        )),
        ("socioeconomic", (
            "escs", "homepos", "wealth", "hisei", "parental education",
            "books at home", "misced", "fisced", "hisced", "socioeconomic",
        )),
        ("demographic", (
            "gender", "immigrant", "migration", "language at home", "age", "grade level",
        )),
        ("student_attitude", (
            "self-efficacy", "motivation", "anxiety", "enjoyment", "belonging",
            "self-concept", "interest", "confidence", "matheff", "anxmat",
        )),
        ("student_behavior", (
            "homework", "absenteeism", "study time", "learning time", "smins", "tmins",
            "reading habit", "workpay", "exerprac",
        )),
        ("teacher", (
            "teacher", "professional development", "instructional", "btbg", "btbm",
            "teaching practice",
        )),
        ("school", (
            "school type", "class size", "school resource", "library", "bcbg",
            "principal", "school climate",
        )),
        ("ict", ("ict", "computer", "digital", "internet", "technology", "ictres")),
        ("curriculum", (
            "curriculum", "instructional time", "content coverage", "famcon", "smins",
        )),
        ("parent_home", (
            "parent", "family support", "home environment", "emosups", "famsup",
        )),
    )
    for category, keywords in keyword_map:
        if any(kw in text for kw in keywords):
            return category
    return "student_behavior"


def _normalize_confounder_code(code: Optional[str], name: str) -> str:
    condensed = _condense_variable_name(name)
    if isinstance(code, str):
        cleaned = code.strip()
        if cleaned.lower() not in _INVALID_VARIABLE_CODES:
            if _is_invented_confounder_code(cleaned):
                return _fix_invented_variable_code(cleaned, condensed)
            return cleaned
    paren = re.search(r"\(([A-Z][A-Z0-9_]{2,})\)", condensed)
    if paren:
        return paren.group(1)
    return _slug_variable_code(condensed)


def _normalize_confounder_dict(entry: dict) -> Optional[dict]:
    name = entry.get("variable_name", "")
    if not isinstance(name, str) or not name.strip():
        return None
    name = _condense_variable_name(name.strip())
    code = _normalize_confounder_code(entry.get("variable_code"), name)
    if code.upper() == "UNDREM":
        name = "Metacognition (understanding, remembering)"
    cat = _coerce_confounder_category(entry.get("category"), name, code)
    return {
        "variable_code": code,
        "variable_name": name,
        "category": cat,
    }


def _expand_confounder_dict(entry: dict) -> list[dict]:
    """Split grouped/list confounders into separate normalized objects."""
    name = entry.get("variable_name", "")
    if not isinstance(name, str) or not name.strip():
        return []

    base_code = entry.get("variable_code")
    if isinstance(base_code, str) and base_code.strip().upper() == "UNDREM":
        normalized = _normalize_confounder_dict(entry)
        return [normalized] if normalized else []
    base_cat = entry.get("category")
    parts = _split_confounder_label(name.strip())
    if not parts:
        return []

    expanded: list[dict] = []
    for part in parts:
        if _looks_like_ilsa_code(part):
            sub_name = part.upper()
            sub_code: Optional[str] = part.upper()
        else:
            sub_name = part
            sub_code = base_code if len(parts) == 1 else None

        normalized = _normalize_confounder_dict({
            "variable_code": sub_code,
            "variable_name": sub_name,
            "category": base_cat if len(parts) == 1 else None,
        })
        if normalized:
            expanded.append(normalized)
    return expanded


def _dedupe_confounders(items: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for item in items:
        key = (item["variable_code"].upper(), item["variable_name"].lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _normalize_confounders_list(
    raw: list,
    *,
    invalid_names: frozenset[str] = frozenset(),
) -> list[dict]:
    """Expand, normalize, and dedupe confounders from LLM or legacy JSON."""
    expanded: list[dict] = []
    for item in raw:
        if isinstance(item, dict):
            expanded.extend(_expand_confounder_dict(item))
        elif isinstance(item, str) and item.strip():
            expanded.extend(_expand_confounder_dict({
                "variable_code": None,
                "variable_name": item.strip(),
                "category": None,
            }))

    filtered = [
        c for c in expanded
        if c["variable_name"] not in invalid_names
        and not _is_micro_process_confounder(
            c.get("variable_name", ""),
            c.get("variable_code") or "",
        )
        and not _is_survey_weight_confounder(
            c.get("variable_name", ""),
            c.get("variable_code") or "",
        )
    ]
    return _dedupe_confounders(_collapse_undrem_duplicates(filtered))


_MEANING_CLAUSE_RE = re.compile(r"\bthis\s+indicates\b", re.IGNORECASE)
_GENERIC_WFI_MARKERS = (
    "no weighting information was explicitly reported",
    "the extraction could not determine the weighting",
    "the extraction could not determine",
)


def _derive_meaning_clause(effect_hint: str | None, conclusion: str) -> str:
    """Build policy/practice implication when the model omits 'This indicates that'."""
    for source in (effect_hint, conclusion):
        if not source or not isinstance(source, str):
            continue
        match = re.search(
            r"this\s+indicates\s+that\s+(.+?)(?:\.|$)",
            source,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip().rstrip(".")
    blob = (effect_hint or conclusion or "").lower()
    if any(k in blob for k in ("policy", "practice", "equity", "instruction", "curriculum")):
        tail = (effect_hint or conclusion or "").strip().rstrip(".")
        if len(tail) > 40:
            return tail[-220:]
    return (
        "these patterns have implications for educational policy and instructional "
        "practice, though cross-sectional ILSA designs cannot establish causality"
    )


def _ensure_conclusion_meaning_clause(
    conclusion: str,
    *,
    effect_hint: str | None = None,
) -> str:
    """Append 'This indicates that …' when the model stops at the finding clause."""
    text = conclusion.strip()
    if not text:
        return text
    if _MEANING_CLAUSE_RE.search(text):
        return _fix_standardized_conclusion_grammar(text)
    base = text.rstrip(".")
    meaning = _derive_meaning_clause(effect_hint, text)
    combined = f"{base}. This indicates that {meaning}."
    return _fix_standardized_conclusion_grammar(combined)


def _is_generic_weight_interpretation(text: str) -> bool:
    lowered = text.strip().lower()
    return any(marker in lowered for marker in _GENERIC_WFI_MARKERS)


def _weight_fields_interpretation_from_outcome(outcome: str) -> str:
    """Synthesize survey-design summary for official reports when WFI is empty."""
    text = outcome.strip()
    if not text:
        return (
            "This official IEA/OECD technical document describes assessment design, "
            "sampling, and scaling procedures rather than student-level ML estimation. "
            "Survey weights and replicate methods should be taken from the "
            "international sampling and weighting manual for the cited cycle."
        )
    sentences = re.split(r"(?<=[.!?])\s+", text)
    picked = [s.strip() for s in sentences if s.strip()][:4]
    body = " ".join(picked) if picked else text[:600]
    if len(body) > 520:
        body = body[:517].rstrip() + "..."
    return (
        f"{body} "
        "As a framework or implementation manual, the document explains how the "
        "assessment is administered and analyzed; apply cycle-specific weight "
        "variables (e.g. W_FSTUWT) when using micro-data from the international database."
    )


def _fix_standardized_conclusion_grammar(text: str) -> str:
    """Fix 'Using X process the study' → 'Using X process data, the study'."""
    if not isinstance(text, str) or not text.strip():
        return text
    if re.search(r"\bprocess\s+data,\s*the\s+study\b", text, re.I):
        return text
    fixed = re.sub(
        r"(Using\s+[^,]+?)\s+process\s+the\s+study\b",
        r"\1 process data, the study",
        text,
        count=1,
        flags=re.I,
    )
    return re.sub(
        r"\bprocess\s+the\s+study\b",
        "process data, the study",
        fixed,
        count=1,
        flags=re.I,
    )


def _dataset_clause(dataset: str) -> str:
    """Avoid 'process data … data,' when dataset_used already contains 'data'."""
    d = dataset.strip().rstrip(".")
    if re.search(r"\bprocess\s+data\b", d, re.IGNORECASE):
        return f"Using {d},"
    if re.search(r"\bdata\b", d, re.IGNORECASE):
        return f"Using {d},"
    if re.search(r"\bprocess\b", d, re.IGNORECASE):
        return f"Using {d} data,"
    return f"Using {d} data,"


def _normalize_finding_key_part(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _finding_row_key(finding: dict) -> tuple[str, str]:
    return (
        _normalize_finding_key_part(finding.get("dataset_used", "")),
        _normalize_finding_key_part(finding.get("target_variable", "")),
    )


_AUXILIARY_TARGET = re.compile(
    r"\b(cluster|k-?means|unsupervised|group|segment|profile|latent\s+class)\b",
    re.IGNORECASE,
)
_PRIMARY_TARGET = re.compile(
    r"\b(correct|accuracy|achievement|score|resilien|binary|predict|performance|"
    r"plausible|pv\d|response|outcome)\b",
    re.IGNORECASE,
)


def _is_auxiliary_finding(finding: dict) -> bool:
    target = finding.get("target_variable", "")
    metrics = finding.get("performance_metrics", "")
    return bool(_AUXILIARY_TARGET.search(f"{target} {metrics}"))


def _merge_finding_into_primary(primary: dict, auxiliary: dict) -> None:
    """Append auxiliary analysis metrics to the primary finding row."""
    extra = auxiliary.get("performance_metrics", "").strip()
    if extra and extra not in primary.get("performance_metrics", ""):
        primary["performance_metrics"] = (
            f"{primary['performance_metrics'].rstrip('.')}. "
            f"Additional analysis ({auxiliary.get('target_variable', 'secondary')}): "
            f"{extra}"
        )


def _dedupe_main_findings(findings: list[dict]) -> list[dict]:
    """
    Remove duplicate rows and merge same-dataset auxiliary analyses (e.g. k-means
    clustering) into the primary supervised finding instead of a second near-copy.
    """
    if not findings:
        return findings

    unique: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    for finding in findings:
        key = _finding_row_key(finding)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(finding)

    by_dataset: dict[str, list[dict]] = {}
    for finding in unique:
        ds_key = _finding_row_key(finding)[0]
        by_dataset.setdefault(ds_key, []).append(finding)

    merged: list[dict] = []
    for group in by_dataset.values():
        if len(group) == 1:
            merged.append(group[0])
            continue

        primaries = [f for f in group if not _is_auxiliary_finding(f)]
        auxiliaries = [f for f in group if _is_auxiliary_finding(f)]

        if not primaries:
            primary = group[0]
            for extra in group[1:]:
                _merge_finding_into_primary(primary, extra)
            merged.append(primary)
            continue

        primary = primaries[0]
        for aux in auxiliaries:
            _merge_finding_into_primary(primary, aux)
        merged.append(primary)
        for extra in primaries[1:]:
            merged.append(extra)

    return merged


def _build_standardized_conclusion(
    dataset: str,
    predictors: list[str],
    target: str,
    *,
    effect_hint: str | None = None,
) -> str:
    """Enforce Dataset → Input → Target → Output sentence template."""
    preds = ", ".join(predictors[:3]) if predictors else "the reported predictors"
    clause = _dataset_clause(dataset)
    if effect_hint and effect_hint.strip():
        effect = effect_hint.strip().rstrip(".")
        for prefix in (
            "the study found that ",
            "finding that ",
            "the study found ",
        ):
            if effect.lower().startswith(prefix):
                effect = effect[len(prefix):].strip()
                break
        if effect.lower().startswith("using "):
            cleaned = effect if effect.endswith(".") else f"{effect}."
            cleaned = re.sub(r"\bdata,\s*the study", "the study", cleaned, flags=re.I)
            return _ensure_conclusion_meaning_clause(
                _fix_standardized_conclusion_grammar(cleaned),
                effect_hint=effect_hint,
            )
    else:
        effect = "results were reported without a clear narrative conclusion in the manuscript"
    built = _fix_standardized_conclusion_grammar(
        f"{clause} the study leveraged {preds} to predict {target}, "
        f"finding that {effect}."
    )
    return _ensure_conclusion_meaning_clause(built, effect_hint=effect_hint)


def _normalize_main_findings_list(raw: list) -> list[dict]:
    """Normalize and validate main_findings entries from LLM or legacy JSON."""
    findings: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        dataset = str(
            item.get("dataset_used") or item.get("ilsa_dataset") or ""
        ).strip()
        if not dataset or dataset in _INVALID_FIELD_STRINGS:
            dataset = "ILSA dataset not specified in manuscript"

        target = str(item.get("target_variable") or "").strip()
        if not target or target in _INVALID_FIELD_STRINGS:
            continue
        predictors = item.get("top_predictors")
        if isinstance(predictors, str):
            predictors = [p.strip() for p in predictors.split(",") if p.strip()]
        elif not isinstance(predictors, list):
            predictors = []
        else:
            predictors = [
                str(p).strip() for p in predictors
                if isinstance(p, (str, int, float)) and str(p).strip() not in _INVALID_FIELD_STRINGS
            ]
        metrics = str(item.get("performance_metrics") or "").strip()
        if not metrics or metrics in _INVALID_FIELD_STRINGS:
            metrics = "Not reported"
        conclusion = str(item.get("standardized_conclusion") or "").strip()
        if not conclusion or conclusion in _INVALID_FIELD_STRINGS:
            conclusion = _build_standardized_conclusion(dataset, predictors, target)
        elif dataset.lower() not in conclusion.lower():
            conclusion = _build_standardized_conclusion(
                dataset, predictors, target, effect_hint=conclusion
            )
        elif re.search(r"\bdata,\s*the study", conclusion, re.IGNORECASE):
            conclusion = _build_standardized_conclusion(
                dataset, predictors, target, effect_hint=conclusion
            )
        elif re.search(r"\bprocess\s+the\s+study\b", conclusion, re.IGNORECASE):
            conclusion = _fix_standardized_conclusion_grammar(conclusion)
        else:
            conclusion = _fix_standardized_conclusion_grammar(conclusion)
        conclusion = _ensure_conclusion_meaning_clause(
            conclusion,
            effect_hint=str(item.get("standardized_conclusion") or ""),
        )
        findings.append({
            "dataset_used": dataset,
            "target_variable": target,
            "top_predictors": predictors[:5],
            "performance_metrics": metrics,
            "standardized_conclusion": conclusion,
        })
    return _dedupe_main_findings(findings)


def _coerce_outcome_summary_text(value) -> str:
    """Normalize outcome_summary to a plain string."""
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, dict):
        text = str(value.get("summary") or "").strip()
        if not text:
            text = " ".join(str(v) for v in value.values() if isinstance(v, str)).strip()
    else:
        text = str(value or "").strip()
    if not text or text in _INVALID_FIELD_STRINGS:
        return ""
    return text


_LEGACY_MIGRATION_MARKER = "legacy migration"
_ILSA_PROGRAM_YEAR_RE = re.compile(
    r"\b(PISA|TIMSS|PIRLS|TALIS|PIAAC|ICILS|ICCS|OECD)\b"
    r"(?:\s*[-–]?\s*(20\d{2}))?",
    re.IGNORECASE,
)
_ILSA_DATASET_PHRASE_RE = re.compile(
    r"\b((?:PISA|TIMSS|PIRLS|TALIS|PIAAC|ICILS|ICCS)\s+20\d{2}"
    r"(?:\s+[\w][\w\s\-]{2,40})?)",
    re.IGNORECASE,
)
_METRICS_SNIPPET_RE = re.compile(
    r"(?:R[²2]\s*[=:]\s*[\d.]+|AUC\s*[=:]\s*[\d.]+|"
    r"(?:accuracy|Accuracy)\s*[=:]\s*[\d.]+%?|RMSE\s*[=:]\s*[\d.]+|"
    r"F1[- ]?score?\s*[=:]\s*[\d.]+|"
    r"(?:\d{1,3}(?:\.\d+)?)\s*%\s+accuracy)",
    re.IGNORECASE,
)
_NON_EMPIRICAL_SOURCE_CATEGORIES = frozenset({
    "review_article", "methodology_paper", "technical_report",
})


def _is_legacy_migration_finding(finding: dict) -> bool:  # noqa: D103
    ds = str(finding.get("dataset_used") or "").lower()
    tv = str(finding.get("target_variable") or "").lower()
    return _LEGACY_MIGRATION_MARKER in ds or _LEGACY_MIGRATION_MARKER in tv


def _collect_migration_context_text(data: dict, metadata: dict | None) -> str:
    """Concatenate extraction fields used for heuristic dataset/target inference."""
    parts: list[str] = []
    meta = metadata or {}
    for key in ("title", "file_name"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    sd = data.get("sample_details") if isinstance(data.get("sample_details"), dict) else {}
    sfc = sd.get("sample_filtering_criteria")
    if isinstance(sfc, str) and sfc.strip():
        parts.append(sfc.strip())
    survey = data.get("survey_design") if isinstance(data.get("survey_design"), dict) else {}
    wfi = survey.get("weight_fields_interpretation")
    if isinstance(wfi, str) and wfi.strip():
        parts.append(wfi.strip())
    for key in ("outcome_summary", "null_fields_interpretation"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return "\n".join(parts)


def _infer_dataset_from_context(data: dict, metadata: dict | None) -> str:
    blob = _collect_migration_context_text(data, metadata)
    phrase_hits = _ILSA_DATASET_PHRASE_RE.findall(blob)
    if phrase_hits:
        return phrase_hits[0].strip().split("\n")[0][:120]

    programs: list[str] = []
    for match in _ILSA_PROGRAM_YEAR_RE.finditer(blob):
        program = match.group(1).upper()
        year = match.group(2) or ""
        label = f"{program} {year}".strip()
        if label not in programs:
            programs.append(label)
    if programs:
        return programs[0][:120]

    sc = (metadata or {}).get("source_category")
    if sc in _NON_EMPIRICAL_SOURCE_CATEGORIES:
        return (
            "ILSA literature synthesis (systematic review; "
            "no single student-level analytic micro-dataset)"
        )
    return "ILSA assessment context not specified in extraction (heuristic migration)"


def _infer_target_from_context(data: dict, metadata: dict | None) -> str:
    outcome = _coerce_outcome_summary_text(data.get("outcome_summary"))
    blob = f"{outcome}\n{_collect_migration_context_text(data, metadata)}"
    target_patterns = (
        r"predict(?:ing|ed|s)?\s+([^.;,\n]{5,70})",
        r"(?:math(?:ematics)?|science|reading)\s+"
        r"(?:achievement|literacy|proficiency|performance|score)s?",
        r"academic\s+resilience",
        r"(?:student\s+)?well[- ]?being",
        r"creative\s+thinking(?:\s+score)?",
        r"problem[- ]solving(?:\s+success)?",
        r"environmental\s+action",
        r"job\s+satisfaction",
        r"item\s+difficulty",
    )
    for pattern in target_patterns:
        match = re.search(pattern, blob, re.IGNORECASE)
        if match:
            return match.group(0).strip()[:80]

    sc = (metadata or {}).get("source_category")
    ml = data.get("ml_techniques") if isinstance(data.get("ml_techniques"), dict) else {}
    has_ml = bool(ml.get("primary")) or bool(ml.get("all_techniques"))
    if sc in _NON_EMPIRICAL_SOURCE_CATEGORIES:
        return "Literature synthesis outcome (not student-level prediction)"
    if has_ml:
        return "Primary analytic outcome (inferred from extraction)"
    return "Primary study outcome (heuristic migration)"


def _infer_top_predictors_from_data(data: dict) -> list[str]:
    predictors: list[str] = []
    for item in data.get("confounders_identified") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("variable_name") or item.get("variable_code")
        if not isinstance(name, str):
            continue
        cleaned = name.strip()
        if cleaned and cleaned.lower() not in _INVALID_FIELD_STRINGS:
            predictors.append(cleaned)
        if len(predictors) >= 5:
            break
    return predictors


def _extract_performance_metrics_from_text(text: str) -> str:
    if not text:
        return "Not reported"
    hits = _METRICS_SNIPPET_RE.findall(text)
    if hits:
        return "; ".join(dict.fromkeys(h.strip() for h in hits))[:220]
    return "Not reported"


_DESCRIPTIVE_TARGET_PATTERNS = (
    re.compile(r"mean\s+(.{3,55}?)\s+score", re.IGNORECASE),
    re.compile(
        r"summaris(?:e|es|ing)\s+(?:[^.]{0,60}?\s+)?performance\s+on\s+(?:the\s+)?"
        r"([^.;,\n]{5,70})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:math(?:ematics)?|science|reading|creative thinking|problem solving|"
        r"financial literacy|collaborative problem solving|environmental action|"
        r"well[- ]?being|job satisfaction|literacy|numeracy|general pedagogical knowledge)"
        r"[^.;]{0,35}?(?:achievement|proficiency|performance|literacy|score|knowledge)s?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"performance\s+in\s+([^.;,\n]{5,60})",
        re.IGNORECASE,
    ),
    re.compile(
        r"performed in\s+([^.;]{8,160})",
        re.IGNORECASE,
    ),
    re.compile(
        r"measures teachers['\u2019]?\s*(.{5,80}?)(?:\s+through|\s+using|\s+via|,|\.)",
        re.IGNORECASE,
    ),
    re.compile(
        r"((?:financial literacy|creative thinking|problem solving|"
        r"collaborative problem solving|global competence)"
        r"[^.;]{0,30}?)\s+was assessed",
        re.IGNORECASE,
    ),
    re.compile(
        r"synthesiz(?:e|es|ing)\s+([^.;]{12,100}?)\s+results",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:addresses|examines|explains)\s+([^.;]{10,90})",
        re.IGNORECASE,
    ),
    re.compile(
        r"associations between .{5,80}?\s+and\s+([^.;]{8,80}?)(?:\s+indicators|\s+outcomes|\.|,)",
        re.IGNORECASE,
    ),
)
_DESCRIPTIVE_METRIC_SNIPPET_RE = re.compile(
    r"(?:mean\s+[\w\s]{2,30}?\s+score\s+is\s+[^.;]{5,40}|"
    r"[\d]{1,3}(?:\.\d+)?\s*%\s+of\s+[^.;]{10,80}|"
    r"correlation[s]?\s+(?:is|are|of)?\s*[\d.]+[^.;]{0,40}|"
    r"[\d.]+\s*\([^)]{5,60}\)|"
    r"significantly\s+(?:below|above)\s+[^.;]{10,60}|"
    r"OECD\s+average[^.;]{0,40})",
    re.IGNORECASE,
)
_ASSOCIATED_VARIABLE_RE = re.compile(
    r"(?:correlation[s]?\s+(?:is|are|of)?\s*)?[\d.]+\s*\(([^)]+)\)|"
    r"attributed\s+to\s+([^.;,\n]{5,60})|"
    r"associated\s+with\s+([^.;,\n]{5,60})",
    re.IGNORECASE,
)
_GROUNDING_SKIP_MARKERS = (
    "not specified",
    "heuristic migration",
    "literature synthesis",
)
_BAD_TARGET_MARKERS = (
    "split into",
    "other group",
    "sampled schools",
    "one group took",
)


def _is_grounded_migration_label(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered or any(m in lowered for m in _GROUNDING_SKIP_MARKERS):
        return False
    if any(m in lowered for m in _BAD_TARGET_MARKERS):
        return False
    return True


def _split_domain_phrase(phrase: str) -> list[str]:
    normalized = re.sub(r"\s+and\s+", ", ", phrase.strip(), flags=re.IGNORECASE)
    return [p.strip() for p in normalized.split(",") if len(p.strip()) >= 3]


def _normalize_descriptive_target_label(raw: str) -> str:
    target = raw.strip().rstrip(".,;")
    if not target:
        return ""
    lowered = target.lower()
    if any(
        tok in lowered
        for tok in (
            "achievement", "proficiency", "performance", "literacy",
            "knowledge", "score", "questionnaire", "results", "outcomes",
        )
    ):
        return target[:80]
    return f"{target} achievement"[:80]


def _fallback_descriptive_target_from_outcome(outcome: str) -> str | None:
    for pattern in (
        r"summariz(?:e|es|ing)\s+(?:[^.]{0,50}?\s+)?(?:results|findings)\s+"
        r"(?:from|on)\s+(?:the\s+)?([^.;]{8,90})",
        r"presents results from\s+(?:the\s+)?([^.;]{8,90})",
        r"reviews how\s+([^.;]{12,100})",
        r"uses\s+[^.;]{0,60}?\s+to\s+(?:describe|explain)\s+([^.;]{10,90})",
    ):
        match = re.search(pattern, outcome, re.IGNORECASE)
        if not match:
            continue
        label = _normalize_descriptive_target_label(match.group(1))
        if label and _is_grounded_migration_label(label):
            return label
    return None


def _outcome_explicitly_non_assessment(outcome: str) -> bool:
    lowered = outcome.lower()
    return any(
        phrase in lowered
        for phrase in (
            "does not describe any student assessment",
            "policy-oriented macroeconomic",
            "rather than an empirical large-scale assessment",
            "no predictive-model performance metrics",
        )
    )


def _infer_domain_list_from_outcome(outcome: str) -> list[str]:
    domain_patterns = (
        r"performed in\s+([^.;]{8,160})",
        r"students?\s+in\s+([^.;]{8,160})",
        r"across\s+([^.;]{8,160})",
        r"trends in\s+([^.;]{8,160})",
        r"performance (?:levels )?in\s+([^.;]{8,160})",
    )
    for pattern in domain_patterns:
        match = re.search(pattern, outcome, re.IGNORECASE)
        if not match:
            continue
        phrase = match.group(1).strip().rstrip(".")
        if len(phrase) > 90:
            continue
        if not re.search(
            r"\b(mathematics|reading|science|literacy|thinking|financial)\b",
            phrase,
            re.I,
        ):
            continue
        domains = _split_domain_phrase(phrase)
        if len(domains) >= 2:
            labels = [_normalize_descriptive_target_label(d) for d in domains[:5]]
            return [label for label in labels if label and _is_grounded_migration_label(label)]
        if domains:
            label = _normalize_descriptive_target_label(domains[0])
            if label and _is_grounded_migration_label(label):
                return [label]
    return []


def _infer_descriptive_targets_from_outcome(outcome: str) -> list[str]:
    if _outcome_explicitly_non_assessment(outcome):
        return []

    domain_targets = _infer_domain_list_from_outcome(outcome)
    if domain_targets:
        return domain_targets

    targets: list[str] = []
    seen: set[str] = set()
    for pattern in _DESCRIPTIVE_TARGET_PATTERNS:
        match = pattern.search(outcome)
        if not match:
            continue
        label = _normalize_descriptive_target_label(match.group(1))
        if not label or not _is_grounded_migration_label(label):
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        targets.append(label)
        if len(targets) >= 5:
            break
    if targets:
        return targets

    fallback = _fallback_descriptive_target_from_outcome(outcome)
    return [fallback] if fallback else []


def _infer_descriptive_target_from_outcome(outcome: str) -> str | None:
    multi = _infer_descriptive_targets_from_outcome(outcome)
    return multi[0] if multi else None


def _extract_descriptive_metric_snippets(outcome: str) -> str:
    hits = [m.group(0).strip() for m in _DESCRIPTIVE_METRIC_SNIPPET_RE.finditer(outcome)]
    ml_hits = _METRICS_SNIPPET_RE.findall(outcome)
    combined: list[str] = []
    for item in hits + ml_hits:
        cleaned = item.strip()
        if cleaned and cleaned not in combined:
            combined.append(cleaned)
    if combined:
        return "; ".join(combined)[:220]
    return "Descriptive statistics only (no ML performance metrics; see outcome_summary)"


def _infer_associated_variables_from_outcome(outcome: str) -> list[str]:
    variables: list[str] = []
    for match in _ASSOCIATED_VARIABLE_RE.finditer(outcome):
        for group in match.groups():
            if not group:
                continue
            label = group.strip().rstrip(".,;")
            if not label or len(label) < 4:
                continue
            label = re.sub(r"\s+", " ", label)
            if label.lower() not in {v.lower() for v in variables}:
                variables.append(label[:80])
            if len(variables) >= 5:
                return variables
    return variables


def _derive_descriptive_meaning_clause(outcome: str) -> str:
    for pattern in (
        r"(significantly\s+(?:below|above)\s+[^.]{10,140}\.)",
        r"((?:below|above)\s+(?:the\s+)?OECD\s+average[^.]{0,80}\.)",
        r"(implications?[^.]{15,160}\.)",
        r"(suggests?\s+that\s+[^.]{15,160}\.)",
    ):
        match = re.search(pattern, outcome, re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(".")[:220]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", outcome.strip()) if s.strip()]
    if len(sentences) >= 2:
        return sentences[-1].rstrip(".")[:220]
    if sentences:
        return sentences[0].rstrip(".")[:220]
    return outcome[:220]


def _build_descriptive_standardized_conclusion(
    dataset: str,
    associated: list[str],
    target: str,
    outcome: str,
) -> str:
    assoc = ", ".join(associated[:3]) if associated else "documented contextual indicators"
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", outcome.strip()) if s.strip()]
    lead = sentences[0] if sentences else outcome[:280]
    if len(lead) > 280:
        lead = lead[:277].rstrip() + "..."
    clause = _dataset_clause(dataset)
    built = _fix_standardized_conclusion_grammar(
        f"{clause} the document summarizes {target} in relation to {assoc}, "
        f"reporting that {lead.rstrip('.')}."
    )
    meaning = _derive_descriptive_meaning_clause(outcome)
    if not _MEANING_CLAUSE_RE.search(built):
        built = f"{built.rstrip('.')}. This indicates that {meaning}."
    return _fix_standardized_conclusion_grammar(built)


def _build_descriptive_findings_from_outcome(
    data: dict,
    metadata: dict | None,
) -> list[dict]:
    """
    Ground main_findings in outcome_summary for official IEA/OECD reports.

    No API calls; no invented ML metrics. Skips when target/dataset cannot be
    inferred from existing extraction text.
    """
    outcome = _coerce_outcome_summary_text(data.get("outcome_summary"))
    if not substantive_outcome_summary(data):
        return []

    targets = _infer_descriptive_targets_from_outcome(outcome)
    if not targets:
        return []

    dataset = _infer_dataset_from_context(data, metadata)
    if not _is_grounded_migration_label(dataset):
        return []

    predictors = _infer_top_predictors_from_data(data)
    if not predictors:
        predictors = _infer_associated_variables_from_outcome(outcome)
    if not predictors:
        predictors = ["See outcome_summary (descriptive associations not coded as predictors)"]

    metrics = _extract_descriptive_metric_snippets(outcome)
    findings: list[dict] = []
    for target in targets:
        findings.append({
            "dataset_used": dataset,
            "target_variable": target,
            "top_predictors": predictors[:5],
            "performance_metrics": metrics,
            "standardized_conclusion": _build_descriptive_standardized_conclusion(
                dataset, predictors, target, outcome,
            ),
        })
    return findings


def _has_empirical_ml_signals(data: dict) -> bool:
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


def _build_migration_skeleton_finding(
    data: dict,
    metadata: dict | None,
    *,
    conclusion: str | None = None,
) -> dict:
    outcome = _coerce_outcome_summary_text(data.get("outcome_summary"))
    dataset = _infer_dataset_from_context(data, metadata)
    target = _infer_target_from_context(data, metadata)
    predictors = _infer_top_predictors_from_data(data)
    metrics = _extract_performance_metrics_from_text(outcome or conclusion or "")
    ml = data.get("ml_techniques") if isinstance(data.get("ml_techniques"), dict) else {}
    primary = ml.get("primary")
    if metrics == "Not reported" and isinstance(primary, str) and primary.strip():
        metrics = f"Best model: {primary.strip()} (see outcome_summary for metrics)"
    effect = (conclusion or outcome or "").strip()
    if not effect:
        effect = (
            "results were synthesized heuristically from outcome_summary during "
            "offline migration (no API re-extraction)"
        )
    return {
        "dataset_used": dataset,
        "target_variable": target,
        "top_predictors": predictors,
        "performance_metrics": metrics,
        "standardized_conclusion": _build_standardized_conclusion(
            dataset, predictors, target, effect_hint=effect,
        ),
    }


def _upgrade_legacy_and_empty_findings(data: dict, metadata: dict | None) -> None:
    """Heuristic migration: never leave main_findings empty when signals exist."""
    if is_official_report_document(data, metadata):
        data["main_findings"] = []
        return
    raw = data.get("main_findings")
    findings: list[dict] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                findings.append(dict(item))

    outcome = _coerce_outcome_summary_text(data.get("outcome_summary"))
    needs_fill = article_requires_main_findings(data, metadata)

    if not findings and needs_fill:
        data["main_findings"] = [_build_migration_skeleton_finding(data, metadata)]
        return

    if not findings:
        return

    upgraded: list[dict] = []
    any_legacy = False
    for finding in findings:
        if _is_legacy_migration_finding(finding):
            any_legacy = True
            skeleton = _build_migration_skeleton_finding(
                data, metadata, conclusion=outcome or finding.get("standardized_conclusion"),
            )
            upgraded.append({
                **finding,
                "dataset_used": skeleton["dataset_used"],
                "target_variable": skeleton["target_variable"],
                "top_predictors": skeleton["top_predictors"] or finding.get("top_predictors") or [],
                "performance_metrics": (
                    skeleton["performance_metrics"]
                    if skeleton["performance_metrics"] != "Not reported"
                    else finding.get("performance_metrics") or "Not reported"
                ),
                "standardized_conclusion": skeleton["standardized_conclusion"],
            })
        else:
            upgraded.append(finding)

    data["main_findings"] = _normalize_main_findings_list(upgraded)

    if needs_fill and not data["main_findings"]:
        data["main_findings"] = [_build_migration_skeleton_finding(data, metadata)]
    elif any_legacy and data["main_findings"]:
        data["main_findings"] = _normalize_main_findings_list(data["main_findings"])


def _normalize_findings_fields(data: dict, metadata: dict | None = None) -> None:
    """Normalize main_findings and outcome_summary; keep both when present."""
    _upgrade_legacy_and_empty_findings(data, metadata)

    raw_findings = data.get("main_findings")
    if isinstance(raw_findings, list):
        data["main_findings"] = _normalize_main_findings_list(raw_findings)
    elif raw_findings is None:
        data["main_findings"] = []
    else:
        data["main_findings"] = []

    outcome = _coerce_outcome_summary_text(data.get("outcome_summary"))

    if not data["main_findings"] and outcome and article_requires_main_findings(data, metadata):
        data["main_findings"] = [_build_migration_skeleton_finding(data, metadata)]

    if not outcome and data["main_findings"]:
        parts = []
        for f in data["main_findings"][:3]:
            parts.append(f.get("standardized_conclusion", ""))
        outcome = " ".join(p for p in parts if p).strip()

    if not outcome:
        outcome = (
            "No narrative outcome summary was extracted. See main_findings for "
            "structured results if available."
        )

    data["outcome_summary"] = outcome

    if article_requires_main_findings(data, metadata) and not data["main_findings"]:
        data["main_findings"] = [_build_migration_skeleton_finding(data, metadata)]
        data["main_findings"] = _normalize_main_findings_list(data["main_findings"])


COUNTRY_NAME_TO_ISO = {
    "türkiye": "TUR", "turkey": "TUR", "usa": "USA",
    "united states": "USA", "germany": "DEU", "deutschland": "DEU", 
    "japan": "JPN", "korea": "KOR", "france": "FRA", 
    "china": "CHN", "brazil": "BRA", "finland": "FIN",
    "singapore": "SGP", "australia": "AUS", "canada": "CAN",
    "uk": "GBR", "united kingdom": "GBR", "england": "GBR",
    "spain": "ESP", "italy": "ITA", "netherlands": "NLD",
    "sweden": "SWE", "norway": "NOR", "denmark": "DNK",
    "israel": "ISR", "new zealand": "NZL", "ireland": "IRL",
    "austria": "AUT", "belgium": "BEL", "switzerland": "CHE",
    "portugal": "PRT", "poland": "POL", "czech republic": "CZE",
    "hungary": "HUN", "greece": "GRC", "romania": "ROU",
    "russia": "RUS", "thailand": "THA", "indonesia": "IDN",
    "malaysia": "MYS", "chile": "CHL", "mexico": "MEX",
    "colombia": "COL", "argentina": "ARG", "india": "IND",
    "south africa": "ZAF", "taiwan": "TWN", "hong kong": "HKG",
    "macao": "MAC", "macau": "MAC", "estonia": "EST",
    "latvia": "LVA", "lithuania": "LTU", "slovakia": "SVK",
    "slovenia": "SVN", "croatia": "HRV", "serbia": "SRB",
    "bulgaria": "BGR", "cyprus": "CYP", "malta": "MLT",
    "luxembourg": "LUX", "iceland": "ISL", "qatar": "QAT",
    "uae": "ARE", "saudi arabia": "SAU", "jordan": "JOR",
    "iran": "IRN", "egypt": "EGY", "morocco": "MAR",
    "tunisia": "TUN", "ghana": "GHA", "kenya": "KEN",
    "nigeria": "NGA", "pakistan": "PAK", "vietnam": "VNM",
    "philippines": "PHL", "peru": "PER", "uruguay": "URY",
    "costa rica": "CRI", "panama": "PAN", 
    "lebanon": "LBN", "dominican republic": "DOM",
    "beijing-shanghai-jiangsu-zhejiang": "CHN",
    "b-s-j-z": "CHN", "bsjz": "CHN", "b-s-j-g": "CHN",
    "northern ireland": "GBR", "türkiye": "TUR",
    "republic of korea": "KOR", "czechia": "CZE",
    "macao sar": "MAC", "hong kong sar": "HKG",
    "chinese mainland": "CHN", "united arab emirates": "ARE",
    "scotland": "GBR", "wales": "GBR", "great britain": "GBR",
    "flemish": "BEL", "flemish community": "BEL",
    "philippine": "PHL", "filipino": "PHL",
    "korean": "KOR", "moroccan": "MAR", "chinese taipei": "TWN",
    "tunisian": "TUN", "b-s-j-z (china)": "CHN",
    "beijing, shanghai, jiangsu, and zhejiang": "CHN",
    "beijing, shanghai, jiangsu, and guangdong": "CHN",
    "b-s-j-g (china)": "CHN", "south korea": "KOR",
    "lebanese": "LBN", "lebanese republic": "LBN",
    "brazilian": "BRA", "spanish": "ESP",
    "german": "DEU", "french": "FRA",
    "japanese": "JPN", "finnish": "FIN",
    "australian": "AUS", "canadian": "CAN",
    "irish": "IRL", "swedish": "SWE",
    "norwegian": "NOR", "danish": "DNK",
    "estonian": "EST", "latvian": "LVA",
    "hungarian": "HUN", "peruvian": "PER",
    "mexican": "MEX", "chilean": "CHL",
    "colombian": "COL", "uruguayan": "URY",
    "singaporean": "SGP", "dutch": "NLD",
    "swiss": "CHE", "belgian": "BEL",
    "polish": "POL", "austrian": "AUT",
    "greek": "GRC", "slovenian": "SVN",
    "italian": "ITA", "portuguese": "PRT",
    "luxembourgish": "LUX", "icelandic": "ISL",
    "qatari": "QAT", "emirati": "ARE",
    "saudi": "SAU", "jordanian": "JOR",
    "iranian": "IRN", "egyptian": "EGY",
    "ghanaian": "GHA", "kenyan": "KEN",
    "nigerian": "NGA", "pakistani": "PAK",
    "vietnamese": "VNM", "thai": "THA",
    "indonesian": "IDN", "indian": "IND",
    "taiwanese": "TWN", "macanese": "MAC",
    "new zealander": "NZL", "british": "GBR",
    "american": "USA", "chinese": "CHN",
    "turkish": "TUR", "czech": "CZE",
    "slovak": "SVK", "croatian": "HRV",
    "serbian": "SRB", "bulgarian": "BGR",
    "cypriot": "CYP", "maltese": "MLT",
    "romanian": "ROU", "russian": "RUS",
    "south african": "ZAF", "panamanian": "PAN",
    "dominican": "DOM", "israeli": "ISR",
    "scandinavian": "NOR",
    "malaysian": "MYS", "lithuanian": "LTU",
    "argentinian": "ARG", "argentine": "ARG",
    "costa rican": "CRI",
    "the netherlands": "NLD",
    "republic of china": "TWN",
    "korea, republic of": "KOR",
}

_VALID_ISO_CODES = frozenset(COUNTRY_NAME_TO_ISO.values())

_TOTAL_STUDENTS_RE = re.compile(
    r"\b(?:N|n)\s*=\s*([\d][\d,]{2,8})\b|"
    r"\b(?:sample\s+of\s+)?([\d][\d,]{3,8})\s+(?:students?|participants?|learners?)\b",
    re.IGNORECASE,
)


def _backfill_countries_from_extracted_text(
    data: dict,
    metadata: dict | None,
) -> None:
    """Ground empty countries[] in already-extracted text (offline, no API)."""
    if is_official_report_document(data, metadata):
        return
    sd = data.get("sample_details")
    if not isinstance(sd, dict):
        return
    existing = sd.get("countries")
    if isinstance(existing, list) and existing:
        return

    blob = _collect_migration_context_text(data, metadata)
    if not blob.strip():
        return

    found: list[dict] = []
    seen: set[str] = set()
    for name, code in sorted(COUNTRY_NAME_TO_ISO.items(), key=lambda x: -len(x[0])):
        if len(name) < 4:
            continue
        if not re.search(rf"\b{re.escape(name)}\b", blob, re.IGNORECASE):
            continue
        if code in seen:
            continue
        seen.add(code)
        found.append({"country_code": code, "n_students": None})

    for match in re.finditer(r"\b([A-Z]{3})\b", blob):
        code = match.group(1).upper()
        if code in _VALID_ISO_CODES and code not in seen:
            seen.add(code)
            found.append({"country_code": code, "n_students": None})

    if found:
        sd["countries"] = found[:80]


def _backfill_total_students_from_extracted_text(
    data: dict,
    metadata: dict | None,
) -> None:
    """Infer total_students only when explicitly stated in extracted text."""
    sd = data.get("sample_details")
    if not isinstance(sd, dict) or sd.get("total_students"):
        return
    blob = _collect_migration_context_text(data, metadata)
    for match in _TOTAL_STUDENTS_RE.finditer(blob):
        raw = match.group(1) or match.group(2)
        if not raw:
            continue
        try:
            value = int(raw.replace(",", ""))
        except ValueError:
            continue
        if 50 <= value <= 10_000_000:
            sd["total_students"] = value
            return

MODEL_NAME = "gpt-5.4-nano"
PRICE_INPUT_PER_1M = 2.50
PRICE_OUTPUT_PER_1M = 10.00

SYSTEM_PROMPT = """You are an expert research analyst specializing in International \
Large-Scale Assessments (ILSA: PISA, TIMSS, PIRLS, TALIS, ICILS, ICCS, PIAAC) \
and related national/regional large-scale assessments (NAEP, CEDRE, INVALSI) \
and the intersection of Machine Learning in educational data mining.

Your task is to extract a highly detailed, structured metadata and methodological \
sheet from an academic article. You must rigorously map academic jargon to the \
strict schema provided, and MINIMIZE null values through deep semantic search \
and expert domain inference.

═══════════════════════════════════════════════════════════════
CORE OBJECTIVE (data → methods → results → meaning)
═══════════════════════════════════════════════════════════════
Every extraction must answer four questions:
  (1) Which ILSA or related large-scale data were used (program, cycle, grade, domain)?
  (2) Which methods (sampling/weights, PV handling, missing data, ML or design docs)?
  (3) What results were obtained (metrics, effects, benchmarks)?
  (4) What do they MEAN for education policy and practice (implications beyond numbers)?

RULE 1 — EMPIRICAL ML / PREDICTIVE PAPERS:
  - Populate main_findings with one StructuredFinding per DISTINCT target_variable.
  - standardized_conclusion MUST use the Dataset→Input→Target→Output template AND end with \
    an implication clause: "... finding that [effect]. This indicates that [policy/practice \
    meaning]." Example ending: "This indicates that strengthening early numeracy support \
    may narrow socioeconomic achievement gaps, though causality cannot be inferred."
  - outcome_summary: 4-5 sentences (~120 words) with metrics and limitations.

RULE 2 — CRASH-PROOF OFFICIAL IEA/OECD TECHNICAL REPORTS & ASSESSMENT FRAMEWORKS:
  Pydantic REJECTS any string outside the exact literals below. Never invent labels \
  like "Official Report", "technical report", or prose in enum fields.
  - metadata.publication_type: EXACTLY "report" (never "Official Report" or variants).
  - metadata.source_category: EXACTLY "technical_report" OR "methodology_paper".
  - data.plausible_values_handling: EXACTLY "not_applicable" (frameworks/manuals; \
    allowed PV literals for empirical papers only: rubin_rules, single_pv, average_pv, \
    all_pv, mitml, wle, irt_theta, not_reported).
  - data.missing_data_handling: EXACTLY "not_reported" (allowed elsewhere: listwise_deletion, \
    pairwise_deletion, mean_imputation, single_imputation, knn_imputation, multiple_imputation).
  - data.research_design_type: JSON null OR EXACTLY "exploratory" — never predictive/causal.
  - handling_not_reported_explanation: REQUIRED 2-3 sentences whenever PV is \
    not_applicable/not_reported OR missing_data_handling is not_reported (explain WHY).
  - null_fields_interpretation: REQUIRED 2-3 sentences for framework/report extractions \
    (no ML sample, no predictive findings — diagnose sparse fields by design).
  - outcome_summary: REQUIRED (~120-150 words) on assessment design, sampling, scaling, \
    instruments, benchmarks, administration — teach how the exam works.
  - main_findings: MUST be [] — do NOT fabricate StructuredFinding rows.
  - confounders_identified: MUST be [] (reference codes are not ML predictors).
  - ml_techniques.primary = null; all_techniques = [] unless the report itself trains ML.
  - student_weights_used / replicate_weights_used / metadata.open_access: JSON null when \
    unknown — NEVER strings "N/A", "na", "unknown", "not applicable" (non-boolean strings fail).
  - weight_fields_interpretation: ALWAYS REQUIRED (3-4 sentences) on sampling, weights, \
    PV/scaling guidance — even when student_weights_used is null.
  - variable_code in confounders (if any): NEVER literal "N/A" — use snake_case slug \
    from variable_name per schema validator (reports should still use []).

═══════════════════════════════════════════════════════════════
CRITICAL EXTRACTION & INFERENCE RULES
═══════════════════════════════════════════════════════════════

1) STRICT ENUMERATIONS & CATEGORIES:
   - publication_type MUST be exactly one of:
     ['journal', 'conference', 'book_chapter', 'preprint', 'report', 'thesis'].
   - source_category MUST be exactly one of:
     ['technical_report', 'review_article', 'methodology_paper', 'peer_reviewed_research'].
   - research_design_type MUST be exactly one of:
     ['predictive', 'causal_observational', 'causal_experimental', 'exploratory'].
     Mapping: prediction/classification/regression → "predictive";
     causal forests, propensity scores, diff-in-diff, IV → "causal_observational";
     randomized experiment → "causal_experimental";
     clustering, topic modeling, EDA, data description → "exploratory".
   - plausible_values_handling MUST be exactly one of:
     ['rubin_rules', 'single_pv', 'average_pv', 'all_pv', 'mitml', 'wle', 'irt_theta',
      'not_applicable', 'not_reported'].
     Synonym table:
       "Rubin's rules" / "Rubin combining rules" / "combined PV estimates" /
       "pooled across PVs" / "PV estimates combined"               → rubin_rules
       "first plausible value" / "PV1 only" / "PV1MATH" / "PV1READ" /
       "PV1SCIE" / "PV2SCIE" / "single PV draw" /
       "one PV per student" / "separate analyses per PV" /
       "PV1 as outcome" / "used PV2SCIE" / "used PV1READ" /
       "target indicator from one PV" /
       "binary variable from PV benchmarks (single draw)"         → single_pv
       "averaged plausible values" / "mean of PVs" / "PV average" /
       "all five PVs averaged" / "all ten PVs averaged" /
       "PV1MATH–PV10MATH averaged" / "BSSSCI01–BSSSCI05 averaged" /
       "mean of 10 plausible values" / "average of plausible values" /
       "average of PV1MATH through PV10MATH" /
       "averaged across all plausible values"                      → average_pv
       "mitml" / "Mplus complex survey" / "multilevel MI"          → mitml
       TALIS/PIAAC without PVs, or DV is Likert/direct measure    → not_applicable
       "WLE" / "Warm's WLE" / "weighted likelihood estimator" /
       "IRT ability estimates" / "theta estimates" / "EAP estimates" /
       "latent trait scores" / "CFA-based scores" /
       "scale scores (not PVs)"                                    → not_applicable
       DV is binary classification (correct/incorrect, high/low) /
       DV is process data (actions, response times) /
       DV is affective/attitudinal (life satisfaction, self-efficacy) /
       DV is curriculum-based (not ILSA achievement)               → not_applicable
    SOFTWARE-BASED PV INFERENCE — when the paper does not state PV handling
    explicitly, infer from the software / R packages mentioned:
      "bifiesurvey" / "repest" / "intsvy" / "EdSurvey" /
      "IEA IDB Analyzer" / "RALSA" / "lavaan.survey" /
      "WeMix" / "mitml"                                              → rubin_rules
      These packages implement Rubin's combining rules internally;
      their use is strong evidence that PVs were handled properly.
      "five plausible values" / "5 PVs" / "all five PVs" /
      "five_pv" / "5_pv" / "ten plausible values" / "10 PVs" /
      "all ten PVs" / "PV1–PV5" / "PV1–PV10" /
      "analyses repeated across PVs and pooled"                      → rubin_rules
    ILSA domain default: PISA/TIMSS/PIRLS always ship PVs for achievement scores.
    If the paper models achievement and never mentions PV handling → average_pv.
    If the paper dichotomizes achievement into binary (e.g. "proficient
    vs not") using PV benchmarks, it still used PVs → infer from context.
   - missing_data_handling MUST be exactly one of:
     ['listwise_deletion', 'pairwise_deletion', 'mean_imputation', 'single_imputation',
      'knn_imputation', 'multiple_imputation', 'not_reported'].
     Synonym table:
       "listwise deletion" / "complete case" / "excluded incomplete" /
       "removed cases with missing" / "cases with missing data were
        removed" / "after exclusion of missing"                    → listwise_deletion
       "pairwise deletion" / "available case analysis"             → pairwise_deletion
       "mean substitution" / "mean replacement" / "imputed with mean" /
       "series mean" / "mode imputation" / "median imputation" /
       "SimpleImputer (mode)" / "SimpleImputer (median)" /
       "substituted mode values" / "imputed with median"           → mean_imputation
       "MICE" / "MI" / "missForest" / "missRanger" / "FIML" /
       "EM algorithm" / "expectation maximization" /
       "chained equations" / "hot-deck" /
       "kNN imputation" / "k-nearest neighbor imputation" /
       "predictive mean matching" / "PMM" /
       "MCMC imputation" / "Markov Chain Monte Carlo" /
       "two-level FCS" / "fully conditional specification" /
       "multivariate imputation" / "RF-based imputation" /
       "rblimp" / "blimp" / "Bayesian imputation" /
       "stochastic regression imputation" /
       any ML-based imputation method                              → multiple_imputation
       "zero imputation" / "zero fill" / "replaced with zero" /
       "filled with zero" / "imputed with zero"                    → mean_imputation
     CAUTION: SMOTE / oversampling / undersampling / data augmentation /
     SMOTETomek / ADASYN / CTGAN / VAE-augmentation are class-balancing
     or synthetic-data techniques, NOT missing data handling.
     Do NOT map them to any missing_data_handling value.
     CAUTION: "winsorized" / "trimmed at percentile" are outlier-treatment
     techniques, NOT missing data handling. Do NOT map them either.

2) COUNTRY CODES (ISO 3166-1 alpha-3):
   - country_code MUST always be a 3-letter UPPERCASE ISO code (e.g. 'TUR', 'USA', \
     'DEU', 'GBR', 'FRA', 'JPN', 'KOR', 'CHN', 'BRA', 'FIN', 'SGP', 'AUS').
   - NEVER write full country names, 2-letter codes, or non-standard abbreviations.
   - SPECIAL ECONOMIES & REGIONS — use these mappings:
     "Beijing-Shanghai-Jiangsu-Zhejiang" / "B-S-J-Z" / "B-S-J-G"
       / "BSJZ" / "Chinese mainland"                               → CHN
     "Chinese Taipei" / "Taiwan"                                   → TWN
     "Hong Kong" / "Hong Kong SAR"                                 → HKG
     "Macao" / "Macau" / "Macao SAR"                               → MAC
     "England" / "Northern Ireland"                                → GBR
     "Republic of Korea" / "South Korea" / "Korea"                 → KOR
     "Türkiye" / "Turkey"                                          → TUR
     "United Arab Emirates" / "UAE"                                → ARE
     "Dominican Republic"                                          → DOM
     "Czech Republic" / "Czechia"                                  → CZE
     "The Netherlands" / "Netherlands"                             → NLD
   - If a study covers 37+ OECD countries or 44+ TIMSS countries or 79+ PISA \
     countries, list ALL countries found in the text or tables. If only a \
     count is given (e.g. "80 countries"), extract every country explicitly \
     named in the manuscript and set n_students to null for unnamed ones. \
     Do NOT leave the countries list empty when the paper clearly analyzed \
     specific nations.

3) ML vs. TRADITIONAL STATISTICS (critical for ml_techniques):
   - For 'all_techniques' and 'primary', extract ONLY Machine Learning / \
     predictive-modeling algorithms. See the COMPREHENSIVE mapping table below.
   - DO NOT include traditional / descriptive / psychometric methods:
     PCA, factor analysis, t-tests, ANOVA, ANCOVA, MANOVA, chi-square,
     basic correlations, descriptive statistics, EFA/CFA, SEM, HLM/mixed-effects
     (unless explicitly used as an ML baseline), ESCS index computations,
     Latent Profile Analysis (LPA), Latent Class Analysis (LCA),
     IRT models, Rasch models, Partial Credit Model, measurement invariance,
     Interpretive Structural Modeling (ISM), bibliometric analysis,
     Shapley value decomposition (standalone; report SHAP only under XAI),
     DBSCAN / k-means / k-medoids / hierarchical clustering /
     Gaussian Mixture Model (GMM) (ONLY if used purely for unsupervised
     exploration without any predictive goal — if combined with a
     prediction pipeline, include the supervised learner, not the
     clustering step).
     Process Mining (Disco, ProM, fuzzy miner) — visualization/discovery
     tools, NOT ML algorithms.
     Finite Mixture Models / Latent Transition Analysis — psychometric
     mixture models, NOT ML.
   - Latent Profile Analysis (LPA) and Latent Class Analysis (LCA) are
     ALWAYS psychometric / mixture modeling methods and NEVER ml_techniques,
     even when used in process-data papers to identify behavioral profiles.
     The same applies to Confirmatory Factor Analysis (CFA), measurement
     invariance testing, and Hierarchical Linear Modeling (HLM) — these
     are statistical methods, not ML.
   - DO NOT include psychometric Diagnostic Classification Models (DCMs):
     HO-DINA, HO-GDINA, DINO, ACDM, LCDM, G-DINA — these are
     psychometric measurement models, NOT machine learning. Only include
     them if the paper explicitly frames them as ML classifiers.
   - DO NOT include Structural Topic Modeling (STM) UNLESS the paper
     uses STM output as features for a supervised prediction task.
     When STM is used solely for exploratory text analysis on abstracts
     or corpora (e.g., in review papers), it is NOT an ML technique.
   - DATA AUGMENTATION methods (SMOTE, CTGAN, VAE-based augmentation)
     are preprocessing steps, NOT ml_techniques. Mention them in
     missing_data_handling or confounders_identified if relevant,
     but never list them as primary or all_techniques entries.
   - COMPREHENSIVE algorithm name mapping (use canonical short names):
     ── TREE & ENSEMBLE ──
     "gradient boosted trees" / "GBT" / "GBM" / "GBDT"    → Gradient Boosting
     "XGBoost" / "extreme gradient boosting" / "XGB"       → XGBoost
     "LightGBM" / "Light GBM" / "LGBM" / "light gradient
      boosting"                                            → LightGBM
     "CatBoost" / "category boosting"                      → CatBoost
     "Histogram GBR" / "HGB" / "HistGradientBoosting"      → Histogram GBR
     "random forests" / "RF"                               → Random Forest
     "Extra Trees" / "ExtraTrees" / "extremely randomized
      trees" / "ET"                                        → Extra Trees
     "AdaBoost" / "adaptive boosting"                      → AdaBoost
     "Decision Tree" / "CART" / "C5.0" / "J48" /
      "classification tree" / "regression tree"            → Decision Tree
     "stacking" / "stacked ensemble" / "meta-model" /
      "stacked generalization"                             → Stacking
     "blending" / "blend"                                  → Blending
     "bagging" / "bootstrap aggregation"                   → Bagging
     "Conditional Inference Trees" / "CIT" / "ctree"       → Conditional Inference Trees
     "Conditional Inference Forests" / "CIF" / "cforest"   → Conditional Inference Forests
     "Boruta" (wraps RF for feature selection)             → Random Forest
     ── LINEAR / REGULARIZED ──
     "LASSO" / "L1 regression" / "glmnet L1"               → LASSO
     "Ridge Regression" / "L2 regression"                  → Ridge Regression
     "Elastic Net" / "L1+L2" / "glmnet" / "Enet"          → Elastic Net
     "Group Mnet" / "group MCP" / "group penalized"        → Group Mnet
     "Logistic Regression" (classification only)           → Logistic Regression
     "Linear Regression" / "MLR" (prediction/baseline)     → Linear Regression
     ── SVM / INSTANCE-BASED ──
     "SVM" / "SVC" / "SVR" / "support vector"              → SVM
     "k-NN" / "KNN" / "k-nearest neighbor"                 → k-NN
     ── PROBABILISTIC ──
     "Naive Bayes" / "GNB" / "NB" / "Gaussian Naive Bayes" → Naive Bayes
     "Bayesian Ridge" / "ARD"                              → Bayesian Ridge
     ── NEURAL NETWORKS & DEEP LEARNING ──
     "ANN" / "MLP" / "deep learning" / "feed-forward NN" /
      "multilayer perceptron"                              → Neural Network
     "LSTM" / "Long Short-Term Memory"                     → LSTM
     "GRU" / "Gated Recurrent Units"                       → GRU
     "CNN" / "Convolutional Neural Network"                → CNN
     "Autoencoder" / "variational autoencoder" / "VAE"     → Autoencoder
     "RNN" / "recurrent neural network"                    → RNN
     "Elman neural network" / "Jordan neural network"      → Neural Network
     ── CAUSAL ML ──
     "BART" / "Bayesian Additive Regression Trees"         → BART
     "BCF" / "Bayesian Causal Forests"                     → BCF
    ── FUZZY / HYBRID ──
    "ANFIS" / "neuro-fuzzy" / "adaptive neuro-fuzzy"      → ANFIS
    ── BAYESIAN ML ──
    "Bayesian Network" / "BN" / "Bayesian classifier" /
     "Bayesian belief network" / "directed acyclic graph
      classifier"                                         → Bayesian Network
    ── PENALIZED MULTILEVEL ──
    "glmmLasso" / "GLMM + LASSO" / "penalized GLMM" /
     "penalized mixed model"                              → glmmLasso
    "blackboost" / "conditional gradient boosting" /
     "mboost" / "model-based boosting"                    → Gradient Boosting
    ── NLP-BASED (when combined with supervised prediction) ──
    "Word2Vec + classifier" / "TF-IDF + classifier" /
     "Doc2Vec + classifier"                               → report the CLASSIFIER
    "RoBERTa" / "BERT" (for scoring/classification)       → report the architecture
    "Bag-of-Words + ANN" / "BoW + Neural Network"         → Neural Network
    ── OTHER CLASSIFIERS ──
    "Discriminant Analysis" / "LDA" / "QDA" /
     "linear discriminant analysis"                       → Discriminant Analysis
    "Gaussian Process" / "GP regression" / "GP classifier" → Gaussian Process
    ── KNOWLEDGE TRACING ──
    "DKT" / "Deep Knowledge Tracing"                       → Deep Knowledge Tracing
    ── SEMI-SUPERVISED / ACTIVE ──
    "active learning" / "semi-supervised learning" /
     "self-training" / "co-training" (when combined with
     a base classifier for label propagation)             → report the base classifier

4) WEIGHTING & REPLICATE DESIGN LOGIC:
   - student_weights_used: set true if ANY of these appear:
     "student weights", "sampling weights", "survey weights",
     "W_FSTUWT", "TOTWGT", "SCHWGT", "HOUWGT", "SENWGT", "MATWGT",
     "SCIWGT", "REAWGT", any variable starting with "W_" or ending in "WGT",
     "final weight", "senate weight", "house weight", "overall weight",
     "analysis weight", "probability weight", "design weight",
     "weighted estimation", "weighted analysis", "weighted mean",
     "adjusting for complex survey design", "adjusting for stratification",
     "adjusting for clustering", "multilevel weighting",
     "population-representative" + mention of weight application.
   - SOFTWARE-BASED WEIGHT INFERENCE: If the paper mentions using any of \
     these weight-aware tools, infer student_weights_used = true unless \
     explicitly contradicted:
     "IEA IDB Analyzer", "IDB Analyzer", "bifiesurvey", "BIFIEsurvey",
     "WeMix", "lavaan.survey", "survey package" (in R), "svy:" (Stata),
     "RALSA", "intsvy", "EdSurvey", "repest".
   - replicate_weights_used: set true if "BRR", "balanced repeated replication", \
     "Fay's method", "jackknife", "JK2", "JRR", "JK1", "replicate weights", \
     "jackknife repeated replication", "Taylor series linearization" appear.
   - weight_variable_name: exact variable name string if mentioned (e.g. 'W_FSTUWT').
   - ILSA domain default: if a study uses ILSA micro-data and never discusses \
     weights → student_weights_used = false (omission = likely unweighted).
   - EXPLICIT NON-USE PATTERN: Many ML-focused studies deliberately ignore \
     survey weights because ML algorithms (RF, XGBoost, SVM, Neural Networks) \
     do not natively support survey weights. If the paper uses ML models \
     on ILSA data and never mentions weights, set student_weights_used = false \
     and FILL weight_fields_interpretation explaining: "The study applied ML \
     algorithms that do not natively incorporate survey weights. The manuscript \
     does not discuss weighting, suggesting an unweighted analysis."
   - weight_fields_interpretation: ALWAYS REQUIRED — this field is NEVER null. \
     Write 3-4 analytical sentences detailing: (a) which dataset and cycle was \
     used and how the sample was filtered/cleaned, (b) whether complex survey \
     weights were applied and which variable (e.g. W_FSTUWT, TOTWGT), (c) if \
     weights were omitted, explain why (ML algorithms lack native weight support, \
     process data study, convenience sample, etc.), (d) any other notable data \
     preprocessing steps (outlier removal, subsample selection, grade filtering). \
     This field serves as a mandatory "Data Preparation Summary" for every paper.

5) NULL FIELDS INTERPRETATION (THE FALLBACK):
   - null_fields_interpretation: trigger ONLY if the overall extraction is \
     extremely sparse — e.g. missing sample sizes, missing ML algorithms, missing \
     PV handling, multiple metadata fields null. Write a structured diagnostic note \
     (plain text) explaining WHY the paper lacks data (e.g. "This is a theoretical \
     review paper, hence no sample size or ML models are evaluated." or "The \
     manuscript is a meta-analysis without original ILSA micro-data analysis.").
   - If the record is reasonably dense (most fields filled), this MUST BE null.

6) EXHAUSTIVE DATA & METHODOLOGY SEARCH (NO LAZY EXTRACTIONS):
   - DataBlock and SurveyDesign are the MOST CRITICAL sections. You must \
     aggressively scan "Methodology", "Data", "Measures", "Analytical Strategy", \
     "Sample", "Participants", "Data Processing", "Data Preprocessing", \
     "Data Cleaning", AND footnotes, table notes, and appendices.
   - EXTENDED WEIGHT SYNONYMS — also look for: "senate weights", "house weights", \
     "overall weights", "SENWGT", "MATWGT", "SCIWGT", "REAWGT", variables starting \
     with "W_" or ending in "WGT". For replicate weights also: "JRR", "jackknife \
     repeated replication", "Taylor series linearization".
   - SOFTWARE-BASED INFERENCE — if the paper mentions using IEA IDB Analyzer, \
     bifiesurvey, WeMix, lavaan.survey, EdSurvey, RALSA, intsvy, repest, or \
     any ILSA-specific analysis tool, these tools inherently apply survey weights \
     → infer student_weights_used = true.
   - INFERRING COMPLEX DESIGN — if the authors mention adjusting for "complex \
     survey design", "stratification", "clustering", or "multilevel weighting", \
     you MUST infer student_weights_used = true.
   - ML-SPECIFIC PATTERN — Many ML studies (RF, XGBoost, SVM, NN) on ILSA data \
     deliberately omit survey weights because these algorithms lack native weight \
     support. If the paper uses ML without mentioning weights, set \
     student_weights_used = false and weight_fields_interpretation must explain \
     this ML-specific omission pattern.
   - REVIEW / NON-EMPIRICAL PAPERS — If the paper is a systematic review, \
     bibliometric analysis, or theoretical framework without original ILSA \
     micro-data analysis, set student_weights_used = null, replicate_weights_used \
     = null, weight_variable_name = null, and explain in weight_fields_interpretation.
   - STRICT FAIL-SAFE ENFORCEMENT — weight_fields_interpretation is ALWAYS \
     REQUIRED regardless of whether weights were used or not. This is the \
     "Data Preparation & Weighting Summary" field. It must describe the dataset, \
     sample filtering, and weighting strategy in every case.
   - *** FATAL ERROR ***: returning weight_fields_interpretation as null or \
     empty is a schema violation. Pydantic will reject the output.

7) OFFICIAL REPORTS & FRAMEWORKS — see RULE 2 above (IEA technical reports, \
   assessment frameworks, encyclopedias, implementation manuals, user guides):
   - Follow RULE 2: rich outcome_summary, main_findings=[], confounders_identified=[].
   - For JSON boolean fields (student_weights_used, replicate_weights_used, \
     metadata.open_access): use JSON null when unknown — NEVER strings "N/A", "na", \
     "not applicable", "unknown", or similar (Pydantic rejects non-boolean strings).
   - For integer fields (total_students, n_students, year): use JSON null when \
     unknown — NEVER "N/A" or other placeholder strings.
   - For array fields (countries, all_techniques, confounders_identified, \
     main_findings): use [] when empty — not null and not "N/A".
   - weight_fields_interpretation remains REQUIRED (3-4 sentences from the manual) even \
     when weights are null/unknown; describe sampling design, PV/scaling, and which \
     weight variables analysts should use in the international database.

8) AGGRESSIVE SAMPLE, COUNTRY, DOI & CONFOUNDER EXTRACTION:
   - total_students: NEVER default to null without an exhaustive search. Scan \
     "Method", "Participants", "Data", "Data Cleaning", and "Results" sections \
     for keywords: "N =", "n =", "final sample", "consisted of", "analytic \
     sample", "valid responses", "after removing", "after exclusion", \
     "remaining students", "total of". Check tables and figure captions too.
   - countries & n_students: identify ALL countries analyzed AND their per-country \
     sample sizes. Aggressively scan "Table 1", "Sample Characteristics", \
     "Participants", and descriptive statistics tables for country-level N. \
     Do NOT leave n_students null if the table shows per-country counts. \
     If the abstract says "using PISA data from the USA", extract \
     country_code = "USA". If a table lists multiple countries, extract ALL \
     of them with ISO 3166-1 alpha-3 codes. Do not leave the list empty if \
     the data source inherently implies a country.
   - *** SAMPLE FILTERING AWARENESS (CRITICAL) ***: ILSA datasets are massive; \
     authors almost NEVER use the entire national or international file. Hunt \
     inclusion/exclusion steps in Method, Participants, Data, Data Cleaning, and \
     Preprocessing: CBA-only subsamples, item/task completion filters, grade bands, \
     school-type restrictions, listwise-deletion rules, process-log completeness, \
     outlier removal. Document precisely in sample_details.sample_filtering_criteria. \
     Do NOT assume the full country sample unless the paper explicitly states it.
   - DOI: Do NOT leave doi null when the document contains one. Scan in order: \
     (1) first-page header/footer and title block, (2) article information / \
     copyright page, (3) footnotes and publisher lines, (4) doi.org or dx.doi.org \
     URLs anywhere in the text. Patterns: "DOI:", "doi:", "https://doi.org/10.…", \
     bare "10.1016/j.cedpsych.2023.102196". If EXTRACTED_DOI_HINT is provided in \
     the user message, copy it exactly (unless the PDF clearly shows a different \
     DOI for this article). Strip URL prefixes; store only the DOI string.
   - confounders_identified: CONCEPTUAL CONTROLS & PREDICTORS — CRITICAL RULES: \
     *** CONCEPTUAL ONLY (NOT ML FEATURE COLUMNS) ***: List questionnaire, \
     background, and high-level process aggregates used as controls or named \
     predictors. NEVER list TF-IDF tokens, Word2Vec dimensions, n-grams, raw log \
     action codes (start/reset/end), slider state codes (0_0_0, 1_2_-2), or \
     per-action frequencies — those belong in main_findings / ml_techniques, \
     NOT here. Process-data papers with only engineered log features may return []. \
     *** NO GROUPING (ANTI-LAZINESS) ***: Create a SEPARATE object for EVERY \
     conceptual variable. NEVER combine variables (e.g. do NOT output "Gender and Age" \
     as one entry; output two objects). Comma-separated lists (ESCS, HOMEPOS) must \
     be separate objects, not one string. \
     *** EXHAUSTIVE (for conceptual vars) ***: Read methodology and variables. \
     Missing a background/control variable = critical failure. Do NOT inflate the \
     list with hundreds of engineered features. \
     Each entry is a STRUCTURED OBJECT with three fields: \
     (a) variable_code — STRICT CODE: use ONLY official ILSA acronyms (ESCS, UNDREM, \
         ST004D01TA) when they appear in the text. Tier 2: exact author-given label \
         (VOTAT, MATHEFF, ICTRES). Tier 3: snake_case slug from variable_name when no \
         official code exists — NOT invented pseudo-codes (e.g. meta_cognition_understanding, \
         remembering). NEVER literal "N/A" (Pydantic rejects it) — Tier 3 snake_case slug only. \
     (b) variable_name — STRICT NAMING: max 8 words; no copy-paste long phrases or \
         sentence starters ("The …", "Teacher-related variables such as …"). Distill to \
         labels like "Test effort", "Gender", "Metacognition (understanding & remembering)". \
     (c) category — exactly ONE of 13 categories (no "other" — mandatory assignment): \
       socioeconomic → ESCS, HOMEPOS, WEALTH, HISEI, BMMJ/BFMJ, parental education, \
                       books at home, family resources, cultural possessions \
       demographic → gender, age, immigration/migrant status, language at home, grade \
       student_attitude → self-efficacy, motivation, anxiety, enjoyment, belonging, \
                         self-concept, interest, value beliefs \
       student_behavior → study time, homework time/frequency, absenteeism, \
                         learning strategies, reading habits, metacognition \
       teacher → qualifications, experience, professional development, teaching \
                strategies, job satisfaction, instructional practices \
       school → school type (public/private), resources, class size, climate, \
               safety, autonomy, leadership, location (urban/rural) \
       ict → ICT resources (ICTRES), computer use, digital access, technology \
            integration in lessons, internet availability \
       curriculum → curriculum type, instructional time (SMINS/TMINS), content \
                   coverage, assessment practices \
       parent_home → parental involvement, parental support (EMOSUPS), home \
                    environment, family structure, homework supervision \
       process_data → ONLY aggregates: total/response time, VOTAT, visits per \
                     item, sequence length — NOT raw clicks, slider codes, TF-IDF \
       prior_achievement → previous test scores, prior-year grades, achievement \
                          in other domains (reading score as math predictor), \
                          WLE/PV scores used as control variables \
       peer_effects → classroom disciplinary climate, peer bullying, class-average \
                     achievement, classroom composition \
       system_level → country-level GDP, education expenditure, tracking age, \
                     national policy variables, GINI coefficient, system-level ratios \
     If unclear, pick the closest category (do NOT list engineered log/ML features; \
     aggregate country indices → system_level or socioeconomic).

9) LOGICAL DEDUCTION FOR ML TECHNIQUES (ml_techniques):
   - primary: DO NOT leave primary null if all_techniques is populated!
     a) If ONLY ONE algorithm is in all_techniques (e.g. ["LASSO"]), that \
        algorithm IS inherently the primary model — copy it to primary.
     b) If MULTIPLE algorithms are listed, scan "Results", "Abstract", or \
        "Conclusion" for: "performed best", "achieved the highest accuracy", \
        "outperformed", "best-performing model", "highest R²/AUC/F1". Assign \
        that winning model to primary.
     c) If the paper genuinely compares models without declaring a winner, \
        pick the one highlighted in the abstract or conclusion.
   - *** FATAL ERROR ***: primary left null while all_techniques has values \
     is a schema violation.

10) ENFORCING THE NULL INTERPRETATION FALLBACK:
   - If after exhaustive search total_students is still null, OR primary is \
     null while all_techniques is empty, you MUST trigger null_fields_interpretation. \
     Write 2-3 sentences diagnosing the omission (e.g. "The study is a scoping \
     review without an empirical sample" or "The authors listed LASSO and Random \
     Forest but did not report which model achieved the best metric.").
   - This rule complements Rule 5 — both may apply simultaneously.

9b) JUSTIFYING 'NOT_REPORTED' OR 'NOT_APPLICABLE' — MANDATORY EXPLANATION:
   - *** FATAL ERROR ***: If you set plausible_values_handling to "not_reported" \
     OR "not_applicable", OR missing_data_handling to "not_reported", you MUST \
     write 2-3 sentences in handling_not_reported_explanation. Leaving this \
     field null when triggered is a schema violation.
   - This rule applies to BOTH "not_reported" AND "not_applicable". Even if PVs \
     are genuinely not applicable, you MUST explain WHY they are not applicable.
   - Act as a critical peer-reviewer. Classify the reason into one of these:
     a) REPORTING GAP (methodological flaw) — the authors performed ML on ILSA \
        cognitive achievement data but completely failed to document how PVs or \
        missing data were handled. Flag this as a severe transparency issue. \
        Example: "The methodology section details the XGBoost architecture \
        extensively but completely fails to report how missing data was imputed \
        or deleted. Given that PISA datasets typically contain 5-15% missing \
        values, this omission represents a severe reporting gap."
     b) AFFECTIVE / NON-COGNITIVE DV — the study predicts Likert-scale items \
        (self-efficacy, anxiety, motivation, attitudes) rather than cognitive \
        achievement scores, so PVs are genuinely not applicable. \
        Example: "The dependent variable is students' awareness of global \
        competence (ST218 Likert items), not a cognitive achievement score. \
        Since PISA generates Plausible Values only for cognitive domains, PVs \
        are not applicable to this affective outcome."
     c) PROCESS DATA STUDY — the DV is binary correctness, IRT theta, or \
        behavioral indicators from log files, not PV-based achievement. \
        Example: "The study classifies problem-solving strategies from PISA \
        process log data using binary correctness as the outcome. PVs are \
        generated for cognitive achievement scales, not for process outcomes."
     d) DATA PAPER / FRAMEWORK / CURRICULUM ANALYSIS — the paper constructs a \
        dataset, theoretical framework, or analyzes curriculum content rather \
        than ILSA micro-data. \
        Example: "This is a dataset construction paper that harmonizes test \
        scores across assessments. It does not analyze individual student-level \
        ILSA micro-data, so PVs and missing data handling are not applicable."
     e) REVIEW / BIBLIOMETRIC — synthesizes literature, not micro-data.
     f) COUNTRY-LEVEL AGGREGATION — the study uses country-mean scores rather \
        than student-level PVs. \
        Example: "The analysis uses OECD-published country-level mean scores \
        rather than student-level Plausible Values. PV handling does not apply \
        to pre-aggregated country-level data."
   - DO NOT write lazy explanations like "It was not mentioned in the text." \
     You must explain the CONTEXT: what is the DV, why PVs don't apply or \
     why missing data handling was omitted, and whether this is a flaw or by design.
   - ONLY set handling_not_reported_explanation to null when BOTH \
     plausible_values_handling is one of {rubin_rules, single_pv, average_pv, \
     all_pv, mitml, wle, irt_theta} AND missing_data_handling is one of \
     {listwise_deletion, pairwise_deletion, mean_imputation, single_imputation, \
     knn_imputation, multiple_imputation}. In ALL other cases, this field is \
     MANDATORY.

11) RESEARCH DESIGN CLASSIFICATION (research_design_type):
   - Papers that predict/classify student outcomes using ML → "predictive"
   - Papers using causal ML (BART, BCF, propensity score matching, \
     diff-in-diff, instrumental variables, causal forests) → "causal_observational"
   - Papers with randomized experiments / RCTs → "causal_experimental"
   - Papers using unsupervised methods ONLY (clustering, LPA, topic modeling, \
     bibliometric analysis, process mining) without a prediction target → "exploratory"
   - Systematic reviews / meta-analyses / theoretical frameworks / \
     methodological papers → "exploratory"
   - Papers that combine prediction AND clustering (e.g. cluster then predict) \
     → "predictive" (the supervised component dominates)
   - Process-data papers that classify engagement or strategy profiles → "predictive"

12) ANTI-HALLUCINATION:
   - Never INVENT: DOIs, exact N, country codes, author names, weight variable \
     names, or algorithm names not present in the text.
   - Inference is allowed ONLY for categorical/boolean/enum fields where ILSA \
     domain knowledge provides a clear default (rules 1 and 4 above).
   - Numeric fields (total_students, n_students, year) MUST come from the text.

13) NATIONAL / REGIONAL LARGE-SCALE ASSESSMENTS AND OTHER ILSAs:
   - Papers using NAEP (USA), CEDRE (France), INVALSI (Italy), or other
     national LSAs should be treated with the SAME extraction rigor as
     ILSA papers. They are valid data sources for this pipeline.
   - NAEP uses plausible values → apply PV handling rules as for PISA/TIMSS.
   - CEDRE and INVALSI may use IRT-based scores rather than PVs → check
     methodology; if WLE or theta scores are used → plausible_values_handling
     = "not_applicable"; if PVs are generated → apply standard PV rules.
   - PIAAC (Programme for the International Assessment of Adult Competencies):
     Uses plausible values for literacy, numeracy, and PS-TRE domains.
     Apply the SAME PV handling inference rules as PISA/TIMSS.
     Process data from PIAAC PS-TRE items (log files, action sequences)
     → plausible_values_handling = "not_applicable" for the process
     component; "rubin_rules" or "average_pv" for the achievement component.
   - ICILS (International Computer and Information Literacy Study):
     Uses plausible values for CIL and CT scores → apply standard PV rules.
   - ICCS (International Civic and Citizenship Education Study):
     Uses plausible values for civic knowledge → apply standard PV rules.
     Engagement/attitude scales are IRT-scaled but NOT PVs →
     plausible_values_handling = "not_applicable" when the study models
     only attitudes/engagement without civic knowledge scores.
   - PISA-VET (OECD's vocational assessment, in development):
     Treat as framework/assessment-design paper unless it reports
     empirical student data; typically non-empirical at this stage.
   - For PSLC DataShop, LUCA simulations, licensure examinations,
     professional certification tests, or other digital learning
     platforms → treat as non-ILSA empirical data; PV handling is
     typically "not_applicable".

14) PROCESS DATA PAPERS (log files, clickstreams, response times, action sequences):
   - These papers analyze HOW students interact with computer-based assessments \
     rather than traditional achievement scores.
   - Typical data: action sequences, response times, mouse clicks, keystrokes, \
     navigation paths, time-on-task variables, VOTAT strategies, N-gram features, \
     directed graph features, network statistics (centralization, density, \
     flow hierarchy), time-to-first-action, number of visits / short visits, \
     Differential Response Time (DRT), Response Time Effort (RTE), \
     behavioral effort indicators, action sequence embeddings (Word2Vec, \
     Doc2Vec on action logs), LCS-based sequence similarity measures.
   - Process analysis tools (NOT ml_techniques): Process Mining (Disco, ProM), \
     action sequence autoencoders (when used purely for representation learning \
     without a downstream prediction task), and profiling via LPA/LCA.
   - plausible_values_handling: usually "not_applicable" because process data \
     studies typically use binary correctness (correct/incorrect), IRT-based \
     ability estimates (EAP, WLE, theta), or behavioral indicators rather \
     than PVs.
     EXCEPTION: when a process data paper ALSO models achievement scores \
     from PVs (e.g., predicting science PV-based performance from behavioral \
     effort), extract PV handling for the achievement component normally \
     (rubin_rules / average_pv / single_pv) and note "not_applicable" only \
     if the DV is purely process-based.
   - student_weights_used: usually false or null — process data studies focus \
     on behavioral patterns, not population-representative estimation.
   - ml_techniques: include ALL ML algorithms used for classification/prediction \
     of process outcomes. Common ones: Random Forest, LSTM, GRU, CNN, SVM, \
     Autoencoder, k-means (if combined with prediction), Neural Network, \
     XGBoost, Gradient Boosting, Logistic Regression, HMM (when used for \
     prediction, NOT when used purely as a psychometric measurement model).
   - DO NOT exclude algorithms just because the DV is process-based rather \
     than achievement-based. Any supervised/semi-supervised learner counts.
   - DO NOT include as ml_techniques: Diagnostic Classification Models \
     (HO-DINA, GDINA, DINO, ACDM), Partial Credit Models, IRT models, \
     Latent Profile Analysis, Process Mining software, or cluster editing \
     algorithms used purely for grouping.
   - research_design_type: "predictive" if classifying engagement/performance \
     from process features; "exploratory" if only clustering/profiling without \
     a prediction target.
   - Capture process-specific features (response time, action counts, \
     time-to-first-action, number of visits, VOTAT score, preparation \
     time, execution time) in confounders_identified with category='process_data'.

15) REVIEW / META-ANALYSIS / BIBLIOMETRIC PAPERS:
   - These papers synthesize existing literature rather than analyzing ILSA \
     micro-data directly.
   - source_category: "review_article" (systematic review, scoping review, \
     meta-analysis, bibliometric analysis, literature survey).
   - research_design_type: "exploratory".
   - total_students: null (no original empirical sample) UNLESS the review \
     reports a pooled sample size from included studies.
   - ml_techniques.primary: null; ml_techniques.all_techniques: [] UNLESS \
     the review itself applies ML (e.g., topic modeling on abstracts, \
     automated text classification of papers).
   - plausible_values_handling: "not_applicable".
   - missing_data_handling: "not_reported" unless the review describes a \
     specific protocol for handling missing studies/data.
   - MUST trigger null_fields_interpretation explaining: "This is a \
     systematic review / meta-analysis / bibliometric study without original \
     ILSA micro-data analysis."
   - student_weights_used: null; replicate_weights_used: null.

16) NON-EMPIRICAL / FRAMEWORK / APP-DEVELOPMENT PAPERS:
   - Papers that develop theoretical frameworks, assessment designs, mobile \
     apps, or methodological proposals without analyzing ILSA student data.
   - Examples: ISM-based cognitive model construction, CAT algorithm design, \
     mobile learning app development, scaling methodology papers, simulation \
     studies.
   - total_students: null (or the expert panel / pilot sample if reported).
   - ml_techniques: extract ONLY if the paper actually trains/evaluates ML \
     models. Framework proposals citing ML concepts do NOT count.
   - plausible_values_handling: "not_applicable".
   - research_design_type: "exploratory" for theoretical/framework papers; \
     "predictive" if simulations test predictive models.
   - MUST trigger null_fields_interpretation explaining the non-empirical nature.

17) ANTI-LAZINESS ENFORCEMENT — MANDATORY EXTRACTION RULES:
   - *** ZERO-TOLERANCE FOR UNNECESSARY NULLS ***
   - The following fields MUST NEVER be null without exhaustive justification:
     a) total_students — scan EVERY section for sample size indicators. \
        If the paper says "N=4,552 students" anywhere, extract 4552.
     b) countries — if the paper names ANY country, economy, or region \
        in connection with data analysis, extract its ISO code. NEVER \
        return an empty countries list for an empirical paper.
     c) sample_filtering_criteria — MUST explain how the analytic subsample was \
        carved from the full ILSA file (task/module filters, exclusions, grade \
        bands). Never leave vague or empty.
     d) n_students (per country) — aggressively scan tables (Table 1, \
        Sample Characteristics) for per-country sample sizes. Do NOT \
        leave n_students null if a table reports country-level counts.
     e) ml_techniques.primary — if all_techniques has ≥1 entry, primary \
        MUST be filled. This is a FATAL ERROR if violated.
     f) main_findings — FATAL ERROR if empty for any empirical ML / predictive study. \
        MUST contain at least one StructuredFinding per distinct target. Each row needs \
        dataset_used (e.g. 'TIMSS 2019 Grade 8 Science'), target_variable, top_predictors, \
        performance_metrics, and standardized_conclusion ending with "This indicates that …". \
        Official IEA/OECD technical reports/frameworks (RULE 2) MUST use [].
     g) outcome_summary — MUST also be 4-5 substantive sentences with specific \
        metrics and limitations. Complements main_findings; do not leave empty for \
        empirical predictive studies.
     h) research_design_type — MUST be classified for every paper.
     i) publication_type and source_category — MUST be classified.
     j) doi — scan headers, footers, footnotes, and copyright notices \
        for "10.xxxx/" patterns. Do NOT leave null if a DOI exists.
     k) confounders_identified — DO NOT return [] if the study has input \
        features/predictors. ONE object per variable. If the paper lists 20 \
        features, output 20 objects. NEVER group. NEVER truncate. \
        variable_code = ILSA code OR author label OR snake_case slug — never "N/A"; \
        variable_name = max 8 words; category = one of 13 literals (no "other").
     l) weight_fields_interpretation — ALWAYS REQUIRED, never null. \
        Write a data preparation summary for every paper.
   - For EVERY null field in your output, ask yourself: "Did I truly search \
     the abstract, methodology, results, tables, footnotes, and appendices?" \
     If not, search again.
   - PENALTY PATTERN: If you return more than 2 null fields in metadata or \
     more than 1 null field in data (excluding null_fields_interpretation), \
     you are being LAZY. Re-read and extract harder.

18) XAI & CAUSALITY SCRUTINY (Explainability ≠ Causality):
   - Many ML papers use SHAP, LIME, Accumulated Local Effects (ALE), or \
     feature importance rankings and then implicitly or explicitly suggest \
     causal relationships. This is methodologically unsound on cross-sectional \
     ILSA data.
   - When extracting, rigorously differentiate between 'predictive feature \
     importance' (what variables improve the model's prediction accuracy) and \
     'causal inference' (what variables actually cause the outcome).
   - If the paper uses only SHAP / LIME / Gini importance / permutation \
     importance → research_design_type stays "predictive", NOT causal.
   - ONLY classify as "causal_observational" if the paper employs actual \
     causal ML methods (BART, BCF, Propensity Score Matching, diff-in-diff, \
     instrumental variables, regression discontinuity) AND explicitly states \
     causal identification assumptions (SUTVA, parallel trends, unconfoundedness).
   - If the authors overstate causality based on predictive ML feature \
     importance alone, capture this in standardized_conclusion as a limitation.
   - XAI technique names (SHAP, LIME, counterfactual XAI, ALE plots) should \
     be mentioned in standardized_conclusion when used but NEVER in ml_techniques \
     (they are interpretation tools, not learning algorithms).

19) HIERARCHICAL DATA AWARENESS (Nested Structure & i.i.d. Violations):
   - ILSA data is strictly nested: students → classrooms → schools → countries.
   - Standard ML algorithms (regular XGBoost, Random Forest, SVM, NN) assume \
     independent and identically distributed (i.i.d.) observations, which is \
     violated by the clustered ILSA sampling design.
   - When extracting methodology, identify EXACTLY how the ML model accounted \
     for hierarchical structure:
     a) Multi-level ML models: Mixed-Effects Random Forests, glmmLasso, \
        multilevel XGBoost, hierarchical neural networks.
     b) Feature aggregation: school-level or country-level averages used as \
        additional predictors alongside student-level features.
     c) Post-hoc corrections: HLM or cluster-robust standard errors applied \
        after ML prediction stage.
     d) Country-stratified modeling: separate models per country or per school.
     e) No adjustment at all: flat ML on raw student-level data.
   - If the study applies standard flat ML on nested student-level data \
     WITHOUT survey weights, WITHOUT hierarchical adjustments, and WITHOUT \
     cluster-stratified modeling, note this in standardized_conclusion as a \
     methodological limitation. Do NOT silently ignore it.
   - student_weights_used is especially important here: ILSA weights partially \
     correct for clustering. If weights are omitted AND hierarchy is ignored, \
     both findings should be flagged.

20) PROCESS DATA DYNAMICS — TIME & SEQUENCE GRANULARITY:
   - Beyond Rule 13's general process data guidance, rigorously extract HOW \
     time and action sequences were operationalized:
   - TIME METRICS — differentiate the following:
     a) Raw total time (crude, loses item-level dynamics).
     b) Time-to-first-action (engagement onset latency).
     c) Item-standardised log response times (accounts for item difficulty).
     d) Effort regulation slope (change in response time across items).
     e) Differential Response Time (DRT = observed − expected time).
     f) Response Time Effort (RTE = binary rapid-guess thresholding).
   - SEQUENCE MINING — differentiate:
     a) Exact chronological sequence modeling (n-grams, HMM, LSTM on action \
        streams, sequence autoencoders, Markov chains).
     b) Lazy frequency aggregation (total clicks, total resets, action counts \
        without order preservation).
     c) Graph-based representations (directed graph features, network \
        statistics from action transitions).
   - STRATEGY INFERENCE — extract how cognitive strategy was operationalized:
     a) VOTAT detection (systematic vs. non-systematic exploration).
     b) Clustering of sequential paths (k-means on action embeddings).
     c) Manual expert coding of strategy types.
   - Capture micro-level operationalization in standardized_conclusion and ml_techniques. \
     confounders_identified gets ONLY high-level aggregates (e.g. VOTAT, total time), \
     never individual action codes or TF-IDF/Word2Vec feature columns.

21) ML ROBUSTNESS, CLASS IMBALANCE & DATA LEAKAGE:
   - Do NOT blindly extract overall model "Accuracy" as the sole metric.
   - CLASS IMBALANCE HANDLING:
     a) If the target variable is skewed (e.g., top 5% resilient students, \
        dropout prediction, cheating detection), extract whether the study \
        used SMOTE, ADASYN, under-sampling, class-weighted loss functions, \
        cost-sensitive learning, or threshold calibration.
     b) Remember: SMOTE/ADASYN/CTGAN/VAE-augmentation are CLASS BALANCING \
        methods, NOT missing_data_handling (see Rule 3 and checklist A).
   - EVALUATION METRICS:
     a) For imbalanced classification, extract F1-Score, Cohen's Kappa, \
        Precision-Recall AUC, Matthews Correlation Coefficient (MCC), or \
        balanced accuracy — these are robust to class skew.
     b) If ONLY "Accuracy" is reported for a known-imbalanced target, note \
        this as a limitation in standardized_conclusion.
   - DATA LEAKAGE:
     a) Check whether imputation, standardization, SMOTE, or feature \
        selection were performed INSIDE cross-validation folds (correct) or \
        on the ENTIRE dataset before splitting (leakage).
     b) If the paper reports suspiciously high performance (e.g., >95% on \
        complex ILSA tasks) without rigorous nested CV, flag potential \
        data leakage in standardized_conclusion.
   - VALIDATION STRATEGY: extract the exact method — k-fold CV, stratified \
     k-fold, leave-one-group-out (LOGO), nested CV, hold-out, repeated \
     random splits — and note it in standardized_conclusion.

═══════════════════════════════════════════════════════════════
OUTPUT SCHEMA
═══════════════════════════════════════════════════════════════

Return a single JSON with exactly two top-level keys: metadata, data.

metadata fields: file_name, title, authors, year, doi, venue, publication_type,
  open_access, source_category.

data fields: survey_design, plausible_values_handling, missing_data_handling,
  handling_not_reported_explanation, sample_details, ml_techniques,
  confounders_identified, main_findings, outcome_summary, research_design_type,
  null_fields_interpretation.

data.survey_design: student_weights_used, replicate_weights_used,
  weight_variable_name, weight_fields_interpretation (ALWAYS REQUIRED — never null).

data.sample_details: total_students, countries (each: country_code, n_students), \
  sample_filtering_criteria (ALWAYS REQUIRED — how the analytic subsample was defined).

data.ml_techniques: primary, all_techniques.

data.confounders_identified: list of objects, each with:
  variable_code (string or null), variable_name (short label), category (literal).

Do not emit any other top-level or nested keys.
No markdown fences, no preamble — valid JSON only.

"""


@dataclass
class ExtractionResult:
    file_name: str
    success: bool
    extraction: ILSAArticleMetadata | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_seconds: float
    error: str | None = None


class GPTExtractor:
    _use_structured: bool | None = None

    def __init__(
        self,
        api_key: str | None = None,
        model: str = MODEL_NAME,
        max_retries: int = 4,
        base_delay: float = 2.0,
    ):
        resolved_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self.client = OpenAI(api_key=resolved_key)
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay

    @staticmethod
    def _apply_doi_hint(
        extraction: ILSAArticleMetadata,
        processed: Optional["ProcessedPDF"] = None,
    ) -> None:
        """Backfill metadata.doi from PDF regex scan when the LLM returned null."""
        if extraction.metadata.doi:
            return
        if processed is None:
            return
        hint = processed.metadata.get("extracted_doi")
        if isinstance(hint, str) and hint.strip().startswith("10."):
            extraction.metadata.doi = hint.strip()
            return
        candidates = processed.metadata.get("doi_candidates")
        if isinstance(candidates, list):
            for c in candidates:
                if isinstance(c, str) and c.strip().startswith("10."):
                    extraction.metadata.doi = c.strip()
                    return

    @staticmethod
    def _prevalidate_structured_payload(
        extraction: ILSAArticleMetadata,
    ) -> ILSAArticleMetadata:
        """Coerce official-report literals before main_findings validator runs."""
        payload = extraction.model_dump(mode="python")
        meta_dict = payload.get("metadata")
        data_dict = payload.get("data")
        if not isinstance(data_dict, dict) or not isinstance(meta_dict, dict):
            return extraction
        report_mode = should_apply_report_literal_coercion(
            data_dict, meta_dict,
        ) or is_official_report_document(data_dict, meta_dict)
        if report_mode:
            _coerce_report_literals(data_dict, meta_dict)
            descriptive = _build_descriptive_findings_from_outcome(
                data_dict, meta_dict,
            )
            if descriptive:
                data_dict["main_findings"] = descriptive
        return ILSAArticleMetadata.model_validate(payload)

    @staticmethod
    def _post_process_model(
        extraction: ILSAArticleMetadata,
        processed: Optional["ProcessedPDF"] = None,
    ) -> ILSAArticleMetadata:
        """Post-process a structured-output extraction in place (mutates caller's model)."""
        validated = GPTExtractor._prevalidate_structured_payload(extraction)
        extraction.metadata = validated.metadata
        extraction.data = validated.data
        GPTExtractor._apply_doi_hint(extraction, processed)
        meta = extraction.metadata
        if not meta.title and meta.file_name:
            backfill = _title_from_file_name(meta.file_name)
            if backfill:
                meta.title = backfill
        for c in extraction.data.sample_details.countries:
            code = c.country_code.strip()
            if len(code) != 3 or not code.isalpha():
                mapped = COUNTRY_NAME_TO_ISO.get(code.lower())
                if mapped:
                    c.country_code = mapped
            c.country_code = c.country_code.upper()

        ml = extraction.data.ml_techniques
        if ml.primary is None and len(ml.all_techniques) == 1:
            ml.primary = ml.all_techniques[0]

        sd = extraction.data.survey_design
        if not sd.weight_fields_interpretation or not sd.weight_fields_interpretation.strip():
            if sd.student_weights_used is True:
                sd.weight_fields_interpretation = (
                    "The study applied survey weights to account for the complex "
                    "sampling design. No further details were extracted."
                )
            else:
                sd.weight_fields_interpretation = (
                    "No weighting information was explicitly reported in the "
                    "manuscript. The extraction could not determine the weighting "
                    "strategy from the available text."
                )

        d = extraction.data
        pv = d.plausible_values_handling
        md = d.missing_data_handling
        needs_explanation = pv in ("not_reported", "not_applicable") or md == "not_reported"
        if needs_explanation and not (
            d.handling_not_reported_explanation
            and d.handling_not_reported_explanation.strip()
        ):
            reasons = []
            if pv == "not_applicable":
                reasons.append(
                    f"plausible_values_handling is '{pv}' — the study likely "
                    "does not analyze cognitive achievement PVs (e.g., it may "
                    "focus on affective/attitudinal outcomes, curriculum data, "
                    "or non-ILSA micro-data)"
                )
            elif pv == "not_reported":
                reasons.append(
                    f"plausible_values_handling is '{pv}' — the authors did "
                    "not document how PVs were handled, which is a reporting gap"
                )
            if md == "not_reported":
                reasons.append(
                    f"missing_data_handling is '{md}' — the manuscript does "
                    "not describe any missing data strategy"
                )
            d.handling_not_reported_explanation = ". ".join(reasons) + "."
        elif not needs_explanation:
            d.handling_not_reported_explanation = None

        sd = d.sample_details
        if not sd.sample_filtering_criteria or not sd.sample_filtering_criteria.strip():
            sd.sample_filtering_criteria = _DEFAULT_SAMPLE_FILTERING

        d.confounders_identified = [
            Confounder.model_validate(c)
            for c in _normalize_confounders_list(
                [conf.model_dump() for conf in d.confounders_identified],
            )
        ]

        d.main_findings = [
            StructuredFinding.model_validate(f)
            for f in _normalize_main_findings_list(
                [f.model_dump() for f in d.main_findings],
            )
        ]
        return extraction

    @staticmethod
    def _catalog_extraction_addon(processed: "ProcessedPDF") -> str:
        """Extra instructions for Scopus/WoS catalogued empirical PDFs."""
        parts: list[str] = []
        folder_hint = processed.metadata.get("folder_organization_hint")
        if folder_hint:
            parts.append(
                "FOLDER_ORGANIZATION_HINT (weak prior for ILSA program/cycle when the "
                f"PDF is silent; NEVER contradict the article text): {folder_hint}\n"
            )
        if processed.source_database not in ("scopus", "web_of_science"):
            return "".join(parts)
        parts.append(
            "\nSCOPUS/WoS META-ANALYSIS PRIORITY — the user must know, per paper:\n"
            "  (1) WHICH DATA: exact ILSA program + cycle + grade/domain in every "
            "main_findings.dataset_used and outcome_summary opening sentence.\n"
            "  (2) WHICH COUNTRIES: data.sample_details.countries with ISO alpha-3 "
            "and n_students for EVERY analyzed economy when tables report counts.\n"
            "  (3) WHICH METHODS: ml_techniques (all models + primary), PV handling, "
            "missing-data strategy, survey weights (survey_design), and preprocessing "
            "in sample_filtering_criteria + weight_fields_interpretation.\n"
            "  (4) WHAT RESULTS: main_findings with numeric performance_metrics from "
            "Results tables; standardized_conclusion must end with 'This indicates that …'.\n"
            "  Peer-reviewed empirical ML papers MUST have ≥1 main_findings row and a "
            "substantive outcome_summary (~120 words) naming dataset, countries, best "
            "model, metrics, and top predictors.\n"
        )
        return "".join(parts)

    def _build_user_message(self, processed: "ProcessedPDF") -> list[dict]:
        sections_label = ", ".join(processed.sections.keys()) or "none"
        catalog_addon = self._catalog_extraction_addon(processed)
        title_hint = ""
        if processed.metadata.get("extracted_title"):
            title_hint = (
                f"\nEXTRACTED_TITLE_HINT (use only if the body never states "
                f"a title; still do not contradict the PDF): "
                f"{processed.metadata['extracted_title']}\n"
            )

        doi_hint = ""
        extracted_doi = processed.metadata.get("extracted_doi")
        doi_candidates = processed.metadata.get("doi_candidates") or []
        if extracted_doi:
            doi_hint = (
                f"\nEXTRACTED_DOI_HINT (regex scan of title page / headers — "
                f"USE for metadata.doi unless the article text proves otherwise): "
                f"{extracted_doi}\n"
            )
        elif doi_candidates:
            doi_hint = (
                f"\nEXTRACTED_DOI_CANDIDATES (pick the DOI for THIS article): "
                f"{', '.join(str(c) for c in doi_candidates[:5])}\n"
            )

        document_text = (
            f"FILE: {processed.file_name}\n"
            f"SOURCE: {processed.source_database}\n"
            f"SECTIONS_DETECTED: {sections_label}\n"
            f"{catalog_addon}{title_hint}{doi_hint}"
            f"--- BEGIN ARTICLE TEXT ---\n\n"
            f"{processed.extraction_text}\n\n"
            f"--- END ARTICLE TEXT ---\n\n"
            "Extract the structured JSON from the article above using the "
            "system prompt rules. Return valid JSON only."
        )

        interpretation_text = (
            "EXPERT INFERENCE CHECKLIST (same article; apply BEFORE finalising JSON):\n\n"

            "A) STRICT ENUMS — publication_type, source_category, research_design_type, "
            "plausible_values_handling, missing_data_handling must each be EXACTLY one "
            "of the allowed values listed in the system prompt. Use the synonym tables "
            "to map academic jargon. Never write free-text descriptions or new slugs.\n"
            "  PV SYNONYM TABLE: 'FIML' → multiple_imputation; 'complete cases' → "
            "listwise_deletion; 'averaged across five PVs' → average_pv; 'PV1' → "
            "single_pv; 'WLE scores' → wle; 'IRT theta' → irt_theta; 'EAP' → "
            "irt_theta; '10 PVs averaged' → all_pv.\n"
            "  MISSING DATA TABLE (map ALL to schema literals):\n"
            "    'MICE' / 'chained equations' / 'FIML' / 'MCMC' / 'PMM' / "
            "'EM algorithm' / 'expectation maximization' / 'hot-deck' / "
            "'rblimp' / 'blimp' / 'Bayesian imputation' / "
            "'stochastic regression imputation' / 'two-level FCS' / "
            "'fully conditional specification' → multiple_imputation.\n"
            "    'kNN imputation' / 'k-nearest neighbor imputation' → knn_imputation.\n"
            "    'missForest' / 'missRanger' / 'RF-based imputation' / "
            "'single regression imputation' / 'deterministic imputation' "
            "→ single_imputation.\n"
            "    'mean substitution' / 'series mean' / 'mode imputation' / "
            "'median imputation' / 'SimpleImputer' / 'zero imputation' / "
            "'replaced with zero' → mean_imputation.\n"
            "    'listwise deletion' / 'complete case' / 'removed missing' / "
            "'excluded incomplete' → listwise_deletion.\n"
            "    'pairwise deletion' / 'available case' → pairwise_deletion.\n"
            "  CAUTION: SMOTE / SMOTETomek / ADASYN / CTGAN / VAE-augmentation "
            "are CLASS BALANCING / synthetic-data techniques, NOT missing data handling.\n"
            "  CAUTION: 'winsorized' / 'trimmed at percentile' are OUTLIER TREATMENT, "
            "NOT missing data handling.\n\n"

            "B) COUNTRY CODES — every country_code must be ISO 3166-1 alpha-3 "
            "(3 uppercase letters). Never write full names or 2-letter codes.\n"
            "  SPECIAL MAPPINGS: B-S-J-Z / Beijing-Shanghai-Jiangsu-Zhejiang → CHN; "
            "B-S-J-G / Beijing-Shanghai-Jiangsu-Guangdong → CHN; "
            "Chinese Taipei / Taiwan → TWN; Hong Kong → HKG; Macao / Macau → MAC; "
            "England / Northern Ireland / Scotland / Wales → GBR; "
            "Türkiye / Turkey → TUR; "
            "Republic of Korea / South Korea → KOR; UAE → ARE; "
            "Czech Republic / Czechia → CZE; Dominican Republic → DOM; "
            "Flemish Community → BEL; Costa Rica → CRI.\n"
            "  NOTE: PIAAC, ICILS, and ICCS use the same ISO codes as PISA/TIMSS.\n\n"

            "C) ML TECHNIQUES ONLY — all_techniques and primary must contain ONLY "
            "Machine Learning / predictive-modeling algorithms. DO NOT include: "
            "PCA, factor analysis, t-tests, ANOVA, chi-square, correlations, "
            "descriptive statistics, EFA/CFA, SEM, HLM (unless ML baseline), "
            "Mantel-Haenszel, IRT model fitting, or ESCS computations.\n"
            "  INCLUDE: Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, "
            "Histogram GBR, SVM/SVR, LASSO, Elastic Net, Ridge Regression, "
            "Group Mnet, glmmLasso, KNN, Naive Bayes, Bayesian Ridge, "
            "Decision Tree / CART / C5.0, Conditional Inference Trees/Forests, "
            "Logistic Regression (when used as ML classifier), "
            "Neural Networks (ANN/MLP/DNN), LSTM, GRU, CNN, RNN, "
            "Autoencoder, BART/BCF (Bayesian causal ML), Bayesian Network, "
            "AdaBoost, Extra Trees, Stacking / Blending / ensemble meta-models, "
            "ANFIS, Discriminant Analysis, Gaussian Process, "
            "Deep Knowledge Tracing, Word2Vec, Doc2Vec, TF-IDF + classifier.\n"
            "  DO NOT INCLUDE: LPA, LCA, k-means/DBSCAN/k-medoids/hierarchical "
            "clustering/GMM (when purely exploratory without a supervised prediction "
            "goal), HLM, CFA/SEM, IRT, DCMs (HO-DINA/GDINA/DINO/ACDM), ISM, "
            "Process Mining (Disco/ProM), finite mixture models, "
            "bibliometric analysis.\n"
            "  Set primary to the best-performing model; if ambiguous pick the one "
            "highlighted in the abstract.\n\n"

            "D) SURVEY WEIGHTS & DATA PREPARATION SUMMARY (system rules 4 + 6):\n"
            "  Aggressively scan methodology, data, footnotes, and table notes for "
            "weight terms (W_FSTUWT, TOTWGT, senate/house weights, BRR, jackknife, "
            "complex survey design, stratification, clustering).\n"
            "  SOFTWARE-BASED INFERENCE: If paper uses IEA IDB Analyzer, bifiesurvey, "
            "WeMix, lavaan.survey, EdSurvey, RALSA, intsvy, or repest → infer "
            "student_weights_used = true (these tools inherently apply weights).\n"
            "  ML-SPECIFIC PATTERN: Many ML studies (RF, XGBoost, SVM, NN) on ILSA "
            "data deliberately omit survey weights. If ML is used without weight "
            "mention → set student_weights_used = false.\n"
            "  *** weight_fields_interpretation is ALWAYS REQUIRED (never null) ***\n"
            "  Write 3-4 sentences covering: (a) dataset/cycle used and sample "
            "filtering, (b) whether survey weights were applied and which variable, "
            "(c) if weights were omitted, why (ML omission pattern, process data, "
            "etc.), (d) notable preprocessing (outlier removal, grade filtering). "
            "This is the 'Data Preparation Summary' — mandatory for every paper.\n\n"

            "E) SAMPLE, N_STUDENTS & DOI (system rule 7) — exhaustively search "
            "Method, Participants, Data, Data Cleaning, Data Preprocessing, and "
            "Results for total N. Look for 'N =', 'final sample', 'analytic "
            "sample', 'valid responses', 'after removing/exclusion', 'remained "
            "for analysis'. Check tables and figure captions.\n"
            "  COUNTRIES + N_STUDENTS: For each country, aggressively scan tables "
            "(Table 1, Sample Characteristics, descriptive stats) for per-country "
            "sample sizes. Do NOT leave n_students null if a table shows the count.\n"
            "  SAMPLE_FILTERING_CRITERIA (CRITICAL): Authors rarely use the entire "
            "ILSA file. Document every inclusion/exclusion step — digital-module only, "
            "specific assessment unit or log task, grade level, school type, complete "
            "cases only, missing-data rules, process-sequence cleaning. If none "
            "reported: 'Used the full available sample for the specified countries.'\n"
            "  DOI: Scan first-page header/footer, title page, article info block, "
            "copyright line, footnotes, and doi.org links for '10.xxxx/' patterns. "
            "If EXTRACTED_DOI_HINT is present above, set metadata.doi to that value "
            "unless contradicted. Do NOT leave doi null when a DOI exists in the PDF.\n\n"

            "F) ML PRIMARY (system rule 8) — *** FATAL ERROR *** to leave primary "
            "null while all_techniques has values. If only ONE algorithm is listed, "
            "it IS the primary. If multiple, scan Results/Abstract/Conclusion for "
            "'performed best', 'highest accuracy/R²/AUC', 'outperformed', 'lowest "
            "RMSE/MAE/MAPE'. If truly ambiguous pick the one highlighted in the "
            "abstract or conclusion.\n\n"

            "G) CONFOUNDERS / PREDICTORS / FEATURES (system rule 7):\n"
            "*** ANTI-LAZINESS — CRITICAL RULES ***:\n"
            "  (1) NO GROUPING: ONE object per variable. If the paper lists 25 "
            "predictors, you MUST output 25 objects. NEVER combine 'Gender and Age' "
            "into a single entry — use two rows. NEVER output comma-separated lists "
            "as one string (ESCS, HOMEPOS → two objects).\n"
            "  (2) EXHAUSTIVE: Read the ENTIRE methodology, variables section, tables, "
            "and results. Do NOT stop after the first few variables. Missing a "
            "variable is a critical extraction failure.\n"
            "  (3) STRICT CODE: official ILSA acronyms only when in text; else author label "
            "or snake_case slug from variable_name (Tier 3). No invented pseudo-codes "
            "(meta_cognition_understanding, remembering). Literal 'N/A' FORBIDDEN.\n"
            "  (3b) STRICT NAMING: variable_name max 8 words — distill, no pasted sentences.\n"
            "  (4) NO MICRO-FEATURES: Do NOT list TF-IDF/Word2Vec columns, n-grams, "
            "raw log actions (start/reset/end), slider codes (0_0_0), or per-action "
            "frequencies. Those are ML inputs — describe them in standardized_conclusion only. "
            "Process-data-only studies may legitimately return [].\n"
            "  (5) category: exactly 13 literals — NO 'other'. Must assign the closest.\n"
            "Each entry has three fields:\n"
            "  variable_code: ILSA code OR conceptual author label OR slug.\n"
            "  variable_name: concise English label (max 8 words). Consistent naming.\n"
            "  category: socioeconomic | demographic | student_attitude | "
            "student_behavior | teacher | school | ict | curriculum | parent_home | "
            "process_data | prior_achievement | peer_effects | system_level.\n"
            "EXAMPLES:\n"
            "  {\"variable_code\": \"ESCS\", \"variable_name\": \"Socioeconomic status (ESCS)\", \"category\": \"socioeconomic\"}\n"
            "  {\"variable_code\": \"ST004Q01TA\", \"variable_name\": \"Gender\", \"category\": \"demographic\"}\n"
            "  {\"variable_code\": \"VOTAT\", \"variable_name\": \"VOTAT navigation behavior\", \"category\": \"process_data\"}\n"
            "  {\"variable_code\": \"public_private\", \"variable_name\": \"School type (public/private)\", \"category\": \"school\"}\n"
            "  {\"variable_code\": \"gdp_per_capita\", \"variable_name\": \"Country GDP per capita\", \"category\": \"system_level\"}\n"
            "Return [] if the paper is a review/theory paper OR uses only engineered "
            "log/ML features with no questionnaire/background controls.\n\n"

            "H) MAIN FINDINGS (Dataset → Input → Target → Output standard):\n"
            "Build a strict main_findings array — ONE object per DISTINCT target_variable. "
            "Do NOT create two near-duplicate rows with the same dataset_used and the same "
            "predictors for the same outcome. Compare methods (TF-IDF vs Word2vec, RF vs SVM) "
            "inside performance_metrics of ONE row. Use a second row ONLY when the "
            "target_variable truly changes (e.g., Math vs Science; student-level vs country-level). "
            "Do NOT add a separate row for k-means/clustering on the same dataset unless it "
            "is the paper's only analysis — otherwise summarize clustering in performance_metrics "
            "or outcome_summary.\n"
            "Map the exact empirical pipeline:\n"
            "  dataset_used: Assessment name + cycle year + grade + domain "
            "(e.g. 'TIMSS 2019 Grade 8 Science', 'PISA 2018 Reading', "
            "'PISA 2012 Problem Solving process data', 'PIAAC 2012 Numeracy').\n"
            "  target_variable: Exactly what was predicted.\n"
            "  top_predictors: Top 3-5 inputs; names MUST match confounders_identified "
            "variable_name when listed there.\n"
            "  performance_metrics: Hard numbers (Accuracy, R², RMSE, AUC, F1) or "
            "'Not reported'.\n"
            "  standardized_conclusion: REQUIRED template — "
            "'Using [dataset_used] data, the study leveraged [top_predictors] to predict "
            "[target_variable], finding that [direction/effect/result]. This indicates that "
            "[education/policy implication].' "
            "Flag SHAP-overstated causality, missing weights, or data leakage in the "
            "finding clause when relevant.\n"
            "EXAMPLE:\n"
            "  {\"dataset_used\": \"TIMSS 2019 Grade 8 Science\", "
            "\"target_variable\": \"Science achievement (Plausible Values)\", "
            "\"top_predictors\": [\"Socioeconomic background (SES)\", "
            "\"Curriculum type (Integrated vs. Separated)\"], "
            "\"performance_metrics\": \"Random Forest — R²: 0.51, RMSE: 74.92\", "
            "\"standardized_conclusion\": \"Using TIMSS 2019 Grade 8 Science data, the "
            "study leveraged socioeconomic background and curriculum type to predict science "
            "achievement, finding that socioeconomic background was the strongest predictor "
            "while curriculum type had only a weak direct effect. This indicates that "
            "equity-focused resource policies may matter more than curriculum structure "
            "alone for narrowing science gaps.\"}\n"
            "Return [] for reviews/theory papers AND official IEA/OECD technical reports "
            "(RULE 2) with no student-level ML prediction.\n\n"

            "H-REPORT) OFFICIAL IEA/OECD REPORTS / FRAMEWORKS (RULE 2):\n"
            "  - outcome_summary ~120-150 words on assessment design (sampling, PVs, items).\n"
            "  - main_findings = []; confounders_identified = [].\n"
            "  - student_weights_used / replicate_weights_used: null unless explicitly stated "
            "(never string 'N/A').\n"
            "  - weight_fields_interpretation: REQUIRED 3-4 sentences from the manual.\n\n"

            "H2) OUTCOME_SUMMARY (narrative companion to main_findings):\n"
            "ALSO write outcome_summary — 4-5 sentences (~120 words max) synthesizing "
            "the study's key results in prose. Include: dataset/subset used, best ML model, "
            "specific performance metrics (Accuracy, R², RMSE, AUC, F1), top predictors, "
            "and methodological caveats (no survey weights, SHAP≠causality, preprocessing "
            "leakage). This is the human-readable summary; main_findings is the tabular "
            "mapping. Do NOT put null-field commentary here.\n\n"

            "I) null_fields_interpretation — trigger if total_students is still "
            "null, or primary is null while all_techniques is empty, or extraction "
            "is extremely sparse. Write a diagnostic note explaining WHY. "
            "If the record is reasonably dense, this MUST be null.\n\n"

            "I2) handling_not_reported_explanation (system rule 9b) — *** FATAL "
            "ERROR IF MISSED ***:\n"
            "  - MANDATORY when plausible_values_handling = 'not_reported' OR "
            "'not_applicable', OR missing_data_handling = 'not_reported'.\n"
            "  - Even 'not_applicable' REQUIRES explanation. You must say WHY:\n"
            "    * AFFECTIVE DV: 'The DV is a Likert-scale attitude measure, not "
            "a cognitive PV-based score.'\n"
            "    * PROCESS DATA: 'The DV is binary correctness or IRT theta from "
            "log data, not PV-based achievement.'\n"
            "    * DATA PAPER: 'This constructs a dataset, not student-level ILSA "
            "micro-data analysis.'\n"
            "    * COUNTRY-LEVEL: 'Uses country-mean scores, not student-level PVs.'\n"
            "    * REPORTING GAP: 'Authors failed to document PV/missing data "
            "strategy — severe transparency issue.'\n"
            "  - DO NOT write 'It was not mentioned.' Explain the CONTEXT.\n"
            "  - null ONLY when PV is {rubin_rules, single_pv, average_pv, all_pv, "
            "mitml, wle, irt_theta} AND missing data is {listwise_deletion, "
            "pairwise_deletion, mean_imputation, single_imputation, knn_imputation, "
            "multiple_imputation}.\n\n"

            "J) ANTI-HALLUCINATION — never invent DOIs, exact N, country codes, "
            "weight variable names, or algorithm names absent from the text. "
            "Inference applies ONLY to categorical/boolean/enum fields.\n\n"

            "K) RESEARCH DESIGN — classify using system rule 10: predictive (ML "
            "prediction/classification), causal_observational (BART, BCF, PSM, "
            "diff-in-diff), causal_experimental (RCT), exploratory (clustering-only, "
            "LPA-only, reviews, theoretical). If paper combines prediction AND "
            "clustering → 'predictive'.\n\n"

            "L) PROCESS DATA PAPERS (system rule 13) — if paper analyzes log files, "
            "clickstreams, response times, action sequences, VOTAT strategies, "
            "N-grams, or mouse/keyboard traces:\n"
            "  - plausible_values_handling → 'not_applicable' (process data uses "
            "binary correctness or IRT ability, not PVs).\n"
            "  - student_weights_used → usually false (process data studies focus "
            "on behavioral patterns, not population estimation).\n"
            "  - ml_techniques: include ALL ML algorithms used for classification "
            "or prediction of process outcomes (RF, LSTM, GRU, CNN, SVM, "
            "Autoencoder, k-means if part of a prediction pipeline, etc.).\n"
            "  - research_design_type → 'predictive' if classifying engagement/"
            "performance; 'exploratory' if only profiling/clustering.\n"
            "  - Capture process-specific features in confounders_identified "
            "with category='process_data'.\n\n"

            "M) REVIEW / META-ANALYSIS / BIBLIOMETRIC PAPERS (system rule 14):\n"
            "  - source_category → 'review_article'.\n"
            "  - research_design_type → 'exploratory'.\n"
            "  - total_students → null (no original sample).\n"
            "  - ml_techniques.primary → null; all_techniques → [] UNLESS the "
            "review itself applies ML (e.g., topic modeling on abstracts).\n"
            "  - plausible_values_handling → 'not_applicable'.\n"
            "  - missing_data_handling → 'not_reported'.\n"
            "  - MUST trigger null_fields_interpretation explaining it is a review.\n\n"

            "N) NON-EMPIRICAL / FRAMEWORK / APP-DEVELOPMENT / DATA PAPERS (system rule 15):\n"
            "  - Papers designing theoretical frameworks, mobile apps, CAT "
            "algorithms, scaling methodologies, or DATA PAPERS that construct/"
            "harmonize datasets without ILSA micro-data analysis.\n"
            "  - publication_type: 'journal' for data papers published in journals "
            "(e.g., 'Data' journal); 'report' for technical data documentation.\n"
            "  - source_category: 'methodology_paper' for data papers and frameworks.\n"
            "  - total_students → null (or expert panel size if applicable).\n"
            "  - plausible_values_handling → 'not_applicable'.\n"
            "  - research_design_type → 'exploratory' for data description / "
            "dataset construction papers.\n"
            "  - MUST trigger null_fields_interpretation explaining the study type.\n\n"

            "O) XAI & CAUSALITY (system rule 17):\n"
            "  - If the paper uses SHAP / LIME / ALE / Gini importance / permutation "
            "importance, report these in standardized_conclusion but NEVER in ml_techniques.\n"
            "  - Do NOT classify as 'causal_observational' unless actual causal methods "
            "(BART, BCF, PSM, diff-in-diff, IV, RDD) with stated assumptions are used.\n"
            "  - If authors claim 'X causes Y' based solely on feature importance, "
            "flag this as overstated causality in standardized_conclusion.\n\n"

            "P) HIERARCHICAL DATA (system rule 18):\n"
            "  - Note in standardized_conclusion whether the ML model accounted for the nested "
            "ILSA data structure (multilevel ML, feature aggregation, cluster-stratified "
            "models, or survey weights).\n"
            "  - If standard flat ML was applied to student-level data with NO "
            "hierarchical adjustments and NO weights, flag it as a methodological "
            "limitation in standardized_conclusion.\n\n"

            "Q) PROCESS DATA GRANULARITY (system rule 19):\n"
            "  - For process data papers, differentiate in standardized_conclusion:\n"
            "    (a) time metric type: raw total time vs. item-standardised log "
            "response times vs. effort regulation slope vs. DRT/RTE;\n"
            "    (b) sequence modeling: exact chronological (n-grams, HMM, LSTM on "
            "actions, Markov) vs. lazy frequency counts (total clicks);\n"
            "    (c) strategy inference: VOTAT detection, path clustering, expert "
            "coding.\n"
            "  - List specific process features in confounders_identified with "
            "category='process_data' (e.g. response time, action count, VOTAT score).\n\n"

            "R) ML ROBUSTNESS & LEAKAGE (system rule 20):\n"
            "  - Extract ALL performance metrics reported (Accuracy, F1, AUC, Kappa, "
            "MCC, RMSE, MAE, R²). If only 'Accuracy' is reported for an imbalanced "
            "classification, note it as a limitation.\n"
            "  - Extract class imbalance handling (SMOTE, under-sampling, class "
            "weights) — remember these are NOT missing_data_handling.\n"
            "  - Note the validation strategy (k-fold CV, nested CV, hold-out, LOGO).\n"
            "  - If preprocessing (imputation, scaling, SMOTE, feature selection) was "
            "done BEFORE train/test split, flag potential data leakage.\n\n"

            "S) FINAL ANTI-LAZINESS CHECK (system rule 16):\n"
            "  Before submitting your JSON, count your null fields:\n"
            "  - total_students null for an empirical paper? → Re-read Method section.\n"
            "  - n_students null for listed countries? → Scan Table 1 again.\n"
            "  - countries list empty for a paper that names countries? → FATAL ERROR.\n"
            "  - sample_filtering_criteria missing or vague? → Re-read Method/Data "
            "Cleaning for inclusion/exclusion rules.\n"
            "  - primary null but all_techniques has entries? → FATAL ERROR.\n"
            "  - doi null? → Check headers, footers, footnotes for 10.xxxx/ patterns.\n"
            "  - confounders_identified empty for an ML study? → ONE object per variable "
            "(code = ILSA/author label/slug — never literal N/A; category from 13 "
            "literals, no 'other'). NEVER group.\n"
            "  - weight_fields_interpretation null or empty? → FATAL ERROR (always required).\n"
            "  - handling_not_reported_explanation null when PV='not_applicable' or "
            "'not_reported', or missing data='not_reported'? → FATAL ERROR.\n"
            "  - main_findings empty for an empirical ML study? → FATAL ERROR; add ≥1 "
            "StructuredFinding row per target.\n"
            "  - outcome_summary vague or <3 sentences? → Add specific metrics and limitations.\n"
            "  - More than 2 null metadata fields? → You are being LAZY. Extract more.\n"
            "  - More than 1 null data field (excl. null_fields_interpretation)? → Re-scan.\n"
            f"{catalog_addon}"
        )

        return [
            {"type": "text", "text": document_text},
            {"type": "text", "text": interpretation_text},
        ]

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * PRICE_INPUT_PER_1M / 1_000_000
            + output_tokens * PRICE_OUTPUT_PER_1M / 1_000_000
        )

    @staticmethod
    def _coerce_pv_literal(value) -> str:
        """Map free-text / invalid PV labels to schema literals."""
        allowed = frozenset({
            "rubin_rules", "single_pv", "average_pv", "mitml",
            "not_applicable", "not_reported",
            "wle", "irt_theta", "all_pv",
        })
        if value in allowed:
            return value
        if not isinstance(value, str):
            return "not_reported"
        t = value.lower().replace("-", "_").replace(" ", "_")
        if "rubin" in t or "combined_estimates" in t:
            return "rubin_rules"
        if "mitml" in t or "mplus" in t:
            return "mitml"
        sw_rubin = (
            "bifiesurvey", "repest", "intsvy", "edsurvey",
            "idb_analyzer", "idb analyzer", "ralsa",
            "lavaan.survey", "wemix",
        )
        if any(pkg in t for pkg in sw_rubin):
            return "rubin_rules"
        if (
            "five_pv" in t or "5_pv" in t
            or "five_plausible" in t
            or "ten_plausible" in t or "10_pv" in t and "pool" in t
            or "pv1_pv5" in t or "pv1_pv10" in t
            or "analyses_repeated_across" in t and "pv" in t
            or "repeated_across_pvs" in t
        ):
            return "rubin_rules"
        if (
            "not_applicable" in t
            or "no_pv" in t
            or "no_pvs" in t
            or "no_plausible" in t
            or "does_not_use" in t
            or "process_data" in t
            or "log_file" in t
            or "review_paper" in t
            or "non_empirical" in t
            or "no_assessment_score" in t
        ):
            return "not_applicable"
        if "wle" in t or "weighted_likelihood" in t or "warm" in t and "estimat" in t:
            return "wle"
        if (
            "irt_theta" in t
            or "eap" in t and ("estimat" in t or "score" in t or "abilit" in t)
            or "theta" in t and ("irt" in t or "estimat" in t or "latent" in t)
            or "latent_trait" in t
        ):
            return "irt_theta"
        if (
            "first_plausible" in t
            or "single_pv" in t
            or "pv1_only" in t
            or t == "pv1"
            or ("separate" in t and "plausible" in t)
            or "per_pv" in t
            or "per_plausible" in t
            or "one_pv" in t
            or ("target" in t and "indicator" in t)
            or ("binary" in t and ("pv" in t or "plausible" in t))
            or "pv1math" in t or "pv1read" in t or "pv1scie" in t
            or "pv2scie" in t or "pv2math" in t
        ):
            return "single_pv"
        if (
            "all_pv" in t
            or "all_10" in t and "pv" in t
            or "ten_pv" in t
            or "10_pv" in t
            or "each_pv" in t and "separate" in t
            or "pv1_through" in t
            or "pv1_to_pv10" in t
            or "all_plausible_values_separately" in t
        ):
            return "all_pv"
        if (
            "average" in t and "pv" in t
            or "all_plausible" in t
            or "across_pv" in t
            or "across_pvs" in t
            or "mean_pv" in t
            or "mean_of" in t and "plausible" in t
            or "averaged" in t and "plausible" in t
        ):
            return "average_pv"
        if "plausible" in t or "_pv" in t or "pv_" in t:
            return "not_reported"
        return "not_reported"

    @staticmethod
    def _coerce_md_literal(value) -> str:
        """Map free-text / invalid missing-data labels to schema literals."""
        allowed = frozenset({
            "listwise_deletion", "pairwise_deletion", "mean_imputation",
            "single_imputation", "knn_imputation",
            "multiple_imputation", "not_reported",
        })
        if value in allowed:
            return value
        if not isinstance(value, str):
            return "not_reported"
        t = value.lower().replace("-", "_").replace(" ", "_")
        if len(t) > 120 or "the_manuscript" in t or "the_paper" in t or "the_dataset" in t:
            return "not_reported"
        if "no_missing" in t or "without_missing" in t or "no_missing_data" in t:
            return "not_reported"
        if (
            "winsoriz" in t
            or "winsoris" in t
            or "trimmed" in t and "percentile" in t
        ):
            return "not_reported"
        if (
            "smote" in t
            or "smotetomek" in t
            or "adasyn" in t
            or "oversampl" in t
            or "undersamp" in t
            or "class_balanc" in t
            or "resampl" in t and ("minority" in t or "imbalanc" in t)
            or "ctgan" in t
            or "vae_augment" in t
            or "synthetic_data" in t and ("generat" in t or "augment" in t or "balanc" in t)
        ):
            return "not_reported"
        if "pairwise" in t:
            return "pairwise_deletion"
        if (
            "listwise" in t
            or "complete_case" in t
            or ("exclusion" in t and "missing" in t)
            or ("removed" in t and "missing" in t)
            or ("deleted" in t and "missing" in t)
            or ("cases_with_missing" in t and ("removed" in t or "excluded" in t or "dropped" in t))
        ):
            return "listwise_deletion"
        if (
            ("mean" in t and "imput" in t)
            or ("mean" in t and "substitut" in t)
            or ("mean" in t and "replac" in t)
            or ("series_mean" in t)
            or ("mode" in t and ("imput" in t or "substitut" in t or "replac" in t))
            or ("median" in t and ("imput" in t or "substitut" in t or "replac" in t))
            or ("simple_imputer" in t and ("mode" in t or "median" in t or "mean" in t))
            or ("simpleimputer" in t)
            or ("substituted_mode" in t)
            or ("substituted_median" in t)
            or ("zero_fill" in t)
            or ("zero_imputation" in t)
            or ("replaced_with_zero" in t)
            or ("filled_with_zero" in t)
        ):
            return "mean_imputation"
        if (
            "knn" in t
            or "k_nearest" in t
            or "k_nn" in t
            or "nearest_neighbor" in t
        ) and "imput" in t:
            return "knn_imputation"
        if (
            "knn_imput" in t
            or "knn imput" in t.replace("_", " ")
        ):
            return "knn_imputation"
        if (
            "missforest" in t
            or "miss_forest" in t
            or "missranger" in t
            or "miss_ranger" in t
            or "rf_based" in t and "imput" in t
            or "random_forest" in t and "imput" in t
            or "single_imput" in t
            or "deterministic_imput" in t
            or "single_regression" in t and "imput" in t
        ):
            return "single_imputation"
        if (
            "imput" in t
            or "mice" in t
            or "fiml" in t
            or "full_information" in t
            or "maximum_likelihood" in t
            or "em_algorithm" in t
            or "hot_deck" in t
            or "hot deck" in t
            or "chained_equations" in t
            or "fully_conditional" in t
            or ("machine_learning" in t and "missing" in t)
            or t == "imputation"
            or "mcmc" in t
            or "markov_chain" in t
            or "pmm" in t
            or "predictive_mean" in t
            or "two_level_fcs" in t
            or "multiple_imput" in t
            or "multivariate_imput" in t
            or "rblimp" in t
            or "blimp" in t
            or "expectation_maximiz" in t
            or "bayesian_imput" in t
            or "stochastic_regress" in t
        ):
            return "multiple_imputation"
        if (
            "dropped_missing" in t
            or "excluded_missing" in t
            or "removed_incomplete" in t
            or "omitted_missing" in t
            or "filtered_out" in t and "missing" in t
            or "discarded" in t and "missing" in t
            or "dropped" in t and "incomplete" in t
            or "cases_removed" in t
        ):
            return "listwise_deletion"
        return "not_reported"

    @staticmethod
    def _sanitize(parsed_data: dict) -> dict:
        """
        Post-process model output to fix known failure modes before Pydantic validation.
        Modifies parsed_data in place and returns it.
        """
        INVALID_STR = frozenset({
            "not_reported", "not_applicable", "N/A", "n/a", "unknown", "",
        })

        def _normalize_literal(value, field_name, allowed, default):
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                text = " ".join(
                    str(v) for v in value.values() if isinstance(v, str)
                ).lower()
                if field_name == "plausible_values_handling":
                    if "plausible" in text or "pv" in text:
                        return "not_reported"
                    if "no" in text or "not" in text and "pv" not in text:
                        return "not_applicable"
                if field_name == "missing_data_handling":
                    if "imput" in text or "mice" in text or "miss" in text:
                        return "multiple_imputation"
                    if "listwise" in text or "complete case" in text or "complete-case" in text:
                        return "listwise_deletion"
                    return "not_reported"
            return default

        DATA_KEYS = (
            "survey_design",
            "plausible_values_handling",
            "missing_data_handling",
            "handling_not_reported_explanation",
            "sample_details",
            "ml_techniques",
            "confounders_identified",
            "main_findings",
            "outcome_summary",
            "research_design_type",
            "null_fields_interpretation",
        )

        if not isinstance(parsed_data.get("data"), dict):
            parsed_data["data"] = {}
        data = parsed_data["data"]

        # Legacy flat JSON → nest under data
        for k in DATA_KEYS:
            if k in parsed_data:
                if k not in data:
                    data[k] = parsed_data.pop(k)
                else:
                    parsed_data.pop(k, None)

        for key in list(parsed_data.keys()):
            if key not in ("metadata", "data"):
                parsed_data.pop(key, None)

        meta_early = parsed_data.get("metadata")
        meta_early_dict = meta_early if isinstance(meta_early, dict) else None
        report_mode = should_apply_report_literal_coercion(
            data, meta_early_dict,
        ) or is_official_report_document(data, meta_early_dict)
        if report_mode:
            _coerce_report_literals(data, meta_early_dict)
            descriptive = _build_descriptive_findings_from_outcome(
                data, meta_early_dict,
            )
            if descriptive:
                data["main_findings"] = _normalize_main_findings_list(descriptive)
        else:
            _normalize_findings_fields(data, meta_early_dict)
            if not data.get("main_findings") and substantive_outcome_summary(data):
                descriptive = _build_descriptive_findings_from_outcome(
                    data, meta_early_dict,
                )
                if descriptive:
                    data["main_findings"] = _normalize_main_findings_list(descriptive)

        for key in list(data.keys()):
            if key not in DATA_KEYS:
                data.pop(key, None)

        ml = data.get("ml_techniques")
        if isinstance(ml, dict):
            for legacy in ("feature_selection", "baseline_model", "xai_method"):
                ml.pop(legacy, None)
            primary = ml.get("primary")
            if primary is None or (isinstance(primary, str) and primary in INVALID_STR):
                ml["primary"] = None
            elif isinstance(primary, list):
                ml["primary"] = None
            if isinstance(ml.get("all_techniques"), list):
                ml["all_techniques"] = [
                    t for t in ml["all_techniques"]
                    if isinstance(t, str) and t not in INVALID_STR
                ]
            elif isinstance(ml.get("all_techniques"), str):
                ml["all_techniques"] = [ml["all_techniques"]]
            elif ml.get("all_techniques") is None:
                ml["all_techniques"] = []

            if ml["primary"] is None and len(ml["all_techniques"]) == 1:
                ml["primary"] = ml["all_techniques"][0]

        sd = data.get("sample_details")
        if isinstance(sd, dict):
            countries = sd.get("countries")
            if isinstance(countries, list):
                cleaned = []
                for c in countries:
                    if not isinstance(c, dict):
                        continue
                    code = c.get("country_code")
                    if not code or not isinstance(code, str):
                        continue
                    code = code.strip()
                    if len(code) != 3 or not code.isalpha():
                        mapped = COUNTRY_NAME_TO_ISO.get(code.lower())
                        if mapped:
                            code = mapped
                        else:
                            continue
                    c["country_code"] = code.upper()
                    n = c.get("n_students")
                    if not isinstance(n, int):
                        c["n_students"] = None
                    cleaned.append(c)
                sd["countries"] = cleaned
            sfc = sd.get("sample_filtering_criteria")
            if not isinstance(sfc, str) or not sfc.strip() or sfc.strip() in INVALID_STR:
                sd["sample_filtering_criteria"] = _DEFAULT_SAMPLE_FILTERING
            else:
                sd["sample_filtering_criteria"] = sfc.strip()
            _backfill_countries_from_extracted_text(data, meta_early_dict)
            _backfill_total_students_from_extracted_text(data, meta_early_dict)

        sdw = data.get("survey_design")
        if isinstance(sdw, dict):
            for bool_key in ("student_weights_used", "replicate_weights_used"):
                if bool_key in sdw:
                    sdw[bool_key] = coerce_optional_bool(sdw[bool_key])
            wfi = sdw.get("weight_fields_interpretation")
            if not isinstance(wfi, str) or wfi.strip() in INVALID_STR or not wfi.strip():
                if sdw.get("student_weights_used") is True:
                    sdw["weight_fields_interpretation"] = (
                        "The study applied survey weights to account for the "
                        "complex sampling design. No further details were "
                        "extracted from the manuscript."
                    )
                else:
                    sdw["weight_fields_interpretation"] = (
                        "No weighting information was explicitly reported. "
                        "The extraction could not determine the weighting "
                        "strategy from the available text."
                    )
            wn = sdw.get("weight_variable_name")
            if isinstance(wn, str) and wn in INVALID_STR:
                sdw["weight_variable_name"] = None

        nfi = data.get("null_fields_interpretation")
        if isinstance(nfi, str) and nfi.strip() in INVALID_STR:
            data["null_fields_interpretation"] = None

        hnre = data.get("handling_not_reported_explanation")
        if isinstance(hnre, str) and hnre.strip() in INVALID_STR:
            data["handling_not_reported_explanation"] = None
        pv = data.get("plausible_values_handling", "")
        md = data.get("missing_data_handling", "")
        needs_explanation = pv in ("not_reported", "not_applicable") or md == "not_reported"
        if needs_explanation and not (isinstance(hnre, str) and hnre.strip() and hnre.strip() not in INVALID_STR):
            data["handling_not_reported_explanation"] = (
                "The extraction pipeline detected that plausible_values_handling "
                f"is '{pv}' and/or missing_data_handling is '{md}', but the LLM "
                "did not provide a diagnostic explanation. This may indicate a "
                "reporting gap in the original manuscript."
            )
        elif not needs_explanation:
            data["handling_not_reported_explanation"] = None

        VALID_PUB_TYPES = frozenset({
            "journal", "conference", "book_chapter", "preprint", "report", "thesis",
        })
        VALID_SOURCE_CATS = frozenset({
            "technical_report", "review_article", "methodology_paper",
            "peer_reviewed_research",
        })
        VALID_DESIGN_TYPES = frozenset({
            "predictive", "causal_observational", "causal_experimental", "exploratory",
        })

        meta = parsed_data.get("metadata")
        if isinstance(meta, dict):
            for legacy in (
                "extraction_timestamp",
                "extraction_cost_usd",
                "prompt_tokens",
                "completion_tokens",
                "catalog_folder_path",
            ):
                meta.pop(legacy, None)
            for field in ("venue", "title"):
                if meta.get(field) in INVALID_STR:
                    meta[field] = None
            if not meta.get("title"):
                backfill = _title_from_file_name(meta.get("file_name", ""))
                if backfill:
                    meta["title"] = backfill
            if not meta.get("doi"):
                from src.extractors.pdf_processor import extract_dois_from_text

                dois = extract_dois_from_text(_doi_backfill_blob(data, meta))
                if dois:
                    meta["doi"] = dois[0]
            doi_val = meta.get("doi")
            if doi_val in INVALID_STR or doi_val == "null":
                meta["doi"] = None
            elif isinstance(doi_val, str):
                doi_val = doi_val.strip()
                for prefix in (
                    "https://doi.org/",
                    "http://doi.org/",
                    "https://dx.doi.org/",
                    "http://dx.doi.org/",
                ):
                    if doi_val.lower().startswith(prefix):
                        doi_val = doi_val[len(prefix):]
                meta["doi"] = doi_val.rstrip(".,;:)>]}") or None
            if not isinstance(meta.get("authors"), list):
                meta["authors"] = []
            pt = meta.get("publication_type")
            if isinstance(pt, str) and pt not in VALID_PUB_TYPES:
                normed = pt.lower().replace("-", "_").replace(" ", "_")
                if normed in VALID_PUB_TYPES:
                    meta["publication_type"] = normed
                else:
                    matched = None
                    if "conference" in normed or "proceeding" in normed or "symposium" in normed:
                        matched = "conference"
                    elif "review" in normed or "survey" in normed or "systematic" in normed:
                        matched = "journal"
                    elif "process_data" in normed or "paper" in normed:
                        matched = "journal"
                    elif "data_paper" in normed or "data_article" in normed:
                        matched = "journal"
                    elif "thesis" in normed or "dissertation" in normed:
                        matched = "thesis"
                    elif "preprint" in normed or "arxiv" in normed or "working_paper" in normed:
                        matched = "preprint"
                    elif "report" in normed or "technical" in normed:
                        matched = "report"
                    elif "book" in normed or "chapter" in normed:
                        matched = "book_chapter"
                    else:
                        for v in VALID_PUB_TYPES:
                            if v in normed or normed.startswith(v):
                                matched = v
                                break
                    meta["publication_type"] = matched

            sc = meta.get("source_category")
            if isinstance(sc, str) and sc not in VALID_SOURCE_CATS:
                normed = sc.lower().replace("-", "_").replace(" ", "_")
                if normed in VALID_SOURCE_CATS:
                    meta["source_category"] = normed
                else:
                    matched = None
                    if (
                        "review" in normed or "systematic" in normed
                        or "scoping" in normed or "bibliometric" in normed
                        or "meta_analysis" in normed or "meta_analytic" in normed
                        or "literature_survey" in normed
                    ):
                        matched = "review_article"
                    elif (
                        "method" in normed or "framework" in normed
                        or "simulation" in normed or "scaling" in normed
                        or "psychometric" in normed or "measurement" in normed
                        or "data_paper" in normed or "dataset" in normed
                    ):
                        matched = "methodology_paper"
                    elif "technical" in normed or "report" in normed:
                        matched = "technical_report"
                    elif (
                        "peer" in normed or "research" in normed
                        or "empirical" in normed or "original" in normed
                    ):
                        matched = "peer_reviewed_research"
                    else:
                        for v in VALID_SOURCE_CATS:
                            if v in normed or normed.startswith(v):
                                matched = v
                                break
                    if matched is None:
                        matched = "peer_reviewed_research"
                    meta["source_category"] = matched

        rdt = data.get("research_design_type")
        if isinstance(rdt, str) and rdt not in VALID_DESIGN_TYPES:
            normed = rdt.lower().replace("-", "_").replace(" ", "_")
            if normed in VALID_DESIGN_TYPES:
                data["research_design_type"] = normed
            else:
                matched = None
                if (
                    "predict" in normed or "classif" in normed
                    or "regress" in normed or "supervis" in normed
                ):
                    matched = "predictive"
                elif (
                    "causal" in normed and ("experiment" in normed or "rct" in normed)
                ):
                    matched = "causal_experimental"
                elif (
                    "causal" in normed or "propensity" in normed
                    or "diff_in_diff" in normed or "instrumental" in normed
                    or "counterfactual" in normed
                ):
                    matched = "causal_observational"
                elif (
                    "explor" in normed or "descript" in normed
                    or "cluster" in normed or "review" in normed
                    or "bibliometric" in normed or "profil" in normed
                    or "unsupervis" in normed or "framework" in normed
                ):
                    matched = "exploratory"
                else:
                    for v in VALID_DESIGN_TYPES:
                        if v in normed or normed.startswith(v):
                            matched = v
                            break
                data["research_design_type"] = matched

        meta_for_rdt = parsed_data.get("metadata")
        meta_rdt_dict = meta_for_rdt if isinstance(meta_for_rdt, dict) else None
        if data.get("research_design_type") is None and (
            is_official_report_document(data, meta_rdt_dict)
            or (
                isinstance(meta_rdt_dict, dict)
                and meta_rdt_dict.get("publication_type") == "report"
            )
        ):
            data["research_design_type"] = "exploratory"

        data["plausible_values_handling"] = _normalize_literal(
            data.get("plausible_values_handling"),
            "plausible_values_handling",
            {
                "rubin_rules", "single_pv", "average_pv", "all_pv",
                "mitml", "wle", "irt_theta",
                "not_applicable", "not_reported",
            },
            "not_reported",
        )
        data["missing_data_handling"] = _normalize_literal(
            data.get("missing_data_handling"),
            "missing_data_handling",
            {
                "listwise_deletion", "pairwise_deletion", "mean_imputation",
                "multiple_imputation", "not_reported",
            },
            "not_reported",
        )

        pv_allowed = frozenset({
            "rubin_rules", "single_pv", "average_pv", "all_pv",
            "mitml", "wle", "irt_theta",
            "not_applicable", "not_reported",
        })
        md_allowed = frozenset({
            "listwise_deletion", "pairwise_deletion", "mean_imputation",
            "multiple_imputation", "not_reported",
        })

        pv_raw = data.get("plausible_values_handling")
        if pv_raw not in pv_allowed:
            data["plausible_values_handling"] = GPTExtractor._coerce_pv_literal(pv_raw)

        md_raw = data.get("missing_data_handling")
        if md_raw not in md_allowed:
            data["missing_data_handling"] = GPTExtractor._coerce_md_literal(md_raw)

        conf = data.get("confounders_identified")
        if not isinstance(conf, list):
            data["confounders_identified"] = []
        else:
            data["confounders_identified"] = _normalize_confounders_list(
                conf,
                invalid_names=INVALID_STR,
            )

        meta = parsed_data.get("metadata")
        if isinstance(meta, dict) and "open_access" in meta:
            meta["open_access"] = coerce_optional_bool(meta.get("open_access"))

        meta_dict = meta if isinstance(meta, dict) else None
        if is_official_report_document(data, meta_dict):
            sdw_report = data.get("survey_design")
            if isinstance(sdw_report, dict):
                wfi_report = sdw_report.get("weight_fields_interpretation")
                outcome_report = _coerce_outcome_summary_text(data.get("outcome_summary"))
                if (
                    not isinstance(wfi_report, str)
                    or not wfi_report.strip()
                    or _is_generic_weight_interpretation(wfi_report)
                ):
                    sdw_report["weight_fields_interpretation"] = (
                        _weight_fields_interpretation_from_outcome(outcome_report)
                    )

        return parsed_data

    def extract(self, processed: "ProcessedPDF") -> ExtractionResult:
        first_error = processed.parse_errors[0] if processed.parse_errors else None
        if not processed.extraction_text:
            return ExtractionResult(
                file_name=processed.file_name,
                success=False,
                extraction=None,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                duration_seconds=0.0,
                error=first_error or "Empty extracted text",
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_message(processed)},
        ]

        last_error = None
        for attempt in range(self.max_retries):
            start = time.perf_counter()
            try:
                # ── Path A: Structured Outputs (token-level schema enforcement) ──
                if GPTExtractor._use_structured is not False:
                    try:
                        response = self.client.beta.chat.completions.parse(
                            model=self.model,
                            messages=messages,
                            temperature=0.0,
                            response_format=ILSAArticleMetadata,
                        )
                        duration = time.perf_counter() - start
                        extraction = response.choices[0].message.parsed
                        if extraction is not None:
                            GPTExtractor._use_structured = True
                            extraction.metadata.file_name = processed.file_name
                            try:
                                self._post_process_model(extraction, processed)
                            except (ValidationError, ValueError) as val_err:
                                logger.warning(
                                    "Structured output failed local validation on "
                                    f"{processed.file_name}: {val_err}; "
                                    "falling back to JSON mode"
                                )
                                last_error = f"Structured validation: {val_err}"
                            else:
                                usage = response.usage
                                prompt_tokens = usage.prompt_tokens if usage else 0
                                completion_tokens = (
                                    usage.completion_tokens if usage else 0
                                )
                                cost = self._calculate_cost(
                                    prompt_tokens, completion_tokens
                                )
                                return ExtractionResult(
                                    file_name=processed.file_name,
                                    success=True,
                                    extraction=extraction,
                                    input_tokens=prompt_tokens,
                                    output_tokens=completion_tokens,
                                    cost_usd=cost,
                                    duration_seconds=duration,
                                )
                        last_error = "Model refused structured output"
                        continue
                    except ValidationError as struct_val_err:
                        logger.warning(
                            "Structured parse validation failed on "
                            f"{processed.file_name}: {struct_val_err}; "
                            "falling back to JSON mode"
                        )
                        last_error = f"Structured parse: {struct_val_err}"
                    except (AttributeError, TypeError):
                        GPTExtractor._use_structured = False
                        logger.info(
                            "Structured outputs unavailable in SDK, "
                            "falling back to JSON mode"
                        )
                    except APIError as struct_err:
                        if GPTExtractor._use_structured is None:
                            GPTExtractor._use_structured = False
                            logger.info(
                                "Model does not support structured outputs "
                                f"({struct_err}), falling back to JSON mode"
                            )
                        else:
                            raise

                # ── Path B: JSON mode with _sanitize + manual validation ──
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                duration = time.perf_counter() - start
                content = response.choices[0].message.content
                parsed_data = json.loads(content) if content else None

                if parsed_data is None:
                    last_error = "Model returned empty response"
                    break

                if isinstance(parsed_data.get("metadata"), dict):
                    parsed_data["metadata"]["file_name"] = processed.file_name

                parsed_data = self._sanitize(parsed_data)

                try:
                    extraction = ILSAArticleMetadata.model_validate(parsed_data)
                except ValidationError as e:
                    last_error = f"Schema validation failed: {e}"
                    logger.warning(
                        f"Validation error on {processed.file_name} "
                        f"attempt {attempt + 1}: {e}"
                    )
                    continue

                self._post_process_model(extraction, processed)

                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                cost = self._calculate_cost(
                    prompt_tokens, completion_tokens
                )
                return ExtractionResult(
                    file_name=processed.file_name,
                    success=True,
                    extraction=extraction,
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    cost_usd=cost,
                    duration_seconds=duration,
                )

            except RateLimitError as e:
                wait = self.base_delay * (2 ** attempt)
                logger.warning(
                    f"Rate limit on {processed.file_name}, "
                    f"retry {attempt + 1}/{self.max_retries} in {wait:.1f}s"
                )
                time.sleep(wait)
                last_error = f"Rate limit: {e}"

            except APITimeoutError as e:
                wait = self.base_delay * (2 ** attempt)
                logger.warning(
                    f"Timeout on {processed.file_name}, "
                    f"retry {attempt + 1} in {wait:.1f}s"
                )
                time.sleep(wait)
                last_error = f"Timeout: {e}"

            except APIError as e:
                last_error = f"API error: {e}"
                logger.error(f"API error on {processed.file_name}: {e}")
                break

            except Exception as e:
                last_error = f"Unexpected: {e}"
                logger.exception(f"Unexpected error on {processed.file_name}")
                break

        return ExtractionResult(
            file_name=processed.file_name,
            success=False,
            extraction=None,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            duration_seconds=0.0,
            error=last_error,
        )
