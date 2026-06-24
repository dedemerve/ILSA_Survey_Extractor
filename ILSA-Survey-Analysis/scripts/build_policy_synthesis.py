#!/usr/bin/env python3
"""
Build Q1_ILSA_Policy_Synthesis.csv — a three-layer policy-oriented synthesis
of the 132 ILSA JSON extractions.

Layers:
    1. Policy Domain Mapping (5 domains, multi-label, keyword rules).
    2. Geopolitical Context (Country-Specific / Multi-Country / Global Trend).
    3. Policy Actionability Score (1-5) + method-family link.

Output columns:
    file_name | canonical_method | canonical_variable | policy_domain |
    target_countries | geopolitical_scope | method_family |
    policy_actionability_score | policy_recommendation_excerpt

Rule-based, deterministic. No LLM calls. If a study is purely methodological
or contains no explicit policy statement, the excerpt is set to
"[Purely Methodological / No Explicit Policy Statement]" and the
actionability score is capped at 2.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
JSON_DIR = ROOT / "ilsa_survey_articles" / "json"
OUT_CSV = ROOT / "outputs" / "Q1_ILSA_Policy_Synthesis.csv"

POLICY_DOMAINS = [
    "Curriculum_and_Instruction",
    "Resource_Allocation_Equity",
    "Teacher_Workforce_Development",
    "Governance_and_School_Climate",
    "Socio-Emotional_Intervention",
]

DOMAIN_RULES: dict[str, list[str]] = {
    # Predictors / levers tied to what teachers teach and how — NOT achievement
    # outcomes (math/reading/science scores) which would over-trigger.
    "Curriculum_and_Instruction": [
        r"\bcurricul", r"\binstructional time", r"weekly (class|lesson|instruction)",
        r"teaching practice", r"teaching method", r"instructional method",
        r"instructional support", r"teacher-directed instruction",
        r"\bpedagog", r"metacogni", r"learning strateg", r"\bhomework",
        r"reading habit", r"reading enjoy", r"joy of reading",
        r"\bSTEM\b.{0,20}(curric|polic|instruct)", r"science curric",
        r"native language curric", r"language curric",
        r"opportunity to learn", r"lesson plan", r"\bphonics",
        r"early literacy", r"inquiry-based", r"adaptive learning",
        r"creative thinking", r"writing process", r"task-based",
        r"reading instruction", r"math instruction", r"science instruction",
        r"differentiated instruction", r"cogac", r"cognitive activation",
        r"discipline.{0,20}climate",
    ],
    "Resource_Allocation_Equity": [
        r"\bESCS\b", r"socio.?economic status", r"\bSES\b", r"\bHISEI\b",
        r"\bICTRES\b", r"ICT resource", r"digital infrastructure",
        r"digital divide", r"digital device", r"home digital",
        r"internet access", r"\bwealth\b", r"books at home",
        r"parental education", r"family resource", r"family wealth",
        r"\brural\b", r"\burban\b", r"metropolitan",
        r"\bregional\b.{0,30}(difference|disparit|inequalit)",
        r"resilien", r"disadvantag", r"inequality of opportunit",
        r"\bIOpE\b", r"school resource", r"\bfunding\b",
        r"school inefficiency", r"educational quality",
        r"immigration status", r"multilingual learner",
        r"emergent biling", r"\bequity\b", r"family involvement",
        r"family engagement", r"home learning",
    ],
    "Teacher_Workforce_Development": [
        r"\bteacher\s+(quality|practice|efficacy|training|qualification|motivation|"
        r"feedback|support|preparation|certification|knowledge|fairness|understanding)",
        r"\bteachers'\s", r"\bteachers\s+who",
        r"\bTALIS\b", r"PISA-?VET",
        r"job satisfaction", r"professional development",
        r"continu(al|ous|ing) professional development", r"\bCPD\b",
        r"team innovativeness", r"professional learning communit",
        r"\bPLC\b", r"team teaching", r"\binduction\b", r"new teacher",
        r"teaching staff", r"shortage of teacher",
        r"vocational education", r"employability skill",
        r"professional competenc", r"teacher engagement",
        r"teacher attitude",
    ],
    "Governance_and_School_Climate": [
        r"school climate", r"school type", r"private school", r"public school",
        r"school autonomy", r"accountability", r"school composition",
        r"school belonging", r"sense of belonging", r"\bbully", r"peer climate",
        r"school governance", r"school district",
        r"school size", r"school selectivity", r"shadow education",
        r"academic track", r"tracking system", r"track recommendation",
        r"grade repetition", r"absenteeism", r"discipline climate",
        r"classroom climate", r"school management", r"principal leadership",
    ],
    "Socio-Emotional_Intervention": [
        r"\banxiety", r"well[\s-]?being", r"life satisfaction",
        r"self.?efficacy", r"self.?concept", r"self.?belief",
        r"\bmotivation", r"mastery goal", r"meaning in life",
        r"engagement", r"disengage", r"test-?taking effort",
        r"affective response", r"psycho.?emotional",
        r"fear of failure", r"creative thinking",
        r"civic engagement", r"civic self-efficacy",
        r"environmental action", r"intercultural", r"digital sport",
        r"physical activity", r"\bstress\b", r"mental health",
        r"sense of belonging", r"collaborative motivation",
        r"resilient student", r"resilience",
        r"emotion regulation", r"affect", r"positive affect", r"negative affect",
    ],
}

METHOD_FAMILY: dict[str, str] = {
    # Ensemble / boosting
    "Random Forest": "Ensemble_Learning",
    "XGBoost": "Ensemble_Learning",
    "LightGBM": "Ensemble_Learning",
    "Gradient Boosting": "Ensemble_Learning",
    "Gradient Boosted Trees": "Ensemble_Learning",
    "Stochastic Gradient Boosting": "Ensemble_Learning",
    "CatBoost": "Ensemble_Learning",
    "Stacking": "Ensemble_Learning",
    "Bagging": "Ensemble_Learning",
    "Conditional Inference Forests": "Ensemble_Learning",
    "Extra Trees": "Ensemble_Learning",
    "BCF": "Ensemble_Learning",
    "BART": "Ensemble_Learning",
    # Deep learning
    "Neural Network": "Deep_Learning",
    "Jordan Neural Network": "Deep_Learning",
    "Elman Neural Network": "Deep_Learning",
    "Autoencoder": "Deep_Learning",
    "LSTM": "Deep_Learning",
    "GRU": "Deep_Learning",
    # Penalized / hybrid stat-ML
    "LASSO": "Hybrid_Penalized",
    "Elastic Net": "Hybrid_Penalized",
    "glmmLasso": "Hybrid_Penalized",
    "Group Mnet": "Hybrid_Penalized",
    "Ridge Regression": "Hybrid_Penalized",
    "ANFIS": "Hybrid_Penalized",
    # Classical
    "Logistic Regression": "Traditional_Stats",
    "Linear Regression": "Traditional_Stats",
    "Decision Tree": "Traditional_Stats",
    "Naive Bayes": "Traditional_Stats",
    "Bayesian Network": "Traditional_Stats",
    "Discriminant Analysis": "Traditional_Stats",
    "k-NN": "Traditional_Stats",
    "KNN": "Traditional_Stats",
    "SVM": "Traditional_Stats",
    "SVR": "Traditional_Stats",
}


def classify_method_family(primary: str, all_techniques: list[str]) -> str:
    if primary in METHOD_FAMILY:
        return METHOD_FAMILY[primary]
    for t in all_techniques or []:
        if t in METHOD_FAMILY:
            return METHOD_FAMILY[t]
    return "Review_or_Methodology"


def classify_policy_domains(haystack: str) -> list[str]:
    """Multi-label domain assignment based on keyword rules."""
    text = haystack.lower()
    hits: list[str] = []
    for dom, patterns in DOMAIN_RULES.items():
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                hits.append(dom)
                break
    return hits


# Policy-signal vocabulary for excerpt extraction & actionability scoring.
# Generic policy framing — fires often, used to detect *any* policy intent.
POLICY_VERBS = re.compile(
    r"\b(policy|policies|policymaker|intervention|implication|"
    r"recommend|recommendation|should|propose|encourage|inform|strategy|"
    r"strategies|programme|program|reform|invest|allocate|target|"
    r"prioritis|prioritiz|actionable|design|implement|deploy|"
    r"to support|to improve|to enhance|to reduce|to mitigate|to address|"
    r"\blever\b|levers?|guidance|inform educational)",
    re.IGNORECASE,
)
# Strong action language — direct call to act, deserves the highest score.
STRONG_ACTION = re.compile(
    r"\b(should|must|recommend|need to|policymakers should|"
    r"intervention(s)? targeting|targeted intervention|"
    r"to improve|to enhance|to reduce|to mitigate|invest in|allocate|"
    r"prioriti[sz]e|implement)\b", re.IGNORECASE,
)
# Numeric effect sizes (percentages, score points, SD, ATE figures).
QUANT_EFFECT = re.compile(
    r"(\d{1,3}(?:\.\d+)?\s*(?:%|points?|percentage points?|SD|"
    r"standard deviation|score points?))|"
    r"\bATE\b\s*[=:]?\s*\d|95%\s*CI", re.IGNORECASE,
)
CAUSAL_LANG = re.compile(
    r"\b(caus(?:al|ed|ing)\s+(infer|forest|effect|model|claim)|"
    r"treatment effect|\bATE\b|counterfactual\s+(simulation|XAI|analys|explan)|"
    r"quasi[- ]experiment|average treatment|propensity score|"
    r"Bayesian causal|bartCause|\bBCF\b|\bBART\b\s+causal)\b",
    re.IGNORECASE,
)


def first_policy_sentence(text: str) -> str:
    """Return the first sentence in text that contains a policy keyword,
    or empty string if none."""
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for s in sentences:
        if POLICY_VERBS.search(s):
            return s.strip()
    return ""


def score_actionability(
    src_category: str,
    design: str,
    method_family: str,
    haystack: str,
    geopolitical_scope: str,
    has_quant_effect: bool,
    has_causal_lang: bool,
    has_strong_action: bool,
    policy_sentence: str,
    policy_domain_count: int,
) -> int:
    """Map empirical signals → 1..5.

    Rubric:
      1 — purely theoretical/methodological, no policy framing at all
      2 — review/methodology paper with generic policy framing
      3 — empirical, predictor importance reported but no actionable lever
      4 — empirical with clear policy-relevant predictor + quantified effect
          in a delimited (country-specific or multi-country) setting
      5 — causal ML / explicit intervention recommendation + quantified
          effect + bounded population
    """
    if src_category in ("review_article", "methodology_paper"):
        if policy_sentence and POLICY_VERBS.search(policy_sentence):
            return 2
        return 1
    if not policy_sentence:
        return 2
    # Empirical baseline.
    score = 3
    bounded = geopolitical_scope in ("Country-Specific", "Multi-Country")
    if has_quant_effect and bounded and policy_domain_count >= 1:
        score = 4
    if has_causal_lang and bounded:
        score = 5
    elif has_strong_action and has_quant_effect and bounded:
        score = 5
    return max(1, min(5, score))


def join_findings(findings: list[dict]) -> dict[str, str]:
    """Aggregate target_variable + standardized_conclusion across findings."""
    targets, conclusions, predictors = [], [], []
    for m in findings or []:
        if not isinstance(m, dict):
            continue
        if m.get("target_variable"):
            targets.append(str(m["target_variable"]))
        if m.get("standardized_conclusion"):
            conclusions.append(str(m["standardized_conclusion"]))
        for p in m.get("top_predictors", []) or []:
            predictors.append(str(p))
    return {
        "targets": " | ".join(dict.fromkeys(targets)),
        "conclusions": " ".join(conclusions),
        "predictors": " | ".join(dict.fromkeys(predictors)),
    }


def fmt_countries(countries: list[dict]) -> tuple[str, str]:
    codes = []
    for c in countries or []:
        if isinstance(c, dict) and c.get("country_code"):
            codes.append(str(c["country_code"]))
    codes = list(dict.fromkeys(codes))
    if not codes:
        return "Unspecified", "Unspecified"
    if len(codes) == 1:
        scope = "Country-Specific"
    elif len(codes) <= 10:
        scope = "Multi-Country"
    else:
        scope = "Global"
    return ";".join(codes), scope


def process_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        doc = json.load(fh)
    md = doc.get("metadata", {}) or {}
    data = doc.get("data", {}) or {}
    src_category = md.get("source_category", "") or ""
    design = data.get("research_design_type", "") or ""
    ml = data.get("ml_techniques", {}) or {}
    primary = ml.get("primary") or ""
    all_techniques = ml.get("all_techniques") or []
    findings = data.get("main_findings", []) or []
    countries = data.get("sample_details", {}).get("countries", []) if isinstance(
        data.get("sample_details", {}), dict) else []
    outcome = data.get("outcome_summary", "") or ""

    agg = join_findings(findings)
    haystack = " ".join([
        outcome, agg["conclusions"], agg["targets"], agg["predictors"],
        md.get("title", ""),
    ]).strip()

    method_family = classify_method_family(primary, all_techniques)
    domains = classify_policy_domains(haystack)
    domain_str = ";".join(domains) if domains else "Cross-Cutting_Methodological"

    country_codes, scope = fmt_countries(countries)

    # Excerpt = first policy-language sentence from standardized_conclusion;
    # fall back to outcome_summary.
    excerpt = first_policy_sentence(agg["conclusions"]) or first_policy_sentence(outcome)
    has_quant = bool(QUANT_EFFECT.search(haystack))
    has_causal = bool(CAUSAL_LANG.search(haystack))
    has_strong = bool(STRONG_ACTION.search(haystack))
    n_domains = 0 if domain_str == "Cross-Cutting_Methodological" else len(domains)

    if not excerpt:
        excerpt = "[Purely Methodological / No Explicit Policy Statement]"
        score = 1 if src_category in ("review_article", "methodology_paper") else 2
    else:
        score = score_actionability(
            src_category=src_category,
            design=design,
            method_family=method_family,
            haystack=haystack,
            geopolitical_scope=scope,
            has_quant_effect=has_quant,
            has_causal_lang=has_causal,
            has_strong_action=has_strong,
            policy_sentence=excerpt,
            policy_domain_count=n_domains,
        )

    return {
        "file_name": md.get("file_name", path.name),
        "canonical_method": primary or "None",
        "canonical_variable": agg["targets"] or "Not_specified",
        "policy_domain": domain_str,
        "target_countries": country_codes,
        "geopolitical_scope": scope,
        "method_family": method_family,
        "policy_actionability_score": score,
        "policy_recommendation_excerpt": excerpt,
    }


def main() -> None:
    rows: list[dict[str, Any]] = []
    for p in sorted(JSON_DIR.glob("*.json")):
        try:
            rows.append(process_json(p))
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] {p.name}: {e}")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "file_name", "canonical_method", "canonical_variable",
            "policy_domain", "target_countries", "geopolitical_scope",
            "method_family", "policy_actionability_score",
            "policy_recommendation_excerpt",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
