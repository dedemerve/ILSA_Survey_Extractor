import re
from typing import List, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.schemas.findings_validation import article_requires_main_findings

_NA_VARIABLE_CODES = frozenset({"n/a", "na", "null", "none", ""})

_BOOL_NULL_STRINGS = frozenset({
    "n/a", "na", "not applicable", "not_applicable", "", "none", "null",
})
_BOOL_TRUE_STRINGS = frozenset({"true", "yes", "1"})
_BOOL_FALSE_STRINGS = frozenset({"false", "no", "0"})


def coerce_optional_bool(value: object) -> Optional[bool]:
    """Coerce LLM output to Optional[bool]; invalid strings become None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized in _BOOL_NULL_STRINGS:
            return None
        if normalized in _BOOL_TRUE_STRINGS:
            return True
        if normalized in _BOOL_FALSE_STRINGS:
            return False
        return None
    return None


def _slug_variable_code(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower()).strip("_")
    return slug[:48] if slug else "unspecified_variable"


class MetadataBlock(BaseModel):
    """Bibliographic fields (no extraction provenance in JSON schema)."""

    model_config = ConfigDict(extra="forbid")

    file_name: str = Field(
        description="Source PDF filename; primary key across all tables."
    )
    title: Optional[str] = Field(
        default=None,
        description="Full article title as it appears in the document."
    )
    authors: Optional[List[str]] = Field(
        default=None,
        description="Ordered list of author full names."
    )
    year: Optional[int] = Field(
        default=None,
        description="Four-digit publication year."
    )
    doi: Optional[str] = Field(
        default=None,
        description=(
            "DOI without URL prefix (e.g. '10.1016/j.foo.2020.01.001'). "
            "MANDATORY when present in the PDF: scan first-page header/footer, "
            "title page, copyright block, article-info box, footnotes, and "
            "doi.org / dx.doi.org links. Use EXTRACTED_DOI_HINT from the user "
            "message when provided. Do NOT leave null if any valid 10.xxxx/… "
            "identifier appears in the document."
        ),
    )
    venue: Optional[str] = Field(
        default=None,
        description="Journal, conference, or repository name."
    )
    publication_type: Optional[Literal[
        "journal", "conference", "book_chapter", "preprint", "report", "thesis"
    ]] = Field(
        default=None,
        description="Strict publication type categorization."
    )
    open_access: Optional[bool] = Field(
        default=None,
        description="True if freely accessible without paywall; null if unknown."
    )
    source_category: Optional[Literal[
        "technical_report", "review_article", "methodology_paper",
        "peer_reviewed_research"
    ]] = Field(
        default=None,
        description="Strict research type categorization."
    )

    @field_validator("doi", mode="before")
    @classmethod
    def strip_doi_prefix(cls, v: Optional[str]) -> Optional[str]:
        if isinstance(v, str):
            for prefix in (
                "https://doi.org/",
                "http://doi.org/",
                "https://dx.doi.org/",
                "http://dx.doi.org/",
            ):
                if v.startswith(prefix):
                    return v[len(prefix):]
        return v


class SurveyDesign(BaseModel):
    """Survey design and weighting methodology."""

    model_config = ConfigDict(extra="forbid")

    student_weights_used: Optional[bool] = Field(
        default=None,
        description="True if student/sampling weights (e.g. W_FSTUWT) were applied."
    )
    replicate_weights_used: Optional[bool] = Field(
        default=None,
        description="True if replicate weights (BRR, Fay) or jackknife were used."
    )
    weight_variable_name: Optional[str] = Field(
        default=None,
        description="Name of weight variable if mentioned (e.g. 'W_FSTUWT', 'TOTWGT')."
    )
    weight_fields_interpretation: str = Field(
        description=(
            "ALWAYS REQUIRED. Write 3-4 sentences detailing the data preparation, "
            "sample selection, and weighting strategy. Explain which dataset was used, "
            "how the data was cleaned or filtered, whether complex survey weights were "
            "applied (and which variable, e.g. W_FSTUWT), and if weights were ignored, "
            "explicitly state that and explain why (e.g. ML algorithms lack native "
            "weight support). This field must never be null."
        ),
    )

    @field_validator("student_weights_used", "replicate_weights_used", mode="before")
    @classmethod
    def coerce_survey_design_bools(cls, v: object) -> Optional[bool]:
        return coerce_optional_bool(v)


class CountrySample(BaseModel):
    """Sample size by country."""

    model_config = ConfigDict(extra="forbid")

    country_code: str = Field(
        description="ISO 3166-1 alpha-3 country code (e.g. 'ESP', 'USA')."
    )
    n_students: Optional[int] = Field(
        default=None,
        description="Number of students from this country in the analytic sample."
    )


class SampleDetails(BaseModel):
    """Detailed sample composition and filtering criteria."""

    model_config = ConfigDict(extra="forbid")

    total_students: Optional[int] = Field(
        default=None,
        description="Total number of students in the final analytic sample.",
    )
    countries: List[CountrySample] = Field(
        default_factory=list,
        description="Breakdown of students by country.",
    )
    sample_filtering_criteria: str = Field(
        description=(
            "CRITICAL: How authors filtered or restricted the original ILSA dataset "
            "to obtain the final analytic sample. Examples: 'Only students who took "
            "the CBA digital module', 'Excluded cases with missing ESCS', "
            "'Focused on 8th-grade students in rural public schools', "
            "'Only students who completed the climate-control problem-solving task'. "
            "If no specific filtering is reported, state: 'Used the full available "
            "sample for the specified countries.'"
        ),
    )


class Confounder(BaseModel):
    """A single independent variable / predictor / feature used in the model.

    EXTRACTION RULES — enforced at the schema level:
    • ONE variable per object. NEVER combine multiple variables into one entry.
    • Extract CONCEPTUAL control/predictor variables only — not ML feature columns.
    • Do NOT list raw log actions, TF-IDF tokens, n-grams, or UI micro-clicks.
    • variable_code must identify the variable — do NOT default to 'N/A' out of laziness.
    • category must be one of exactly 13 literals (no 'other').
    """

    model_config = ConfigDict(extra="forbid")

    variable_code: str = Field(
        description=(
            "Stable identifier for tabulation. NEVER output the literal string 'N/A'. "
            "STRICT: Do NOT extract raw log sequences, n-grams, TF-IDF/Word2Vec tokens, "
            "slider states, or action codes (e.g. '0_0_0', '1_2_-2', 'reset', 'start'). "
            "If a variable is a micro-level ML input, OMIT it entirely — do not list it. "
            "Use tier (1) Official ILSA code (ESCS, ST004Q01TA); "
            "(2) high-level construct label (VOTAT, MATHEFF, gdp_per_capita); "
            "(3) snake_case slug from a CONCEPTUAL variable_name. "
            "Do NOT invent fake ILSA codes."
        ),
    )
    variable_name: str = Field(
        description=(
            "A concise, standardised English label (max 8 words) for a CONCEPTUAL "
            "variable (e.g. 'Gender', 'Total time on task', 'VOTAT score', 'ESCS'). "
            "NOT an ML feature or UI action (e.g. 'Top slider +2', 'TF-IDF weight', "
            "'Action triple', 'Word2vec embedding'). If only micro-features exist, "
            "return fewer confounders rather than listing model inputs."
        ),
    )
    category: Literal[
        "socioeconomic",
        "demographic",
        "student_attitude",
        "student_behavior",
        "teacher",
        "school",
        "ict",
        "curriculum",
        "parent_home",
        "process_data",
        "prior_achievement",
        "peer_effects",
        "system_level",
    ] = Field(
        description=(
            "The domain category that BEST fits the variable. You MUST assign "
            "exactly one of the 13 categories — there is no 'other'. Mapping guide: "
            "socioeconomic → ESCS, HOMEPOS, WEALTH, HISEI, parental education, "
            "books at home, cultural possessions, family resources; "
            "demographic → gender, age, immigration/migrant status, language at home, grade; "
            "student_attitude → self-efficacy, motivation, anxiety, enjoyment, "
            "belonging, self-concept, interest, value beliefs, intrinsic motivation; "
            "student_behavior → study time, homework time/frequency, absenteeism, "
            "learning strategies, reading habits, metacognition; "
            "teacher → qualifications, experience, professional development, "
            "teaching practices, job satisfaction, instructional strategies; "
            "school → school type (public/private), resources, class size, climate, "
            "safety, autonomy, leadership, location (urban/rural); "
            "ict → ICT resources (ICTRES), computer use, digital access, "
            "technology integration in lessons, internet availability; "
            "curriculum → curriculum type, instructional time (SMINS/TMINS), "
            "content coverage, assessment practices; "
            "parent_home → parental involvement/support (EMOSUPS), home environment, "
            "family structure, homework supervision; "
            "process_data → ONLY high-level aggregates (total/response time on task, "
            "total action counts, VOTAT score, visits per item). NEVER raw clicks, "
            "slider codes, n-grams, TF-IDF/Word2Vec features, or state transitions; "
            "prior_achievement → previous test scores, prior-year grades, "
            "achievement in other domains (e.g. reading score used as predictor "
            "for math), WLE/PV scores used as control variables; "
            "peer_effects → classroom disciplinary climate, peer bullying, "
            "class-average achievement, classroom composition; "
            "system_level → country-level GDP, education expenditure, tracking age, "
            "national policy variables, GINI coefficient, teacher-student ratio "
            "at system level. If unclear, choose the closest category (e.g. aggregate "
            "indices → socioeconomic; classroom climate → peer_effects). Do NOT "
            "categorize micro ML features as process_data."
        ),
    )

    @model_validator(mode="after")
    def coerce_na_variable_code_to_slug(self) -> "Confounder":
        """Schema-level guard: 'N/A' is never a valid final code when name exists."""
        if self.variable_code.strip().lower() in _NA_VARIABLE_CODES:
            self.variable_code = _slug_variable_code(self.variable_name)
        return self


class StructuredFinding(BaseModel):
    """Dataset → Input → Target → Output pipeline for meta-analysis tabulation."""

    model_config = ConfigDict(extra="forbid")

    dataset_used: str = Field(
        description=(
            "The ILSA/PIAAC assessment, cycle year, grade level, and domain for this "
            "finding (e.g. 'TIMSS 2019 Grade 8 Science', 'PISA 2022 Creative Thinking', "
            "'PIRLS 2016 Reading', 'PISA 2012 Problem Solving process data'). "
            "Be specific — include program name, year, grade, and subject/domain."
        ),
    )
    target_variable: str = Field(
        description=(
            "The dependent variable / outcome being predicted or analyzed "
            "(e.g. 'Math achievement (PVs)', 'Resilience (top 2%)')."
        ),
    )
    top_predictors: List[str] = Field(
        description=(
            "Top 3-5 most important independent variables driving the model. "
            "Names MUST match variable_name entries in confounders_identified when "
            "those variables appear there; otherwise use the exact label from the paper."
        ),
    )
    performance_metrics: str = Field(
        description=(
            "Reported model metrics (e.g. 'Accuracy: 85%, AUC: 0.82, R²: 0.45'). "
            "Write 'Not reported' if the paper gives no metrics for this target."
        ),
    )
    standardized_conclusion: str = Field(
        description=(
            "2-3 sentences connecting the full pipeline. MUST follow: "
            "'Using [dataset_used] data, the study leveraged [top_predictors] to predict "
            "[target_variable], finding that [key finding/effect]. This indicates that "
            "[education/policy implication].' "
            "Note methodological limitations (SHAP overstated causality, no weights) "
            "inside the finding clause when relevant."
        ),
    )


class MLTechniques(BaseModel):
    """ML algorithms and methodological components."""

    model_config = ConfigDict(extra="forbid")

    primary: Optional[str] = Field(
        default=None,
        description=(
            "Primary/best-performing ML algorithm (e.g. 'XGBoost', 'Random Forest'). "
            "MUST NOT be null if all_techniques has values — deduce the best model "
            "from results, or copy the only technique if just one is listed."
        ),
    )
    all_techniques: List[str] = Field(
        description="All ML algorithms evaluated (NOT preprocessing/stats methods)."
    )


class DataBlock(BaseModel):
    """Methodological and analytic extraction fields."""

    model_config = ConfigDict(extra="forbid")

    survey_design: SurveyDesign = Field(
        description="Survey weighting and replicate design methodology."
    )
    plausible_values_handling: Literal[
        "rubin_rules", "single_pv", "average_pv", "all_pv",
        "mitml", "wle", "irt_theta",
        "not_applicable", "not_reported"
    ] = Field(
        description="How plausible values (PVs) were handled in analysis."
    )
    missing_data_handling: Literal[
        "listwise_deletion", "pairwise_deletion",
        "mean_imputation", "single_imputation", "knn_imputation",
        "multiple_imputation", "not_reported"
    ] = Field(
        description=(
            "How missing data was addressed. Map missForest/RF-based imputation "
            "to single_imputation; kNN imputation to knn_imputation."
        ),
    )
    handling_not_reported_explanation: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED IF plausible_values_handling OR missing_data_handling is "
            "'not_reported' or 'not_applicable'. Write 2-3 sentences as a critical "
            "peer-reviewer diagnosing WHY the information is missing. Is it a "
            "reporting gap (authors failed to document their strategy), or is it "
            "the study's nature (e.g., only Likert-scale responses, no cognitive "
            "PVs)? Must be null when both PV and missing data handling are explicitly "
            "reported."
        ),
    )
    sample_details: SampleDetails = Field(
        description="Total sample size and breakdown by country."
    )
    ml_techniques: MLTechniques = Field(
        description="ML algorithms and methodological components."
    )
    confounders_identified: List[Confounder] = Field(
        default_factory=list,
        description=(
            "Conceptual independent variables, predictors, and controls used in the "
            "study — NOT ML feature-engineering columns (TF-IDF tokens, n-grams, "
            "raw log action codes). Each conceptual variable is one Confounder object. "
            "Omit micro-level process/log features entirely. DO NOT leave empty if the "
            "study uses questionnaire or background controls — scan methodology and "
            "variables sections. Process-data papers may legitimately return []."
        ),
    )
    main_findings: List[StructuredFinding] = Field(
        description=(
            "Structured findings mapping inputs to targets. One object per distinct "
            "dependent variable analyzed (e.g. separate rows for Math vs Science). "
            "Each object links top_predictors (from confounders_identified when possible) "
            "to target_variable with performance_metrics and standardized_conclusion. "
            "Required non-empty when outcome_summary is substantive or empirical ML "
            "signals exist (see ILSAArticleMetadata validator). Empty [] only for "
            "reviews/theory with no predictive results and no substantive outcome."
        ),
    )
    outcome_summary: str = Field(
        description=(
            "4-5 sentence narrative summary (max ~120 words) of key findings and model "
            "performance, grounded only in the article text. Complements main_findings with "
            "a readable prose overview: best model, key metrics, main predictors, limitations "
            "(weights, causality overstatement, data leakage). Do NOT duplicate the entire "
            "main_findings table — synthesize across targets. Do NOT put null-field "
            "commentary here (use null_fields_interpretation)."
        ),
    )
    research_design_type: Optional[Literal[
        "predictive", "causal_observational", "causal_experimental", "exploratory"
    ]] = Field(
        default=None,
        description="Strict research design categorization."
    )
    null_fields_interpretation: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED when total_students is null, or primary ML model is null while "
            "all_techniques is empty, or the extraction is extremely sparse. 2-3 "
            "sentences diagnosing the omission (e.g. theoretical review with no "
            "empirical data, or authors listed models but never reported which "
            "performed best). Must be null when the record is reasonably dense."
        ),
    )


class ILSAArticleMetadata(BaseModel):
    """Top-level extraction record for ILSA ML papers (nested metadata + data)."""

    model_config = ConfigDict(extra="forbid")

    metadata: MetadataBlock = Field(
        description="Bibliographic identification fields."
    )
    data: DataBlock = Field(
        description="Survey design, sample, ML, and outcome fields."
    )

    @model_validator(mode="after")
    def validate_main_findings_when_required(self) -> "ILSAArticleMetadata":
        data_dict = self.data.model_dump(mode="python")
        meta_dict = self.metadata.model_dump(mode="python")
        if article_requires_main_findings(data_dict, meta_dict) and not self.data.main_findings:
            raise ValueError(
                "data.main_findings must contain at least one StructuredFinding when "
                "outcome_summary is substantive or empirical ML signals are present"
            )
        return self


def validate_public_article_json(raw: dict) -> ILSAArticleMetadata:
    """Validate on-disk article JSON (``{"metadata": ..., "data": ...}``).

    ``ILSAArticleMetadata`` is the root model for the full file. Do not pass
    ``raw["metadata"]`` alone — that block is a ``MetadataBlock``, not the
    top-level record.
    """
    return ILSAArticleMetadata.model_validate(raw)
