"""
Three-layer storage for extraction results.

v5.4 — SQLite persistence aligned with nested ILSAArticleMetadata (metadata + data).
Changes from v5.2:
  - metadata: added source_category
  - core_data: removed obsolete ilsa_type/ilsa_year;
    added replicate_weights_used, weight_variable_name,
    plausible_values_handling, missing_data_handling,
    feature_selection, baseline_model, xai_method
  - renamed: sample_size -> total_students, survey_weights_used -> student_weights_used
  - new junction table: core_countries (country_code, n_students)
  - removed: core_ilsa_types_all (no longer in schema)
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.schemas.models import (
    ILSAArticleMetadata,
    MetadataBlock,
    SurveyDesign,
    SampleDetails,
    MLTechniques,
    DataBlock,
    StructuredFinding,
)
from src.extractors.gpt_extractor import ExtractionResult

logger = logging.getLogger(__name__)

# Written into main_findings[].standardized_conclusion when extraction fails.
# Lets --resume retry failed runs while skipping successful flat JSON files.
EXTRACTION_FAILED_PREFIX = "__EXTRACTION_FAILED__: "


def _failure_text_from_data_block(data_block: dict) -> str | None:
    """Return pipeline failure message if present in main_findings or legacy field."""
    for finding in data_block.get("main_findings") or []:
        if isinstance(finding, dict):
            text = finding.get("standardized_conclusion")
            if isinstance(text, str) and text.startswith(EXTRACTION_FAILED_PREFIX):
                return text
    legacy = data_block.get("outcome_summary")
    if isinstance(legacy, str) and legacy.startswith(EXTRACTION_FAILED_PREFIX):
        return legacy
    return None


def json_looks_like_public_extraction(data: dict) -> bool:
    """True if JSON is the public shape: top-level metadata + data only."""
    return isinstance(data, dict) and "metadata" in data and "data" in data


def is_pipeline_failure_payload(data: dict) -> bool:
    if not json_looks_like_public_extraction(data):
        return False
    return _failure_text_from_data_block(data.get("data") or {}) is not None


def should_skip_resume_for_json(data: dict) -> bool:
    """Whether an existing JSON file means the PDF can be skipped on --resume."""
    if not isinstance(data, dict):
        return False
    if "success" in data:
        return bool(data.get("success"))
    if json_looks_like_public_extraction(data):
        return not is_pipeline_failure_payload(data)
    return False


def extraction_payload_for_disk(extraction: ILSAArticleMetadata) -> dict:
    """Serialize extraction for on-disk JSON (matches public metadata + data shape)."""
    out = extraction.model_dump(mode="json")
    meta = out.setdefault("metadata", {})
    if meta.get("authors") is None:
        meta["authors"] = []
    return out


def failed_extraction_payload_for_disk(result: ExtractionResult) -> dict:
    """Same top-level keys as a successful file; error in main_findings."""
    msg = result.error or "Extraction failed."
    if not msg.startswith(EXTRACTION_FAILED_PREFIX):
        msg = EXTRACTION_FAILED_PREFIX + msg
    model = ILSAArticleMetadata(
        metadata=MetadataBlock(
            file_name=result.file_name,
            title=None,
            authors=[],
            year=None,
            doi=None,
            venue=None,
            publication_type=None,
            open_access=None,
            source_category=None,
        ),
        data=DataBlock(
            survey_design=SurveyDesign(),
            plausible_values_handling="not_reported",
            missing_data_handling="not_reported",
            sample_details=SampleDetails(
                countries=[],
                sample_filtering_criteria="Not applicable (extraction failed).",
            ),
            ml_techniques=MLTechniques(primary=None, all_techniques=[]),
            confounders_identified=[],
            main_findings=[],
            outcome_summary=msg,
            research_design_type=None,
        ),
    )
    return extraction_payload_for_disk(model)


# ---------------------------------------------------------------------------
# JSON layer
# ---------------------------------------------------------------------------

def save_json(result: ExtractionResult, output_dir: Path) -> Path:
    """
    Write one JSON file with only `metadata` and `data` top-level keys
    (same public shape as a hand-edited extraction file).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.file_name).stem
    json_path = output_dir / f"{stem}.json"

    if result.success and result.extraction is not None:
        payload = extraction_payload_for_disk(result.extraction)
    else:
        payload = failed_extraction_payload_for_disk(result)

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return json_path


# ---------------------------------------------------------------------------
# Parquet layer
# ---------------------------------------------------------------------------

def flatten_extraction(extraction, file_name: str) -> dict:
    d = extraction if isinstance(extraction, dict) else extraction.model_dump()
    flat = {"file_name": file_name}
    for section_key, section_val in d.items():
        if isinstance(section_val, dict):
            for inner_key, inner_val in section_val.items():
                flat[f"{section_key}__{inner_key}"] = inner_val
        else:
            flat[section_key] = section_val
    return flat


def build_master_parquet(json_dir: Path, parquet_path: Path) -> pd.DataFrame:
    rows = []
    for json_file in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
        except json.JSONDecodeError:
            logger.warning(f"Skipping malformed JSON: {json_file}")
            continue

        # Public on-disk shape: { "metadata": {...}, "data": {...} }
        if json_looks_like_public_extraction(data) and "success" not in data:
            file_name = (data.get("metadata") or {}).get("file_name") or json_file.stem
            if is_pipeline_failure_payload(data):
                summary = _failure_text_from_data_block(data.get("data") or {}) or ""
                err = (
                    summary[len(EXTRACTION_FAILED_PREFIX):]
                    if summary.startswith(EXTRACTION_FAILED_PREFIX)
                    else summary
                )
                rows.append({
                    "file_name": file_name,
                    "extraction_success": False,
                    "error": err,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                })
                continue
            extraction = ILSAArticleMetadata.model_validate(data)
            flat = flatten_extraction(extraction, file_name)
            flat["extraction_success"] = True
            flat["error"] = None
            flat["input_tokens"] = 0
            flat["output_tokens"] = 0
            flat["cost_usd"] = 0.0
            rows.append(flat)
            continue

        if not data.get("success") or data.get("extraction") is None:
            rows.append({
                "file_name": data.get("file_name"),
                "extraction_success": False,
                "error": data.get("error"),
                "input_tokens": data.get("input_tokens", 0),
                "output_tokens": data.get("output_tokens", 0),
                "cost_usd": data.get("cost_usd", 0.0),
            })
            continue

        extraction = ILSAArticleMetadata(**data["extraction"])
        flat = flatten_extraction(extraction, data["file_name"])
        flat["extraction_success"] = True
        flat["error"] = None
        flat["input_tokens"] = data.get("input_tokens", 0)
        flat["output_tokens"] = data.get("output_tokens", 0)
        flat["cost_usd"] = data.get("cost_usd", 0.0)
        rows.append(flat)

    df = pd.DataFrame(rows)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False, compression="snappy")
    logger.info(f"Wrote {len(df)} rows to {parquet_path}")
    return df


# ---------------------------------------------------------------------------
# SQLite layer
# ---------------------------------------------------------------------------

def build_sqlite_database(parquet_path: Path, db_path: Path) -> None:
    json_dir = parquet_path.parent / "json"
    storage = StorageManager(str(db_path))

    inserted = 0
    skipped = 0

    for json_file in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
        except json.JSONDecodeError:
            logger.warning(f"Skipping malformed JSON: {json_file.name}")
            skipped += 1
            continue

        extraction_model: ILSAArticleMetadata | None = None
        legacy_file_name: str | None = None
        legacy_tokens: tuple[int, int, float, float] = (0, 0, 0.0, 0.0)

        if json_looks_like_public_extraction(data) and "success" not in data:
            if is_pipeline_failure_payload(data):
                skipped += 1
                continue
            try:
                extraction_model = ILSAArticleMetadata.model_validate(data)
                legacy_file_name = extraction_model.metadata.file_name
            except Exception as e:
                logger.warning(f"Skipping {json_file.name}: {e}")
                skipped += 1
                continue
        elif not data.get("success") or data.get("extraction") is None:
            skipped += 1
            continue
        else:
            try:
                extraction_model = ILSAArticleMetadata(**data["extraction"])
                legacy_file_name = data.get("file_name")
                legacy_tokens = (
                    data.get("input_tokens", 0),
                    data.get("output_tokens", 0),
                    data.get("cost_usd", 0.0),
                    data.get("duration_seconds", 0.0),
                )
            except Exception as e:
                logger.warning(f"Skipping {json_file.name}: {e}")
                skipped += 1
                continue

        try:
            from src.extractors.gpt_extractor import ExtractionResult as _ER

            in_t, out_t, cost, dur = legacy_tokens
            result = _ER(
                file_name=legacy_file_name or extraction_model.metadata.file_name,
                success=True,
                extraction=extraction_model,
                input_tokens=in_t,
                output_tokens=out_t,
                cost_usd=cost,
                duration_seconds=dur,
            )
            storage.insert_article(result)
            inserted += 1
        except Exception as e:
            logger.warning(f"Skipping {json_file.name}: {e}")
            skipped += 1

    storage.close()
    logger.info(f"Built SQLite at {db_path}: {inserted} inserted, {skipped} skipped")


class StorageManager:
    """SQLite storage aligned with ILSAArticleMetadata (nested metadata + data)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def _create_tables(self):
        """Create tables for flattened extraction fields."""
        self.cursor.executescript("""
        PRAGMA foreign_keys = ON;

        -- Bibliographic and extraction provenance
        CREATE TABLE IF NOT EXISTS metadata (
            file_name           TEXT PRIMARY KEY,
            title               TEXT,
            authors             TEXT,
            year                INTEGER,
            doi                 TEXT,
            venue               TEXT,
            publication_type    TEXT,
            source_category     TEXT,
            open_access         INTEGER,
            extraction_timestamp TEXT,
            extraction_cost_usd REAL,
            prompt_tokens       INTEGER,
            completion_tokens   INTEGER
        );

        -- Core extraction fields (one row per article)
        CREATE TABLE IF NOT EXISTS core_data (
            file_name                TEXT PRIMARY KEY,
            total_students           INTEGER,
            research_design_type     TEXT,
            student_weights_used     INTEGER,
            replicate_weights_used   INTEGER,
            weight_variable_name     TEXT,
            plausible_values_handling TEXT,
            missing_data_handling    TEXT,
            ml_technique_primary     TEXT,
            feature_selection        TEXT,
            baseline_model           TEXT,
            xai_method               TEXT,
            main_findings            TEXT,
            FOREIGN KEY (file_name) REFERENCES metadata (file_name) ON DELETE CASCADE
        );

        -- Junction: authors
        CREATE TABLE IF NOT EXISTS metadata_authors (
            file_name   TEXT NOT NULL,
            ordinal     INTEGER NOT NULL,
            author_name TEXT NOT NULL,
            PRIMARY KEY (file_name, ordinal),
            FOREIGN KEY (file_name) REFERENCES metadata (file_name) ON DELETE CASCADE
        );

        -- Junction: all ML techniques
        CREATE TABLE IF NOT EXISTS core_ml_techniques_all (
            file_name TEXT NOT NULL,
            ordinal   INTEGER NOT NULL,
            technique TEXT NOT NULL,
            PRIMARY KEY (file_name, ordinal),
            FOREIGN KEY (file_name) REFERENCES metadata (file_name) ON DELETE CASCADE
        );

        -- Junction: confounders
        CREATE TABLE IF NOT EXISTS core_confounders (
            file_name       TEXT NOT NULL,
            ordinal         INTEGER NOT NULL,
            confounder_name TEXT NOT NULL,
            PRIMARY KEY (file_name, ordinal),
            FOREIGN KEY (file_name) REFERENCES metadata (file_name) ON DELETE CASCADE
        );

        -- Junction: country-level sample sizes
        CREATE TABLE IF NOT EXISTS core_countries (
            file_name    TEXT NOT NULL,
            ordinal      INTEGER NOT NULL,
            country_code TEXT NOT NULL,
            n_students   INTEGER,
            PRIMARY KEY (file_name, ordinal),
            FOREIGN KEY (file_name) REFERENCES metadata (file_name) ON DELETE CASCADE
        );

        -- Indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_metadata_year
            ON metadata (year);
        CREATE INDEX IF NOT EXISTS idx_metadata_source_category
            ON metadata (source_category);
        CREATE INDEX IF NOT EXISTS idx_core_research_design
            ON core_data (research_design_type);
        CREATE INDEX IF NOT EXISTS idx_core_ml_technique
            ON core_data (ml_technique_primary);
        CREATE INDEX IF NOT EXISTS idx_core_pv_handling
            ON core_data (plausible_values_handling);
        CREATE INDEX IF NOT EXISTS idx_core_xai
            ON core_data (xai_method);
        """)
        self.conn.commit()

    @staticmethod
    def _bool_to_int(value: bool | None) -> int | None:
        """Map True->1, False->0, None->NULL."""
        return None if value is None else int(value)

    def _insert_junction(
        self, file_name: str, table: str, column: str, values: list[str]
    ) -> None:
        """Insert rows into a junction table."""
        for ordinal, value in enumerate(values):
            self.cursor.execute(
                f"INSERT OR IGNORE INTO {table} (file_name, ordinal, {column}) "
                f"VALUES (?, ?, ?)",
                (file_name, ordinal, str(value)),
            )

    def insert_article(self, result: ExtractionResult) -> None:
        """Persist a successful ExtractionResult as a single atomic transaction."""
        if not result.success or result.extraction is None:
            raise ValueError(
                f"Cannot insert failed extraction for '{result.file_name}'"
            )

        m = result.extraction.metadata
        file_name = m.file_name

        d = result.extraction.data
        survey = d.survey_design
        sample = d.sample_details
        ml = d.ml_techniques

        try:
            # metadata table
            self.cursor.execute("""
            INSERT OR REPLACE INTO metadata (
                file_name, title, authors, year, doi, venue,
                publication_type, source_category, open_access,
                extraction_timestamp, extraction_cost_usd,
                prompt_tokens, completion_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_name,
                m.title,
                json.dumps(m.authors or [], ensure_ascii=False),
                m.year,
                m.doi,
                m.venue,
                m.publication_type,
                m.source_category,
                self._bool_to_int(m.open_access),
                datetime.now(timezone.utc).isoformat(),
                result.cost_usd,
                result.input_tokens,
                result.output_tokens,
            ))

            # core_data table
            self.cursor.execute("""
            INSERT OR REPLACE INTO core_data (
                file_name, total_students, research_design_type,
                student_weights_used, replicate_weights_used,
                weight_variable_name, plausible_values_handling,
                missing_data_handling, ml_technique_primary,
                feature_selection, baseline_model, xai_method,
                main_findings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_name,
                sample.total_students if sample else None,
                d.research_design_type,
                self._bool_to_int(
                    survey.student_weights_used if survey else None
                ),
                self._bool_to_int(
                    survey.replicate_weights_used if survey else None
                ),
                survey.weight_variable_name if survey else None,
                d.plausible_values_handling,
                d.missing_data_handling,
                ml.primary if ml else None,
                None,
                None,
                None,
                json.dumps(
                    [f.model_dump(mode="json") for f in d.main_findings],
                    ensure_ascii=False,
                ),
            ))

            # junction tables
            self._insert_junction(
                file_name, "metadata_authors", "author_name",
                m.authors or [],
            )
            self._insert_junction(
                file_name, "core_ml_techniques_all", "technique",
                ml.all_techniques if ml else [],
            )
            self._insert_junction(
                file_name, "core_confounders", "confounder_name",
                d.confounders_identified or [],
            )

            # country-level samples
            countries = sample.countries if sample else []
            for ordinal, cs in enumerate(countries):
                self.cursor.execute(
                    "INSERT OR IGNORE INTO core_countries "
                    "(file_name, ordinal, country_code, n_students) "
                    "VALUES (?, ?, ?, ?)",
                    (file_name, ordinal, cs.country_code, cs.n_students),
                )

            self.conn.commit()
            logger.info(f"Inserted '{file_name}' (${result.cost_usd:.4f})")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Rolled back '{file_name}': {e}")
            raise

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# Backward compatibility alias
ILSAStorage = StorageManager