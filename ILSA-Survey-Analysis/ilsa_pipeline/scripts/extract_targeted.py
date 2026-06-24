"""
Targeted extraction for 8 specific ILSA papers.

Run from the project root:
    cd /Users/mrved/Desktop/ILSA_LLMs
    python ilsa_pipeline/scripts/extract_targeted.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import glob

from dotenv import load_dotenv
from tqdm import tqdm

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))        # ilsa_pipeline/
sys.path.insert(0, str(_here.parent.parent)) # ILSA_LLMs/

from extractors.gpt_extractor import GPTExtractor, ExtractionResult, MODEL_NAME
from extractors.pdf_processor import process_pdf
from utils.storage import (
    save_json,
    build_master_parquet,
    build_sqlite_database,
    should_skip_resume_for_json,
)

MODEL = MODEL_NAME
OUTPUT_DIR = _here.parent.parent / "outputs" / "deneme2"

# Build PDF_TARGETS from every *.pdf under ~/Desktop/deneme2/
PDF_TARGETS = [Path(p) for p in glob.glob(str(Path.home() / "Desktop" / "deneme2" / "*.pdf"))]

# (doc_root, subdir, filename) — resolved at runtime via os.listdir so that
# special characters in the WoS folder names never need to be hardcoded.
_TARGETS = [
    ("ILSA Documents copy", "TIMSS",      "data-10-00130.pdf"),
    ("ILSA Documents",      "PISA",       "Kalaycı Alas and Tezer - 2024 - Artificial Neural Network and Adaptive Neuro Fuzzy Inference System Hybridized Models in the Sustain.pdf"),
    ("ILSA Documents copy", "PISA",       "978-981-99-3043-2.pdf"),
    ("ILSA Documents",      "TALIS 2018", "McJames et al. (2023). Factors affecting teacher job satisfaction  a causal inference machine learning approach using data from TALIS 2018.pdf"),
    ("ILSA Documents",      "PIRLS 2016", "schwerter-et-al-2025-metropolitan-urban-and-rural-regions-how-regional-differences-affect-elementary-school-students-in.pdf"),
    ("ILSA Documents",      "PIAAC 2012", "s40536-024-00194-y.pdf"),
    ("ILSA Documents",      "PISA",       "Robitzsch & Lüdtke. (2022). Some thoughts on analytical choices in the scaling model for test scores in international large‑scale assessment studies.pdf"),
    ("ILSA Documents",      "TIMSS",      "Rutkowski et al. - 2024 - The limits of inference reassessing causality in international assessments.pdf"),
]

_DESKTOP = Path.home() / "Desktop"


def _resolve_paths() -> list[Path]:
    """Walk the WoS directory to resolve each (doc_root, subdir, filename) triple."""
    resolved: list[Path] = []
    for doc_root, subdir, filename in _TARGETS:
        wos_base = _DESKTOP / doc_root / "Web of Science"
        found = False
        for wos_dir in sorted(wos_base.iterdir()):
            candidate = wos_dir / subdir / filename
            if candidate.is_file():
                resolved.append(candidate)
                found = True
                break
        if not found:
            # Report as a missing-path placeholder so callers can log it
            resolved.append(_DESKTOP / doc_root / subdir / filename)
    return resolved


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    return logging.getLogger("extract_targeted")


def already_done(pdf_path: Path, json_dir: Path) -> bool:
    candidate = json_dir / f"{pdf_path.stem}.json"
    if not candidate.exists():
        return False
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
        return should_skip_resume_for_json(data)
    except Exception:
        return False


def process_one(
    pdf_path: Path,
    extractor: GPTExtractor,
    json_dir: Path,
) -> ExtractionResult:
    processed = process_pdf(pdf_path, source_database="wos")
    result = extractor.extract(processed)
    save_json(result, json_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    model = args.model
    output_dir = args.output_dir

    load_dotenv(dotenv_path=_here.parent / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set.")

    json_dir = output_dir / "json"
    parquet_path = output_dir / "ilsa_master.parquet"
    sqlite_path = output_dir / "ilsa_knowledge_base.db"
    log_path = output_dir / "logs" / "extract_targeted.log"

    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_path)
    logger.info(f"Model: {model}  |  Output: {output_dir}")

    all_paths = PDF_TARGETS

    pdfs_to_run: list[Path] = []
    for p in all_paths:
        if not p.exists():
            logger.error(f"NOT FOUND — skipping: {p.name}")
            continue
        if already_done(p, json_dir):
            logger.info(f"Already done, skipping: {p.name}")
            continue
        pdfs_to_run.append(p)

    logger.info(f"PDFs to process: {len(pdfs_to_run)} / {len(PDF_TARGETS)}")
    if not pdfs_to_run:
        logger.info("Nothing to do.")
        return

    extractor = GPTExtractor(api_key=api_key, model=model)

    total_cost = 0.0
    successes = 0
    failures = 0

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(process_one, pdf, extractor, json_dir): pdf
            for pdf in pdfs_to_run
        }
        with tqdm(total=len(futures), desc="Extracting") as pbar:
            for future in as_completed(futures):
                pdf = futures[future]
                try:
                    result = future.result()
                    if result.success:
                        successes += 1
                        logger.info(f"OK   {pdf.name}  cost=${result.cost_usd:.4f}")
                    else:
                        failures += 1
                        logger.warning(f"FAIL {pdf.name}: {result.error}")
                    total_cost += result.cost_usd
                except Exception as exc:
                    failures += 1
                    logger.exception(f"Worker exception on {pdf.name}: {exc}")
                pbar.update(1)

    logger.info(
        f"Done — successes: {successes}, failures: {failures},"
        f" total cost: ${total_cost:.4f}"
    )

    completed_jsons = list(json_dir.glob("*.json"))
    if not completed_jsons:
        logger.warning("No JSON files found; skipping aggregation.")
        return

    logger.info("Building master Parquet ...")
    df = build_master_parquet(json_dir, parquet_path)
    if len(df) > 0:
        logger.info("Building SQLite database ...")
        build_sqlite_database(parquet_path, sqlite_path)
        logger.info(f"Outputs written to: {output_dir}")
    else:
        logger.warning("Parquet is empty; SQLite step skipped.")


if __name__ == "__main__":
    main()
