"""
Main orchestration script for the ILSA literature extraction pipeline.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent))          # ilsa_pipeline/ — for extractors.*, utils.*
sys.path.insert(0, str(_here.parent.parent.parent))   # ILSA_LLMs/    — for src.*

from extractors.gpt_extractor import GPTExtractor, ExtractionResult, MODEL_NAME
from extractors.pdf_processor import process_pdf
from utils.storage import (
    save_json,
    build_master_parquet,
    build_sqlite_database,
    StorageManager,
    should_skip_resume_for_json,
)

KNOWN_SOURCES = {"wos", "scopus", "oecd", "iea"}


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def find_pdfs(pdf_dir: Path) -> list[Path]:
    return sorted(pdf_dir.rglob("*.pdf"))


def infer_source(pdf_path: Path) -> str:
    parent = pdf_path.parent.name.lower()
    return parent if parent in KNOWN_SOURCES else "unknown"


def already_processed(pdf_path: Path, json_dir: Path) -> bool:
    json_file = json_dir / f"{pdf_path.stem}.json"
    if not json_file.exists():
        return False
    try:
        data = json.loads(json_file.read_text())
        return should_skip_resume_for_json(data)
    except Exception:
        return False


def process_one(pdf_path: Path, extractor: GPTExtractor, json_dir: Path, sqlite_path: Path) -> ExtractionResult:
    source = infer_source(pdf_path)
    processed = process_pdf(pdf_path, source)
    result = extractor.extract(processed)
    save_json(result, json_dir)
    if result.success and result.extraction:
        storage = StorageManager(sqlite_path)
        try:
            storage.insert_article(result)
        finally:
            storage.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the ILSA literature extraction pipeline.")
    parser.add_argument("--pdf-dir", type=Path, required=True, help="Directory containing PDFs (recursive).")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"), help="Output directory.")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent worker threads.")
    parser.add_argument("--model", default=MODEL_NAME, help="OpenAI model name.")
    parser.add_argument("--resume", action="store_true", help="Skip PDFs already extracted.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N PDFs.")
    parser.add_argument("--skip-aggregation", action="store_true", help="Skip Parquet/SQLite build.")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set in environment or .env file.")

    json_dir = args.output_dir / "json"
    parquet_path = args.output_dir / "ilsa_master.parquet"
    sqlite_path = args.output_dir / "ilsa_knowledge_base.db"
    log_path = args.output_dir / "logs" / "pipeline.log"

    setup_logging(log_path)
    logger = logging.getLogger("pipeline")

    pdfs = find_pdfs(args.pdf_dir)
    logger.info(f"Found {len(pdfs)} PDFs in {args.pdf_dir}")

    if args.resume:
        before = len(pdfs)
        pdfs = [p for p in pdfs if not already_processed(p, json_dir)]
        logger.info(f"Resuming: {before - len(pdfs)} already done, {len(pdfs)} remaining")

    if args.limit:
        pdfs = pdfs[:args.limit]
        logger.info(f"Limit applied: processing {len(pdfs)} PDFs")

    if not pdfs:
        logger.info("Nothing to process. Building aggregated outputs.")
        if not args.skip_aggregation and any(json_dir.glob("*.json")):
            build_master_parquet(json_dir, parquet_path)
            build_sqlite_database(parquet_path, sqlite_path)
        return

    extractor = GPTExtractor(api_key=api_key, model=args.model)

    total_cost = 0.0
    successes = 0
    failures = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, pdf, extractor, json_dir, sqlite_path): pdf for pdf in pdfs}
        with tqdm(total=len(futures), desc="Extracting") as pbar:
            for future in as_completed(futures):
                pdf = futures[future]
                try:
                    result = future.result()
                    if result.success:
                        successes += 1
                    else:
                        failures += 1
                        logger.warning(f"FAILED: {pdf.name} -> {result.error}")
                    total_cost += result.cost_usd
                    pbar.set_postfix({"ok": successes, "fail": failures, "cost_$": f"{total_cost:.2f}"})
                except Exception as e:
                    failures += 1
                    logger.exception(f"Worker exception on {pdf.name}: {e}")
                pbar.update(1)

    logger.info(f"Extraction complete. Successes: {successes}, Failures: {failures}, Total cost: ${total_cost:.2f}")

    if not args.skip_aggregation and successes > 0:
        logger.info("Building master Parquet ...")
        df = build_master_parquet(json_dir, parquet_path)
        if len(df) > 0:
            logger.info("Building SQLite database ...")
            build_sqlite_database(parquet_path, sqlite_path)
        logger.info(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
