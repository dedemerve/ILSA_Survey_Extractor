#!/usr/bin/env python3
"""
Batch extraction: all PDFs under each immediate subfolder of a root (recursive).

Çıktı yolu kök altındaki ilk seviye klasörü ve iç içe göreli yolu korur:
``outputs/<IEA|OECD|…>/<iç_yol>/json/<pdf_stem[:80]>.json``

Usage / Kullanım
----------------
From project root::

    python scripts/run_batch_folders.py --dry-run
    python scripts/run_batch_folders.py --root ~/Desktop/ILSA\\ Documents\\ LLM\\ Extraction
    python scripts/run_batch_folders.py --workers 4
    python scripts/run_batch_folders.py --top-folder IEA
    python scripts/run_batch_folders.py --top-folder IEA --top-folder OECD
    python scripts/run_batch_folders.py --build-db

Dry-run lists subdirs, PDF counts, and target JSON paths (no API calls).
Skips existing JSON when ``should_skip_resume_for_json`` says the file is complete.
Log: ``outputs/batch_folders_extraction.log`` (stdout + file, thread-safe).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ilsa_pipeline"))

from src.extractors.gpt_extractor import GPTExtractor, ExtractionResult, MODEL_NAME
from src.extractors.pdf_processor import process_pdf
from utils.storage import (
    build_master_parquet,
    build_sqlite_database,
    extraction_payload_for_disk,
    failed_extraction_payload_for_disk,
    should_skip_resume_for_json,
)

DEFAULT_ROOT = Path.home() / "Desktop" / "ILSA Documents LLM Extraction"
ROOT_CANDIDATES = (
    DEFAULT_ROOT,
    Path.home() / "Desktop" / "ILSA Survey Paper LLM Extraction",
)
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
BATCH_LOG_NAME = "batch_folders_extraction.log"
PROGRESS_LOG_EVERY = 25  # emit PROGRESS line every N completions (OK+FAIL)


@dataclass(frozen=True)
class PdfJob:
    pdf_path: Path
    subdir_name: str
    nested_label: str  # e.g. "IEA/foo/bar" for display
    json_path: Path
    source_database: str


class ThreadSafeBatchLog:
    """Append timestamped lines to batch log and mirror to stdout."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _line(self, message: str) -> str:
        ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        return f"{ts} | {message}"

    def write(self, message: str) -> None:
        line = self._line(message)
        with self._lock:
            print(line, flush=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def expand_path(path: Path) -> Path:
    return path.expanduser().resolve()


def resolve_default_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return expand_path(explicit)
    for candidate in ROOT_CANDIDATES:
        if candidate.is_dir():
            return candidate.expanduser().resolve()
    return DEFAULT_ROOT.expanduser().resolve()


def list_immediate_subdirs(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def filter_top_subdirs(subdirs: list[Path], top_folders: list[str] | None) -> list[Path]:
    """Keep only immediate subdirs whose names match top_folders (case-sensitive)."""
    if not top_folders:
        return subdirs
    allowed = {name.strip() for name in top_folders if name.strip()}
    return [p for p in subdirs if p.name in allowed]


def safe_json_stem(pdf_path: Path) -> str:
    stem = pdf_path.stem[:80]
    return stem.replace("/", "_").replace("\\", "_")


def nested_relative_dir(pdf_path: Path, top_subdir: Path) -> Path:
    """Parent path of PDF relative to top-level subdir (IEA, OECD, …)."""
    try:
        rel = pdf_path.parent.relative_to(top_subdir)
    except ValueError:
        rel = Path(".")
    if str(rel) in (".", ""):
        return Path()
    return rel


def nested_label(subdir_name: str, rel_parent: Path) -> str:
    if not rel_parent.parts:
        return subdir_name
    return f"{subdir_name}/{'/'.join(rel_parent.parts)}"


def json_output_path(
    pdf_path: Path,
    top_subdir: Path,
    subdir_name: str,
    outputs_root: Path = OUTPUTS_ROOT,
) -> Path:
    rel_parent = nested_relative_dir(pdf_path, top_subdir)
    base = outputs_root / subdir_name
    if rel_parent.parts:
        base = base / rel_parent
    return base / "json" / f"{safe_json_stem(pdf_path)}.json"


def source_database_for_subdir(subdir_name: str) -> str:
    return subdir_name.strip().lower().replace(" ", "_")


def should_skip_pdf(json_path: Path) -> bool:
    if not json_path.exists():
        return False
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return should_skip_resume_for_json(data)
    except Exception:
        return False


def iter_pdfs_under_subdirs(
    root: Path,
    top_folders: list[str] | None = None,
) -> list[tuple[Path, Path]]:
    """(top_subdir_path, pdf_path) for every PDF under each immediate subdir."""
    out: list[tuple[Path, Path]] = []
    subdirs = filter_top_subdirs(list_immediate_subdirs(root), top_folders)
    for subdir in subdirs:
        for pdf_path in sorted(subdir.rglob("*.pdf")):
            if pdf_path.is_file():
                out.append((subdir, pdf_path))
    return out


def discover_jobs(
    root: Path,
    *,
    outputs_root: Path = OUTPUTS_ROOT,
    force: bool = False,
    top_folders: list[str] | None = None,
) -> list[PdfJob]:
    jobs: list[PdfJob] = []
    for top_subdir, pdf_path in iter_pdfs_under_subdirs(root, top_folders=top_folders):
        subdir_name = top_subdir.name
        source = source_database_for_subdir(subdir_name)
        rel_parent = nested_relative_dir(pdf_path, top_subdir)
        out = json_output_path(pdf_path, top_subdir, subdir_name, outputs_root)
        if not force and should_skip_pdf(out):
            continue
        jobs.append(
            PdfJob(
                pdf_path=pdf_path,
                subdir_name=subdir_name,
                nested_label=nested_label(subdir_name, rel_parent),
                json_path=out,
                source_database=source,
            )
        )
    return jobs


def count_all_pdfs(root: Path, top_folders: list[str] | None = None) -> int:
    return len(iter_pdfs_under_subdirs(root, top_folders=top_folders))


def save_result_to_path(result: ExtractionResult, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if result.success and result.extraction is not None:
        payload = extraction_payload_for_disk(result.extraction)
    else:
        payload = failed_extraction_payload_for_disk(result)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    return logging.getLogger("batch_folders")


def process_one(job: PdfJob, extractor: GPTExtractor) -> ExtractionResult:
    processed = process_pdf(job.pdf_path, job.source_database)
    if job.nested_label:
        # Prompt-only hint; never written to on-disk JSON metadata schema.
        processed.metadata["folder_organization_hint"] = job.nested_label
    result = extractor.extract(processed)
    save_result_to_path(result, job.json_path)
    return result


def build_databases(outputs_root: Path, subdir_names: list[str], logger: logging.Logger) -> None:
    for name in subdir_names:
        json_glob = list((outputs_root / name).rglob("json/*.json"))
        if not json_glob:
            logger.info("Skip DB build (no JSON): %s", name)
            continue
        json_dir = outputs_root / name / "json"
        if not json_dir.is_dir():
            logger.info("Skip DB build (no flat json dir): %s — use per-subdir paths", name)
            continue
        parquet_path = outputs_root / name / "ilsa_master.parquet"
        sqlite_path = outputs_root / name / "ilsa_knowledge_base.db"
        logger.info("Building Parquet for %s …", name)
        df = build_master_parquet(json_dir, parquet_path)
        if len(df) > 0:
            logger.info("Building SQLite for %s …", name)
            build_sqlite_database(parquet_path, sqlite_path)
        logger.info("Done: %s (%d rows)", name, len(df))


def print_dry_run(
    root: Path,
    outputs_root: Path,
    top_folders: list[str] | None = None,
) -> tuple[int, int, int, int]:
    subdirs = filter_top_subdirs(list_immediate_subdirs(root), top_folders)
    total_pdfs = 0
    to_process = 0
    skipped = 0
    print(f"Root: {root}")
    print(f"Outputs: {outputs_root}")
    print(f"Log (on run): {outputs_root / BATCH_LOG_NAME}")
    print(f"Immediate subdirs: {len(subdirs)}\n")
    for subdir in subdirs:
        pdfs = sorted(p for p in subdir.rglob("*.pdf") if p.is_file())
        pending: list[Path] = []
        skip_here = 0
        for pdf_path in pdfs:
            out = json_output_path(pdf_path, subdir, subdir.name, outputs_root)
            if should_skip_pdf(out):
                skip_here += 1
            else:
                pending.append(pdf_path)
        total_pdfs += len(pdfs)
        to_process += len(pending)
        skipped += skip_here
        print(f"[{subdir.name}] {len(pdfs)} PDF(s), {len(pending)} would run, {skip_here} skipped")
        for pdf_path in pending[:5]:
            rel = nested_relative_dir(pdf_path, subdir)
            folder = nested_label(subdir.name, rel)
            out = json_output_path(pdf_path, subdir, subdir.name, outputs_root)
            print(f"  Folder: {folder}")
            print(f"  PDF:    {pdf_path.name[:80]}")
            print(f"  ->     {out.relative_to(outputs_root)}")
        if len(pending) > 5:
            print(f"  … and {len(pending) - 5} more pending in this subdir")
        print()
    print(
        f"Summary: {len(subdirs)} subdir(s), {total_pdfs} PDF(s) total, "
        f"{to_process} would be extracted, {skipped} skipped (valid JSON)."
    )
    return len(subdirs), total_pdfs, to_process, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=None,
        help="Root folder with immediate subdirs (default: ILSA Documents LLM Extraction)",
    )
    parser.add_argument(
        "--root",
        dest="root_flag",
        type=Path,
        default=None,
        help="Same as positional root (overrides positional if both given)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent extraction threads (default: 4)",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="OpenAI model name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N PDFs (across all subdirs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List subdirs/PDFs and output paths without calling the API",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even when a resumable JSON already exists",
    )
    parser.add_argument(
        "--build-db",
        action="store_true",
        help="After extraction, build Parquet + SQLite per subdir under outputs/",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=OUTPUTS_ROOT,
        help=f"Base outputs directory (default: {OUTPUTS_ROOT})",
    )
    parser.add_argument(
        "--top-folder",
        action="append",
        dest="top_folders",
        metavar="NAME",
        help=(
            "Process only this immediate subdirectory under root (e.g. IEA). "
            "Repeat for multiple: --top-folder IEA --top-folder OECD"
        ),
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    explicit_root = args.root_flag if args.root_flag is not None else args.root
    root = resolve_default_root(explicit_root)
    outputs_root = expand_path(args.outputs_root)

    if not root.is_dir():
        print(f"ERROR: root directory not found: {root}")
        sys.exit(1)

    if args.top_folders:
        all_sub = {p.name for p in list_immediate_subdirs(root)}
        missing = [n for n in args.top_folders if n.strip() and n.strip() not in all_sub]
        if missing:
            print(f"ERROR: --top-folder not found under root: {missing}")
            print(f"Available: {sorted(all_sub)}")
            sys.exit(1)

    if args.dry_run:
        print_dry_run(root, outputs_root, top_folders=args.top_folders)
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set in environment or .env file.")

    batch_log_path = outputs_root / BATCH_LOG_NAME
    batch_log = ThreadSafeBatchLog(batch_log_path)
    detail_log_path = outputs_root / "logs" / "batch_folders.log"
    logger = setup_logging(detail_log_path)

    total_pdfs = count_all_pdfs(root, top_folders=args.top_folders)
    jobs = discover_jobs(
        root,
        outputs_root=outputs_root,
        force=args.force,
        top_folders=args.top_folders,
    )
    skipped_upfront = total_pdfs - len(jobs)
    subdirs = filter_top_subdirs(list_immediate_subdirs(root), args.top_folders)

    batch_log.write(f"START | root={root} | workers={args.workers} | queue={len(jobs)} | total_pdfs={total_pdfs} | skipped_existing={skipped_upfront}")
    logger.info("Root: %s", root)
    logger.info("Batch log: %s", batch_log_path)
    logger.info("Subdirs: %s", [s.name for s in subdirs])
    logger.info("PDFs under subdirs: %d", total_pdfs)
    logger.info("Skipped (resumable JSON): %d", skipped_upfront)
    logger.info("Queue: %d PDF(s) to extract", len(jobs))

    if args.limit is not None:
        jobs = jobs[: args.limit]
        batch_log.write(f"LIMIT | processing first {len(jobs)} PDF(s) only")
        logger.info("Limit applied: %d PDF(s)", len(jobs))

    if not jobs:
        batch_log.write("DONE | nothing to extract")
        logger.info("Nothing to extract.")
        if args.build_db:
            build_databases(outputs_root, [s.name for s in subdirs], logger)
        return

    extractor = GPTExtractor(api_key=api_key, model=args.model)
    total_cost = 0.0
    successes = 0
    failures = 0
    total_jobs = len(jobs)
    progress_lock = threading.Lock()
    completed_count = 0
    finished_count = 0
    last_finished_label = ""

    def maybe_log_progress() -> None:
        if total_jobs <= 0:
            return
        if finished_count % PROGRESS_LOG_EVERY != 0 and finished_count != total_jobs:
            return
        pct = int(100 * finished_count / total_jobs)
        batch_log.write(
            f"PROGRESS | {finished_count}/{total_jobs} ({pct}%) | "
            f"ok={successes} fail={failures} skip={skipped_upfront} | "
            f"${total_cost:.2f} | last={last_finished_label}"
        )

    def run_job(job: PdfJob) -> tuple[PdfJob, ExtractionResult | None, BaseException | None]:
        try:
            with progress_lock:
                nonlocal completed_count
                completed_count += 1
                idx = completed_count
            batch_log.write(
                f"PROCESS | [{job.nested_label}] {job.pdf_path.name} ({idx}/{total_jobs})"
            )
            result = process_one(job, extractor)
            return job, result, None
        except BaseException as exc:
            return job, None, exc

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_job, job): job for job in jobs}
            with tqdm(total=len(futures), desc="Extracting") as pbar:
                for future in as_completed(futures):
                    job, result, exc = future.result()
                    if exc is not None:
                        failures += 1
                        last_finished_label = f"[{job.nested_label}] {job.pdf_path.name}"
                        batch_log.write(
                            f"FAIL | [{job.nested_label}] {job.pdf_path.name} | error={exc!s}"
                        )
                        logger.exception(
                            "Worker exception on %s: %s", job.pdf_path.name, exc,
                        )
                    elif result is not None:
                        last_finished_label = f"[{job.nested_label}] {job.pdf_path.name}"
                        if result.success:
                            successes += 1
                            status = "OK"
                        else:
                            failures += 1
                            status = "FAIL"
                        total_cost += result.cost_usd
                        batch_log.write(
                            f"{status} | [{job.nested_label}] {job.pdf_path.name} | "
                            f"cost=${result.cost_usd:.4f} | cumulative=${total_cost:.2f} | "
                            f"-> {job.json_path.relative_to(outputs_root)}"
                        )
                        if not result.success:
                            logger.warning(
                                "FAILED: %s -> %s",
                                job.pdf_path.name,
                                result.error,
                            )
                    finished_count += 1
                    maybe_log_progress()
                    pbar.set_postfix(
                        {
                            "ok": successes,
                            "fail": failures,
                            "skip": skipped_upfront,
                            "cost_$": f"{total_cost:.2f}",
                        }
                    )
                    pbar.update(1)
    finally:
        batch_log.write(
            f"FINISH | extracted_ok={successes} | failed={failures} | "
            f"skipped_at_start={skipped_upfront} | total_pdfs={total_pdfs} | "
            f"total_cost=${total_cost:.2f}"
        )
    logger.info(
        "Extraction complete. Successes: %d, Failures: %d, Skipped: %d, Total cost: $%.2f",
        successes,
        failures,
        skipped_upfront,
        total_cost,
    )

    if args.build_db:
        build_databases(outputs_root, [s.name for s in subdirs], logger)


if __name__ == "__main__":
    main()
