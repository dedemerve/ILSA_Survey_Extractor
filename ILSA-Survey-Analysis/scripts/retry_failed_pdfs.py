#!/usr/bin/env python3
"""
Re-extract PDFs that failed or have invalid/missing JSON (skip valid outputs).

Mirrors path layout from ``run_batch_folders.py``. Skips JSON that already passes
``validate_public_article_json`` or ``should_skip_resume_for_json``. Optionally
uses ``batch_folders_extraction.log`` to find PDFs marked FAIL.

Usage::

    python scripts/retry_failed_pdfs.py --dry-run
    python scripts/retry_failed_pdfs.py --top-folder IEA --workers 4
    python scripts/retry_failed_pdfs.py --root ~/Desktop/ILSA\\ Documents --top-folder IEA
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ilsa_pipeline"))

from scripts.run_batch_folders import (  # noqa: E402
    BATCH_LOG_NAME,
    OUTPUTS_ROOT,
    PdfJob,
    ThreadSafeBatchLog,
    expand_path,
    iter_pdfs_under_subdirs,
    json_output_path,
    list_immediate_subdirs,
    nested_label,
    nested_relative_dir,
    process_one,
    resolve_default_root,
    safe_json_stem,
    save_result_to_path,
    setup_logging,
    source_database_for_subdir,
)
from src.extractors.gpt_extractor import GPTExtractor, MODEL_NAME  # noqa: E402
from src.schemas.models import validate_public_article_json  # noqa: E402
from utils.storage import should_skip_resume_for_json  # noqa: E402

DEFAULT_ROOT = Path.home() / "Desktop" / "ILSA Documents"
ROOT_CANDIDATES = (
    DEFAULT_ROOT,
    Path.home() / "Desktop" / "ILSA Documents LLM Extraction",
    Path.home() / "Desktop" / "ILSA Survey Paper LLM Extraction",
)

_FAIL_LINE_RE = re.compile(
    r"\|\s*FAIL\s*\|\s*\[[^\]]+\]\s+(?P<name>.+?)\s*\|",
)


def resolve_retry_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return expand_path(explicit)
    for candidate in ROOT_CANDIDATES:
        if candidate.is_dir():
            return candidate.expanduser().resolve()
    return DEFAULT_ROOT.expanduser().resolve()


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def json_is_resumable(data: dict) -> bool:
    if should_skip_resume_for_json(data):
        return True
    try:
        validate_public_article_json(data)
        return True
    except (ValidationError, ValueError, TypeError):
        return False


def load_failed_pdf_names(log_path: Path) -> set[str]:
    """PDF file names (and stems) that appear on FAIL lines in the batch log."""
    names: set[str] = set()
    if not log_path.is_file():
        return names
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _FAIL_LINE_RE.search(line)
        if not m:
            continue
        pdf_name = m.group("name").strip()
        names.add(pdf_name)
        names.add(Path(pdf_name).stem)
        names.add(safe_json_stem(Path(pdf_name)))
    return names


def pdf_marked_failed_in_log(pdf_path: Path, failed_names: set[str]) -> bool:
    if not failed_names:
        return False
    stem = safe_json_stem(pdf_path)
    return (
        pdf_path.name in failed_names
        or pdf_path.stem in failed_names
        or stem in failed_names
    )


def classify_pdf(
    pdf_path: Path,
    top_subdir: Path,
    *,
    outputs_root: Path,
    failed_names: set[str],
) -> str:
    """Return 'skip' or 'retry'."""
    subdir_name = top_subdir.name
    json_path = json_output_path(pdf_path, top_subdir, subdir_name, outputs_root)

    if pdf_marked_failed_in_log(pdf_path, failed_names):
        return "retry"

    if not json_path.exists():
        return "retry"

    data = load_json(json_path)
    if data is None:
        return "retry"

    if json_is_resumable(data):
        return "skip"

    return "retry"


def discover_retry_jobs(
    root: Path,
    *,
    outputs_root: Path = OUTPUTS_ROOT,
    top_folders: list[str] | None = None,
    log_path: Path | None = None,
) -> tuple[list[PdfJob], int, int]:
    failed_names = load_failed_pdf_names(log_path) if log_path else set()
    jobs: list[PdfJob] = []
    skipped = 0
    total = 0

    for top_subdir, pdf_path in iter_pdfs_under_subdirs(root, top_folders=top_folders):
        total += 1
        subdir_name = top_subdir.name
        rel_parent = nested_relative_dir(pdf_path, top_subdir)
        json_path = json_output_path(pdf_path, top_subdir, subdir_name, outputs_root)
        decision = classify_pdf(
            pdf_path,
            top_subdir,
            outputs_root=outputs_root,
            failed_names=failed_names,
        )
        if decision == "skip":
            skipped += 1
            continue
        jobs.append(
            PdfJob(
                pdf_path=pdf_path,
                subdir_name=subdir_name,
                nested_label=nested_label(subdir_name, rel_parent),
                json_path=json_path,
                source_database=source_database_for_subdir(subdir_name),
            )
        )
    return jobs, skipped, total


def print_dry_run(
    root: Path,
    outputs_root: Path,
    top_folders: list[str] | None,
    log_path: Path | None,
) -> None:
    jobs, skipped, total = discover_retry_jobs(
        root,
        outputs_root=outputs_root,
        top_folders=top_folders,
        log_path=log_path,
    )
    failed_names = load_failed_pdf_names(log_path) if log_path else set()
    print(f"Root: {root}")
    print(f"Outputs: {outputs_root}")
    print(f"Log: {log_path or '(none)'}")
    print(f"FAIL names in log: {len(failed_names)}")
    print(f"PDFs scanned: {total}")
    print(f"Would skip (valid JSON): {skipped}")
    print(f"Would retry: {len(jobs)}\n")
    for job in jobs[:15]:
        print(f"  RETRY [{job.nested_label}] {job.pdf_path.name}")
        print(f"       -> {job.json_path.relative_to(outputs_root)}")
    if len(jobs) > 15:
        print(f"  … and {len(jobs) - 15} more")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root with immediate subdirs (default: first existing ILSA Documents*)",
    )
    parser.add_argument(
        "--top-folder",
        action="append",
        dest="top_folders",
        metavar="NAME",
        help="Only process this top-level folder (e.g. IEA). Repeat for multiple.",
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
        "--outputs-root",
        type=Path,
        default=OUTPUTS_ROOT,
        help=f"Base outputs directory (default: {OUTPUTS_ROOT})",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help=f"Batch log to parse for FAIL lines (default: outputs/{BATCH_LOG_NAME})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List skip/retry counts without calling the API",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N retry PDFs",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    root = resolve_retry_root(args.root)
    outputs_root = expand_path(args.outputs_root)
    log_path = expand_path(args.log) if args.log else outputs_root / BATCH_LOG_NAME

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
        print_dry_run(root, outputs_root, args.top_folders, log_path)
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set in environment or .env file.")

    jobs, skipped, total = discover_retry_jobs(
        root,
        outputs_root=outputs_root,
        top_folders=args.top_folders,
        log_path=log_path,
    )
    if args.limit is not None:
        jobs = jobs[: args.limit]

    batch_log_path = outputs_root / "retry_failed_pdfs.log"
    batch_log = ThreadSafeBatchLog(batch_log_path)
    detail_log_path = outputs_root / "logs" / "retry_failed_pdfs.log"
    logger = setup_logging(detail_log_path)
    logger.setLevel(logging.DEBUG)

    batch_log.write(
        f"START | root={root} | workers={args.workers} | "
        f"total_pdfs={total} | skipped={skipped} | retry_queue={len(jobs)} | log={log_path}"
    )
    logger.info("Root: %s", root)
    logger.info("Skipped (valid JSON): %d / %d", skipped, total)
    logger.info("Retry queue: %d", len(jobs))

    if not jobs:
        batch_log.write("DONE | nothing to retry")
        print(f"Skipped: {skipped} | Retried: 0 | OK: 0 | FAIL: 0")
        return

    extractor = GPTExtractor(api_key=api_key, model=args.model)
    successes = 0
    failures = 0
    total_cost = 0.0
    progress_lock = threading.Lock()
    completed_count = 0
    total_jobs = len(jobs)

    def run_job(job: PdfJob):
        nonlocal completed_count
        with progress_lock:
            completed_count += 1
            idx = completed_count
        batch_log.write(
            f"PROCESS | [{job.nested_label}] {job.pdf_path.name} ({idx}/{total_jobs})"
        )
        logger.info("Retry %s -> %s", job.pdf_path, job.json_path)
        return process_one(job, extractor)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        with tqdm(total=len(futures), desc="Retrying") as pbar:
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                    if result.success:
                        successes += 1
                        batch_log.write(
                            f"OK | [{job.nested_label}] {job.pdf_path.name} | "
                            f"cost=${result.cost_usd:.4f}"
                        )
                    else:
                        failures += 1
                        batch_log.write(
                            f"FAIL | [{job.nested_label}] {job.pdf_path.name} | "
                            f"error={result.error}"
                        )
                        logger.warning("FAILED: %s — %s", job.pdf_path.name, result.error)
                    total_cost += result.cost_usd
                except BaseException as exc:
                    failures += 1
                    batch_log.write(
                        f"FAIL | [{job.nested_label}] {job.pdf_path.name} | error={exc!s}"
                    )
                    logger.exception("Worker exception on %s", job.pdf_path.name)
                pbar.set_postfix({"ok": successes, "fail": failures})
                pbar.update(1)

    batch_log.write(
        f"FINISH | skipped={skipped} | retried={len(jobs)} | ok={successes} | fail={failures} | "
        f"cost=${total_cost:.2f}"
    )
    print(
        f"Skipped: {skipped} | Retried: {len(jobs)} | OK: {successes} | FAIL: {failures}"
    )


if __name__ == "__main__":
    main()
