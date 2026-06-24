#!/usr/bin/env python3
"""
Extract ILSA articles from ~/Desktop/articles into outputs/articles/json/.

Output naming (matches tests/run_articles_3_5_6_8_10.py, NOT run_pipeline.py):
  {pdf_path.stem[:80]}.json
  — numbered prefix + truncated stem so filenames stay stable across OS limits.

run_pipeline.py uses full pdf_path.stem without truncation; do not mix strategies
in the same output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.extractors.gpt_extractor import GPTExtractor
from src.extractors.pdf_processor import process_pdf

ARTICLES_DIR = Path.home() / "Desktop" / "articles"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "articles" / "json"
DEFAULT_MISSING = ("1", "2", "4", "7", "9")


def _article_num(pdf_path: Path) -> str:
    return pdf_path.name.split(".")[0].strip()


def _json_output_path(pdf_path: Path) -> Path:
    safe_name = pdf_path.stem[:80].replace("/", "_").replace("\\", "_")
    return OUTPUT_DIR / f"{safe_name}.json"


def _already_done(pdf_path: Path) -> bool:
    return _json_output_path(pdf_path).exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nums",
        nargs="+",
        default=list(DEFAULT_MISSING),
        help=f"Article number prefixes to extract (default: {' '.join(DEFAULT_MISSING)})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if JSON output already exists",
    )
    args = parser.parse_args()
    target_nums = {n.strip() for n in args.nums}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(
        p for p in ARTICLES_DIR.glob("*.pdf")
        if _article_num(p) in target_nums
    )
    if not pdfs:
        print(f"No PDFs for articles {sorted(target_nums)} in {ARTICLES_DIR}")
        return

    print(f"Extracting up to {len(pdfs)} PDFs → {OUTPUT_DIR}\n")
    extractor = GPTExtractor()
    total_cost = 0.0
    ok = 0

    for i, pdf_path in enumerate(pdfs, 1):
        out_path = _json_output_path(pdf_path)
        if out_path.exists() and not args.force:
            print(f"[{i}/{len(pdfs)}] SKIP (exists): {out_path.name}")
            ok += 1
            continue

        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(pdfs)}] {_article_num(pdf_path)} — {pdf_path.name[:65]}")
        print("=" * 70, flush=True)

        t0 = time.perf_counter()
        processed = process_pdf(pdf_path, source_database="articles")
        if not processed.extraction_text:
            print(f"  SKIP: No text ({processed.parse_errors})")
            continue

        result = extractor.extract(processed)
        elapsed = time.perf_counter() - t0

        if not result.success:
            print(f"  FAILED: {result.error}")
            continue

        total_cost += result.cost_usd
        ok += 1
        output = result.extraction.model_dump(mode="json")
        out_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        meta = output["metadata"]
        conf = output["data"]["confounders_identified"]
        print(f"  Duration: {elapsed:.1f}s | Cost: ${result.cost_usd:.4f}")
        print(f"  Title: {(meta.get('title') or 'N/A')[:75]}")
        print(f"  Confounders: {len(conf)}")
        print(f"  Saved: {out_path.name}", flush=True)

    print(f"\n{'=' * 70}")
    print(f"DONE — {ok}/{len(pdfs)} OK | Total cost: ${total_cost:.4f}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
