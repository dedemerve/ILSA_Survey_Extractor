#!/usr/bin/env python3
"""
Extract ILSA articles from the Survey Paper LLM Extraction folder.

Skips numbered papers 1–10 (already in outputs/articles/json/).
Output naming: {pdf_path.stem[:80]}.json (matches extract_missing_articles.py).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from pydantic import ValidationError

from src.extractors.gpt_extractor import GPTExtractor
from src.extractors.pdf_processor import process_pdf
from src.schemas import ILSAArticleMetadata

DEFAULT_SOURCE = Path.home() / "Desktop" / "ILSA Survey Paper LLM Extraction"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "articles" / "json"
EXCLUDE_NUMS = frozenset(range(1, 11))


def _norm_stem(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", s.lower())[:100]


def _article_num(name: str) -> int | None:
    m = re.match(r"^(\d+)\.", name)
    return int(m.group(1)) if m else None


def _json_output_path(pdf_path: Path) -> Path:
    safe_name = pdf_path.stem[:80].replace("/", "_").replace("\\", "_")
    return OUTPUT_DIR / f"{safe_name}.json"


def _load_existing_index(json_dir: Path) -> tuple[set[str], set[str]]:
    stems: set[str] = set()
    dois: set[str] = set()
    for jf in json_dir.glob("*.json"):
        stems.add(_norm_stem(jf.stem))
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            doi = (data.get("metadata") or {}).get("doi")
            if doi and str(doi).strip().lower() not in ("", "none", "n/a", "null"):
                dois.add(str(doi).strip().lower())
        except Exception:
            pass
    return stems, dois


def discover_pdfs(
    source_dir: Path,
    json_dir: Path,
    *,
    exclude_nums: frozenset[int] = EXCLUDE_NUMS,
) -> tuple[list[Path], dict[str, list[Path]]]:
    """Return (to_extract, skip_reasons)."""
    existing_stems, existing_dois = _load_existing_index(json_dir)
    skip: dict[str, list[Path]] = {
        "numbered_1_10": [],
        "json_exists": [],
        "stem_match": [],
        "duplicate_source_stem": [],
    }
    to_extract: list[Path] = []
    seen_source: set[str] = set()

    for pdf in sorted(source_dir.rglob("*.pdf")):
        num = _article_num(pdf.name)
        if num is not None and num in exclude_nums:
            skip["numbered_1_10"].append(pdf)
            continue
        out = _json_output_path(pdf)
        if out.exists():
            skip["json_exists"].append(pdf)
            continue
        pn = _norm_stem(pdf.stem[:80])
        if pn in existing_stems:
            skip["stem_match"].append(pdf)
            continue
        if pn in seen_source:
            skip["duplicate_source_stem"].append(pdf)
            continue
        seen_source.add(pn)
        to_extract.append(pdf)

    return to_extract, skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Folder with PDFs (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N PDFs (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List PDFs to extract without calling the API",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if JSON output exists",
    )
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        print(f"ERROR: source dir not found: {args.source_dir}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.force:
        pdfs = sorted(args.source_dir.rglob("*.pdf"))
        pdfs = [p for p in pdfs if (_article_num(p.name) or 0) not in EXCLUDE_NUMS]
        skip = {}
    else:
        pdfs, skip = discover_pdfs(args.source_dir, OUTPUT_DIR)

    if args.limit:
        pdfs = pdfs[: args.limit]

    total_in_folder = len(list(args.source_dir.rglob("*.pdf")))
    print(f"Source: {args.source_dir}")
    print(f"PDFs in folder: {total_in_folder}")
    if skip:
        for reason, items in skip.items():
            if items:
                print(f"  Skip ({reason}): {len(items)}")
    print(f"Queue: {len(pdfs)} PDFs → {OUTPUT_DIR}\n")

    if args.dry_run:
        for p in pdfs:
            print(p.name)
        return

    if not pdfs:
        print("Nothing to extract.")
        return

    extractor = GPTExtractor()
    total_cost = 0.0
    ok = 0
    failed: list[tuple[str, str]] = []

    for i, pdf_path in enumerate(pdfs, 1):
        out_path = _json_output_path(pdf_path)
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(pdfs)}] {pdf_path.name[:70]}")
        print("=" * 70, flush=True)

        t0 = time.perf_counter()
        try:
            processed = process_pdf(pdf_path, source_database="survey_paper")
        except Exception as e:
            failed.append((pdf_path.name, f"pdf_process: {e}"))
            print(f"  FAILED (pdf): {e}")
            continue

        if not processed.extraction_text:
            failed.append((pdf_path.name, f"no_text: {processed.parse_errors}"))
            print(f"  SKIP: No text ({processed.parse_errors})")
            continue

        result = extractor.extract(processed)
        elapsed = time.perf_counter() - t0

        if not result.success:
            failed.append((pdf_path.name, result.error or "unknown"))
            print(f"  FAILED: {result.error}")
            continue

        try:
            ILSAArticleMetadata.model_validate(result.extraction.model_dump())
        except ValidationError as e:
            failed.append((pdf_path.name, f"validation: {e}"))
            print(f"  FAILED validation: {e}")
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
    print(f"DONE — {ok}/{len(pdfs)} OK | Failures: {len(failed)} | Total cost: ${total_cost:.4f}")
    if failed:
        print("\nFailures:")
        for name, err in failed:
            print(f"  - {name[:60]}: {err[:120]}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
