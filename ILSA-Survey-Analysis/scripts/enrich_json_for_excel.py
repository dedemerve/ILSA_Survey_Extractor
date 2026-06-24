#!/usr/bin/env python3
"""
Gap-fill on-disk article JSON for Excel export (anti-hallucination, deterministic).

Re-applies GPTExtractor sanitize/post-process plus Tier A/B rules from
``src/enrichment/json_gap_fill.py``. Does not call OpenAI.

Default: all ``*.json`` under ``outputs/`` recursively.

After completion, rebuild Excel::

    python3 scripts/build_tabular_dataset.py --workers 16
    python3 scripts/report_excel_completeness.py
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.enrichment.json_gap_fill import enrich_article_dict

DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _process_one(path: Path, *, dry_run: bool) -> tuple[str, str | None]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "metadata" not in raw:
            return path.name, "SKIP: not article JSON"
        out = enrich_article_dict(raw)
        if not dry_run:
            path.write_text(
                json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return path.name, None
    except Exception as exc:
        return path.name, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help=f"Root to rglob for JSON (default: {DEFAULT_OUTPUTS_DIR})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate gap-fill without writing files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-file progress.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Optional specific JSON paths; default: all under --outputs-dir",
    )
    args = parser.parse_args()

    outputs_dir = args.outputs_dir.resolve()
    if args.files:
        paths = [Path(p).resolve() for p in args.files]
    else:
        paths = sorted(outputs_dir.rglob("*.json"))

    if not paths:
        print(f"No JSON files under {outputs_dir}")
        return

    print(f"Gap-fill {len(paths)} JSON file(s) (dry_run={args.dry_run})", flush=True)
    ok = fail = 0
    workers = max(1, min(args.workers, len(paths)))

    if workers == 1:
        for i, path in enumerate(paths, 1):
            name, err = _process_one(path, dry_run=args.dry_run)
            if err and err.startswith("SKIP"):
                if args.verbose:
                    print(f"SKIP {name}: {err}")
            elif err:
                print(f"FAIL {name}: {err}")
                fail += 1
            else:
                ok += 1
                if args.verbose and (i % 100 == 0 or i == len(paths)):
                    print(f"OK {i}/{len(paths)} {name}")
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(_process_one, p, dry_run=args.dry_run): p for p in paths
            }
            for fut in as_completed(futs):
                done += 1
                name, err = fut.result()
                if err and err.startswith("SKIP"):
                    if args.verbose:
                        print(f"SKIP {name}")
                elif err:
                    print(f"FAIL {name}: {err}")
                    fail += 1
                else:
                    ok += 1
                    if args.verbose and (done % 100 == 0 or done == len(paths)):
                        print(f"OK {done}/{len(paths)}")

    print(f"\nGap-fill complete: {ok} OK, {fail} failed, {len(paths)} total.")
    if not args.dry_run and ok:
        print("Next: python3 scripts/build_tabular_dataset.py --workers 16")


if __name__ == "__main__":
    main()
