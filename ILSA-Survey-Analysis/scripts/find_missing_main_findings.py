#!/usr/bin/env python3
"""
Report (and optionally fix) article JSON files with missing/weak main_findings.

Default: report only. Use --fix to re-run GPTExtractor._sanitize migration locally
(no OpenAI / API calls).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas.findings_validation import article_requires_main_findings
from src.schemas.models import validate_public_article_json

DEFAULT_JSON_DIR = PROJECT_ROOT / "outputs" / "articles" / "json"
_LEGACY_MARKER = "legacy migration"


def audit_file(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    data = raw.get("data") or {}
    metadata = raw.get("metadata") or {}
    issues: list[str] = []

    if "main_findings" not in data:
        issues.append("missing_key")
    mf = data.get("main_findings")
    if not isinstance(mf, list):
        issues.append("not_list")
        mf = []
    if len(mf) == 0:
        issues.append("empty_list")
    else:
        for i, row in enumerate(mf):
            if not isinstance(row, dict):
                issues.append(f"entry_{i}_not_dict")
                continue
            if not str(row.get("target_variable") or "").strip():
                issues.append(f"entry_{i}_no_target")
            if not str(row.get("dataset_used") or "").strip():
                issues.append(f"entry_{i}_no_dataset")
            ds = str(row.get("dataset_used") or "").lower()
            tv = str(row.get("target_variable") or "").lower()
            if _LEGACY_MARKER in ds or _LEGACY_MARKER in tv:
                issues.append("legacy_placeholder")

    osum = str(data.get("outcome_summary") or "").strip()
    if osum and not mf and not osum.startswith("EXTRACTION_FAILED"):
        issues.append("outcome_without_findings")

    requires = article_requires_main_findings(data, metadata)
    if requires and not mf:
        issues.append("required_but_empty")

    return {
        "path": path,
        "issues": issues,
        "requires_findings": requires,
        "n_findings": len(mf) if isinstance(mf, list) else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=DEFAULT_JSON_DIR,
        help=f"Article JSON directory (default: {DEFAULT_JSON_DIR})",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Re-sanitize files in place (migration only, no API)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan all *.json under --json-dir recursively",
    )
    args = parser.parse_args()

    if args.recursive:
        paths = sorted(args.json_dir.rglob("*.json"))
    else:
        paths = sorted(args.json_dir.glob("*.json"))
    if not paths:
        print(f"No JSON files in {args.json_dir}")
        return

    if args.fix:
        from scripts.resanitize_json_outputs import resanitize_file

        fixed = 0
        for path in paths:
            try:
                out = resanitize_file(path)
                path.write_text(
                    json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                fixed += 1
            except Exception as exc:
                print(f"FAIL {path.name}: {exc}")
        print(f"Resanitized {fixed}/{len(paths)} files.")
        if args.recursive:
            paths = sorted(args.json_dir.rglob("*.json"))
        else:
            paths = sorted(args.json_dir.glob("*.json"))

    missing_key = empty = legacy = required_empty = invalid = 0
    flagged: list[tuple[str, list[str]]] = []

    for path in paths:
        result = audit_file(path)
        if not result["issues"]:
            continue
        flagged.append((path.name, result["issues"]))
        if "missing_key" in result["issues"]:
            missing_key += 1
        if "empty_list" in result["issues"] or "required_but_empty" in result["issues"]:
            empty += 1
        if "legacy_placeholder" in result["issues"]:
            legacy += 1
        if "required_but_empty" in result["issues"]:
            required_empty += 1
        if any(i.startswith("entry_") for i in result["issues"]):
            invalid += 1

    print(f"Scanned {len(paths)} JSON files under {args.json_dir}")
    print(f"Files with any issue: {len(flagged)}")
    print(f"  missing main_findings key: {missing_key}")
    print(f"  empty / required-but-empty: {empty}")
    print(f"  legacy placeholder rows: {legacy}")
    print(f"  invalid entries: {invalid}")
    print(f"  required_but_empty: {required_empty}")

    if flagged:
        print("\nFlagged files:")
        for name, issues in flagged[:40]:
            print(f"  {name}: {', '.join(issues)}")
        if len(flagged) > 40:
            print(f"  ... and {len(flagged) - 40} more")

    # Public JSON shape: {"metadata": MetadataBlock, "data": DataBlock}.
    # Validate the full root object via ILSAArticleMetadata — not raw["metadata"].
    schema_fail = 0
    for path in paths:
        try:
            validate_public_article_json(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except Exception:
            schema_fail += 1
    print(f"\nPydantic validation failures: {schema_fail}")


if __name__ == "__main__":
    main()
