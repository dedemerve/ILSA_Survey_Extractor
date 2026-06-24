#!/usr/bin/env python3
"""Report cell fill rate per sheet in ILSA_Meta_Analysis_Dataset.xlsx."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = PROJECT_ROOT / "outputs" / "ILSA_Meta_Analysis_Dataset.xlsx"


def _blank_mask(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


def report_sheet(name: str, df: pd.DataFrame) -> None:
    if df.empty:
        print(f"\n=== {name} === (empty)")
        return
    cells = df.size
    blanks = int(_blank_mask(df).sum().sum())
    filled = cells - blanks
    pct = 100.0 * filled / cells if cells else 0.0
    print(f"\n=== {name} ===")
    print(f"  Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"  Filled cells: {filled}/{cells} ({pct:.1f}%)")
    print("  Blanks by column:")
    for col in df.columns:
        n_blank = int(_blank_mask(df[col]).sum())
        if n_blank:
            print(f"    {col}: {n_blank} blank ({100 * n_blank / len(df):.1f}% of rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=DEFAULT_XLSX,
        help=f"Excel path (default: {DEFAULT_XLSX})",
    )
    args = parser.parse_args()
    path = args.xlsx.resolve()
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        report_sheet(sheet, pd.read_excel(path, sheet_name=sheet))


if __name__ == "__main__":
    main()
