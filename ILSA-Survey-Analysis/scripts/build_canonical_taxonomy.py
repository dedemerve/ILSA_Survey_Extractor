#!/usr/bin/env python3
"""
Build canonical taxonomy artifacts for RAG semantic depth (Stage-1, no API).

Reads outputs/ILSA_Meta_Analysis_Dataset_CLEAN.xlsx and writes:
  - outputs/taxonomy_map.json
  - outputs/variable_taxonomy_map.json
  - outputs/knowledge_synthesis.csv
  - outputs/synthesis_audit.log
  - Appends sheet Canonical_View to CLEAN workbook

Does not modify on-disk JSON extractions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.enrichment.canonical_taxonomy import (
    build_canonical_view,
    build_knowledge_synthesis,
    build_taxonomy_maps,
    write_taxonomy_artifacts,
)

DEFAULT_CLEAN = PROJECT_ROOT / "outputs" / "ILSA_Meta_Analysis_Dataset_CLEAN.xlsx"
DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"


def _collect_variable_names(findings: pd.DataFrame, confounders: pd.DataFrame) -> list[str]:
    names: list[str] = []
    if "target_variable" in findings.columns:
        names.extend(findings["target_variable"].dropna().astype(str).tolist())
    if "variable_name" in confounders.columns:
        names.extend(confounders["variable_name"].dropna().astype(str).tolist())
    return names


def _append_sheet_to_workbook(xlsx_path: Path, sheet_name: str, df: pd.DataFrame) -> None:
    with pd.ExcelWriter(
        xlsx_path,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-xlsx", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS)
    args = parser.parse_args()

    xlsx = args.input_xlsx.resolve()
    out_dir = args.outputs_dir.resolve()

    master = pd.read_excel(xlsx, sheet_name="1_Articles_Master")
    findings = pd.read_excel(xlsx, sheet_name="2_Main_Findings")
    confounders = pd.read_excel(xlsx, sheet_name="3_Confounders")

    # Merge ml_techniques onto findings for synthesis when missing on child rows.
    if not master.empty and "file_name" in master.columns:
        ml_cols = [c for c in ("file_name", "ml_techniques", "ml_primary") if c in master.columns]
        findings = findings.merge(
            master[ml_cols].drop_duplicates(subset=["file_name"]),
            on="file_name",
            how="left",
            suffixes=("", "_master"),
        )

    var_names = _collect_variable_names(findings, confounders)
    taxonomy_map, flat_map = build_taxonomy_maps(var_names)
    write_taxonomy_artifacts(out_dir, taxonomy_map, flat_map)

    synthesis_df, audit_lines, _unresolved = build_knowledge_synthesis(findings, master, version="v1")
    synthesis_path = out_dir / "knowledge_synthesis.csv"
    synthesis_df.to_csv(synthesis_path, index=False)

    audit_path = out_dir / "synthesis_audit.log"
    header = (
        "# synthesis_audit.log — canonical taxonomy build\n"
        "# [IGNORE] = None/NaN/sentinel skipped (zero imputation).\n"
    )
    audit_path.write_text(
        header + "\n".join(audit_lines[:5000]) + ("\n" if audit_lines else ""),
        encoding="utf-8",
    )

    canonical_view = build_canonical_view(master, findings, confounders)
    _append_sheet_to_workbook(xlsx, "Canonical_View", canonical_view)

    n_cat = len(taxonomy_map.get("canonical_categories", {}))
    n_mapped = len(flat_map)
    n_uncat = sum(1 for v in flat_map.values() if v == "Uncategorized_Contextual")
    print(f"Wrote {out_dir / 'taxonomy_map.json'} ({n_cat} categories)")
    print(f"Wrote {out_dir / 'variable_taxonomy_map.json'} ({n_mapped} variable keys)")
    print(f"  Uncategorized_Contextual: {n_uncat} ({100*n_uncat/max(n_mapped,1):.1f}%)")
    print(f"Wrote {synthesis_path} ({len(synthesis_df)} synthesis rows)")
    print(f"Wrote {audit_path} ({len(audit_lines)} audit lines)")
    print(f"Updated {xlsx} — sheet Canonical_View ({len(canonical_view)} rows)")


if __name__ == "__main__":
    main()
