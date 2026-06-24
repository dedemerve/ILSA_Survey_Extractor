#!/usr/bin/env python3
"""
Build Semantic Knowledge Base v2 (Smart Domain Resolver + harmonized synthesis).

Reads ILSA_Meta_Analysis_Dataset_CLEAN.xlsx; writes:
  - outputs/final_knowledge_synthesis_v2.csv
  - outputs/canonical_codebook.csv / canonical_codebook.md
  - outputs/semantic_knowledge_base.json
  - outputs/taxonomy_map.json, variable_taxonomy_map.json (refreshed)
  - Appends unresolved tail to outputs/audit_log.txt

Does not modify on-disk JSON extractions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.enrichment.canonical_taxonomy import (
    THEORETICAL_META_SYNTHESIS,
    UNCATEGORIZED,
    build_canonical_codebook,
    build_knowledge_synthesis,
    build_rag_navigation_map,
    build_semantic_knowledge_base,
    build_taxonomy_maps,
    write_synthesis_audit_log,
    write_taxonomy_artifacts,
    write_unresolved_audit,
)

DEFAULT_CLEAN = PROJECT_ROOT / "outputs" / "ILSA_Meta_Analysis_Dataset_CLEAN.xlsx"
DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"
DEFAULT_AUDIT = DEFAULT_OUTPUTS / "audit_log.txt"


def _collect_variable_names(findings: pd.DataFrame, confounders: pd.DataFrame) -> list[str]:
    names: list[str] = []
    if "target_variable" in findings.columns:
        names.extend(findings["target_variable"].dropna().astype(str).tolist())
    if "variable_name" in confounders.columns:
        names.extend(confounders["variable_name"].dropna().astype(str).tolist())
    return names


def _write_codebook_md(df: pd.DataFrame, path: Path) -> None:
    lines = ["# ILSA Canonical Codebook (Appendix)\n"]
    for _, row in df.iterrows():
        lines.append(f"## {row['Canonical_Category']}\n")
        lines.append(f"**Examples:** {row['Original_Source_Examples']}\n")
        lines.append(f"**Definition:** {row['Operational_Definition']}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-xlsx", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--audit-log", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--version",
        choices=("v2", "v4"),
        default="v4",
        help="Synthesis export version tag (default: v4)",
    )
    args = parser.parse_args()

    out_dir = args.outputs_dir.resolve()
    xlsx = args.input_xlsx.resolve()

    master = pd.read_excel(xlsx, sheet_name="1_Articles_Master")
    findings = pd.read_excel(xlsx, sheet_name="2_Main_Findings")
    confounders = pd.read_excel(xlsx, sheet_name="3_Confounders")

    if not master.empty and "file_name" in master.columns:
        ml_cols = [c for c in ("file_name", "ml_techniques", "ml_primary") if c in master.columns]
        findings = findings.merge(
            master[ml_cols].drop_duplicates(subset=["file_name"]),
            on="file_name",
            how="left",
        )

    var_names = _collect_variable_names(findings, confounders)
    taxonomy_map, flat_map = build_taxonomy_maps(var_names)
    write_taxonomy_artifacts(out_dir, taxonomy_map, flat_map)

    synthesis_df, audit_lines, unresolved = build_knowledge_synthesis(
        findings, master, version=args.version,
    )

    synthesis_df.to_csv(out_dir / "knowledge_synthesis.csv", index=False)
    meta_stats = write_synthesis_audit_log(
        findings,
        synthesis_df,
        out_dir / "synthesis_audit.log",
        example_n=8,
    )

    # RAG export: exclude Uncategorized_Contextual buckets (v2/v4).
    synthesis_export = synthesis_df[
        synthesis_df["Canonical_Variable"] != UNCATEGORIZED
    ].copy()
    synthesis_export["Aggregate_Effect_Trend"] = synthesis_export[
        "Aggregate_Effect_Trend"
    ].map(lambda t: t if t in ("Positive", "Negative", "Null") else "Null")

    export_name = f"final_knowledge_synthesis_{args.version}.csv"
    export_path = out_dir / export_name
    synthesis_export.to_csv(export_path, index=False)

    n_syn_uncat = int((synthesis_df["Canonical_Variable"] == UNCATEGORIZED).sum())
    n_syn_meta = int((synthesis_df["Canonical_Variable"] == THEORETICAL_META_SYNTHESIS).sum())
    n_export_uncat = int((synthesis_export["Canonical_Variable"] == UNCATEGORIZED).sum())

    codebook_df = build_canonical_codebook(flat_map)
    codebook_df.to_csv(out_dir / "canonical_codebook.csv", index=False)
    _write_codebook_md(codebook_df, out_dir / "canonical_codebook.md")

    navigation_map = build_rag_navigation_map()
    skb = build_semantic_knowledge_base(
        synthesis_export,
        codebook_df,
        taxonomy_map,
        navigation_map=navigation_map,
    )
    skb["version"] = args.version
    skb_path = out_dir / f"semantic_knowledge_base_{args.version}.json"
    skb_path.write_text(
        json.dumps(skb, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "semantic_knowledge_base.json").write_text(
        json.dumps(skb, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    audit_path = args.audit_log.resolve()
    header = (
        "# audit_log.txt — Semantic Knowledge Base v2\n"
        "# Smart Domain Resolver + effect harmonization\n"
    )
    if not audit_path.exists():
        audit_path.write_text(header, encoding="utf-8")
    write_unresolved_audit(unresolved, audit_path, tail_n=20)

    n_mapped = len(flat_map)
    n_uncat = sum(1 for v in flat_map.values() if v == UNCATEGORIZED)
    print(f"Variable map: {n_mapped} keys, Uncategorized_Contextual {n_uncat} ({100*n_uncat/max(n_mapped,1):.1f}%)")
    print(
        f"Synthesis rows total: {len(synthesis_df)}, "
        f"Theoretical_and_Meta_Synthesis rows: {n_syn_meta}, "
        f"Uncategorized rows: {n_syn_uncat}"
    )
    print(
        f"Meta isolation audit: {meta_stats['finding_rows']} finding rows, "
        f"{meta_stats['unique_articles']} articles -> {out_dir / 'synthesis_audit.log'}"
    )
    print(
        f"Exported RAG synthesis: {len(synthesis_export)} rows, "
        f"Uncategorized in export: {n_export_uncat} -> {export_path}"
    )
    print(
        f"Validation: synthesis Uncategorized rows (pre-export) = {n_syn_uncat} "
        f"(target <= 3: {'OK' if n_syn_uncat <= 3 else 'REVIEW'}); "
        f"export uncategorized = {n_export_uncat}"
    )
    print(f"Wrote {out_dir / 'canonical_codebook.csv'} and canonical_codebook.md")
    print(f"Wrote {skb_path} and semantic_knowledge_base.json")
    print(f"Unresolved tail (max 20) appended to {audit_path}")


if __name__ == "__main__":
    main()
