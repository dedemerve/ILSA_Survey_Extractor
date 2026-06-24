#!/usr/bin/env python3
"""
Rule-based academic prose from final_knowledge_synthesis_v4.csv (no LLM / no API).

Reads the pre-aggregated matrix (Canonical_Method × Canonical_Variable ×
Aggregate_Effect_Trend × Study_Count) and emits template-bound Turkish sentences
so every claim is traceable to Study_Count in the synthesis table.

Usage:
  python scripts/generate_academic_synthesis.py
  python scripts/generate_academic_synthesis.py --method Traditional_Stats \\
      --variable Teacher_Efficacy_Workforce
  python scripts/generate_academic_synthesis.py --lang en --output outputs/academic_synthesis_en.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.enrichment.canonical_taxonomy import (  # noqa: E402
    META_SYNTHESIS_CANONICAL_METHOD,
    METADATA_FILTER_META,
    THEORETICAL_META_SYNTHESIS,
)

DEFAULT_CSV = PROJECT_ROOT / "outputs" / "final_knowledge_synthesis_v4.csv"
DEFAULT_OUT_SENTENCES = PROJECT_ROOT / "outputs" / "academic_synthesis_sentences.csv"
DEFAULT_OUT_REPORT = PROJECT_ROOT / "outputs" / "academic_synthesis_report_tr.md"

METHOD_LABELS: dict[str, dict[str, str]] = {
    "Traditional_Stats": {"tr": "geleneksel istatistik", "en": "traditional statistics"},
    "Deep_Learning": {"tr": "derin öğrenme", "en": "deep learning"},
    "Ensemble_Learning": {"tr": "ensemble öğrenme", "en": "ensemble learning"},
    "Supervised_General": {"tr": "genel denetimli öğrenme", "en": "supervised machine learning"},
    "Unsupervised": {"tr": "denetimsiz öğrenme", "en": "unsupervised learning"},
    META_SYNTHESIS_CANONICAL_METHOD: {
        "tr": "teorik ve meta-sentez",
        "en": "theoretical and meta-synthesis",
    },
}

TEMPLATES: dict[str, dict[str, str]] = {
    "tr": {
        "Positive": (
            "{n} çalışma, {method} yöntemiyle {variable} için pozitif bir eğilim "
            "raporlamıştır (Study_Count={n})."
        ),
        "Negative": (
            "{n} çalışma, {method} yöntemiyle {variable} için negatif bir eğilim "
            "raporlamıştır (Study_Count={n})."
        ),
        "Null": (
            "{n} çalışma, {method} yöntemiyle {variable} için net yönlü bir etki "
            "bildirmemiş veya null/belirsiz sonuç raporlamıştır (Study_Count={n})."
        ),
        "meta": (
            "{n} kayıt, {method} katmanında {variable} bağlamında literatür düzeyinde "
            "sentez/tartışma içermektedir; bu satırlar ampirik etki büyüklüğü değil, "
            "meta-bilişsel özet niteliğindedir (Study_Count={n})."
        ),
    },
    "en": {
        "Positive": (
            "{n} studies using {method} reported a positive trend for {variable} "
            "(Study_Count={n})."
        ),
        "Negative": (
            "{n} studies using {method} reported a negative trend for {variable} "
            "(Study_Count={n})."
        ),
        "Null": (
            "{n} studies using {method} reported null or non-directional findings "
            "for {variable} (Study_Count={n})."
        ),
        "meta": (
            "{n} records in the {method} layer address {variable} as theoretical or "
            "literature-level synthesis rather than student-level effect sizes "
            "(Study_Count={n})."
        ),
    },
}


def _display_method(method: str, lang: str) -> str:
    if method in METHOD_LABELS:
        return METHOD_LABELS[method][lang]
    return method.replace("_", " ").lower()


def _display_variable(variable: str, lang: str) -> str:
    if variable == THEORETICAL_META_SYNTHESIS:
        return (
            "teorik ve meta-sentez içeriği"
            if lang == "tr"
            else "theoretical and meta-synthesis content"
        )
    return variable.replace("_", " ")


def sentence_for_row(row: pd.Series, *, lang: str = "tr") -> str:
    """One template-bound sentence for a single synthesis matrix row."""
    method = str(row["Canonical_Method"])
    variable = str(row["Canonical_Variable"])
    trend = str(row["Aggregate_Effect_Trend"])
    n = int(row["Study_Count"])
    flag = str(row.get("Metadata_Filter_Flag", "empirical_finding"))

    tpl_bank = TEMPLATES[lang]
    if flag == METADATA_FILTER_META or variable == THEORETICAL_META_SYNTHESIS:
        key = "meta"
    else:
        key = trend if trend in ("Positive", "Negative", "Null") else "Null"
    template = tpl_bank[key]

    return template.format(
        n=n,
        method=_display_method(method, lang),
        variable=_display_variable(variable, lang),
        trend=trend,
    )


def build_sentences_df(df: pd.DataFrame, *, lang: str = "tr") -> pd.DataFrame:
    out = df.copy()
    out["Academic_Sentence"] = out.apply(lambda r: sentence_for_row(r, lang=lang), axis=1)
    return out


def build_grouped_report(df: pd.DataFrame, *, lang: str = "tr") -> str:
    """Aggregate sentences into method → variable paragraphs."""
    lines: list[str] = []
    title = (
        "# ILSA Meta-Analiz — Kural Tabanlı Akademik Sentez (API yok)\n"
        if lang == "tr"
        else "# ILSA Meta-Analysis — Rule-Based Academic Synthesis (no API)\n"
    )
    lines.append(title)
    lines.append(
        "_Her cümle `final_knowledge_synthesis_v4.csv` içindeki Study_Count değerinden "
        "türetilmiştir; LLM kullanılmamıştır._\n"
        if lang == "tr"
        else "_Each sentence is derived from Study_Count in `final_knowledge_synthesis_v4.csv`; "
        "no LLM was used._\n"
    )

    for method in sorted(df["Canonical_Method"].unique()):
        mdf = df[df["Canonical_Method"] == method]
        lines.append(f"\n## {_display_method(method, lang).title()} ({method})\n")
        for variable in sorted(mdf["Canonical_Variable"].unique()):
            vdf = mdf[mdf["Canonical_Variable"] == variable]
            lines.append(f"\n### {_display_variable(variable, lang)}\n")
            for _, row in vdf.sort_values("Aggregate_Effect_Trend").iterrows():
                lines.append(f"- {sentence_for_row(row, lang=lang)}")
            total = int(vdf["Study_Count"].sum())
            if lang == "tr":
                lines.append(
                    f"\n_Ozet (bu ikili): toplam {total} study-count birimi, "
                    f"{len(vdf)} trend kovası._\n"
                )
            else:
                lines.append(
                    f"\n_Summary (pair): {total} study-count units across "
                    f"{len(vdf)} trend buckets._\n"
                )
    return "\n".join(lines)


def filter_df(
    df: pd.DataFrame,
    *,
    method: str | None = None,
    variable: str | None = None,
    trend: str | None = None,
    metadata_flag: str | None = None,
) -> pd.DataFrame:
    out = df
    if method:
        out = out[out["Canonical_Method"] == method]
    if variable:
        out = out[out["Canonical_Variable"] == variable]
    if trend:
        out = out[out["Aggregate_Effect_Trend"] == trend]
    if metadata_flag:
        out = out[out["Metadata_Filter_Flag"] == metadata_flag]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-sentences", type=Path, default=DEFAULT_OUT_SENTENCES)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUT_REPORT)
    parser.add_argument("--lang", choices=("tr", "en"), default="tr")
    parser.add_argument("--method", default=None, help="Filter Canonical_Method")
    parser.add_argument("--variable", default=None, help="Filter Canonical_Variable")
    parser.add_argument("--trend", default=None, choices=("Positive", "Negative", "Null"))
    parser.add_argument(
        "--metadata-flag",
        default=None,
        choices=("empirical_finding", "theoretical_meta_synthesis"),
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print filtered sentences to stdout; skip writing files",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required = {
        "Canonical_Method",
        "Canonical_Variable",
        "Aggregate_Effect_Trend",
        "Study_Count",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {sorted(missing)}")

    filtered = filter_df(
        df,
        method=args.method,
        variable=args.variable,
        trend=args.trend,
        metadata_flag=args.metadata_flag,
    )
    if filtered.empty:
        print("No rows match filters.")
        sys.exit(1)

    sentences_df = build_sentences_df(filtered, lang=args.lang)

    has_filter = any([args.method, args.variable, args.trend, args.metadata_flag])

    if has_filter or args.stdout_only:
        print("\n--- Filtered synthesis ---\n" if has_filter else "")
        for sent in sentences_df["Academic_Sentence"]:
            print(sent)
        if args.stdout_only or has_filter:
            return

    full_sentences = build_sentences_df(df, lang=args.lang)
    full_sentences.to_csv(args.output_sentences, index=False)
    report = build_grouped_report(df, lang=args.lang)
    args.output_report.write_text(report + "\n", encoding="utf-8")

    print(f"Wrote {len(full_sentences)} sentences -> {args.output_sentences}")
    print(f"Wrote grouped report -> {args.output_report}")


if __name__ == "__main__":
    main()
