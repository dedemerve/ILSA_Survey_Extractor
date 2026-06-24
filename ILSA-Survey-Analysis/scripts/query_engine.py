#!/usr/bin/env python3
"""
ILSA Semantic Knowledge Base — vector index and RAG query engine.

Indexes outputs/final_knowledge_synthesis_v4.csv into ChromaDB with Metadata_Filter_Flag
for filtered retrieval. Answers queries using retrieved synthesis rows (LLM optional).

Usage:
  python scripts/query_engine.py index
  python scripts/query_engine.py query "Your question here"
  python scripts/query_engine.py stress-test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SYNTHESIS = PROJECT_ROOT / "outputs" / "final_knowledge_synthesis_v4.csv"
DEFAULT_PERSIST = PROJECT_ROOT / "outputs" / "chroma_ilsa_synthesis"
COLLECTION_NAME = "ilsa_knowledge_synthesis_v4"

SYSTEM_PROMPT = """Sen bir eğitim araştırmacısı ve ILSA meta-analiz uzmanısın.

Kullanıcı bir soru sorduğunda:
1. empirical_finding etiketli kaynaklardan Aggregate_Effect_Trend (Positive / Negative / Null)
   ve Study_Count üzerinden ampirik eğilimi özetle.
2. Soru literatür tartışması, politika veya metodoloji içeriyorsa theoretical_meta_synthesis
   etiketli kaynaklardan destekleyici bağlam ekle.
3. Yalnızca verilen bağlamdaki sentez satırlarına dayan; uydurma yapma.
4. Türkçe, akademik ve net yaz."""

# Chroma cosine distance: strong matches ~0.55–0.62; weak/unrelated ~0.80+.
MAX_RELEVANCE_DISTANCE = 0.72

STRESS_TEST_QUERY = (
    "Literatürde Teacher_Efficacy_Workforce ile ilgili çalışmalar, öğretmenlerin iş tatmini "
    "ve verimliliği konusunda ne tür bir eğilim gösteriyor? Özellikle 'Traditional_Stats' "
    "metodolojisi ile elde edilen bulgular nelerdir?"
)


def _row_to_document(row: pd.Series) -> str:
    return (
        f"Canonical_Method: {row['Canonical_Method']}. "
        f"Canonical_Variable: {row['Canonical_Variable']}. "
        f"Aggregate_Effect_Trend: {row['Aggregate_Effect_Trend']}. "
        f"Study_Count: {row['Study_Count']}. "
        f"Metadata_Filter_Flag: {row.get('Metadata_Filter_Flag', 'empirical_finding')}."
    )


def _row_metadata(row: pd.Series) -> dict[str, Any]:
    return {
        "canonical_method": str(row["Canonical_Method"]),
        "canonical_variable": str(row["Canonical_Variable"]),
        "aggregate_effect_trend": str(row["Aggregate_Effect_Trend"]),
        "study_count": int(row["Study_Count"]),
        "metadata_filter_flag": str(
            row.get("Metadata_Filter_Flag", "empirical_finding"),
        ),
    }


def load_synthesis(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "Canonical_Method",
        "Canonical_Variable",
        "Aggregate_Effect_Trend",
        "Study_Count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Synthesis CSV missing columns: {sorted(missing)}")
    if "Metadata_Filter_Flag" not in df.columns:
        df["Metadata_Filter_Flag"] = "empirical_finding"
    return df


def build_index(
    synthesis_path: Path,
    persist_dir: Path,
    *,
    reset: bool = True,
) -> int:
    """Index synthesis rows into ChromaDB."""
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError as exc:
        raise SystemExit(
            "Install RAG dependencies: pip install -r requirements-rag.txt",
        ) from exc

    df = load_synthesis(synthesis_path)
    persist_dir.mkdir(parents=True, exist_ok=True)

    if reset and persist_dir.exists():
        import shutil

        shutil.rmtree(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for i, row in df.iterrows():
        ids.append(f"syn_{i}")
        documents.append(_row_to_document(row))
        metadatas.append(_row_metadata(row))

    # Chroma batch limit — chunk if needed
    batch = 100
    for start in range(0, len(ids), batch):
        end = start + batch
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    manifest = {
        "source_csv": str(synthesis_path),
        "n_records": len(df),
        "collection": COLLECTION_NAME,
        "persist_dir": str(persist_dir),
    }
    (persist_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(df)


def _infer_filters(question: str) -> dict[str, Any]:
    """Lightweight query routing from natural language (no extra LLM call)."""
    q = question.lower()
    where: dict[str, Any] = {"metadata_filter_flag": "empirical_finding"}

    # Meta layer only when the user asks about discourse/reviews — not "literatürde X çalışmaları".
    meta_markers = (
        "literatür tartışması",
        "literature debate",
        "policy debate",
        "politika tartışması",
        "systematic review",
        "metodoloji tartışması",
        "methodology debate",
        "framework discourse",
        "tarihsel gelişim",
        "historical development",
    )
    if any(m in q for m in meta_markers):
        where["metadata_filter_flag"] = "theoretical_meta_synthesis"

    if (
        "traditional_stats" in q
        or "traditional stats" in q
        or "traditional_Stats" in question
        or "geleneksel istatistik" in q
    ):
        where["canonical_method"] = "Traditional_Stats"

    wants_math = (
        "math_achievement" in q
        or "mathematics achievement" in q
        or "matematik başarı" in q
    )
    wants_ses = bool(
        re.search(r"\bses\b", q)
        or "socioeconomic" in q
        or "sosyoekonomik" in q
        or "escs" in q
        or "hisei" in q
        or "homepos" in q
        or "ses-başarı" in q
        or "ses-achievement" in q
    )
    wants_teacher = (
        "teacher_efficacy" in q
        or "teacher efficacy" in q
        or "teacher_efficacy_workforce" in q
        or "iş tatmini" in q
        or ("öğretmen" in q and ("efficacy" in q or "yeterlik" in q or "tatmin" in q))
    )

    # Single-variable filter only when one construct is requested (avoid last-wins overwrite).
    n_vars = sum([wants_math, wants_ses, wants_teacher])
    if n_vars == 1:
        if wants_math:
            where["canonical_variable"] = "Math_Achievement"
        elif wants_ses:
            where["canonical_variable"] = "SES"
        elif wants_teacher:
            where["canonical_variable"] = "Teacher_Efficacy_Workforce"

    return where


def _filter_relevant_hits(
    hits: list[dict[str, Any]],
    question: str,
) -> list[dict[str, Any]]:
    """Drop weak semantic matches (hallucination guard)."""
    if not hits:
        return []
    relevant = [
        h for h in hits
        if h.get("distance") is not None and h["distance"] <= MAX_RELEVANCE_DISTANCE
    ]
    return relevant


def retrieve(
    question: str,
    persist_dir: Path,
    *,
    n_results: int = 12,
) -> list[dict[str, Any]]:
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError as exc:
        raise SystemExit(
            "Install RAG dependencies: pip install -r requirements-rag.txt",
        ) from exc

    client = chromadb.PersistentClient(path=str(persist_dir))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
    )
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    where = _infer_filters(question)
    kwargs: dict[str, Any] = {
        "query_texts": [question],
        "n_results": n_results,
    }
    if where:
        if len(where) == 1:
            kwargs["where"] = where
        else:
            kwargs["where"] = {
                "$and": [{k: v} for k, v in where.items()],
            }

    result = collection.query(**kwargs)
    hits: list[dict[str, Any]] = []
    if not result["ids"] or not result["ids"][0]:
        return hits

    for i, doc_id in enumerate(result["ids"][0]):
        meta = (result.get("metadatas") or [[{}]])[0][i] or {}
        hits.append(
            {
                "id": doc_id,
                "document": (result.get("documents") or [[""]])[0][i],
                "distance": (result.get("distances") or [[None]])[0][i],
                "metadata": meta,
            }
        )
    return _filter_relevant_hits(hits, question)


def _no_evidence_message(question: str) -> str:
    return (
        "## Kanıt bulunamadı (hallucination guard)\n\n"
        "Bu bilgi tabanında (174 sentez satırı) sorunuzla anlamlı şekilde eşleşen "
        "empirik sentez kaydı **yoktur**. Retrieval boş döndü veya benzerlik eşiği "
        f"({MAX_RELEVANCE_DISTANCE}) altında kaldı.\n\n"
        "**Yorum:** Bu konuda bulgu raporlanmamıştır; modelin dış bilgi üretmesi "
        "engellenmiştir.\n\n"
        f"_Soru:_ {question}"
    )


def _summarize_hits(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "İlgili sentez kaydı bulunamadı."

    lines = ["## Ampirik sentez (retrieval bağlamı)\n"]
    by_trend: dict[str, int] = {}
    total_studies = 0
    for h in hits:
        m = h["metadata"]
        trend = m.get("aggregate_effect_trend", "Null")
        cnt = int(m.get("study_count", 0))
        by_trend[trend] = by_trend.get(trend, 0) + cnt
        total_studies += cnt
        lines.append(
            f"- **{m.get('canonical_method')}** × **{m.get('canonical_variable')}** "
            f"→ **{trend}** (Study_Count={cnt})"
        )

    lines.append("\n### Trend özeti (Study_Count ağırlıklı)")
    for trend in ("Positive", "Negative", "Null"):
        if trend in by_trend:
            lines.append(f"- {trend}: {by_trend[trend]} çalışma")
    lines.append(f"- Toplam (çakışan kovalar): {total_studies} study-count birimi")
    return "\n".join(lines)


def answer_with_llm(question: str, context: str) -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=os.environ.get("ILSA_RAG_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Soru:\n{question}\n\nBağlam:\n{context}",
            },
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def run_query(
    question: str,
    persist_dir: Path,
    *,
    use_llm: bool = True,
) -> str:
    hits = retrieve(question, persist_dir)
    if not hits:
        return _no_evidence_message(question)
    context = _summarize_hits(hits)

    if use_llm:
        llm_answer = answer_with_llm(question, context)
        if llm_answer:
            return f"{llm_answer}\n\n---\n{context}"

    return (
        f"{context}\n\n"
        "_Not: OPENAI_API_KEY tanımlı değilse yalnızca retrieval özeti döner. "
        "Tam RAG yanıtı için `export OPENAI_API_KEY=...` sonra tekrar çalıştırın._"
    )


def run_stress_test(persist_dir: Path, *, use_llm: bool = False) -> int:
    print("=== ILSA RAG Stress Test ===\n")
    print(f"Soru:\n{STRESS_TEST_QUERY}\n")
    answer = run_query(STRESS_TEST_QUERY, persist_dir, use_llm=use_llm)
    print(answer)

    hits = retrieve(STRESS_TEST_QUERY, persist_dir, n_results=20)
    te_trad = [
        h
        for h in hits
        if h["metadata"].get("canonical_variable") == "Teacher_Efficacy_Workforce"
        and h["metadata"].get("canonical_method") == "Traditional_Stats"
    ]
    ok = len(te_trad) > 0 and all(h.get("distance", 1) <= MAX_RELEVANCE_DISTANCE for h in te_trad)
    print(
        f"\n=== Doğrulama: Teacher_Efficacy_Workforce + Traditional_Stats "
        f"retrieval → {'PASS' if ok else 'FAIL'} ({len(te_trad)} hit)"
    )
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("index", "query", "stress-test"),
        help="index | query | stress-test",
    )
    parser.add_argument("question", nargs="?", default="", help="Query text (query mode)")
    parser.add_argument("--synthesis-csv", type=Path, default=DEFAULT_SYNTHESIS)
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST)
    parser.add_argument("--no-llm", action="store_true", help="Retrieval-only answer")
    args = parser.parse_args()

    if args.command == "index":
        n = build_index(args.synthesis_csv, args.persist_dir, reset=True)
        print(f"Indexed {n} rows -> {args.persist_dir} (collection={COLLECTION_NAME})")
        return

    if not args.persist_dir.exists():
        print("Index missing. Run: python scripts/query_engine.py index")
        sys.exit(1)

    if args.command == "stress-test":
        sys.exit(run_stress_test(args.persist_dir, use_llm=not args.no_llm))

    if args.command == "query":
        if not args.question.strip():
            print("Provide a question string after 'query'")
            sys.exit(1)
        print(run_query(args.question, args.persist_dir, use_llm=not args.no_llm))


if __name__ == "__main__":
    main()
