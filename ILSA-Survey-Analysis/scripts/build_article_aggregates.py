#!/usr/bin/env python3
"""Build master Parquet + SQLite from outputs/articles/json/."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ilsa_pipeline"))

from ilsa_pipeline.utils.storage import build_master_parquet, build_sqlite_database

JSON_DIR = PROJECT_ROOT / "outputs" / "articles" / "json"
PARQUET_PATH = PROJECT_ROOT / "outputs" / "articles" / "ilsa_master.parquet"
SQLITE_PATH = PROJECT_ROOT / "outputs" / "articles" / "ilsa_knowledge_base.db"


def main() -> None:
    df = build_master_parquet(JSON_DIR, PARQUET_PATH)
    print(f"Parquet: {len(df)} rows → {PARQUET_PATH}")
    if len(df) > 0:
        build_sqlite_database(PARQUET_PATH, SQLITE_PATH)
        print(f"SQLite: {SQLITE_PATH}")


if __name__ == "__main__":
    main()
