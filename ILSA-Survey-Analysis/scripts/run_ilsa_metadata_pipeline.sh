#!/usr/bin/env bash
# Full ILSAArticleMetadata local pipeline: resanitize → Excel → taxonomy → git push.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
LOG="${ROOT}/outputs/ilsa_metadata_pipeline.log"
DO_GIT=1
GIT_ONLY=0

usage() {
  cat <<'EOF'
Usage: run_ilsa_metadata_pipeline.sh [--no-git] [--git-only]

  --no-git    Run pipeline steps only; skip git add/commit/push
  --git-only  Stage, commit (if changes), and push without re-running pipeline
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-git)  DO_GIT=0; shift ;;
    --git-only) GIT_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"
}

run_step() {
  log "STEP: $*"
  "$@"
}

if [[ "$GIT_ONLY" -eq 0 ]]; then
  : > "$LOG"
  log "=== ILSA metadata pipeline start ==="
  log "Root: $ROOT"

  run_step "$PYTHON" scripts/resanitize_json_outputs.py --json-dir outputs --recursive
  run_step "$PYTHON" scripts/find_missing_main_findings.py --json-dir outputs --recursive
  run_step "$PYTHON" scripts/build_tabular_dataset.py
  run_step "$PYTHON" scripts/build_structured_meta_analysis.py
  run_step "$PYTHON" scripts/build_canonical_taxonomy.py
  run_step "$PYTHON" scripts/build_semantic_knowledge_base_v2.py --version v4
  run_step "$PYTHON" scripts/generate_academic_synthesis.py
  run_step "$PYTHON" -m pytest \
    tests/red_team_confounders_static.py \
    tests/test_doi_extraction.py \
    tests/test_main_findings_migration.py \
    -q --tb=short

  log "=== Pipeline steps complete ==="
fi

if [[ "$DO_GIT" -eq 1 ]]; then
  log "=== Git stage / commit / push ==="
  export GIT_DIR="${GIT_DIR:-$ROOT/.git}"
  export GIT_WORK_TREE="${GIT_WORK_TREE:-$ROOT}"
  GIT="${GIT:-/usr/bin/git}"

  bash "$ROOT/scripts/_batch_git_add_outputs.sh" "$LOG"

  # Code + derived root artifacts
  "$GIT" add \
    src/ scripts/ tests/ prompts/ ilsa_pipeline/ \
    outputs/*.xlsx outputs/*.csv outputs/*.json outputs/*.md outputs/*.txt \
    .cursor/skills/ \
    2>/dev/null || true

  if "$GIT" diff --cached --quiet; then
    log "Nothing to commit (working tree clean after stage)."
  else
    "$GIT" commit -m "$(cat <<'EOF'
Rebuild ILSAArticleMetadata pipeline outputs after resanitize and aggregation.

Refreshes JSON sanitization, Excel meta-analysis workbooks, taxonomy, and synthesis artifacts.
EOF
)"
    log "Committed: $("$GIT" rev-parse --short HEAD)"
  fi

  "$GIT" push origin HEAD
  log "Pushed to origin."
fi

log "=== Done ==="
