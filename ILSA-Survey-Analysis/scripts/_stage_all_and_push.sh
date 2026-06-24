#!/usr/bin/env bash
set -euo pipefail
export GIT_DIR="${GIT_DIR:-/Users/mrved/Desktop/ILSA_LLMs/.git}"
export GIT_WORK_TREE="${GIT_WORK_TREE:-/Users/mrved/Desktop/ILSA_LLMs}"
export GIT_OPTIONAL_LOCKS=1
GIT=/usr/bin/git
ROOT=/Users/mrved/Desktop/ILSA_LLMs
LOG="$ROOT/outputs/_stage_all_push.log"
BATCH=25
AUTHOR_NAME="Merve Dede"
AUTHOR_EMAIL="140520079+dedemerve@users.noreply.github.com"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

cd "$ROOT"
log "=== stage all untracked (batch=$BATCH) ==="

LIST=$(mktemp)
comm -23 \
  <(find . -type f \
      ! -path './.git/*' \
      ! -path './.venv/*' \
      ! -path './.venv-rag/*' \
      ! -path './venv/*' \
      ! -path './env/*' \
      ! -path './outputs/chroma_ilsa_synthesis/*' \
      ! -path './outputs/logs/*' \
      ! -path './logs/*' \
      ! -name '.DS_Store' \
      ! -name '.env' \
      ! -name '.env.local' \
    | sed 's|^\./||' | sort) \
  <($GIT ls-files | sort) >"$LIST"
TOTAL=$(wc -l <"$LIST" | tr -d ' ')
log "untracked files: $TOTAL"
batch=()
n=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  batch+=("$f")
  if ((${#batch[@]} >= BATCH)); then
    if $GIT add "${batch[@]}"; then
      n=$((n + ${#batch[@]}))
      log "staged $n / $TOTAL"
    else
      log "WARN batch failed at $n — retrying one-by-one"
      for one in "${batch[@]}"; do
        $GIT add "$one" 2>/dev/null || log "FAIL $one"
        n=$((n + 1))
      done
    fi
    batch=()
    sleep 1
  fi
done <"$LIST"
if ((${#batch[@]} > 0)); then
  $GIT add "${batch[@]}" || true
  n=$((n + ${#batch[@]}))
fi
rm -f "$LIST"
log "staging done (~$n files)"

log "=== stage modifications ==="
$GIT add -u :/
log "modifications staged"

STAGED=$($GIT diff --cached --name-only 2>/dev/null | wc -l | tr -d ' ')
log "cached entries: $STAGED"

if [ "$STAGED" -eq 0 ]; then
  log "nothing to commit"
  exit 0
fi

log "=== commit as $AUTHOR_NAME ==="
$GIT -c user.name="$AUTHOR_NAME" -c user.email="$AUTHOR_EMAIL" commit -m "$(cat <<'EOF'
Add full ILSA extraction corpus and project outputs

Track IEA, OECD, Scopus, Web of Science, and survey article JSON under
outputs/, reference datasets and synthesis artifacts, and remaining
project source files.
EOF
)"

log "=== push origin main ==="
$GIT push origin main
log "=== done ==="
$GIT log -1 --format='%H %an <%ae> %s'
