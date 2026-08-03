#!/usr/bin/env bash
# Nightly wiki update for ~/knowledge/.
#
# Designed for crontab; intended to run shortly after the bridge's own
# nightly scheduler (default 04:00) finishes, so any newly sanitized KCs
# from that night are picked up before the wiki rebuild.
#
# Workflow:
#   1. Source API keys from /home/dev/gpu-shared/.env (the only file
#      that carries them; the bridge container reads them too).
#   2. Run tools/build_wiki.py --mode incremental against ~/knowledge/.
#   3. If the build changed any tracked files, commit + push to origin.
#
# Crontab line:
#   30 5 * * * /home/dev/omi_whisper_transcribe/tools/cron_wiki_update.sh (bridge の nightly sanitize 完了後)
#
# Logs append to ~/knowledge/.build.log (gitignored).

set -uo pipefail

REPO_ROOT="/home/dev/omi_whisper_transcribe"
KNOWLEDGE_DIR="/home/dev/knowledge"
ENV_FILE="/home/dev/gpu-shared/.env"
LOG="${KNOWLEDGE_DIR}/.build.log"

mkdir -p "$KNOWLEDGE_DIR"
touch "$LOG"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "$(ts) $*" >> "$LOG"; }

log "=== cron_wiki_update.sh start ==="

# --- 1. Load API keys ----------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
    log "ERROR: env file not found at $ENV_FILE; aborting"
    exit 2
fi

set +e
OPENAI_API_KEY=$(grep '^OPENAI_API_KEY=' "$ENV_FILE" | head -n1 | cut -d= -f2-)
GEMINI_API_KEY=$(grep '^GEMINI_API_KEY=' "$ENV_FILE" | head -n1 | cut -d= -f2-)
set -e
export OPENAI_API_KEY GEMINI_API_KEY

if [[ -z "${OPENAI_API_KEY}" || -z "${GEMINI_API_KEY}" ]]; then
    log "ERROR: missing API keys in $ENV_FILE; aborting"
    exit 2
fi

# --- 2. Run incremental build -------------------------------------------
cd "$REPO_ROOT" || { log "ERROR: cannot cd $REPO_ROOT"; exit 2; }

log "build_wiki.py --mode incremental starting"
build_output=$(python3 tools/build_wiki.py --mode incremental 2>&1)
build_rc=$?
echo "$build_output" >> "$LOG"

if [[ $build_rc -ne 0 ]]; then
    log "ERROR: build_wiki.py exited with code $build_rc"
    exit $build_rc
fi
log "build_wiki.py finished OK"

# --- 3. Commit + push if anything changed -------------------------------
cd "$KNOWLEDGE_DIR" || { log "ERROR: cannot cd $KNOWLEDGE_DIR"; exit 2; }

if [[ -z "$(git status --porcelain)" ]]; then
    log "No wiki changes; nothing to commit"
    log "=== cron_wiki_update.sh done (no changes) ==="
    exit 0
fi

log "Wiki changed; committing"
git add -A
if ! git -c commit.gpgsign=false commit -q \
        -m "Wiki incremental update $(date +%Y-%m-%d)" 2>>"$LOG"; then
    log "ERROR: git commit failed"
    exit 3
fi

log "Pushing to origin"
if ! git push origin main 2>>"$LOG"; then
    log "ERROR: git push failed"
    exit 4
fi

log "Push successful"
log "=== cron_wiki_update.sh done (pushed) ==="
