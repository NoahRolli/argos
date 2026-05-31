#!/usr/bin/env bash
# pre-commit-hook.sh — ARGOS. Bash 3.2 compatible (macOS has no `mapfile`).
set -euo pipefail

RED=$'\033[0;31m'; YEL=$'\033[0;33m'; GRN=$'\033[0;32m'; NC=$'\033[0m'
BLOCK=0
fail() { echo "${RED}  BLOCK ${NC} $1"; BLOCK=1; }
warn() { echo "${YEL}  WARN  ${NC} $1"; }

STAGED="$(git diff --cached --name-only --diff-filter=ACM)"
[ -z "$STAGED" ] && exit 0
echo "Scanning staged files..."

while IFS= read -r f; do
  [ -z "$f" ] && continue
  base="$(basename "$f")"
  case "$base" in
    .env.example|.env.sample) : ;;
    .env|.env.*) fail "env file staged: $f" ;;
  esac
  case "$f" in
    *.key|*.pem|*.crt) fail "key/cert staged: $f" ;;
    *.db|*.sqlite|*.sqlite3) fail "database staged: $f" ;;
    local_storage/*|.keys/*|backups/*|secrets/*) fail "sensitive dir staged: $f" ;;
  esac
  case "$f" in
    assets/*|tests/fixtures/*) : ;;
    *.jpg|*.jpeg|*.png|*.bmp|*.gif|*.webp|*.mp4|*.mov|*.avi|*.mkv) fail "media file staged: $f" ;;
    *.npy|*.npz|*.pkl|*.pickle|*.faiss|*.index) fail "embedding/data blob staged: $f" ;;
    *.pt|*.pth|*.onnx|*.task|*.tflite|*.mlpackage|*.mlmodel|*.engine) fail "model weight staged: $f" ;;
  esac
done <<< "$STAGED"

scan_block() {
  local re="$1" label="$2" flags="${3:-}" f
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    if git show ":$f" 2>/dev/null | grep -E $flags -q "$re"; then fail "$label in: $f"; fi
  done <<< "$STAGED"
}
scan_block 'AKIA[0-9A-Z]{16}' "AWS access key"
scan_block 'ghp_[A-Za-z0-9]{36}' "GitHub PAT"
scan_block 'github_pat_[A-Za-z0-9_]{82}' "GitHub fine-grained PAT"
scan_block 'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+' "JWT token"
scan_block 'BEGIN (RSA |EC |OPENSSH |)PRIVATE KEY' "private key block"
scan_block '(api[_-]?key|secret[_-]?key)[[:space:]]*[:=][[:space:]]*["'"'"'][A-Za-z0-9_-]{20,}' "generic API/secret key" "-i"

scan_warn() {
  local re="$1" label="$2" f
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    if git show ":$f" 2>/dev/null | grep -E -q "$re"; then warn "$label in: $f"; fi
  done <<< "$STAGED"
}
scan_warn '/Users/[A-Za-z0-9._-]+/' "macOS user path"
scan_warn '/(mnt|srv)/[A-Za-z0-9._-]+/' "server-specific path"
scan_warn '(192\.168\.|10\.)[0-9]{1,3}\.[0-9]{1,3}' "private IP"
scan_warn '(password|passwd|pwd)[[:space:]]*[:=]' "hardcoded password (verify it is a schema field)"

echo "---------------------------------------------"
if [ "$BLOCK" -eq 1 ]; then
  echo "${RED}Commit blocked.${NC} Fix the BLOCK items, or use --no-verify if truly a false positive."
  exit 1
fi
echo "${GRN}Pre-commit checks passed.${NC}"
exit 0