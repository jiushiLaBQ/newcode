#!/usr/bin/env bash
# Push this repo to GitHub. Token comes from argv or GITHUB_TOKEN env, never stored on disk.
# Usage:  bash scripts/push_github.sh <token>
#     or  GITHUB_TOKEN=<token> bash scripts/push_github.sh
set -euo pipefail

REMOTE_URL="https://github.com/jiushiLaBQ/newcode.git"
GITHUB_TOKEN="${1:-${GITHUB_TOKEN:-}}"
: "${GITHUB_TOKEN:?usage: bash scripts/push_github.sh <token>  (or set GITHUB_TOKEN)}"

cd "$(dirname "$0")/.."

if [ ! -d .git ]; then
  git init -b main
fi

git config user.name  >/dev/null 2>&1 || git config user.name "LaBQ"
git config user.email >/dev/null 2>&1 || git config user.email "LaBQ@users.noreply.github.com"

git add -A
if ! git diff --cached --quiet; then
  git commit -m "Initial commit: ABSA competition pipeline"
fi

git remote add origin "$REMOTE_URL" 2>/dev/null || git remote set-url origin "$REMOTE_URL"

# Token travels in the URL for this command only; remote stays clean afterwards.
AUTH_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/jiushiLaBQ/newcode.git"
if ! git push "$AUTH_URL" main; then
  echo "Push rejected — remote has commits. Rebasing onto remote main..."
  git pull --rebase "$AUTH_URL" main --allow-unrelated-histories
  git push "$AUTH_URL" main
fi

git remote set-url origin "$REMOTE_URL"
echo "Done: $REMOTE_URL"
