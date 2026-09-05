#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [[ "${1:-}" == "--help" ]]; then
  echo 'Usage: ./scripts/evaluate-and-push.sh'
  echo 'Requires committed changes and AI_CREDITS in .env. Evaluates 100 random questions,'
  echo 'commits generated results and dashboard, then pushes to origin default branch.'
  echo 'Running this command makes paid AI Credits calls. --help makes no calls.'
  exit 0
fi
if [[ $# -ne 0 ]]; then echo 'Only --help is supported.' >&2; exit 2; fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'Commit your code and README edits first; the working tree must be clean.' >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then python3.12 -m venv .venv; fi
source .venv/bin/activate
python - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit('Python 3.12 or newer is required; recreate .venv with Python 3.12.')
PY
python scripts/evaluate_and_push.py
