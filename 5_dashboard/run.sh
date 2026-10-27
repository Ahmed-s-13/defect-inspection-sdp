#!/usr/bin/env bash
# Start the dashboard: builds the web app the first time, then serves everything on http://127.0.0.1:8765
#   ./run.sh                     REBUILD=1 ./run.sh (after changing web/src)
#   PYTHON=/path/to/python ./run.sh   (any Python with the repo's requirements.txt; default: the repo's .venv)
#   PORT=8766 ./run.sh  another port      NO_OPEN=1 ./run.sh  don't open the browser
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=${PYTHON:-}
if [[ -z "$PY" ]]; then
  if [[ -x "$HERE/../.venv/bin/python" ]]; then PY="$HERE/../.venv/bin/python"; else PY=python3; fi
fi
"$PY" -c "import fastapi, uvicorn, torch" 2>/dev/null || {
  echo "$PY is missing packages: run  pip install -r requirements.txt  in the repo root (see 5_dashboard/README.md)"; exit 1; }
if [[ ! -d "$HERE/web/dist" || -n "$REBUILD" ]]; then
  command -v npm >/dev/null || { echo "Node.js (npm) is needed once to build the web app: https://nodejs.org"; exit 1; }
  (cd "$HERE/web" && npm install --no-audit --no-fund && npm run build)
fi
URL="http://127.0.0.1:${PORT:-8765}"
[[ -n "$NO_OPEN" ]] || (sleep 3 && open "$URL" 2>/dev/null || xdg-open "$URL" 2>/dev/null || true) &
cd "$HERE" && exec "$PY" -m uvicorn server.app:app --host 127.0.0.1 --port ${PORT:-8765}
