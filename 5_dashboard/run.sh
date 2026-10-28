#!/usr/bin/env bash
# Start the dashboard: builds the web app when it is missing or older than web/ (e.g. after a git pull), then serves
# everything on http://127.0.0.1:8765
#   ./run.sh                     REBUILD=1 ./run.sh (force a rebuild)
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
stale() { [[ ! -f "$HERE/web/dist/index.html" ]] || [[ -n "$(cd "$HERE/web" && find src public index.html package.json \
  -newer dist/index.html -print -quit 2>/dev/null)" ]]; }
if [[ -n "$REBUILD" ]] || stale; then
  command -v npm >/dev/null || { echo "Node.js (npm) is needed once to build the web app: https://nodejs.org"; exit 1; }
  (cd "$HERE/web" && npm install --no-audit --no-fund && npm run build)
fi
URL="http://127.0.0.1:${PORT:-8765}"
[[ -n "$NO_OPEN" ]] || (sleep 3 && open "$URL" 2>/dev/null || xdg-open "$URL" 2>/dev/null || true) &
cd "$HERE" && exec "$PY" -m uvicorn server.app:app --host 127.0.0.1 --port ${PORT:-8765}
