#!/bin/zsh
set -euo pipefail
cd "${0:A:h}"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/setup_paths.py
.venv/bin/python scripts/check_ready.py
print "Setup complete. Follow README.md to train."
