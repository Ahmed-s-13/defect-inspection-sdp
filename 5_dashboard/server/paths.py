"""Where things are. Importing this puts the repo's main-model folder on sys.path (read-only use)."""
import sys
from pathlib import Path

DASH = Path(__file__).resolve().parents[1]
REPO = DASH.parent                                  # the Project Code repo
REPO_MAIN = REPO / "1_main_model_resnet18"
SAMPLES = REPO_MAIN / "samples"
EXTRA_SAMPLES = DASH / "samples"  # no-defect test crops of d1_gc7 (scripts/no_defect_samples.py)
CACHE = DASH / "cache"
WEB_DIST = DASH / "web" / "dist"

if str(REPO_MAIN) not in sys.path:
    sys.path.insert(0, str(REPO_MAIN))
