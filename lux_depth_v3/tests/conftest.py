from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable so `import lux_depth_v3...` works in tests.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
