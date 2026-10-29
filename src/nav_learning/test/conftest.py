"""Load the project's CSV logging guard before importing SAC in tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nav_learning  # noqa: F401,E402
