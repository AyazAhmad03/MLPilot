import sys
from pathlib import Path

# Make the project root importable (tests/ is a sibling of core/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))