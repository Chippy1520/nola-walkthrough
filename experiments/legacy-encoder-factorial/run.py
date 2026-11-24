"""Entry point for the historical protocol; arguments are passed unchanged."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    repo = Path(__file__).resolve().parents[2]
    raise SystemExit(subprocess.call([sys.executable, str(repo / 'siglip-replacement/factorial.py'), *sys.argv[1:]], cwd=repo))
