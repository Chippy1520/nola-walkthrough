"""Expand the same seed pool using CLIP confidence; matched control for DINO retrieval."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '_shared'))
from runner import run

if __name__ == '__main__':
    run('anchors', 'clip_expansion')
