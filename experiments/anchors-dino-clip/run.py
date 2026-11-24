"""Reweight the same DINO graph with CLIP posterior overlap."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '_shared'))
from runner import run

if __name__ == '__main__':
    run('anchors', 'dino_clip')
