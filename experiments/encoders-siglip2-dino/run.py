"""Independent SigLIP 2 nomination with original DINO alignment features."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '_shared'))
from runner import run

if __name__ == '__main__':
    run('encoders', 'siglip2_dino')
