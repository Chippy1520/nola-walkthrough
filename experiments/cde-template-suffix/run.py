"""Place each class template before each description before encoding."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '_shared'))
from runner import run

if __name__ == '__main__':
    run('cde', 'template_suffix')
