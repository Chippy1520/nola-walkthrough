from pathlib import Path
from zipfile import ZipFile
import subprocess
import sys
import tempfile
import nbformat as nb

out = Path(__file__).resolve().parent
root = out.parent
m = nb.v4.new_markdown_cell
c = nb.v4.new_code_cell

runner = (out/'nola_eval.py').read_text().split('\n', 1)[1]
runner = runner.replace('DATASETS', 'RELEASE_DATASETS')
runner = runner.replace('wrapper_sha256=sha(Path(__file__)),', "runner='inline-evaluator-v3',")

blocks = [
    ('Imports and data loading', 'import csv'),
    ('Frozen encoders and CDE', 'def load_backbones('),
    ('Anchor selection and DINO alignment', 'def select_anchors('),
    ('Visual prompts and training', 'class PromptedCLIP('),
    ('Baseline stage wrappers and exports', 'def frozen_baseline('),
    ('Dataset runner', 'def run_dataset('),
]
definitions = []
for i, (title, marker) in enumerate(blocks):
    start = runner.index(marker)
    end = runner.index(blocks[i+1][1]) if i+1 < len(blocks) else len(runner)
    definitions.extend([m(f'### {title}'), c(runner[start:end].strip())])

cells = [
    m('# 02 · NoLA full-dataset evaluator'),
    m('## Settings and setup'),
    c('''from pathlib import Path
import subprocess
import sys

COLAB = True
DATASETS = ['eurosat', 'oxford_pets', 'cifar10', 'dtd']
SMOKE = False
RUN_ID = 'baseline-smoke-v4' if SMOKE else 'baseline-full-v4'
SEED = 7777
EPOCHS = 1 if SMOKE else 50
HEAD_EPOCHS = 2 if SMOKE else 50
BATCH = 32 if SMOKE else 512
MICRO = 16 if SMOKE else 32

if COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = Path('/content/drive/MyDrive/NoLA-Final-Evaluation')
    ROOT = Path('/content/nola')
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'gdown', 'ftfy', 'wget'], check=True)
else:
    ROOT = Path.home() / 'nola-walkthrough'
    BASE = ROOT / 'final-evaluation'

sys.path.insert(0, str(BASE))
from setup_inputs import setup

ROOT, DATA = setup(ROOT, ROOT/'data', [], source=BASE/'source.zip')'''),
    m('## Evaluator definitions'),
    *definitions,
    m('## Run and export'),
    c('''OUT = BASE / 'runs' / RUN_ID
for dataset in DATASETS:
    if (OUT/dataset/'complete.json').is_file():
        continue
    print('Running:', dataset)
    setup(ROOT, DATA, [dataset], source=BASE/'source.zip')
    run_dataset(dataset, ROOT, DATA, OUT/dataset,
                seed=SEED, epochs=EPOCHS, head_epochs=HEAD_EPOCHS,
                batch=BATCH, micro=MICRO, smoke=SMOKE)

for filename in ('metrics.csv', 'anchor_summary.csv', 'anchors_by_class.csv', 'learning_curve.csv'):
    records = []
    for dataset in DATASETS:
        with (OUT/dataset/filename).open() as file:
            records.extend(csv.DictReader(file))
    write_csv(OUT/filename, records)

reference = [dict(dataset=d, paper_top1_pct=RELEASE_DATASETS[d][2],
                  source='BMVC 2025 Table 1', comparable='not yet verified') for d in DATASETS]
write_csv(OUT/'paper_reference.csv', reference)
print('CSV folder:', OUT)'''),
]
notebook = nb.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name':'python3','language':'python','display_name':'Python 3'}})
nb.write(notebook, out/'02-NoLA-Full-Evaluator.ipynb')

if '--notebook-only' in sys.argv:
    raise SystemExit

with tempfile.TemporaryDirectory() as directory, ZipFile(out/'source.zip', 'w') as bundle:
    for name in ('upstream', 'dino-source'):
        path = Path(directory)/f'{name}.zip'
        subprocess.run(['git', '-C', str(root/name), 'archive', '--format=zip', f'--prefix={name}/',
                        '-o', str(path), 'HEAD'], check=True)
        with ZipFile(path) as source:
            for item in source.infolist():
                if not item.is_dir():
                    bundle.writestr(item, source.read(item))
