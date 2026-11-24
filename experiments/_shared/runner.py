"""Execute one NoLA variant, on one or more datasets, without a notebook."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TAGS = ('eurosat', 'oxford_pets', 'cifar10', 'dtd')


def export_class_results(out_dir, names, write_csv):
    with (out_dir / 'predictions.csv').open(newline='') as f:
        counts = Counter((r['split'], r['stage'], r['arm'], int(r['truth']), int(r['prediction']))
                         for r in csv.DictReader(f))
    totals = Counter()
    for (split, stage, arm, truth, _), count in counts.items():
        totals[split, stage, arm, truth] += count
    write_csv(out_dir / 'per_class.csv', [dict(split=s, stage=t, arm=a, class_id=y, class_name=names[y],
              correct=counts[s,t,a,y,y], n=n, recall_pct=100*counts[s,t,a,y,y]/n)
              for (s,t,a,y), n in sorted(totals.items())])
    write_csv(out_dir / 'confusions.csv', [dict(split=s, stage=t, arm=a, truth=y, prediction=p, count=n)
              for (s,t,a,y,p), n in sorted(counts.items())])


def run(family, arm):
    parser = argparse.ArgumentParser(description=f'NoLA {family}: {arm}')
    parser.add_argument('--datasets', nargs='+', choices=TAGS, default=list(TAGS))
    parser.add_argument('--root', type=Path, default=REPO)
    parser.add_argument('--output', type=Path, default=HERE / 'runs')
    parser.add_argument('--micro', type=int, choices=[2, 4, 8, 16, 32], default=32)
    args = parser.parse_args()
    import torch
    from torch.nn.functional import normalize
    helpers = REPO / 'final-evaluation'
    sys.path.insert(0, str(helpers))
    from setup_inputs import setup
    ROOT = args.root.resolve()
    BASE = args.output.resolve()
    SEED = 7777
    HEAD_EPOCHS = PROMPT_EPOCHS = 50
    BATCH, MICRO, EXTRACT_BATCH = 512, args.micro, args.micro
    RUN_NAME = f'{family}-{arm}-micro{MICRO}'
    archive = next((p for p in (ROOT / 'source.zip', ROOT / 'final-evaluation/source.zip') if p.is_file()), None)
    if not (ROOT / 'upstream').is_dir() and archive is None:
        parser.error('Missing pinned sources; run python3 scripts/fetch_sources.py first')
    ROOT, DATA = setup(ROOT, ROOT / 'data', [], source=archive)
    import nola_eval
    import experiments as ex
    from class_geometry import export_geometry
    if MICRO < 32:
        nola_eval.loader = lambda rows, source, transform, batch=32, shuffle=False: torch.utils.data.DataLoader(
            nola_eval.Images(rows, source, transform), batch_size=min(batch, MICRO), shuffle=shuffle, num_workers=0)
    if not torch.cuda.is_available():
        parser.error('GPU required: install CUDA-enabled PyTorch and select a CUDA device')
    stages = ['stages', 'cde-base']
    stages += ['text-change'] if family == 'cde' and arm != 'release' else []
    if family != 'cde':
        stages += ['full-prefix', 'dino-prefix']
        stages += ['anchor-graph', 'anchor-retrieval'] if family == 'anchors' else ['encoder-change']
    for TAG in args.datasets:
        out_dir = BASE / 'runs' / '03-experiments' / RUN_NAME / TAG
        marker = out_dir / 'complete.json'
        if marker.exists():
            print(TAG, 'already complete', flush=True)
            continue
        setup(ROOT, DATA, [TAG], source=ROOT / 'source.zip')
        ns = dict(globals(), **locals())
        for stage in stages:
            print(TAG, stage, flush=True)
            path = HERE / 'stages' / (stage.replace('-', '_') + '.py')
            exec(compile(path.read_text(), str(path), 'exec'), ns)
        if family == 'cde':
            data = ns['rows']['val']
            pred = (ns['release_val']['val'] if arm == 'release' else ex.score(ns['clip_val']['val'], ns['text_banks'][arm])).argmax(1)
            truth = ns['truth']['val']
            key = dict(**ns['key'], arm=arm, split='val', stage='cde')
            correct, n = nola_eval.top1_counts(pred, truth)
            nola_eval.write_csv(out_dir / 'metrics.csv', [dict(**key, correct=correct, n=n, top1_pct=100*correct/n)])
            nola_eval.write_csv(out_dir / 'predictions.csv', [dict(**key, image_id=item[0], truth=y, prediction=p)
                                for item, y, p in zip(data, truth.tolist(), pred.tolist())])
            export_geometry(out_dir, ns['key'], 'clip', ns['clip_val']['val'], truth, data, ns['names'])
            epoch = 0
        else:
            if family == 'anchors':
                if arm not in ns['anchor_arms']:
                    status = dict(dataset=TAG, arm=arm, status='not_applicable', reason='No eligible additional anchors')
                    (out_dir / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
                    print(TAG, status['reason'], flush=True)
                    continue
                anchors, features, logits = ns['anchor_arms'][arm], ns['dino_features'], ns['release_logits']
            else:
                nomination, vision = arm.split('_')
                anchors = ns['matched_anchors'][nomination]
                features = ns['dino_features' if vision == 'dino' else 'v3_features']
                logits = ns['nomination'][nomination]
            aligned = ns['train_dino_head'](features, anchors, len(ns['names']))
            model, epoch, curve = ns['fine_tune_clip'](ns['load_clip'](), ns['release_cde'], ns['rows'], ns['source'],
                                                       aligned['train'], ns['truth'], out_dir)
            ns['evaluate_full_run'](out_dir, arm, ns['rows'], ns['source'], ns['truth'], ns['names'], logits,
                                    aligned, model, anchors, epoch, curve)
            del model
        export_class_results(out_dir, ns['names'], nola_eval.write_csv)
        marker.write_text(json.dumps(dict(dataset=TAG, arm=arm, epoch=epoch)) + '\n')
        print(TAG, 'complete', out_dir, flush=True)
        ns.clear()
        torch.cuda.empty_cache()
