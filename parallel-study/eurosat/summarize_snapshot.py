"""Summarize matched validation milestones from a Colab report snapshot."""
import argparse
import csv
import json
import zipfile
from pathlib import Path


def summarize(archive, output):
    seen = {}
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        config = json.loads(z.read('config.json'))
        for name in ('baseline_inherited_events.jsonl', 'events.jsonl'):
            for line in z.read(name).splitlines():
                rec = json.loads(line)
                if rec['stage'] not in ('teacher_done', 'epoch_done'):
                    continue
                key = rec['stage'], rec.get('epoch', 0), rec['arm']
                seen[key] = rec
    rows = []
    for stage, epoch in sorted({(s, e) for s, e, _ in seen}):
        a = seen.get((stage, epoch, 'original'))
        b = seen.get((stage, epoch, 'templated075'))
        if a is None or b is None:
            continue
        n = config['val']
        key = 'dino_val_correct' if stage == 'teacher_done' else 'val_correct'
        if stage == 'epoch_done':
            assert a['updates'] == b['updates'] and a['n_train'] == b['n_train']
            assert a['val_n'] == b['val_n'] == n
        rows.append(dict(split='validation', stage=stage, epoch=epoch or '', n=n,
                         original_correct=a[key], templated_correct=b[key],
                         original_pct=100*a[key]/n, templated_pct=100*b[key]/n,
                         delta_pp=100*(b[key]-a[key])/n))
    if not rows:
        raise ValueError('No matched milestones')
    with Path(output).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot')
    p.add_argument('output_csv')
    a = p.parse_args()
    for row in summarize(a.snapshot, a.output_csv):
        print(row)
