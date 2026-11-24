"""Audit complete-linkage prompt merges against the class-local SVD projection."""
import csv
import io
import itertools
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
import torch

root = Path(__file__).resolve().parent
with zipfile.ZipFile(root/'NoLA-EuroSAT-Prompt-Bank-v1.zip') as z:
    X = torch.load(io.BytesIO(z.read('description_embeddings.pt')), weights_only=True).double().numpy()
    P = torch.load(io.BytesIO(z.read('prototype_embeddings.pt')), weights_only=True).double().numpy()
    rows = list(csv.DictReader(io.StringIO(z.read('description_index.csv').decode())))
    saved = json.loads(z.read('prototypes.json'))
cutoff = saved['min_cos']
distance_cutoff = np.sqrt(2*(1-cutoff))
by_class = list(dict.fromkeys(r['parent'] for r in rows))
all_pairs, all_groups, summaries = [], [], []
for parent in by_class:
    idx = [i for i,r in enumerate(rows) if r['parent']==parent]
    desc = [rows[i] for i in idx]
    x = X[idx]
    cos = np.clip(x @ x.T, -1, 1)
    dist = 1-cos
    np.fill_diagonal(dist, 0)
    groups = fcluster(linkage(squareform(dist), method='complete'), 1-cutoff, criterion='distance')
    centered = x-x.mean(0)
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    energy = np.cumsum(s*s)/np.sum(s*s)
    rank = int(np.searchsorted(energy,.9))+1
    coords = centered @ vh[:rank].T
    original = {frozenset(p['members']):p for p in saved['clusters'] if p['parent']==parent}
    np.testing.assert_allclose(np.linalg.norm(x,axis=1),1,atol=1e-6)
    member_sets = set()
    for g in np.unique(groups):
        js = np.flatnonzero(groups==g)
        members = frozenset(desc[j]['id'] for j in js)
        member_sets.add(members)
        minimum = min((cos[j,k] for j,k in itertools.combinations(js,2)),default=1.)
        assert minimum >= cutoff-1e-6
        entry = original[members]
        centroid = x[js].mean(0); centroid /= np.linalg.norm(centroid)
        np.testing.assert_allclose(centroid,P[entry['prototype_id']],atol=1e-6)
        all_groups.append(dict(parent=parent,prototype_id=entry['prototype_id'],n=len(js),min_cos=minimum,descriptions=[desc[j]['description'] for j in js]))
    assert member_sets==set(original)
    pairs = []
    for j,k in itertools.combinations(range(len(desc)),2):
        projected = float(np.linalg.norm(coords[j]-coords[k]))
        full = float(np.linalg.norm(x[j]-x[k]))
        assert projected <= full+1e-9
        pairs.append(dict(parent=parent,first=desc[j]['description'],second=desc[k]['description'],svd_distance=projected,full_distance=full,cosine=float(cos[j,k]),svd_only=bool(projected<=distance_cutoff and cos[j,k]<cutoff),merged=bool(groups[j]==groups[k])))
    pairs.sort(key=lambda p:p['svd_distance'])
    all_pairs.extend(pairs)
    summaries.append((parent,len(desc),len(member_sets),sum(p['svd_only'] for p in pairs),rank))

with (root/'svd-merge-pairs.csv').open('w',newline='') as f:
    w = csv.DictWriter(f,fieldnames=list(all_pairs[0])); w.writeheader(); w.writerows(all_pairs)
report = ['# EuroSAT prompt merges: all ten classes','',f'Complete linkage on full-space cosine ≥ {cutoff:.2f}; class-local SVD at 90% centered energy ranks pairs and flags disagreements. Projected-distance screening uses {distance_cutoff:.6f}, the full-space cutoff expressed as distance for unit vectors.','', '| Class | Input | Prototypes | SVD-only close pairs | SVD directions |','|---|---:|---:|---:|---:|']
for p,n,k,flags,rank in summaries: report.append(f'| {p} | {n} | {k} | {flags} | {rank} |')
report.extend(['', 'SVD-only close pairs pass the projected cutoff but not the full-space cosine cutoff. They were not merged.', '', '## Every prototype and its source descriptions'])
for parent in by_class:
    report.extend(['',f'### {parent}'])
    for g in sorted((g for g in all_groups if g['parent']==parent),key=lambda g:(-g['n'],g['prototype_id'])):
        suffix = f" (merged; min cosine {g['min_cos']:.3f})" if g['n']>1 else ''
        report.append(f"- **P{g['prototype_id']:02d}**{suffix}: " + ' / '.join(g['descriptions']))
report.extend(['','## Closest SVD-only pairs (not merged)'])
for p in sorted((p for p in all_pairs if p['svd_only']),key=lambda p:p['svd_distance'])[:10]:
    report.append(f"- **{p['parent']}**: {p['first']} ↔ {p['second']} (SVD distance {p['svd_distance']:.3f}; full cosine {p['cosine']:.3f}).")
(root/'svd-merge-results.md').write_text('\n'.join(report)+'\n')
assert len(all_pairs)==450 and len(all_groups)==72 and sum(g['n'] for g in all_groups)==100
print('\n'.join(report[4:16]))
print('Merged groups:',sum(g['n']>1 for g in all_groups),'| SVD-only pairs rejected:',sum(p['svd_only'] for p in all_pairs))
for g in all_groups:
    if g['n']>1: print(g['parent'],f"{g['min_cos']:.3f}",' / '.join(g['descriptions']))
