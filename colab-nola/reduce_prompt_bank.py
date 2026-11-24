"""Build and verify smaller same-prompt EuroSAT banks using complete linkage."""
import csv
import io
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
import torch

root = Path(__file__).resolve().parent
with zipfile.ZipFile(root/'NoLA-EuroSAT-Prompt-Bank-v1.zip') as z:
    x = torch.load(io.BytesIO(z.read('description_embeddings.pt')), weights_only=True).double().numpy()
    rows = list(csv.DictReader(io.StringIO(z.read('description_index.csv').decode())))
    baseline = json.loads(z.read('prototypes.json'))
parents = list(dict.fromkeys(r['parent'] for r in rows))
thresholds = (.88, .85, .80)
banks = {t: [] for t in thresholds}
vectors = {t: [] for t in thresholds}
report = ['# Smaller EuroSAT prompt banks', '', 'Same 100 descriptions and CLIP vectors. Complete linkage in the original normalized CLIP space; SVD does not override the cosine cutoff. The 0.88 bank is preserved.', '', '| Class | 0.88 baseline | 0.85 moderate | 0.80 aggressive |', '|---|---:|---:|---:|']
changes = []
for parent in parents:
    indices = [i for i,r in enumerate(rows) if r['parent']==parent]
    xx = x[indices]; cos = np.clip(xx @ xx.T, -1, 1)
    d = 1-cos; np.fill_diagonal(d,0)
    tree = linkage(squareform(d),method='complete')
    previous = {frozenset(p['members']) for p in baseline['clusters'] if p['parent']==parent}
    counts=[]
    for t in thresholds:
        labels = fcluster(tree,1-t,criterion='distance')
        groups = set()
        for g in sorted(set(labels)):
            ii = np.flatnonzero(labels==g)
            source = [rows[indices[i]] for i in ii]
            members = [r['id'] for r in source]
            groups.add(frozenset(members))
            floor = min((float(cos[i,j]) for i,j in itertools.combinations(ii,2)),default=1.)
            assert floor>=t-1e-6
            u=xx[ii].mean(0); u/=np.linalg.norm(u)
            entry=dict(prototype_id=len(banks[t]),parent=parent,members=members,descriptions=[r['description'] for r in source],min_cos=floor)
            banks[t].append(entry); vectors[t].append(u)
            if t!=.88 and frozenset(members) not in previous:
                changes.append((t,entry))
        if t==.88: assert groups==previous
        else: assert all(sum(p<=g for g in groups)==1 for p in previous)
        previous=groups
        counts.append(len(groups))
    report.append(f'| {parent} | '+ ' | '.join(map(str,counts))+' |')
report.append('| Total | '+' | '.join(str(len(banks[t])) for t in thresholds)+' |')
print('\n'.join(report))
for t in thresholds[1:]:
    report.extend(['',f'## New groups at {t:.2f} relative to previous cutoff'])
    new=[p for threshold,p in changes if threshold==t]
    weakest=sorted(new,key=lambda p:p['min_cos'])[:2]
    for p in weakest: print(f'{t:.2f}',p['parent'],f"min={p['min_cos']:.3f}", ' / '.join(p['descriptions']))
    for p in new: report.append(f"- {p['parent']} (minimum cosine {p['min_cos']:.3f}): " + ' / '.join(p['descriptions']))
report.extend(['','No image predictions or training were run. These configurations are candidates, not tuned thresholds.'])
report_text='\n'.join(report)+'\n'
(root/'smaller-prompt-banks.md').write_text(report_text)
archive=root/'NoLA-EuroSAT-Smaller-Prompt-Banks.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for t in thresholds[1:]:
        p=banks[t]; v=torch.from_numpy(np.stack(vectors[t]).astype('float32'))
        manifest=dict(min_cos=t,templates=baseline['templates'],clip_sha256=baseline['clip_sha256'],clusters=p)
        z.writestr(f'cos_{t:.2f}/prototypes.json',json.dumps(manifest,indent=2)+'\n')
        b=io.BytesIO(); torch.save(v,b); z.writestr(f'cos_{t:.2f}/prototype_embeddings.pt',b.getvalue())
    z.writestr('smaller-prompt-banks.md',report_text)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for t in thresholds[1:]:
        m=json.loads(z.read(f'cos_{t:.2f}/prototypes.json'))['clusters']
        v=torch.load(io.BytesIO(z.read(f'cos_{t:.2f}/prototype_embeddings.pt')),weights_only=True)
        assert v.shape==(len(m),512) and torch.allclose(v.norm(dim=1),torch.ones(len(m)),atol=1e-6)
        assert sorted(i for p in m for i in p['members'])==sorted(r['id'] for r in rows)
print('Verified:',archive)
