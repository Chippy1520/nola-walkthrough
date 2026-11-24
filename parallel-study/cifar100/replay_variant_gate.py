"""Replay the fixed CIFAR text clusters on cached validation features with a new selection gate."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent
REF = ROOT/'frozen_hybrid_075'
GATE = float(sys.argv[1]); assert 0 < GATE <= 1
TAU = .05
OUT = ROOT/f'frozen_hybrid_gate_{GATE:.2f}'


def table(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


with np.load(REF/'text_bank.npz') as b, np.load(REF/'validation_features.npz') as v, np.load(REF/'validation_scores.npz') as ref, np.load(ROOT/'audit_manifest.npz') as m:
    x, ids = v['features'], v['row_ids']
    assert np.array_equal(ids,ref['row_ids']) and np.array_equal(ids,m['validation_indices'])
    means, vectors = b['means'],b['wrapped_description_vectors']
    y,fine=m['train_coarse'][ids],m['train_fine'][ids]
    coarse_names,fine_names=m['coarse_names'],m['fine_names']
    expected = ref['same_prompt_mean']
source=json.loads((REF/'prompt_bank.json').read_text())
keys=source['class_order']; assert list(coarse_names)==keys
chunks=np.split(vectors,np.cumsum([len(source['descriptions'][k]) for k in keys])[:-1]); assert len(chunks)==20
candidates=table(REF/'candidate_audit.csv'); assert len(candidates)==27
base=x@means.T
np.testing.assert_allclose(base,expected,atol=3e-7)
assert np.array_equal(base.argmax(1),expected.argmax(1))
hybrid=base.copy()
for i,vv in enumerate(chunks):
    p,w=[means[i]],[len(vv)]
    for r in (r for r in candidates if int(r['coarse_id'])==i):
        idx=json.loads(r['description_ids']); raw=vv[idx].mean(0)
        sim=float(raw.astype('float64')@means[i].astype('float64')/np.linalg.norm(raw)/np.linalg.norm(means[i]))
        np.testing.assert_allclose(sim,float(r['cosine_to_mean']),atol=2e-7)
        r['retained']=sim<GATE
        if r['retained']: p.append(raw);w.append(len(idx))
    if len(p)>1:
        z=x@np.stack(p).T/TAU
        hybrid[:,i]=TAU*(logsumexp(z+np.log(w),axis=1)-np.log(sum(w)))
a,b=base.argmax(1),hybrid.argmax(1); assert len(y)==5000

def stats(mask):
    u,v=(a==y)[mask],(b==y)[mask]; n=len(u)
    fix,loss=int((~u&v).sum()),int((u&~v).sum()); c0,c1=int(u.sum()),int(v.sum())
    assert c1-c0==fix-loss
    return dict(n=n,mean_correct=c0,hybrid_correct=c1,mean_pct=100*c0/n,hybrid_pct=100*c1/n,delta_pp=100*(c1-c0)/n,fixed=fix,lost=loss,prediction_changes=int((a[mask]!=b[mask]).sum()))

OUT.mkdir(exist_ok=True)
retained=[r for r in candidates if r['retained']]
s=dict(stage='frozen CLIP ViT-B/32',split='validation',test_scored=False,selection_gate=GATE,clustering_cutoff=.75,tau=TAU,n_candidates=len(candidates),n_retained=len(retained),retained=[dict(class_name=r['coarse_name'],cluster_id=int(r['cluster_id']),cosine_to_mean=float(r['cosine_to_mean'])) for r in retained],**stats(np.ones(len(y),bool)))
write(OUT/'candidate_audit.csv',candidates)
write(OUT/'predictions.csv',[dict(row_id=int(j),fine_id=int(fi),truth_coarse_id=int(yi),same_prompt_mean=int(ai),hybrid=int(bi)) for j,fi,yi,ai,bi in zip(ids,fine,y,a,b)])
write(OUT/'by_coarse.csv',[dict(class_name=k,**stats(y==i)) for i,k in enumerate(coarse_names)])
write(OUT/'by_fine_subtype.csv',[dict(subtype=k,**stats(fine==i)) for i,k in enumerate(fine_names)])
(OUT/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(s,indent=2))
