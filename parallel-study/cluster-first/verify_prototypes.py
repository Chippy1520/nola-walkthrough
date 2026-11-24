import csv,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
p=Path(sys.argv[1]); s=json.loads((p/'summary.json').read_text());assert not s['test_scored']
def table(f):
    with (p/f).open() as h:return list(csv.DictReader(h))
a=table('anchors.csv');v=table('predictions.csv');tr=table('clusters.csv');g={r['id']:r['cluster'] for r in tr};assert len(g)==len(tr)==s['train_n']
counts={r['class_id']:int(r['budget']) for r in table('budgets.csv')};assert sum(counts.values())==s['anchors']
truth={};hits={}; all_classes=table('per_class.csv'); weights={}
for r in table('prototype_assignments.csv'):
    w=float(r['weight']);assert 0<w<=1.000001
    key=r['cluster'],r['class_id'];weights[key]=weights.get(key,0)+w
assert len(weights)==s['cluster_count']*len(counts) and np.allclose(list(weights.values()),1)
for arm,stats in s['arms'].items():
    ar=[r for r in a if r['arm']==arm]; vr=[r for r in v if r['arm']==arm]; m={r['id']:int(r['truth']) for r in vr};assert len(m)==len(vr)==s['n'] and not set(g).intersection(m)
    if not truth:truth=m
    assert m==truth and len({r['id'] for r in ar})==len(ar)==s['anchors'];assert Counter(r['pseudo'] for r in ar)==Counter({j:q for j,q in counts.items() if q});assert all(g[r['id']]==r['group'] for r in ar)
    hits[arm]={r['id']:int(r['pred']==r['truth']) for r in vr};assert sum(hits[arm].values())==stats['val_correct'];assert sum(r['pseudo']==r['truth'] for r in ar)==stats['anchor_correct']
    for r in (r for r in all_classes if r['arm']==arm):
        j=r['class_id'];assert sum(q['pred']==q['truth']==j for q in vr)==int(r['val_correct']);assert sum(q['pseudo']==q['truth']==j for q in ar)==int(r['correct_anchors'])
if s['stage']=='validation':assert (s['train_n'],s['n'])==(13500,5400)
pairs=[('original','original_clusters'),('wrapped_mean','wrapped_mean_clusters'),('prototypes','prototypes_clusters'),('original_clusters','prototypes_clusters'),('wrapped_mean_clusters','prototypes_clusters'),('prototypes_clusters','mapped_clusters'),('original','mapped_clusters')]; y=np.array(list(truth.values())); report={}
for ref,arm in pairs:
    d=np.array([hits[arm][i]-hits[ref][i] for i in truth]); rng=np.random.default_rng(42);boot=np.zeros(10000,dtype=int)
    for j in np.unique(y):
        z=d[y==j]; m=rng.multinomial(len(z),np.bincount(z+1,minlength=3)/len(z),size=len(boot));boot+=m[:,2]-m[:,0]
    report[f'{arm} vs {ref}']=dict(delta_pp=round(float(100*d.mean()),3),fixes=int((d==1).sum()),regressions=int((d==-1).sum()),ci95_pp=np.round(100*np.quantile(boot,[.025,.975])/len(d),3).tolist())
out=dict(stage=s['stage'],anchors=s['anchors'],arms={a:{**r,'accuracy_pct':round(r['val_correct']*100/s['n'],3)} for a,r in s['arms'].items()},pairwise=report,herbaceous=[r for r in all_classes if r['name'].startswith('Herbaceous')]); print(json.dumps(out,indent=2))
