"""Independently verify saved selector and teacher reports."""
import csv,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
p=Path(sys.argv[1]); s=json.loads((p/'summary.json').read_text()); assert not s['test_scored']
def table(name):
    with (p/name).open() as f:return list(csv.DictReader(f))
a=table('anchors.csv'); v=table('predictions.csv'); candidates=table('candidates.csv'); clusters=table('cluster_assignments.csv')
all_ids={r['id'] for r in clusters};assert len(clusters)==len(all_ids)==s['train_n']
ref=None; counts={}; mapped={};
for arm,summary in s['arms'].items():
    ar=[r for r in a if r['arm']==arm]; val=[r for r in v if r['arm']==arm]; pred={r['id']:r for r in val}
    assert len({r['id'] for r in ar})==len(ar)==summary['anchors']
    assert set(r['id'] for r in ar)<=all_ids and not all_ids.intersection(pred)
    assert len(pred)==len(val)==s['val_n'];assert sum(r['pseudo']==r['truth'] for r in ar)==summary['correct_anchors'];assert sum(r['pred']==r['truth'] for r in val)==summary['val_correct']
    if ref is None:ref={k:r['truth'] for k,r in pred.items()}
    assert ref=={k:r['truth'] for k,r in pred.items()}
    counts[arm]=Counter(r['pseudo'] for r in ar); mapped[arm]=pred
assert counts['expanded_clip']==counts['dino_clusters']
for r in candidates:
    j=r['class_id']; assert counts['baseline'][j]==int(r['base_count']); assert counts['expanded_clip'][j]==int(r['base_count'])+int(r['additions'])
if s['stage']=='validation':assert (s['train_n'],s['val_n'])==(13500,5400)
base=mapped['baseline']; report={}
for arm,pred in mapped.items():
    d=np.array([int(r['pred']==r['truth'])-int(base[k]['pred']==r['truth']) for k,r in pred.items()]);y=np.array([int(r['truth']) for r in pred.values()]);boot=np.zeros(10000,dtype=int);rng=np.random.default_rng(42)
    for j in np.unique(y):
        z=d[y==j]; m=rng.multinomial(len(z),np.bincount(z+1,minlength=3)/len(z),size=len(boot));boot+=m[:,2]-m[:,0]
    report[arm]=dict(delta_pp=float(100*d.mean()),fixes=int((d==1).sum()),regressions=int((d==-1).sum()),paired_95ci_pp=(100*np.quantile(boot,[.025,.975])/len(d)).tolist())
print(json.dumps(dict(stage=s['stage'],n=s['val_n'],arms=s['arms'],paired=report,herbaceous=[r for r in table('per_class.csv') if r['class_name'].startswith('Herbaceous')]),indent=2))
