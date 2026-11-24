"""Recompute matched metrics and pairwise validation intervals from stored rows."""
import csv,itertools,json,sys
from collections import Counter
from pathlib import Path
import numpy as np

def verify(folder):
    path=Path(folder); s=json.loads((path/'summary.json').read_text()); assert not s['test_scored']
    keys=['clip+dino','siglip2+dino','clip+convnext','siglip2+convnext']; assert set(keys)==set(s['results'])
    a=list(csv.DictReader((path/'anchors.csv').open())); p=list(csv.DictReader((path/'predictions.csv').open())); truth=None; base=None; hits={}
    for key in keys:
        aa=[r for r in a if r['arm']==key]; vv=[r for r in p if r['arm']==key]
        expect={j:q for j,q in enumerate(s['quota']) if q}; assert Counter(int(r['pseudo']) for r in aa if int(r['added']))==Counter(expect)
        current={(r['id'],r['pseudo'],r['truth']) for r in aa if not int(r['added'])}; assert len(current)==sum(s['base_budgets']) and (base is None or current==base); base=current
        assert len({r['id'] for r in aa})==len(aa)==sum(s['base_budgets'])+sum(s['quota'])
        v={r['id']:int(r['truth']) for r in vv}; assert len(vv)==len(v)==s['val_n'] and not (v.keys() & {r['id'] for r in aa}); assert truth is None or v==truth; truth=v
        h={r['id']:int(r['pred']==r['truth']) for r in vv};hits[key]=np.array([h[i] for i in sorted(v)]);
        assert sum(h.values())==s['results'][key]['val_correct'];assert sum(int(r['added']) and r['pseudo']==r['truth'] for r in aa)==s['results'][key]['correct_additions']
    classes=np.array([truth[i] for i in sorted(truth)]); out={}
    for b,a in itertools.combinations(keys,2):
        d=hits[a]-hits[b];rng=np.random.default_rng(42); boot=np.zeros(10000,dtype=int)
        for c in np.unique(classes):
            z=d[classes==c];m=rng.multinomial(len(z),np.bincount(z+1,minlength=3)/len(z),size=len(boot));boot+=m[:,2]-m[:,0]
        out[f'{a} vs {b}']={'pp':100*float(d.mean()),'ci95_pp':(100*np.quantile(boot,[.025,.975])/len(d)).tolist()}
    report=dict(dataset=s['dataset'],smoke=s['smoke'],added=sum(s['quota']),val_n=s['val_n'],results=s['results'],paired=out);text=json.dumps(report,indent=2);(path/'verification.json').write_text(text+'\n');print(text)

if __name__=='__main__':
    for folder in sys.argv[1:]: verify(folder)
