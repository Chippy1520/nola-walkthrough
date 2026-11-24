"""Verify downloaded DINO selector archives and produce cross-dataset paired statistics."""
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT=Path(__file__).resolve().parent
EXPECTED={'eurosat':(13500,5400),'resisc45':(24930,6255),'cifar100':(45000,5000)}

def verify(path):
    with ZipFile(path) as z:
        assert z.testzip() is None
        s=json.loads(z.read('summary.json'))
        def read(n):return list(csv.DictReader(io.StringIO(z.read(n).decode())))
        rows=read('predictions.csv'); anchors=read('anchor_audit.csv'); by_class=read('by_class.csv')
        if s['name']=='cifar100':
            sub=read('by_subtype.csv'); assert sum(r['split']=='validation' for r in sub)==400
    assert (s['n_train_unique'],s['n_val'])==EXPECTED[s['name']]
    assert not s['metadata']['test_decoded'] and s['evaluation_split']=='validation'
    n=s['n_val']; baseline={r['val_id']:r for r in rows if r['arm']=='confidence'}; assert len(baseline)==n
    ref=list(baseline); out=[]
    for k,m in s['arms'].items():
        b={r['val_id']:r for r in rows if r['arm']==k}; assert set(b)==set(ref)
        a=[r for r in anchors if r['arm']==k]; assert len(a)==len({r['train_id'] for r in a})==m['anchor_count']==sum(s['realized_per_class'])
        assert not set(b).intersection(r['train_id'] for r in a)
        assert all(int(r['correct'])==(r['true_class']==r['pseudo_class']) for r in a)
        assert all(int(b[i]['correct'])==(b[i]['true_class']==b[i]['pred_class']) and b[i]['true_class']==baseline[i]['true_class'] for i in ref)
        assert [sum(int(r['pseudo_class'])==j for r in a) for j in range(len(s['classes']))]==s['realized_per_class']
        d=np.array([int(b[i]['correct'])-int(baseline[i]['correct']) for i in ref]); yy=np.array([int(b[i]['true_class']) for i in ref]);
        assert (int((d==1).sum()),int((d==-1).sum()))==(m['val_fixed_vs_confidence'],m['val_broken_vs_confidence'])
        assert round(m['val_accuracy']*n)==sum(int(b[i]['correct']) for i in ref)
        assert round(m['anchor_precision']*len(a))==sum(int(r['correct']) for r in a)
        rng=np.random.default_rng(42); boot=np.zeros(10000,dtype=np.int64)
        for j in np.unique(yy):
            v=d[yy==j]; x=rng.multinomial(len(v),np.bincount(v+1,minlength=3)/len(v),size=len(boot));boot+=x[:,2]-x[:,0]
        lo,hi=100*np.quantile(boot,[.025,.975])/n
        macro=100*np.mean([float(r['val_accuracy']) for r in by_class if r['arm']==k and r['val_accuracy']])
        out.append(dict(dataset=s['name'],method=k,train_pool=s['n_train_unique'],val_n=n,anchors=len(a),anchor_precision_pct=100*m['anchor_precision'],val_accuracy_pct=100*m['val_accuracy'],val_macro_pct=macro,delta_pp=100*d.mean(),paired_ci_low_pp=lo,paired_ci_high_pp=hi,fixed=int((d==1).sum()),regressions=int((d==-1).sum()),archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return out

if __name__=='__main__':
    rows=[]
    for p in sorted((ROOT/'reports').glob('*-report.zip')):
        rows.extend(verify(p))
    assert rows
    with (ROOT/'reports'/'combined.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    for r in rows:
        print(f"{r['dataset']:8} {r['method']:20} n={r['train_pool']:5} anchors={r['anchors']:4} precision={r['anchor_precision_pct']:5.2f} val={r['val_accuracy_pct']:5.2f} delta={r['delta_pp']:+5.2f} [{r['paired_ci_low_pp']:+.2f},{r['paired_ci_high_pp']:+.2f}] fixed/lost={r['fixed']}/{r['regressions']}")
