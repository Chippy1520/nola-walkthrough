import csv,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
p=Path(sys.argv[1]);s=json.loads((p/'summary.json').read_text());assert not s['test_scored']
def read(f):
    with (p/f).open() as h:return list(csv.DictReader(h))
a=read('anchors.csv');v=read('predictions.csv');b=read('by_class.csv');g=read('clusters.csv'); arms=sorted({r['arm'] for r in a}); report={};hits={};truth=None;base=None; part={}
for m,d in s['diagnostics'].items():
    gr=[r for r in g if r['algorithm']==m]; ids={r['id'] for r in gr}; assert len(gr)==len(ids)==s['train_n'];part[m]=ids
    c=Counter(int(r['cluster']) for r in gr if int(r['cluster'])>=0);assert dict(clusters=len(c),noise=len(gr)-sum(c.values()),largest=max(c.values(),default=0))==d
for m in arms:
    ar=[r for r in a if r['arm']==m];vr=[r for r in v if r['arm']==m]; tr={r['id']:r['truth'] for r in vr}; x={(r['id'],r['pseudo']) for r in ar if r['added']=='0'}; ext=[r for r in ar if r['added']=='1']; cc=Counter(int(r['pseudo']) for r in ext)
    assert [cc[j] for j in range(len(s['base_budgets']))]==([0]*len(s['base_budgets']) if m=='baseline' else s['add_quotas'])
    assert len({r['id'] for r in ar})==len(ar) and len(tr)==len(vr)==s['val_n']; train=next(iter(part.values()));assert {r['id'] for r in ar}<=train and not train.intersection(tr)
    if base is None: base,truth=x,tr
    assert x==base and tr==truth
    h={r['id']:int(r['pred']==r['truth']) for r in vr};hits[m]=h
    report[m]=dict(anchors=len(ar),added=len(ext),added_correct=sum(r['pseudo']==r['truth'] for r in ext),beyond_top3=sum(int(r['target_rank'])>3 for r in ext),val_correct=sum(h.values()),accuracy_pct=round(100*sum(h.values())/len(h),3))
    for r in (r for r in b if r['arm']==m):
        j=r['class_id']; assert sum(z['pseudo']==z['truth']==j for z in ext)==int(r['added_correct']);assert sum(z['pred']==z['truth']==j for z in vr)==int(r['val_correct'])
if not s['smoke']:
    expected={'eurosat':(13500,5400),'cifar100':(4096,1024)}
    assert (s['train_n'],s['val_n'])==expected[s.get('dataset','eurosat')]
y=np.array(list(truth.values()));paired={}
for m in arms:
    if m=='clip_expansion':continue
    d=np.array([hits[m][i]-hits['clip_expansion'][i] for i in truth]);rng=np.random.default_rng(42); boot=np.zeros(10000,dtype=int)
    for j in np.unique(y):
        z=d[y==j];marg=rng.multinomial(len(z),np.bincount(z+1,minlength=3)/len(z),size=len(boot));boot+=marg[:,2]-marg[:,0]
    paired[m]=dict(delta_pp=round(100*d.mean(),3),ci95_pp=np.round(100*np.quantile(boot,[.025,.975])/len(d),3).tolist())
print(json.dumps(dict(smoke=s['smoke'],diagnostics=s['diagnostics'],results=report,vs_clip=paired,target_classes=[r for r in b if s['add_quotas'][int(r['class_id'])]]),indent=2))
