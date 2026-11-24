from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

root = Path(__file__).resolve().parent
out = root/'weakness-audit'
out.mkdir(exist_ok=True)
p = pd.read_csv(root/'predictions.csv')
a = pd.read_csv(root/'anchors.csv')
features = np.load(root/'dino-pca.npz')['train_features']
classes = json.loads((root/'metrics.json').read_text())['class_order']
train = p.query("stage == 'cde' and split == 'adaptation'").sort_values('row').reset_index(drop=True)
coords = pd.read_csv(root/'dino-pca-coordinates.csv').query("split == 'adaptation'").sort_values('row')
assert np.array_equal(train.image_path,coords.image_path) and len(features)==len(train)
assert np.array_equal(train.iloc[a.row].path if 'path' in train else train.iloc[a.row].image_path, a.path)
table = []
for c in classes:
    selected = a[a.pseudo_class == c]
    entry = dict(class_name=c, adaptation_true=int(train.true_class.eq(c).sum()), clip_top1=int(train.predicted_class.eq(c).sum()), anchors=len(selected), anchors_correct=int(selected.correct.sum()), fallback_count=int(selected.assigned_rank.gt(1).sum()))
    for stage,name in [('cde','cde'),('dino_head','dino'),('prompted_epoch_4','nola')]:
        x=p[(p.stage==stage)&(p.split=='validation')&(p.true_class==c)]
        entry[f'{name}_correct']=int(x.correct.sum()); entry['val_n']=len(x)
    table.append(entry)
class_table=pd.DataFrame(table); class_table.to_csv(out/'per-class.csv',index=False)

# Global cosine neighbors on adaptation rows only; retain mutual, same-predicted-class edges.
v=features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
s=v@v.T; np.fill_diagonal(s,-np.inf)
neighbor=np.argsort(-s,axis=1,kind='stable')[:,:5]
edges=np.zeros_like(s,dtype=bool); edges[np.arange(len(v))[:,None],neighbor]=True
labels=train.predicted_class.to_numpy()
graph=edges & edges.T & (labels[:,None]==labels)
_,group=connected_components(graph,directed=False)
groups=[]; supports={}
for i in np.unique(group):
    ids=np.flatnonzero(group==i); part=train.iloc[ids]; c=labels[ids[0]]
    matching=a[a.row.isin(ids) & a.pseudo_class.eq(c)]
    groups.append(dict(group_id=int(i),predicted_class=c,n=len(ids),supported=len(ids)>=4,
                       correct_cde=int(part.correct.sum()),anchors=len(matching),anchor_correct=int(matching.correct.sum()),
                       rows=' '.join(map(str,ids.tolist()))))
groups=pd.DataFrame(groups).sort_values(['predicted_class','n'],ascending=[True,False]); groups.to_csv(out/'dino-regions.csv',index=False)
assert groups.n.sum()==len(train)
x=groups[groups.supported]; missed=x[x.anchors.eq(0)]
print(class_table.to_string(index=False)); print('\nSupported regions, mutual 5-NN, min 4:',len(x),'; zero anchors:',len(missed)); print(missed[['predicted_class','n','correct_cde']].to_string(index=False))
for split in ['validation','fresh_test']:
    wide=p[p.split.eq(split)].pivot(index='image_path',columns='stage',values='correct')
    d=wide.dino_head.astype(bool); c=wide.prompted_epoch_4.astype(bool)
    print(split, 'DINO correct / NoLA wrong',int((d&~c).sum()),'NoLA correct / DINO wrong',int((c&~d).sum()))
