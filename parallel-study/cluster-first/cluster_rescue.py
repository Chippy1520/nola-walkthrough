"""EuroSAT DINO-first cluster rescue. Frozen backbones, teacher-only training."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import normalize

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'dino-anchors'))
from anchor_inputs import prepare
from anchor_eval import _allocate, _teacher


def discover(x, k, device):
    x=normalize(torch.from_numpy(x).to(device),dim=1)
    rng=np.random.default_rng(42)
    cent=x[rng.choice(len(x),k,replace=False)].clone()
    for _ in range(20):
        assignment=(x@cent.T).argmax(1)
        sums=torch.zeros_like(cent).index_add_(0,assignment,x)
        nonzero=sums.norm(dim=1)>0
        cent[nonzero]=normalize(sums[nonzero],dim=1)
    score,group=(x@cent.T).max(1)
    return group.cpu().numpy(),score.cpu().numpy()


def select(logits, groups, centrality, k, scale):
    n,c=logits.shape
    order=np.argsort(-logits,axis=1,kind='stable')[:,:3]
    top=order[:,0]
    pools,budget,_,logp,_=_allocate(logits,k,scale)
    base=[p[:j] for p,j in zip(pools,budget)]
    used=np.zeros(n,bool);used[np.concatenate(base)]=True
    # An image is offered to only one underfilled class: its highest-scoring eligible class.
    scoregap=logits[np.arange(n),top,None]-logits[np.arange(n)[:,None],order]
    eligible=(scoregap<=.04)&(budget[order]<k)&(~used[:,None]); eligible[:,0]=False
    any_ok=eligible.any(1)
    owner=np.where(any_ok,order[np.arange(n),eligible.argmax(1)],-1)
    group_n=np.bincount(groups); soft=np.exp(logp)
    votes=np.zeros((len(group_n),c)); evidence=np.zeros_like(votes)
    np.add.at(votes,(groups,top),1.);np.add.at(evidence,groups,soft)
    denom=np.maximum(group_n[:,None],1); frac=votes/denom; mean=evidence/denom
    arms={'baseline':base,'expanded_clip':[],'dino_clusters':[]}; audit=[]
    for j in range(c):
        ids=np.flatnonzero(owner==j)
        # Abstain unless several independent top-1 votes and mean soft support exist.
        g=groups[ids]; good=(votes[g,j]>=2)&(frac[g,j]>=.08)&(mean[g,j]>=.10)&(group_n[g]>=5)
        keep=ids[good]; count=min(k-budget[j],len(keep)); assert count<=len(ids)
        by=ids[np.argsort(-logp[ids,j],kind='stable')][:count]
        rank=logp[keep,j]+np.log(mean[groups[keep],j])+centrality[keep]
        by_group={}
        for row in keep[np.argsort(-rank,kind='stable')]: by_group.setdefault(int(groups[row]),[]).append(int(row))
        strat=[rows[i] for i in range(max(map(len,by_group.values()),default=0)) for rows in by_group.values() if i<len(rows)][:count]
        arms['expanded_clip'].append(np.r_[base[j],by]);arms['dino_clusters'].append(np.r_[base[j],np.asarray(strat,dtype=int)])
        audit.append(dict(class_id=j,base_count=int(budget[j]),expanded_candidate_count=len(ids),cluster_supported_candidate_count=len(keep),cluster_supported_groups=len(by_group),additions=int(count)))
    for arm in arms.values(): assert len(np.concatenate(arm))==len(np.unique(np.concatenate(arm)))
    assert list(map(len,arms['expanded_clip']))==list(map(len,arms['dino_clusters']))
    return arms,audit


def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(smoke):
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    p=HERE.parent.parent; size=(64,32) if smoke else (None,None)
    d=prepare('eurosat',work_dir=HERE/'cache',archive=p/'data/EuroSAT_RGB.zip',weights_dir=p/'weights',device='cuda',batch_size=8,workers=0,chunk_size=256,train_limit=size[0],val_limit=size[1]); print('Frozen features ready',len(d['train_ids']),len(d['val_ids']),flush=True)
    ids=d['train_ids']; sort=np.argsort(ids); ids=ids[sort]; scores=d['train_clip_scores'][sort]; x=d['train_dino'][sort]
    assert not np.intersect1d(ids,d['val_ids']).size
    out=HERE/('smoke' if smoke else 'eurosat');out.mkdir(exist_ok=True)
    device=torch.device('cuda'); c=len(d['classes']); k=max(16,len(x)//(5*c)); groups,centrality=discover(x,min(len(x),2*c if smoke else 12*c),device)
    arms,audit=select(scores,groups,centrality,k,d['metadata']['clip_logit_scale']); print('Selector ready',[(key,sum(map(len,val))) for key,val in arms.items()],flush=True)
    # Fit every teacher before reading any auditing label.
    frozen={}
    for label,picks in arms.items():
        rows=np.concatenate([np.sort(j) for j in picks]); pseudo=np.repeat(np.arange(c),list(map(len,picks)))
        pred,_=_teacher(torch.from_numpy(x[rows]).to(device),torch.as_tensor(pseudo,dtype=torch.long,device=device),torch.from_numpy(d['val_dino']),c,device)
        frozen[label]=(rows,pseudo,pred);print('Teacher done',label,flush=True)
    y=d['audit_train_y'][sort];vy=d['val_y']; anchor_rows=[];val_rows=[];stats=[]
    for label,(rows,pseudo,pred) in frozen.items():
        anchor_rows.extend(dict(arm=label,id=ids[i],cluster=int(groups[i]),pseudo=int(j),truth=int(y[i])) for i,j in zip(rows,pseudo))
        val_rows.extend(dict(arm=label,id=i,pred=int(j),truth=int(z)) for i,j,z in zip(d['val_ids'],pred,vy))
        stats.extend(dict(arm=label,class_name=name,anchors=int((pseudo==j).sum()),correct_anchors=int(((pseudo==j)&(y[rows]==j)).sum()),val_n=int((vy==j).sum()),val_correct=int(((vy==j)&(pred==j)).sum())) for j,name in enumerate(d['classes']))
    write(out/'anchors.csv',anchor_rows);write(out/'predictions.csv',val_rows);write(out/'per_class.csv',stats);write(out/'candidates.csv',audit);write(out/'cluster_assignments.csv',[dict(id=i,cluster=int(g),cosine=float(z)) for i,g,z in zip(ids,groups,centrality)])
    summary=dict(stage='smoke_only' if smoke else 'validation',test_scored=False,teacher_epochs=50,clusters=int(len(np.unique(groups))),train_n=len(y),val_n=len(vy),metadata=d['metadata'],selector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),arms={a:dict(anchors=len(r),correct_anchors=int((y[r]==pj).sum()),val_correct=int((pr==vy).sum())) for a,(r,pj,pr) in frozen.items()})
    tmp=out/'summary.tmp';tmp.write_text(json.dumps(summary,indent=2));tmp.replace(out/'summary.json');print(json.dumps(summary['arms']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');run(p.parse_args().smoke)
