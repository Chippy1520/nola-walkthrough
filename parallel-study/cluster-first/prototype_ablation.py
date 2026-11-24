"""Matched frozen-EuroSAT text-bank x DINO-cluster ablation."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import normalize
from prototype_scores import HERE,ROOT,BANK,bank
from cluster_rescue import discover,write
from anchor_inputs import prepare,get_dataset,_read_chunk
from anchor_eval import _teacher,_allocate


def run(smoke=False):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda'); size=64 if smoke else None
    params=dict(work_dir=HERE/'cache',archive=ROOT/'data/EuroSAT_RGB.zip')
    d=prepare('eurosat',**params,weights_dir=ROOT/'weights',device='cuda',batch_size=8,workers=0,chunk_size=256,train_limit=size,val_limit=32 if smoke else None)
    ds=get_dataset('eurosat',**params); b=bank(); assert list(d['classes'])==b['class_order']; assert sum(map(len,b['sizes']))==16 and all(sum(q)==25 for q in b['sizes'])
    ids=d['train_ids']; chunks=[]
    for j in range(0,len(ids),256):
        z=ids[j:j+256]; x=_read_chunk(HERE/'clip_vectors'/('smoke' if smoke else 'full')/f'{j:06d}.npz',z,{'clip':(len(z),512)}); assert x is not None, (j,len(z)); chunks.append(x['clip'])
    emb=torch.from_numpy(np.concatenate(chunks)); np.testing.assert_allclose((emb@torch.from_numpy(ds.text_mean).T).numpy(),d['train_clip_scores'],atol=3e-6,rtol=0)
    assert not np.intersect1d(ids,d['val_ids']).size
    order=np.argsort(ids); ids=ids[order]; emb=emb[order]; x=d['train_dino'][order]; n,c=d['train_clip_scores'].shape; k=16 if smoke else 270
    group,central=discover(x,20 if smoke else 120,device); gn=np.bincount(group); active=gn[group]>0
    assert active.all()
    gcount=len(gn); tau=b['tau']; proto=[]; mean=[]; mapped=[]; align=[]
    for j,(vs,sz) in enumerate(zip(b['vectors'],b['sizes'])):
        v=vs.float(); w=torch.tensor(sz,dtype=torch.float32); sim=emb@v.T
        mean.append(emb@normalize((v*w[:,None]).sum(0),dim=0))
        base=tau*(torch.logsumexp(sim/tau+w.log(),dim=1)-w.sum().log());proto.append(base)
        # Learn within-parent variant responsibilities from TRAIN cluster members only.
        # A 50/50 blend with global text weights limits influence of a noisy cluster.
        resp=(sim/tau+w.log()).softmax(1); counts=torch.zeros((gcount,len(w))).index_add_(0,torch.from_numpy(group),resp)
        weights=.5*(counts/torch.from_numpy(gn.clip(1))[:,None])+.5*(w/w.sum())[None,:]
        mapped.append(tau*torch.logsumexp(sim/tau+weights[group].log(),dim=1))
        for g in np.flatnonzero(gn):
            for vj,weight in enumerate(weights[g]):align.append(dict(cluster=int(g),cluster_size=int(gn[g]),class_id=j,variant=vj,weight=float(weight)))
    banks=dict(original=d['train_clip_scores'][order],wrapped_mean=torch.stack(mean,1).numpy(),prototypes=torch.stack(proto,1).numpy(),mapped=torch.stack(mapped,1).numpy()); scale=d['metadata']['clip_logit_scale']
    pool={z:_allocate(s,k,scale) for z,s in banks.items()}; budgets=np.min([entry[1] for entry in pool.values()],axis=0)
    assert budgets.sum()>0
    assignments={}; config={}
    for name,key,strat in [('original','original',False),('original_clusters','original',True),('wrapped_mean','wrapped_mean',False),('wrapped_mean_clusters','wrapped_mean',True),('prototypes','prototypes',False),('prototypes_clusters','prototypes',True),('mapped_clusters','mapped',True)]:
        # Cluster use is confined to ranking/coverage inside each bank's candidate pools.
        pools,_,_,logp,_=pool[key]; mass=np.zeros((gcount,c));np.add.at(mass,group,np.exp(logp));mass/=gn.clip(1)[:,None]
        picks=[]
        for j,(p,q) in enumerate(zip(pools,budgets)):
            if not strat or q==0: rows=p[:q]
            else:
                ranked=p[np.argsort(-(logp[p,j]+np.log(mass[group[p],j].clip(1e-15))+central[p]),kind='stable')]; by={}
                for row in ranked: by.setdefault(int(group[row]),[]).append(int(row))
                rows=np.asarray([rr[level] for level in range(max(map(len,by.values()))) for rr in by.values() if level<len(rr)][:q])
            picks.append(np.sort(rows))
        rows=np.concatenate(picks);assert len(np.unique(rows))==len(rows)==budgets.sum()
        assignments[name]=rows;config[name]=dict(bank=key,stratified=strat)
    py=np.repeat(np.arange(c),budgets); preds={}
    for name,rows in assignments.items():
        preds[name],_=_teacher(torch.from_numpy(x[rows]).to(device),torch.as_tensor(py,dtype=torch.long,device=device),torch.from_numpy(d['val_dino']),c,device)
        print('Head complete',name,len(rows),flush=True)
    y=d['audit_train_y'][order]; vy=d['val_y']; out=HERE/('prototype_smoke' if smoke else 'prototype_eurosat');out.mkdir(exist_ok=True)
    anchors=[]; predrows=[]; counts=[]; report={}
    for arm,rows in assignments.items():
        pr=preds[arm]; report[arm]=dict(anchor_correct=int((y[rows]==py).sum()),val_correct=int((pr==vy).sum()))
        anchors.extend(dict(arm=arm,id=ids[r],group=int(group[r]),pseudo=int(j),truth=int(y[r])) for r,j in zip(rows,py))
        predrows.extend(dict(arm=arm,id=i,pred=int(j),truth=int(z)) for i,j,z in zip(d['val_ids'],pr,vy))
        counts.extend(dict(arm=arm,class_id=j,name=name,anchors=int(budgets[j]),correct_anchors=int(((py==j)&(y[rows]==j)).sum()),val_n=int((vy==j).sum()),val_correct=int(((vy==j)&(pr==j)).sum())) for j,name in enumerate(d['classes']))
    write(out/'anchors.csv',anchors);write(out/'predictions.csv',predrows);write(out/'per_class.csv',counts);write(out/'prototype_assignments.csv',align)
    write(out/'budgets.csv',[dict(class_id=j,name=name,budget=int(budgets[j]),**{z:int(v[1][j]) for z,v in pool.items()}) for j,name in enumerate(d['classes'])]);write(out/'clusters.csv',[dict(id=i,cluster=int(g),cosine=float(t)) for i,g,t in zip(ids,group,central)])
    summary=dict(stage='smoke' if smoke else 'validation',test_scored=False,n=len(vy),train_n=n,teachers='50 epochs, same fixed seed',anchors=int(budgets.sum()),cluster_count=int(len(np.unique(group))),bank_sha256=hashlib.sha256(BANK.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),features=d['metadata'],config=config,arms=report)
    p=out/'summary.tmp';p.write_text(json.dumps(summary,indent=2));p.replace(out/'summary.json');print(report,flush=True)

if __name__=='__main__':run('--smoke' in sys.argv)
