"""Training-only sparse spectral rescue with/without CLIP edge weights."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix,diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
import torch
from torch.nn.functional import normalize
from clustering_sweep import HERE,prepare,_allocate,_teacher,write,seed_scores,group_evidence


def graph(x,k):
    n=len(x); cols=[]; vals=[]
    for start in range(0,n,256):
        z=x[start:start+256]@x.T; r=torch.arange(len(z),device=x.device);z[r,start+r]=-torch.inf
        v,j=z.topk(k,dim=1);cols.append(j.cpu().numpy());vals.append(v.cpu().numpy())
    col=np.concatenate(cols); dist=np.maximum(1-np.concatenate(vals),1e-6); scale=dist[:,-1]
    w=np.exp(-dist**2/(scale[:,None]*scale[col])).astype('float64')
    a=csr_matrix((w.ravel(),(np.repeat(np.arange(n),k),col.ravel())),shape=(n,n));a=a.maximum(a.T).tocsr();assert (a!=a.T).nnz==0 and not np.any(a.diagonal())
    return a


def reweight(a,p):
    r,c=a.nonzero();v=np.sqrt(p);overlap=np.einsum('ij,ij->i',v[r],v[c]).clip(0,1)
    b=a.copy();b.data*=.25+.75*overlap
    assert np.array_equal(a.indices,b.indices) and np.array_equal(a.indptr,b.indptr) and (b!=b.T).nnz==0
    return b


def spectral(a,k,dev):
    rng=np.random.default_rng(42); d=diags(np.asarray(a.sum(1)).ravel()**-.5)
    _,u=eigsh(d@a@d,k=k,which='LA',tol=1e-5,v0=rng.normal(size=a.shape[0]));z=normalize(torch.from_numpy(u.astype('float32')).to(dev),dim=1)
    cent=z[rng.choice(len(z),k,replace=False)].clone()
    for _ in range(30):
        labels=(z@cent.T).argmax(1);s=torch.zeros_like(cent).index_add_(0,labels,z);ok=s.norm(dim=1)>0;cent[ok]=normalize(s[ok],dim=1)
    return (z@cent.T).argmax(1).cpu().numpy()


def run(dataset,smoke):
    t=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;dev=torch.device('cuda');root=HERE.parent.parent
    ec=dataset=='eurosat';limits=(64,32) if smoke else (None,None) if ec else (4096,1024)
    d=prepare(dataset,work_dir=HERE/'cache',archive=root/'data/EuroSAT_RGB.zip' if ec else HERE.parent/'cifar100/.cache/cifar-100-python.tar.gz',weights_dir=root/'weights',device='cuda',batch_size=8,workers=0,chunk_size=256,train_limit=limits[0],val_limit=limits[1]);n=len(d['train_ids']);c=len(d['classes']); k=16 if smoke else n//(5*c);kc=8 if smoke else 12*c if ec else 80;cap=8 if smoke else min(64,max(8,k//2));kn=10 if smoke else 20
    ix=np.argsort(d['train_ids']); ids=d['train_ids'][ix]; s=d['train_clip_scores'][ix];raw=d['train_dino'][ix];x=normalize(torch.from_numpy(raw).to(dev),dim=1);assert not np.intersect1d(ids,d['val_ids']).size
    pools,base_n,_,lp,_=_allocate(s,k,d['metadata']['clip_logit_scale']);base=[p[:b] for p,b in zip(pools,base_n)];used=np.isin(np.arange(n),np.concatenate(base));under=np.flatnonzero(base_n<k)
    if not len(under):
        # Explore scarcity explicitly if no class naturally falls below the anchor target.
        scarcity='simulated';under=np.argsort(np.bincount(s.argmax(1),minlength=c))[:2];base_n[under]=max(1,k//10)
        for j in under:base[j]=base[j][:base_n[j]]
        # Exclude released anchors from all rescue pools, rather than letting methods find those again.
    else:scarcity='natural'
    own=under[lp[:,under].argmax(1)];score=s[np.arange(n),own]; ranks=1+(s>score[:,None]).sum(1);gap=s.max(1)-score; candidates=[np.flatnonzero((own==j)&~used&(gap<=.06)) for j in range(c)];quota=np.minimum(np.minimum(k-base_n,cap),list(map(len,candidates)));assert quota.sum()>0
    a=graph(x,kn);p=np.exp(lp);b=reweight(a,p);print(dataset,'graph edges',a.nnz,'components',connected_components(a,directed=False,return_labels=False),'quotas',quota.tolist(),'scarc.',scarcity,flush=True)
    groups={m:spectral(g,kc,dev) for m,g in [('spectral_dino',a),('spectral_dino_clip',b)]};bonus={m:group_evidence(g,p) for m,g in groups.items()}
    diag={m:dict(clusters=int(len(np.unique(g))),noise=0,largest=int(np.bincount(g).max())) for m,g in groups.items()};print(dataset,'partitions',diag,flush=True)
    arms={}; preds={}
    for m in ['baseline','clip_expansion','dino_seed_retrieval',*groups]:
        sets=[]
        for j,(rows,q) in enumerate(zip(candidates,quota)):
            r=lp[rows,j].copy()
            if q and m in bonus:r+=.5*bonus[m][rows,j]
            if q and m=='dino_seed_retrieval':r+=.5*seed_scores(x,base,j,rows)
            sets.append(rows[np.argsort(-r,kind='stable')][:0 if m=='baseline' else q])
        picks=[np.sort(np.r_[p,q]) for p,q in zip(base,sets)];idx=np.concatenate(picks);y=torch.as_tensor(np.repeat(np.arange(c),list(map(len,picks))),device=dev);assert len(np.unique(idx))==len(idx)
        preds[m],_=_teacher(torch.from_numpy(raw[idx]).to(dev),y,torch.from_numpy(d['val_dino']),c,dev); arms[m]=sets;print(dataset,'teacher complete',m,len(idx),flush=True)
    out=HERE/('spectral_smoke' if smoke else f'spectral_{dataset}');out.mkdir(exist_ok=True);y=d['audit_train_y'][ix];vy=d['val_y'];anc=[];pred=[];stats=[]
    for m,sets in arms.items():
        q=preds[m];pred.extend(dict(arm=m,id=i,pred=int(z),truth=int(h)) for i,z,h in zip(d['val_ids'],q,vy))
        for j,(old,new) in enumerate(zip(base,sets)):
            anc.extend(dict(arm=m,id=ids[i],pseudo=j,truth=int(y[i]),added=int(extra),target_rank=int(ranks[i]) if extra else '') for part,extra in [(old,False),(new,True)] for i in part)
            stats.append(dict(arm=m,class_id=j,name=d['classes'][j],base=len(old),added=len(new),added_correct=int(np.sum(y[new]==j)),pool=len(candidates[j]),val_n=int(np.sum(vy==j)),val_correct=int(np.sum((q==j)&(vy==j)))))
    write(out/'anchors.csv',anc);write(out/'predictions.csv',pred);write(out/'by_class.csv',stats);write(out/'clusters.csv',[dict(algorithm=m,id=i,cluster=int(z)) for m,g in groups.items() for i,z in zip(ids,g)])
    cfg=dict(smoke=smoke,dataset=dataset,train_n=n,val_n=len(vy),test_scored=False,scarcity=scarcity,graph_k=kn,graph_edges=a.nnz,components=int(connected_components(a,directed=False,return_labels=False)),cluster_count_target=kc,diagnostics=diag,base_budgets=base_n.tolist(),add_quotas=quota.tolist(),metadata=d['metadata'],seconds=time.monotonic()-t,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());tmp=out/'summary.tmp';tmp.write_text(json.dumps(cfg,indent=2));tmp.replace(out/'summary.json')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('dataset',choices=['eurosat','cifar100']);a.add_argument('--smoke',action='store_true');args=a.parse_args();run(args.dataset,args.smoke)
