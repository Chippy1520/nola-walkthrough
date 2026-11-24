"""Matched cluster algorithms for low-anchor rescue; train features only for selection."""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch.nn.functional import normalize
from cluster_rescue import HERE,prepare,_allocate,_teacher,write


def density(x, k=12):
    # Exact cosine-radius DBSCAN, choosing radius from training-only kth-neighbor distances.
    n=len(x); kth=[]
    for a in range(0,n,256):
        s=x[a:a+256]@x.T
        kth.append(s.topk(k+1,dim=1).values[:,-1].cpu().numpy())
    cutoff=float(np.quantile(np.concatenate(kth),.65)); edges=[]
    for a in range(0,n,256):
        r,c=torch.where((x[a:a+256]@x.T)>=cutoff); edges.append((r.cpu().numpy()+a,c.cpu().numpy()))
    r=np.concatenate([p[0] for p in edges]);c=np.concatenate([p[1] for p in edges]); graph=csr_matrix((np.ones(len(r),dtype=bool),(r,c)),shape=(n,n));graph=graph.maximum(graph.T); graph.setdiag(False);graph.eliminate_zeros()
    core=np.diff(graph.indptr)>=k; out=np.full(n,-1,dtype=int)
    out[core]=connected_components(graph[core][:,core],directed=False)[1]
    for i in np.flatnonzero(~core):
        near=graph.indices[graph.indptr[i]:graph.indptr[i+1]]; near=near[core[near]]
        if len(near):
            j=int((x[i]@x[near].T).argmax());out[i]=out[near[j]]
    return out,cutoff


def spherical(x,k):
    rng=np.random.default_rng(42); mu=x[rng.choice(len(x),k,replace=False)].clone()
    for _ in range(20):
        g=(x@mu.T).argmax(1); m=torch.zeros_like(mu).index_add_(0,g,x); active=m.norm(dim=1)>0;mu[active]=normalize(m[active],dim=1)
    return (x@mu.T).argmax(1).cpu().numpy()


def ward(x,k):
    # Full unweighted Ward on all rows after train-only PCA, not a microcluster approximation.
    z=x.cpu().numpy();z=z-z.mean(0);_,v=np.linalg.eigh(z.T@z);p=z@v[:,-32:]
    return cut_tree(linkage(p,method='ward'),n_clusters=k).ravel()


def group_evidence(g,p):
    # Leave each image out of its own group's vote. Shrink small groups toward no bonus.
    ok=g>=0; q=np.zeros_like(p); sums=np.zeros((int(g.max())+1,p.shape[1]));np.add.at(sums,g[ok],p[ok]);n=np.bincount(g[ok],minlength=len(sums)); idx=g[ok]
    other=(sums[idx]-p[ok]+8*p[ok])/(n[idx,None]-1+8)
    q[ok]=(np.log(other.clip(1e-12))-np.log(p[ok].clip(1e-12))).clip(-2,2)
    return q


def seed_scores(x,base,cls,rows):
    positive=x[base[cls]];negative=x[np.concatenate([p for j,p in enumerate(base) if j!=cls])]
    if not len(positive) or not len(negative):return np.zeros(len(rows))
    q=[]
    for a in range(0,len(rows),256):
        z=x[rows[a:a+256]];ps=(z@positive.T).topk(min(3,len(positive)),dim=1).values.mean(1);ns=(z@negative.T).topk(min(3,len(negative)),dim=1).values.mean(1)
        q.append(((ps-ns)*10).clamp(-2,2).cpu().numpy())
    return np.concatenate(q) if q else np.empty(0)


def run(smoke):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;dev=torch.device('cuda');root=HERE.parent.parent
    d=prepare('eurosat',work_dir=HERE/'cache',archive=root/'data/EuroSAT_RGB.zip',weights_dir=root/'weights',device='cuda',batch_size=8,workers=0,chunk_size=256,train_limit=64 if smoke else None,val_limit=32 if smoke else None)
    order=np.argsort(d['train_ids']);ids=d['train_ids'][order];s=d['train_clip_scores'][order]; raw=d['train_dino'][order];x=normalize(torch.from_numpy(raw).to(dev),dim=1);n,c=s.shape
    assert not np.intersect1d(ids,d['val_ids']).size
    k=16 if smoke else 270; kk=8 if smoke else 120; maxadd=8 if smoke else 64
    pools,budgets,_,lp,_=_allocate(s,k,d['metadata']['clip_logit_scale']);base=[p[:b] for p,b in zip(pools,budgets)]; used=np.isin(np.arange(n),np.concatenate(base))
    # Assign candidates to their strongest underfilled class, allowing ranks beyond top 3.
    under=np.flatnonzero(budgets<k); assert len(under)
    owner=under[lp[:,under].argmax(1)]; gap=s.max(1)-s[np.arange(n),owner]; ranks=1+(s>s[np.arange(n),owner,None]).sum(1)
    candidates=[np.flatnonzero((owner==j)&(~used)&(gap<=.06)) for j in range(c)];quota=np.minimum(np.minimum(k-budgets,maxadd),list(map(len,candidates)))
    assert np.sum(quota)>0
    print('Base / added',budgets.tolist(),quota.tolist(),flush=True)
    groups={'spherical_kmeans':spherical(x,kk),'ward_pca32':ward(x,kk)}
    groups['dbscan'],eps=density(x,4 if smoke else 12)
    def sizes(g):
        a=np.bincount(g[g>=0]);return dict(clusters=int((a>0).sum()),noise=int((g<0).sum()),largest=int(a.max(initial=0)))
    diag={m:sizes(g) for m,g in groups.items()};print('Clustering complete',diag,flush=True)
    bonus={m:group_evidence(g,np.exp(lp)) for m,g in groups.items()}
    arms={'baseline':[(p,np.empty(0,dtype=int)) for p in base]}
    for m in ['clip_expansion','dino_seed_retrieval',*groups]:
        arms[m]=[]
        for j,(p,rows,q) in enumerate(zip(base,candidates,quota)):
            score=lp[rows,j].copy()
            if q and m in bonus: score+=.5*bonus[m][rows,j]
            elif q and m=='dino_seed_retrieval': score+=.5*seed_scores(x,base,j,rows)
            extra=rows[np.argsort(-score,kind='stable')][:q];arms[m].append((p,extra))
    preds={};out=HERE/('clustering_sweep_smoke' if smoke else 'clustering_sweep_eurosat');out.mkdir(exist_ok=True)
    for m,sets in arms.items():
        rows=np.concatenate([np.sort(np.r_[p,q]) for p,q in sets]);pseudo=np.repeat(np.arange(c),[len(p)+len(q) for p,q in sets]);assert len(np.unique(rows))==len(rows)
        preds[m],_=_teacher(torch.from_numpy(raw[rows]).to(dev),torch.from_numpy(pseudo).to(dev),torch.from_numpy(d['val_dino']),c,dev); print('Head complete',m,len(rows),flush=True)
    y=d['audit_train_y'][order]; vy=d['val_y'];predrows=[];anc=[];stats=[]; clusterrows=[]
    for m,g in groups.items():clusterrows.extend(dict(algorithm=m,id=i,cluster=int(j)) for i,j in zip(ids,g))
    for m,sets in arms.items():
        pred=preds[m];predrows.extend(dict(arm=m,id=i,pred=int(p),truth=int(t)) for i,p,t in zip(d['val_ids'],pred,vy))
        for j,(p,q) in enumerate(sets):
            anc.extend(dict(arm=m,id=ids[i],pseudo=j,truth=int(y[i]),added=int(extra),target_rank=int(ranks[i]) if extra else '') for group,extra in [(p,False),(q,True)] for i in group)
            stats.append(dict(arm=m,class_id=j,name=d['classes'][j],base=len(p),added=len(q),added_correct=int((y[q]==j).sum()),pool=len(candidates[j]),val_n=int((vy==j).sum()),val_correct=int(((pred==j)&(vy==j)).sum())))
    write(out/'anchors.csv',anc);write(out/'predictions.csv',predrows);write(out/'by_class.csv',stats);write(out/'clusters.csv',clusterrows)
    cfg=dict(smoke=smoke,train_n=n,val_n=len(vy),test_scored=False,cluster_count_target=kk,density_cosine_cutoff=eps,diagnostics=diag,base_budgets=budgets.tolist(),add_quotas=quota.tolist(),feature_signature=d['metadata']['cache_signature'],code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    tmp=out/'summary.tmp';tmp.write_text(json.dumps(cfg,indent=2));tmp.replace(out/'summary.json')

if __name__=='__main__':run('--smoke' in sys.argv)
