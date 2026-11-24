"""Independent vision-language nomination, with a matched rerank control."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import binomtest
from torch.nn.functional import normalize
HERE=Path(__file__).resolve().parent
CF=HERE.parent/'parallel-study/cluster-first';sys.path.insert(0,str(CF))
from compare_siglip import REV,SHA
from convnext_retrieval import ROOT,setup
from clustering_sweep import _allocate,_teacher,prepare,seed_scores,write
from anchor_eval import EPOCHS,SEED,BATCH_SIZE,LR,HIDDEN
from anchor_inputs import _read_chunk

CL='clip_only'; RR='clip_pool_siglip_rank'; SG='siglip_only'
def select(cs,ss,x,k,cap,scale):
    n,c=cs.shape; ri=np.arange(n)
    def nominate(s):
        pools,b,rank,lp,top=_allocate(s,k,scale)
        base=[p[:q] for p,q in zip(pools,b)]
        def candidates(base, under):
            if not len(under): return [np.empty(0,dtype=int) for _ in range(c)]
            used=np.isin(ri,np.concatenate(base)); owner=under[s[:,under].argmax(1)]
            gap=s.max(1)-s[ri,owner]
            return [ri[(owner==j)&(~used)&(gap<=.06)] for j in range(c)]
        return base,b,rank,lp,top,candidates
    ca,sa=nominate(cs),nominate(ss)
    quota=np.minimum(ca[1],sa[1]); under=np.flatnonzero(quota<k)
    def expand(src,ranklog,base,cand,q):
        result=[]
        for j,(rows,cnt) in enumerate(zip(cand,q)):
            r=ranklog[rows,j]+.5*seed_scores(x,base,j,rows) if cnt else np.empty(0)
            p=rows[np.argsort(-r,kind='stable')[:cnt]] if cnt else rows[:0]
            result.append(p)
        return base,result
    cb,sb=([v[:q] for v,q in zip(t[0],quota)] for t in [ca,sa]); cc,sc=ca[-1](cb,under),sa[-1](sb,under)
    extra=np.minimum.reduce([k-quota,np.full(c,cap),np.array(list(map(len,cc))),np.array(list(map(len,sc)))])
    arms={CL:expand(ca,ca[3],cb,cc,extra),RR:expand(ca,sa[3],cb,cc,extra),SG:expand(sa,sa[3],sb,sc,extra)}
    for name,p in [('clip_native',ca),('siglip_native',sa)]:
        base,b,_,lp,_,fn=p; cands=fn(base,np.flatnonzero(b<k)); add=np.minimum.reduce([k-b,np.full(c,cap),np.array(list(map(len,cands)))])
        arms[name]=expand(p,lp,base,cands,add)
    return arms,(ca,sa),quota,extra,cc,sc

def compare(a,b,y):
    x=a==y; z=b==y; d=z.astype(int)-x.astype(int); f=int((d==1).sum()); l=int((d==-1).sum());n=len(y)
    rng=np.random.default_rng(42); draws=np.zeros(10000,int)
    for j in np.unique(y):
        q=d[y==j]; h=rng.multinomial(len(q),np.bincount(q+1,minlength=3)/len(q),size=len(draws));draws+=h[:,2]-h[:,0]
    return dict(delta_pp=100*(f-l)/n,fixes=f,losses=l,ci95_pp=(100*np.quantile(draws,[.025,.975])/n).tolist(),mcnemar_p=float(binomtest(f,f+l).pvalue) if f+l else 1.)

def verify(out):
    s=json.loads((out/'summary.json').read_text()); a=list(csv.DictReader((out/'anchors.csv').open()));p=list(csv.DictReader((out/'predictions.csv').open()));n=s['val_n']; preds={}
    for m,v in s['arms'].items():
        z=[r for r in a if r['arm']==m]; w=[r for r in p if r['arm']==m]; assert len(z)==len({r['id'] for r in z})==v['anchors'];assert len(w)==len({r['id'] for r in w})==n
        assert not ({r['id'] for r in z}&{r['id'] for r in w}); assert sum(r['pseudo']==r['truth'] for r in z)==v['correct_anchors'];assert sum(r['pred']==r['truth'] for r in w)==v['val_correct']
        if m in (CL,RR,SG):
            for j,(q,h) in enumerate(zip(s['matched_base'],s['matched_extra'])):
                assert sum(int(r['pseudo'])==j and r['stage']=='seed' for r in z)==q
                assert sum(int(r['pseudo'])==j and r['stage']=='extra' for r in z)==h
        preds[m]=w
    assert {(r['id'],r['pseudo']) for r in a if r['arm']==CL and r['stage']=='seed'}=={(r['id'],r['pseudo']) for r in a if r['arm']==RR and r['stage']=='seed'}
    for k,stat in s['paired'].items():
        b,a=k.split(' vs ');x,z=preds[a],preds[b]; assert [(r['id'],r['truth']) for r in x]==[(r['id'],r['truth']) for r in z]
        f=sum(a['pred']!=a['truth'] and b['pred']==b['truth'] for a,b in zip(x,z));l=sum(a['pred']==a['truth'] and b['pred']!=b['truth'] for a,b in zip(x,z));assert (f,l)==(stat['fixes'],stat['losses'])
    (out/'verified.json').write_text(json.dumps(s,indent=2)+'\n'); return s

def run(name,smoke):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False; dev=torch.device('cuda')
    tr,va=(64,32) if smoke else (None,None) if name=='eurosat' else (4096,1024)
    d=prepare(name,**setup(name),weights_dir=ROOT/'weights',device='cuda',batch_size=8,workers=0,chunk_size=256,train_limit=tr,val_limit=va)
    ids0=d['train_ids']; c=len(d['classes']); n=len(ids0); parts=[]; cachehash=hashlib.sha256()
    for i in range(0,n,256):
        ids=ids0[i:i+256]; p=CF/'siglip_scores'/REV/name/('smoke' if smoke else 'full')/'siglip2/train'/f'{i:06d}.npz';r=_read_chunk(p,ids,{'scores':(len(ids),c)});assert r is not None,p;parts.append(r['scores']);cachehash.update(p.read_bytes())
    idx=np.argsort(ids0);ids=ids0[idx];vid=d['val_ids']; assert len(np.unique(ids))==n and not np.intersect1d(ids,vid).size
    raw=torch.from_numpy(d['train_dino'][idx]).to(dev); val=torch.from_numpy(d['val_dino']);scale=d['metadata']['clip_logit_scale']; k=16 if smoke else n//(5*c); cap=8 if smoke else min(64,max(8,k//2)); cs=d['train_clip_scores'][idx];ss=np.concatenate(parts)[idx]
    start=time.monotonic(); arms,pools,q,h,cc,sc=select(cs,ss,normalize(raw,dim=1),k,cap,scale); print(name,'matched quotas',q.tolist(),h.tolist(),flush=True)
    runs={};reuse={}
    for m,(base,extra) in arms.items():
        rows=np.concatenate([np.sort(np.r_[b,e]) for b,e in zip(base,extra)]); labels=np.repeat(np.arange(c),[len(b)+len(e) for b,e in zip(base,extra)]);assert len(np.unique(rows))==len(rows)>0
        key=(rows.tobytes(),labels.tobytes())
        if key not in reuse: reuse[key]=_teacher(raw[rows],torch.from_numpy(labels).to(dev),val,c,dev)[0]
        runs[m]=reuse[key]; print(name,m,len(rows),round(time.monotonic()-start,1),'s',flush=True)
    y=d['audit_train_y'][idx];v=d['val_y'];a=[];pred=[];per=[]; stats={}; paired={}; audit=[]
    ct,st=(p[4] for p in pools)
    for j,nm in enumerate(d['classes']):
        cb,sb=(p[0][j] for p in pools); ca,sa=(x[(y==j)&(top==j)] for x,top in [(np.arange(n),ct),(np.arange(n),st)])
        audit.append(dict(class_id=j,name=nm,fit_true=int((y==j).sum()),clip_top1=int((ct==j).sum()),siglip_top1=int((st==j).sum()),clip_correct_top1=len(ca),siglip_correct_top1=len(sa),clip_unique_correct_top1=len(np.setdiff1d(ca,sa)),siglip_unique_correct_top1=len(np.setdiff1d(sa,ca)),clip_seed_count=len(cb),siglip_seed_count=len(sb),clip_rescue_candidates=len(cc[j]),siglip_rescue_candidates=len(sc[j]),matched_seeds=int(q[j]),matched_extras=int(h[j])))
    for m,(base,extra) in arms.items():
        pr=runs[m]; pred.extend(dict(arm=m,id=i,pred=int(p),truth=int(z)) for i,p,z in zip(vid,pr,v));count=correct=correct_extra=unique_correct=0
        for j,(b,e) in enumerate(zip(base,extra)):
            row=np.r_[b,e]; ok=y[row]==j; count+=len(row);correct+=int(ok.sum());correct_extra+=int((y[e]==j).sum());unique_correct+=int((ok&(ct[row]!=j)).sum())
            fine=d['metadata'].get('audit_train_fine_y'); diversity=len(set(np.asarray(fine)[idx][row[ok]])) if fine is not None and len(fine) else ''
            per.append(dict(arm=m,class_id=j,name=d['classes'][j],anchors=len(row),correct=int(ok.sum()),correct_fine_subtypes=diversity,val_n=int((v==j).sum()),val_correct=int(((v==j)&(pr==j)).sum())))
            for phase,rr in [('seed',b),('extra',e)]: a.extend(dict(arm=m,id=ids[i],pseudo=j,truth=int(y[i]),stage=phase,clip_top1=int(ct[i]),siglip_top1=int(st[i])) for i in rr)
        stats[m]=dict(anchors=count,correct_anchors=correct,correct_extras=correct_extra,correct_anchors_not_clip_top1=unique_correct,val_correct=int((pr==v).sum()),head_updates=EPOCHS*((count+BATCH_SIZE-1)//BATCH_SIZE))
    for b,a0 in [(SG,CL),(RR,CL),(SG,RR),('siglip_native','clip_native')]:paired[b+' vs '+a0]=compare(runs[a0],runs[b],v)
    out=HERE/('smoke' if smoke else name);out.mkdir(exist_ok=True)
    for f,rows in [('anchors',a),('predictions',pred),('by_class',per),('nomination_audit',audit)]: write(out/(f+'.csv'),rows)
    s=dict(dataset=name,smoke=smoke,stage='DINO teacher validation only',test_scored=False,train_n=n,val_n=len(v),k=k,cap=cap,score_gap=.06,geometry_weight=.5,scale=scale,matched_base=q.tolist(),matched_extra=h.tolist(),teacher=dict(seed=SEED,epochs=EPOCHS,batch=BATCH_SIZE,lr=LR,hidden=HIDDEN),siglip_rev=REV,siglip_weights_sha=SHA,siglip_cache_sha=cachehash.hexdigest(),feature_signature=d['metadata']['cache_signature'],code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),arms=stats,paired=paired)
    tmp=out/'summary.tmp';tmp.write_text(json.dumps(s,indent=2));tmp.replace(out/'summary.json');verify(out);print('VERIFIED',name,s['arms'],s['paired'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['eurosat','cifar100']);p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.dataset,a.smoke)
