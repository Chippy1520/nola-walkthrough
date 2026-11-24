"""Full 2×2 independent VLM nomination × visual retriever. Labels are audit-only."""
import argparse, csv, hashlib, itertools, json, time
import numpy as np
import torch
from run import HERE, CF, REV, SHA, ROOT, setup, prepare, _read_chunk, _allocate, _teacher, seed_scores, write, normalize, compare, EPOCHS, SEED, LR, BATCH_SIZE
from convnext_retrieval import SHA as CONV_SHA

VLM=('clip','siglip2'); RET=('dino','convnext')
ARMS=[f'{v}+{g}' for g in RET for v in VLM]

def read(p):
    with p.open() as f: return list(csv.DictReader(f))

def verify(out):
    s=json.loads((out/'summary.json').read_text()); a=read(out/'anchors.csv'); p=read(out/'predictions.csv'); seeds={}; vals={}
    for key,v in s['arms'].items():
        b=[r for r in a if r['arm']==key]; q=[r for r in p if r['arm']==key]; assert len(b)==len({r['id'] for r in b})==v['anchors']; assert len(q)==len({r['id'] for r in q})==s['val_n']; assert not ({r['id'] for r in b}&{r['id'] for r in q})
        for typ,quota in [('seed',s['matched_seed']),('extra',s['matched_extra'])]:
            assert np.bincount([int(r['pseudo']) for r in b if r['stage']==typ],minlength=len(quota)).tolist()==quota
        assert sum(r['pseudo']==r['truth'] for r in b)==v['correct_anchors']; assert sum(r['pseudo']==r['truth'] and r['stage']=='extra' for r in b)==v['correct_extra']; assert sum(r['pred']==r['truth'] for r in q)==v['correct']
        seeds[key]={(r['id'],r['pseudo']) for r in b if r['stage']=='seed'};vals[key]=q
    for m in VLM: assert seeds[m+'+dino']==seeds[m+'+convnext']
    for k,d in s['paired'].items():
        b,a=k.split(' vs '); x,y=vals[a],vals[b]; assert [(r['id'],r['truth']) for r in x]==[(r['id'],r['truth']) for r in y]
        f=sum(q['pred']==q['truth'] and r['pred']!=r['truth'] for r,q in zip(x,y)); l=sum(q['pred']!=q['truth'] and r['pred']==r['truth'] for r,q in zip(x,y)); assert (f,l)==(d['fixes'],d['losses'])
    (out/'verified.json').write_text(json.dumps(s,indent=2)+'\n'); return s

def run(ds,smoke):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;dev=torch.device('cuda'); tr,va=(64,32) if smoke else (4096,1024) if ds=='cifar100' else (None,None)
    d=prepare(ds,**setup(ds),weights_dir=ROOT/'weights',device='cuda',batch_size=8,workers=0,chunk_size=256,train_limit=tr,val_limit=va)
    raw_ids=d['train_ids']; ix=np.argsort(raw_ids);ids=raw_ids[ix]; vid=d['val_ids'];n=len(ix);c=len(d['classes']);k=16 if smoke else n//(5*c);cap=8 if smoke else min(64,max(8,k//2));assert not np.intersect1d(ids,vid).size
    cachepaths=[CF/'siglip_scores'/REV/ds/('smoke' if smoke else 'full')/'siglip2/train',CF/'convnext_features'/CONV_SHA/ds/('smoke' if smoke else 'train')]
    arrays=[];hashes=[]
    for root,field,width in zip(cachepaths,['scores','features'],[c,768]):
        parts=[];digest=hashlib.sha256()
        for i in range(0,n,256):
            pid=raw_ids[i:i+256];p=root/f'{i:06d}.npz';r=_read_chunk(p,pid,{field:(len(pid),width)});assert r is not None,p;parts.append(r[field]);digest.update(p.read_bytes())
        arrays.append(np.concatenate(parts)[ix]);hashes.append(digest.hexdigest())
    sig,conv=arrays; raw=torch.from_numpy(d['train_dino'][ix]).to(dev); geo=dict(dino=normalize(raw,dim=1),convnext=normalize(torch.from_numpy(conv).to(dev),dim=1)); s=dict(clip=d['train_clip_scores'][ix],siglip2=sig); scale=d['metadata']['clip_logit_scale'];nom={m:_allocate(s[m],k,scale) for m in VLM}
    prior=CF/f'siglip_factorial_{ds}'; cfg=None if smoke else json.loads((prior/'summary.json').read_text()); stress=([i for i,b in enumerate(cfg['base_budgets']) if b<k] if ds=='cifar100' and not smoke else [])
    full={m:[v[:b] for v,b in zip(nom[m][0],nom[m][1])] for m in VLM};q=np.minimum(nom['clip'][1],nom['siglip2'][1]);q[stress]=np.minimum(q[stress],max(1,k//10));under=np.array([int(q.argmax())]) if smoke else np.flatnonzero(q<k); gap_cap=2.01 if smoke else .06
    if smoke: q[under]=1
    base={m:[v[:b] for v,b in zip(full[m],q)] for m in VLM}; cand={}
    for m in VLM:
        # In the scarcity stress test, reserve removed natural anchors from this model's rescue pool.
        used=np.isin(np.arange(n),np.concatenate(full[m] if stress else base[m])); scores=s[m]
        if len(under):
            owner=under[scores[:,under].argmax(1)];gap=scores.max(1)-scores[np.arange(n),owner];cand[m]=[np.flatnonzero((owner==j)&(~used)&(gap<=gap_cap)) for j in range(c)]
        else: cand[m]=[np.empty(0,dtype=int) for _ in range(c)]
    h=np.minimum.reduce([k-q,np.full(c,cap),*[np.array(list(map(len,cand[m]))) for m in VLM]]);assert h.sum()>0; print(ds,'seed',sum(q),'additional',sum(h),'stress classes',stress,flush=True)
    fits={};selected={}; parity={}; val=torch.from_numpy(d['val_dino']); start=time.monotonic()
    p_a=[] if smoke else read(prior/'anchors.csv'); p_v=[] if smoke else read(prior/'predictions.csv')
    for g,m in itertools.product(RET,VLM):
        key=m+'+'+g; chosen=[]
        for j,(rows,b) in enumerate(zip(cand[m],h)):
            z=nom[m][3][rows,j]+.5*seed_scores(geo[g],base[m],j,rows) if b else np.empty(0);chosen.append(rows[np.argsort(-z,kind='stable')[:b]] if b else rows[:0])
        selected[key]=(base[m],chosen); rr=np.concatenate([np.sort(np.r_[a,b]) for a,b in zip(base[m],chosen)]); assert len(np.unique(rr))==len(rr)==sum(q+h)
        fits[key]=_teacher(raw[rr],torch.from_numpy(np.repeat(np.arange(c),q+h)).to(dev),val,c,dev)[0]
        if m=='clip' and not smoke:
            past={(r['id'],int(r['pseudo']),int(r['added'])) for r in p_a if r['arm']==key}; cur={(ids[i],j,t) for j,(a,b) in enumerate(zip(base[m],chosen)) for rr,t in [(a,0),(b,1)] for i in rr}; p={r['id']:int(r['pred']) for r in p_v if r['arm']==key}; parity[key]=dict(anchor_ids_equal=cur==past,predictions_equal=np.array_equal(fits[key],[p[i] for i in vid]));
            if parity[key]['anchor_ids_equal']: assert parity[key]['predictions_equal']
        print(ds,key,round(time.monotonic()-start,1),'s',flush=True)
    y=d['audit_train_y'][ix];vy=d['val_y']; arows=[]; prows=[]; per=[]; arms={}
    for key,(base,ext) in selected.items():
        pr=fits[key]; prows.extend(dict(arm=key,id=i,pred=int(p),truth=int(y)) for i,p,y in zip(vid,pr,vy)); arms[key]=dict(correct=int((pr==vy).sum()),anchors=int(sum(q+h)),correct_anchors=0,correct_extra=0)
        for j,(b,e) in enumerate(zip(base,ext)):
            c0=int((y[b]==j).sum());c1=int((y[e]==j).sum());arms[key]['correct_anchors']+=c0+c1;arms[key]['correct_extra']+=c1
            for ph,r in [('seed',b),('extra',e)]: arows.extend(dict(arm=key,id=ids[i],pseudo=j,truth=int(y[i]),stage=ph) for i in r)
            per.append(dict(arm=key,class_id=j,name=d['classes'][j],correct_seeds=c0,correct_extra=c1,val_correct=int(((pr==j)&(vy==j)).sum()),val_n=int((vy==j).sum())))
    contrasts={f'{b} vs {a}':compare(fits[a],fits[b],vy) for a,b in itertools.combinations(ARMS,2)}
    out=HERE/'factorial'/('smoke' if smoke else ds);out.mkdir(parents=True,exist_ok=True)
    for k0,r in [('anchors',arows),('predictions',prows),('by_class',per)]:write(out/(k0+'.csv'),r)
    rpt=dict(dataset=ds,smoke=smoke,train_n=n,val_n=len(vy),arms=arms,paired=contrasts,stage='original-DINO teacher validation',teacher=dict(epochs=EPOCHS,seed=SEED,batch=BATCH_SIZE,lr=LR,updates=int(EPOCHS*((sum(q+h)+BATCH_SIZE-1)//BATCH_SIZE))),test_scored=False,matched_seed=q.tolist(),matched_extra=h.tolist(),raw_seed_counts={m:nom[m][1].tolist() for m in VLM},matched_candidate_counts={m:list(map(len,cand[m])) for m in VLM},cifar_simulated_scarcity_classes=stress,parity_previous_clip_controls=parity,feature_signature=d['metadata']['cache_signature'],siglip_rev=REV,siglip_sha=SHA,convnext_sha=CONV_SHA,siglip_cache_sha=hashes[0],convnext_cache_sha=hashes[1],code_sha=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),clip_logit_scale=scale,cosine_gap=.06,retrieval_weight=.5)
    rpt['cosine_gap']=gap_cap
    (out/'summary.json').write_text(json.dumps(rpt,indent=2)+'\n');verify(out);print('VERIFIED',ds,smoke,arms,parity,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['eurosat','cifar100']);p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.dataset,a.smoke)
