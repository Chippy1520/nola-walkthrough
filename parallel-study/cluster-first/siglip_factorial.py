"""2x2 rescue ranking: text scores x image retrieval, on unchanged baseline pools."""
import argparse,csv,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import normalize
from compare_siglip import scores,REV,SHA
from convnext_retrieval import HERE,ROOT,setup,SHA as CONV_SHA
from clustering_sweep import prepare,_allocate,_teacher,write,seed_scores
from anchor_inputs import get_dataset,_read_chunk
from verify_siglip_factorial import verify

def run(name,smoke):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;dev=torch.device('cuda'); ec=name=='eurosat'
    tr,va=(64,32) if smoke else (None,None) if ec else (4096,1024)
    ds=get_dataset(name,**setup(name)); d=prepare(name,**setup(name),weights_dir=ROOT/'weights',device='cuda',batch_size=8,workers=0,chunk_size=256,train_limit=tr,val_limit=va)
    ref=HERE/('convnext_smoke' if smoke else f'convnext_{name}'); orig=json.loads((ref/'summary.json').read_text()); reference={m:orig['results'][m] for m in ['dino_seed_retrieval','convnext_v3']}
    cid=d['train_ids'];cv=[]
    for i in range(0,len(cid),256):
        z=cid[i:i+256];r=_read_chunk(HERE/'convnext_features'/CONV_SHA/name/('smoke' if smoke else 'train')/f'{i:06d}.npz',z,{'features':(len(z),768)});assert r is not None;cv.append(r['features'])
    sid,sg=scores(ds,'siglip2',{'train':tr},smoke)['train'];assert np.array_equal(cid,sid);torch.cuda.empty_cache()
    ix=np.argsort(cid);ids=cid[ix];cs=d['train_clip_scores'][ix];sg=sg[ix];x=d['train_dino'][ix];c=len(ds.classes); k=16 if smoke else len(x)//(5*c);cap=8 if smoke else min(64,max(8,k//2));scale=d['metadata']['clip_logit_scale']
    pools,bn,_,lc,_=_allocate(cs,k,scale);base=[p[:j] for p,j in zip(pools,bn)];used=np.isin(np.arange(len(ids)),np.concatenate(base));under=np.flatnonzero(bn<k);natural=bool(len(under))
    if not len(under):
        under=np.argsort(np.bincount(cs.argmax(1),minlength=c))[:2];bn[under]=max(1,k//10)
        for j in under:base[j]=base[j][:bn[j]]
    owner=under[lc[:,under].argmax(1)];gap=cs.max(1)-cs[np.arange(len(ids)),owner];cand=[np.flatnonzero((owner==j)&(~used)&(gap<=.06)) for j in range(c)];quota=np.minimum(np.minimum(k-bn,cap),list(map(len,cand)));assert quota.tolist()==orig['add_quotas']
    ss=(sg.astype(np.float64)*scale); ls=ss-np.logaddexp.reduce(ss,axis=1,keepdims=True)
    geo={'dino':normalize(torch.from_numpy(x).to(dev),dim=1),'convnext':normalize(torch.from_numpy(np.concatenate(cv)[ix]).to(dev),dim=1)}
    olda=list(csv.DictReader((ref/'anchors.csv').open()));oldv=list(csv.DictReader((ref/'predictions.csv').open()));vid=d['val_ids']; assert not np.intersect1d(vid,ids).size
    preds={};picked={}
    for g,features in geo.items():
        bonuses=[.5*seed_scores(features,base,j,rows) if q else np.empty(0) for j,(rows,q) in enumerate(zip(cand,quota))]
        for bank,l in [('clip',lc),('siglip2',ls)]:
            arm=f'{bank}+{g}'; rows=[r[np.argsort(-(l[r,j]+bonuses[j]),kind='stable')][:q] if q else np.empty(0,int) for j,(r,q) in enumerate(zip(cand,quota))];picked[arm]=rows
            if bank=='clip':
                ctl='dino_seed_retrieval' if g=='dino' else 'convnext_v3';expected={(r['id'],int(r['pseudo']),int(r['added'])) for r in olda if r['arm']==ctl};actual={(ids[i],j,ext) for j in range(c) for rows0,ext in [(base[j],0),(rows[j],1)] for i in rows0};assert actual==expected,arm
                p={r['id']:int(r['pred']) for r in oldv if r['arm']==ctl};assert p.keys()==set(vid);preds[arm]=np.array([p[i] for i in vid]);assert int((preds[arm]==d['val_y']).sum())==reference[ctl]['val_correct']
            else:
                r=np.concatenate([np.sort(np.r_[v,w]) for v,w in zip(base,rows)]);assert len(np.unique(r))==len(r);yy=np.repeat(np.arange(c),bn+quota)
                preds[arm],_=_teacher(torch.from_numpy(x[r]).to(dev),torch.from_numpy(yy).to(dev),torch.from_numpy(d['val_dino']),c,dev)
            print('Arm complete',name,arm,flush=True)
    out=HERE/('siglip_factorial_smoke' if smoke else f'siglip_factorial_{name}');out.mkdir(exist_ok=True);a=[];v=[];crows=[];results={};y=d['audit_train_y'][ix];vy=d['val_y'];fine=d['metadata'].get('audit_train_fine_y');fine=np.asarray(fine)[ix] if fine else None
    for arm,rows in picked.items():
        p=preds[arm];v.extend(dict(arm=arm,id=i,pred=int(j),truth=int(z)) for i,j,z in zip(vid,p,vy));results[arm]=dict(correct_additions=int(sum((y[r]==j).sum() for j,r in enumerate(rows))),val_correct=int((p==vy).sum()))
        for j,(old,new) in enumerate(zip(base,rows)):
            a.extend(dict(arm=arm,id=ids[i],pseudo=j,truth=int(y[i]),added=q) for r,q in [(old,0),(new,1)] for i in r)
            if not quota[j]: continue
            good=np.r_[old,new];good=good[y[good]==j];crows.append(dict(arm=arm,name=ds.classes[j],added_correct=int((y[new]==j).sum()),val_correct=int(((p==j)&(vy==j)).sum()),val_n=int((vy==j).sum()),subtypes=';'.join(sorted(set(fine[good]))) if fine is not None else ''))
    assert crows; write(out/'anchors.csv',a);write(out/'predictions.csv',v);write(out/'by_class.csv',crows)
    report=dict(dataset=name,smoke=smoke,train_n=len(ids),val_n=len(vy),natural_scarcity=natural,quota=quota.tolist(),base_budgets=bn.tolist(),candidate_counts=list(map(len,cand)),text_scale=scale,checkpoint_sha=SHA,convnext_sha=CONV_SHA,source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),test_scored=False,results=results);tmp=out/'summary.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(out/'summary.json');verify(out);print('COMPLETE',name,smoke,results,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('dataset',choices=['eurosat','cifar100']);a.add_argument('--smoke',action='store_true');q=a.parse_args();run(q.dataset,q.smoke)
