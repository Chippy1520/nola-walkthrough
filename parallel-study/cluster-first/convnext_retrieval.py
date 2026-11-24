"""Pinned DINOv3 ConvNeXt-Tiny retrieval; replay the existing spectral-study controls."""
import argparse,csv,hashlib,json,time
from collections import Counter
import numpy as np
import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader,Subset
from clustering_sweep import HERE,prepare,_allocate,_teacher,write,seed_scores
from anchor_inputs import get_dataset,_read_chunk,_atomic_npz

SHA='a7637753a2e85d709d1a5474e13382d39e7bc581a35868b477585c970f36847f'
REV='39c7bb32f83c11bbb13cb23058f8852a4dac88ed'
MODEL='convnext_tiny.dinov3_lvd1689m'
ROOT=HERE.parent.parent

def setup(name):
    ec=name=='eurosat'
    return dict(work_dir=HERE/'cache',archive=ROOT/'data/EuroSAT_RGB.zip' if ec else HERE.parent/'cifar100/.cache/cifar-100-python.tar.gz')

def extract(name,smoke):
    import timm
    from safetensors.torch import load_file
    weights=HERE/'models'/f'{MODEL}.safetensors'
    with weights.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==SHA
    net=timm.create_model(MODEL,pretrained=False,num_classes=0)
    net.load_state_dict(load_file(str(weights)),strict=True);net.eval().requires_grad_(False).cuda()
    cfg=timm.data.resolve_model_data_config(net);print(name,'model',MODEL,'transform',cfg,flush=True)
    tf=timm.data.create_transform(**cfg,is_training=False)
    ds=get_dataset(name,**setup(name)); limit=64 if smoke else None if name=='eurosat' else 4096
    im=ds.dataset('train',tf,limit);ids=np.array([r[0] for r in im.rows]); start=time.monotonic()
    base=HERE/'convnext_features'/SHA/name/('smoke' if smoke else 'train'); arrays=[]
    for a in range(0,len(im),256):
        z=ids[a:a+256];path=base/f'{a:06d}.npz'; data=_read_chunk(path,z,{'features':(len(z),768)})
        if data is None:
            pos=a; chunks=[]
            with torch.inference_mode():
                for image, names,_ in DataLoader(Subset(im,range(a,a+len(z))),batch_size=16,num_workers=0):
                    assert np.array_equal(names,ids[pos:pos+len(image)]);pos+=len(image)
                    result=net(image.cuda());assert result.is_cuda and result.shape==(len(image),768)
                    chunks.append(result.float().cpu().numpy())
            f=np.concatenate(chunks);assert pos==a+len(z) and np.isfinite(f).all() and (np.linalg.norm(f,axis=1)>0).all()
            _atomic_npz(path,ids=z,features=f);data={'features':f}
        arrays.append(data['features']);
        if a==0 or a//256%10==0 or a+len(z)==len(ids):print(name,'features',a+len(z),len(ids),'elapsed',round(time.monotonic()-start,1),'peak MiB',round(torch.cuda.max_memory_allocated()/2**20),flush=True)
    return ids,np.concatenate(arrays),cfg

def run(name,smoke):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;dev=torch.device('cuda'); ec=name=='eurosat'
    ids,v3,cfg=extract(name,smoke);limits=(64,32) if smoke else (None,None) if ec else (4096,1024)
    d=prepare(name,**setup(name),weights_dir=ROOT/'weights',device='cuda',batch_size=8,workers=0,chunk_size=256,train_limit=limits[0],val_limit=limits[1]); assert np.array_equal(d['train_ids'],ids)
    ix=np.argsort(ids);ids=ids[ix];s=d['train_clip_scores'][ix]; x=d['train_dino'][ix];c=len(d['classes']); k=16 if smoke else len(ids)//(5*c);cap=8 if smoke else min(64,max(8,k//2))
    pools,bn,_,lp,_=_allocate(s,k,d['metadata']['clip_logit_scale']);base=[p[:q] for p,q in zip(pools,bn)];used=np.isin(np.arange(len(ids)),np.concatenate(base));under=np.flatnonzero(bn<k)
    if not len(under):
        under=np.argsort(np.bincount(s.argmax(1),minlength=c))[:2];bn[under]=max(1,k//10)
        for j in under:base[j]=base[j][:bn[j]]
    own=under[lp[:,under].argmax(1)];gap=s.max(1)-s[np.arange(len(ids)),own]; cand=[np.flatnonzero((own==j)&~used&(gap<=.06)) for j in range(c)];quota=np.minimum(np.minimum(k-bn,cap),list(map(len,cand)))
    geom={'clip_expansion':None,'dino_seed_retrieval':normalize(torch.from_numpy(x).to(dev),dim=1),'convnext_v3':normalize(torch.from_numpy(v3[ix]).to(dev),dim=1)}
    selected={}
    for arm,g in geom.items():
        rows=[]
        for j,(r,q) in enumerate(zip(cand,quota)):
            z=lp[r,j].copy()
            if q and g is not None:z+=.5*seed_scores(g,base,j,r)
            rows.append(r[np.argsort(-z,kind='stable')][:q])
        selected[arm]=rows
    ref=HERE/('spectral_smoke' if smoke else f'spectral_{name}'); sm=json.loads((ref/'summary.json').read_text());assert sm['base_budgets']==bn.tolist() and sm['add_quotas']==quota.tolist()
    assert sm['metadata']['cache_signature']==d['metadata']['cache_signature']
    oldrows=list(csv.DictReader((ref/'anchors.csv').open())); oldpred=list(csv.DictReader((ref/'predictions.csv').open()))
    for arm in ['clip_expansion','dino_seed_retrieval']:
        actual={(r['id'],int(r['pseudo']),int(r['added'])) for r in oldrows if r['arm']==arm}
        rebuilt={(ids[i],j,a) for j in range(c) for chunk,a in [(base[j],0),(selected[arm][j],1)] for i in chunk};assert actual==rebuilt,arm
    yy=np.repeat(np.arange(c),bn+quota);rr=np.concatenate([np.sort(np.r_[p,q]) for p,q in zip(base,selected['convnext_v3'])]);assert len(rr)==len(np.unique(rr)); assert not np.intersect1d(ids,d['val_ids']).size
    pr,_=_teacher(torch.from_numpy(x[rr]).to(dev),torch.as_tensor(yy,device=dev),torch.from_numpy(d['val_dino']),c,dev)
    v=d['val_y'];y=d['audit_train_y'][ix]; id_val=dict(zip(d['val_ids'],v));out=HERE/('convnext_smoke' if smoke else f'convnext_{name}');out.mkdir(exist_ok=True)
    anchors=[r for r in oldrows if r['arm'] in geom]; preds=[r for r in oldpred if r['arm'] in geom]; sub=dict(zip(ids,d['metadata'].get('audit_train_fine_y',[]) and np.asarray(d['metadata']['audit_train_fine_y'])[ix])) if not ec else {}
    for j,rows in enumerate(selected['convnext_v3']):
        for chunk,a in [(base[j],0),(rows,1)]:
            for i in chunk:
                rank=1+int((s[i]>s[i,j]).sum()) if a else '';anchors.append(dict(arm='convnext_v3',id=ids[i],pseudo=j,truth=int(y[i]),added=a,target_rank=rank))
    preds.extend(dict(arm='convnext_v3',id=i,pred=int(p),truth=int(t)) for i,p,t in zip(d['val_ids'],pr,v))
    result={};hits={}
    for arm in geom:
        a=[r for r in anchors if r['arm']==arm];w=[r for r in preds if r['arm']==arm]; good=[r for r in a if int(r['pseudo'])==int(r['truth'])]; ext=[r for r in a if int(r['added'])]; assert len({r['id'] for r in a})==len(yy) and {r['id']:int(r['truth']) for r in w}==id_val and len(w)==len(v);assert Counter(int(r['pseudo']) for r in ext)==Counter({j:int(q) for j,q in enumerate(quota) if q})
        hits[arm]={r['id']:int(r['pred'])==int(r['truth']) for r in w}; result[arm]=dict(correct_additions=sum(int(r['pseudo'])==int(r['truth']) for r in ext),val_correct=sum(hits[arm].values()),target_classes={d['classes'][j]:dict(new_correct=sum(int(r['pseudo'])==int(r['truth'])==j for r in ext),val_correct=sum(int(r['pred'])==int(r['truth'])==j for r in w),val_n=int((v==j).sum()),subtypes=sorted({sub[r['id']] for r in good if int(r['pseudo'])==j}) if sub else None) for j in under.tolist()})
    paired={}
    for arm in ['clip_expansion','dino_seed_retrieval']:
        delta=np.array([int(hits['convnext_v3'][i])-int(hits[arm][i]) for i in d['val_ids']]);rng=np.random.default_rng(42);boot=np.zeros(10000,int)
        for j in np.unique(v):
            zz=delta[v==j]; m=rng.multinomial(len(zz),np.bincount(zz+1,minlength=3)/len(zz),size=len(boot));boot+=m[:,2]-m[:,0]
        paired[arm]=dict(delta_pp=100*delta.mean(),ci95_pp=(100*np.quantile(boot,[.025,.975])/len(v)).tolist())
    write(out/'anchors.csv',anchors);write(out/'predictions.csv',preds)
    summary=dict(dataset=name,smoke=smoke,model=MODEL,revision=REV,sha256=SHA,transform=cfg,train_n=len(ids),val_n=len(v),matched_additions=int(sum(quota)),reference=str(ref),add_quotas=quota.tolist(),results=result,paired=paired);tmp=out/'summary.tmp';tmp.write_text(json.dumps(summary,indent=2));tmp.replace(out/'summary.json');print('Completed',name,smoke,result,paired,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('dataset',choices=['eurosat','cifar100']);a.add_argument('--smoke',action='store_true');args=a.parse_args();run(args.dataset,args.smoke)
