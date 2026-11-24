"""Compare pinned CLIP and SigLIP 2 text banks and matched frozen DINO heads."""
import argparse,csv,hashlib,json,time
import numpy as np
import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader,Subset
from torchvision import transforms as T
from convnext_retrieval import HERE,ROOT,setup
from clustering_sweep import prepare,_allocate,_teacher,write
from anchor_inputs import get_dataset,_asset,_load_file_module,PINNED,_read_chunk,_atomic_npz

REV='75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'
SHA='612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b'

def scores(ds, key, splits, smoke):
    sig=key=='siglip2'; start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    if sig:
        from transformers import AutoTokenizer,SiglipModel
        root=HERE/'models/siglip2-base-patch16-224'
        with (root/'model.safetensors').open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==SHA
        tok=AutoTokenizer.from_pretrained(root,local_files_only=True);net=SiglipModel.from_pretrained(root,local_files_only=True,attn_implementation='eager').float().eval().requires_grad_(False).cuda()
        tf=T.Compose([T.Resize((224,224),interpolation=T.InterpolationMode.BILINEAR),T.ToTensor(),T.Normalize((.5,)*3,(.5,)*3)])
        bank=[]
        with torch.inference_mode():
            for cls in ds.classes:
                x=[]
                for a in range(0,len(ds.text_prompts[cls]),32):
                    p=list(ds.text_prompts[cls][a:a+32]); inp=tok(p,padding='max_length',max_length=64,truncation=True,return_tensors='pt'); x.append(normalize(net.get_text_features(input_ids=inp['input_ids'].cuda()).float(),dim=-1))
                bank.append(normalize(torch.cat(x).mean(0),dim=0))
        bank=torch.stack(bank)
    else:
        w=_asset('clip',HERE/'cache',ROOT/'weights/ViT-B-32.pt'); cm=_load_file_module('_siglip_compare_clip',PINNED/'clip/model.py');net=cm.build_model(torch.jit.load(str(w),map_location='cpu').state_dict()).float().eval().requires_grad_(False).cuda()
        bank=torch.from_numpy(ds.text_mean).cuda();tf=T.Compose([T.Resize(224,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    out={}
    for split,limit in splits.items():
        im=ds.dataset(split,tf,limit); ids=np.array([r[0] for r in im.rows]); parts=[]
        for j in range(0,len(ids),256):
            z=ids[j:j+256]; p=HERE/'siglip_scores'/REV/ds.name/('smoke' if smoke else 'full')/key/split/f'{j:06d}.npz'; cached=_read_chunk(p,z,{'scores':(len(z),len(ds.classes))})
            if cached is None:
                chunks=[]; i=j
                with torch.inference_mode():
                    for xs,names,_ in DataLoader(Subset(im,range(j,j+len(z))),batch_size=8,num_workers=0):
                        assert np.array_equal(names,ids[i:i+len(xs)]);i+=len(xs); xs=xs.cuda();f=net.get_image_features(pixel_values=xs) if sig else net.encode_image(xs); assert f.is_cuda
                        chunks.append((normalize(f.float(),dim=1)@bank.T).cpu().numpy())
                x=np.concatenate(chunks);assert i==j+len(z) and np.isfinite(x).all();_atomic_npz(p,ids=z,scores=x);cached={'scores':x}
            parts.append(cached['scores']);
            if j==0 or j//256%8==0 or j+len(z)==len(ids):print(key,ds.name,split,j+len(z),len(ids),'s',round(time.monotonic()-start,1),'peak',round(torch.cuda.max_memory_allocated()/2**20),flush=True)
        out[split]=(ids,np.concatenate(parts))
    return out

def run(name,smoke):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False; dev=torch.device('cuda')
    tr,va=(64,32) if smoke else (4096,1024) if name=='cifar100' else (None,None)
    ds=get_dataset(name,**setup(name)); d=prepare(name,**setup(name),weights_dir=ROOT/'weights',device='cuda',batch_size=8,workers=0,chunk_size=256,train_limit=tr,val_limit=va)
    cl=scores(ds,'clip',{'val':va},smoke);torch.cuda.empty_cache();sg=scores(ds,'siglip2',{'train':tr,'val':va},smoke);torch.cuda.empty_cache()
    assert np.array_equal(cl['val'][0],sg['val'][0]) and np.array_equal(d['val_ids'],cl['val'][0]) and np.array_equal(d['train_ids'],sg['train'][0]);assert not np.intersect1d(d['train_ids'],d['val_ids']).size
    ix=np.argsort(d['train_ids']); ids=d['train_ids'][ix];x=d['train_dino'][ix]; c=len(ds.classes); k=16 if smoke else len(x)//(5*c)
    pools={m:_allocate(z[ix],k,d['metadata']['clip_logit_scale']) for m,z in [('clip',d['train_clip_scores']),('siglip2',sg['train'][1])]}; quotas=np.minimum(pools['clip'][1],pools['siglip2'][1]); arms={}
    for m,z in [('clip',cl),('siglip2',sg)]:
        zz=z['val'][1].argmax(1)
        for mode,q in [('natural',pools[m][1]),('matched',quotas)]:
            a=np.concatenate([np.sort(r[:n]) for r,n in zip(pools[m][0],q)]);assert len(np.unique(a))==len(a);pseudo=np.repeat(np.arange(c),q)
            if mode=='matched' and np.array_equal(q,pools[m][1]): pred=arms[m+'_natural'][-1]
            else:pred,_=_teacher(torch.from_numpy(x[a]).to(dev),torch.from_numpy(pseudo).to(dev),torch.from_numpy(d['val_dino']),c,dev)
            arms[m+'_'+mode]=(a,pseudo,zz,pred);print('head completed',name,m,mode,len(a),flush=True)
    v=d['val_y']; y=d['audit_train_y'][ix];out=HERE/('siglip_comparison_smoke' if smoke else f'siglip_comparison_{name}');out.mkdir(exist_ok=True); ar=[];pr=[];by=[];summary={}
    for m,(a,pseudo,zz,p) in arms.items():
        summary[m]=dict(n_anchors=len(a),correct_anchors=int((pseudo==y[a]).sum()),zero_shot_correct=int((zz==v).sum()),head_correct=int((p==v).sum()))
        ar.extend(dict(arm=m,id=ids[i],pseudo=int(j),truth=int(y[i])) for i,j in zip(a,pseudo));pr.extend(dict(arm=m,id=i,zero=int(z),head=int(h),truth=int(t)) for i,z,h,t in zip(d['val_ids'],zz,p,v))
        by.extend(dict(arm=m,class_id=j,name=n,anchors=int((pseudo==j).sum()),correct_anchors=int(((pseudo==j)&(y[a]==j)).sum()),val_n=int((v==j).sum()),zero_correct=int(((zz==j)&(v==j)).sum()),head_correct=int(((p==j)&(v==j)).sum())) for j,n in enumerate(ds.classes))
    write(out/'anchors.csv',ar);write(out/'predictions.csv',pr);write(out/'by_class.csv',by)
    result=dict(dataset=name,smoke=smoke,train_n=len(y),val_n=len(v),test_scored=False,revision=REV,weight_sha=SHA,clip_weights_sha=d['metadata']['clip_weights_sha256'],cache_signature=d['metadata']['cache_signature'],source_sha=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),quotas=quotas.tolist(),per_class_target=k,shared_anchor_ranking_scale=d['metadata']['clip_logit_scale'],results=summary)
    tmp=out/'summary.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(out/'summary.json');print('COMPLETE',name,smoke,summary,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('dataset',choices=['eurosat','cifar100']);a.add_argument('--smoke',action='store_true');arg=a.parse_args();run(arg.dataset,arg.smoke)
