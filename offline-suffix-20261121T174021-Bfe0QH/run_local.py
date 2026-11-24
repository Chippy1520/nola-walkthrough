"""Offline, matched NoLA pilot using pinned CLIP/DINO and the released anchor selector."""
import argparse, hashlib, json, math, os, random, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
# NoLA's utils is a package and already exposes the trunc_normal_ DINO imports.
sys.path[:0] = [str(ROOT/'upstream'), str(ROOT/'dino-source'), str(ROOT/'.deps')]
from clip import clip
from vision_transformer import vit_base
from utils.data_utils import ds_specific_templates
from utils.model_utils import get_random_transform, top_k_indices_per_class2

SEED = 7777
MEAN, STD = (.48145466,.4578275,.40821073), (.26862954,.26130258,.27577711)
WEAK = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
ARMS = ('baseline','suffix')
CLIP_HASH = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
DINO_HASH = 'bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e'

def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def rev(path):
    return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()

def write(path, data):
    temp = path.with_suffix(path.suffix+'.part')
    temp.write_text(json.dumps(data,indent=2)+'\n'); temp.replace(path)

def checkpoint(path, obj):
    tmp = path.with_suffix('.part'); torch.save(obj,tmp); tmp.replace(path)

def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)

class Images(Dataset):
    def __init__(self, rows, source, tx): self.rows,self.source,self.tx = rows,source,tx
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        key,y,_ = self.rows[i]
        if isinstance(self.source,Path):
            with Image.open(self.source/key) as f: im=f.convert('RGB')
        else:
            part,j=key.split('/'); im=Image.fromarray(self.source[part].data[int(j)])
        return self.tx(im),i

def inputs(args):
    if args.dataset=='eurosat':
        base=ROOT/'data/eurosat'; split=base/'split_zhou_EuroSAT.json'; source=base/'EuroSAT_RGB'
        digest=sha(split); assert digest=='541503a77d0b26b735d85ec7a3e672804fe0aa466c2e5f07da772cc6c0af8ad2'
        rows=json.loads(split.read_text()); ids=[r[0] for v in rows.values() for r in v]
        assert len(set(ids))==len(ids)==27000 and all((source/key).is_file() for key in ids)
        by_id={y:name for v in rows.values() for _,y,name in v}; names=[by_id[i] for i in range(10)]
        assert all(by_id[y]==name for v in rows.values() for _,y,name in v)
        return rows,source,names,'EuroSAT',dict(split=str(split),split_sha256=digest,image_root=str(source))
    if args.cifar_root is None: raise SystemExit('CIFAR-10 cache not provided; set --cifar-root to the directory containing cifar-10-batches-py (no downloads).')
    train,test=[torchvision.datasets.CIFAR10(str(args.cifar_root),train=p,download=False) for p in (True,False)]
    names=train.classes; assert names==test.classes and len(train)==50000 and len(test)==10000
    rows={'train':[], 'val':[], 'test':[[f'test/{i}',y,names[y]] for i,y in enumerate(test.targets)]}
    for y in range(10):
        ids=[i for i,k in enumerate(train.targets) if k==y]; random.Random(SEED+y).shuffle(ids)
        for part,js in [('val',ids[:500]),('train',ids[500:])]: rows[part].extend([[f'train/{j}',y,names[y]] for j in js])
    return rows,{'train':train,'test':test},names,'CIFAR10_local',dict(image_root=str(args.cifar_root),split='official test; 500/class from official train reserved for validation',batch_sha256={p.name:sha(p) for p in sorted((args.cifar_root/'cifar-10-batches-py').iterdir()) if p.is_file()})

def subset(rows, n, s):
    if n==0 or n>=len(rows): return rows
    assert n%10==0
    rng=random.Random(s)
    ids=sorted(i for c in range(10) for i in rng.sample([j for j,r in enumerate(rows) if r[1]==c],n//10))
    return [rows[i] for i in ids]

def prompted(model,x,p):
    v=model.visual; x=v.conv1(x).flatten(2).transpose(1,2)
    x=torch.cat((v.class_embedding.expand(len(x),1,-1).to(x.dtype),x),1)
    x=v.ln_pre(x+v.positional_embedding.to(x.dtype))
    if p is not None: x=torch.cat((x[:,:1],p.expand(len(x),-1,-1).to(x.dtype),x[:,1:]),1)
    x=v.transformer(x.transpose(0,1)).transpose(0,1)
    return v.ln_post(x[:,0])@v.proj

def accuracy(p,y):
    assert p.shape==y.shape
    return {'correct':int((p==y).sum()),'n':y.numel(),'accuracy':float((p==y).double().mean())}

def paired(p,q,y):
    a,b=p==y,q==y
    return dict(fixed=int((b&~a).sum()),lost=int((a&~b).sum()),changed=int((p!=q).sum()),delta_pp=100*int(b.sum()-a.sum())/len(y))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['eurosat','cifar10'],required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--cifar-root',type=Path); ap.add_argument('--train',type=int,default=540); ap.add_argument('--val',type=int,default=200); ap.add_argument('--test',type=int,default=300)
    ap.add_argument('--epochs',type=int,default=2); ap.add_argument('--microbatch',type=int,default=8); ap.add_argument('--effective-batch',type=int,default=32); ap.add_argument('--feature-batch',type=int,default=16)
    a=ap.parse_args(); assert a.effective_batch%a.microbatch==0 and a.epochs>0
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available(); seed(SEED)
    out=a.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    t0=time.monotonic()
    def event(stage,**kw):
        d=dict(stage=stage,elapsed_s=round(time.monotonic()-t0,1),**kw); msg=json.dumps(d); print(msg,flush=True)
        with (out/'events.jsonl').open('a') as f: f.write(msg+'\n')
    event('start',pid=os.getpid(),gpu=torch.cuda.get_device_name(),free_bytes=torch.cuda.mem_get_info()[0]); rows,source,names,tag,provenance=inputs(a)
    rows={k:subset(v,getattr(a,k),SEED+i) for i,(k,v) in enumerate(rows.items())}; y={k:torch.tensor([r[1] for r in v]) for k,v in rows.items()}
    assert len({r[0] for v in rows.values() for r in v})==sum(map(len,rows.values()))
    k=max(1,math.floor(.2*len(rows['train'])/len(names)))
    weights=ROOT/'weights'; cpath=weights/'ViT-B-32.pt'; dpath=weights/'dino_vitbase16_pretrain.pth'
    assert sha(cpath)==CLIP_HASH and sha(dpath)==DINO_HASH
    dp=ROOT/'upstream/descriptions/generic'/f'{tag}.json'; desc=json.loads(dp.read_text()); templates=ds_specific_templates[tag]; assert set(names)==set(desc)
    texts={arm:{} for arm in ARMS}
    for name in names:
        templates_c=[t.format(name) for t in templates]
        texts['baseline'][name]=desc[name]+templates_c
        texts['suffix'][name]=[f'{d} {t}' for d in desc[name] for t in templates_c]
    write(out/'manifest.json',rows); write(out/'prompts.json',texts)
    cfg=dict(argv=sys.argv,seed=SEED,counts={k:len(v) for k,v in rows.items()},anchors_per_class=k,provenance=provenance,classes=names,templates=templates,descriptions_sha256=sha(dp),manifest_sha256=sha(out/'manifest.json'),prompts_sha256=sha(out/'prompts.json'),source_sha256=sha(Path(__file__)),upstream=rev(ROOT/'upstream'),dino_commit=rev(ROOT/'dino-source'),clip_sha256=CLIP_HASH,dino_sha256=DINO_HASH,gpu=torch.cuda.get_device_name(),torch=torch.__version__,torchvision=torchvision.__version__,precision='FP32 masters with CUDA FP16 autocast and student GradScaler',feature_batch=a.feature_batch,epochs=a.epochs,microbatch=a.microbatch,effective_batch=a.effective_batch,teacher=dict(widths=[768,256,10],epochs=50,batch=32,optimizer='Adam(0.001)'),student=dict(visual_tokens=16,trainable='visual LayerNorms + visual prompts + class head',optimizer='AdamW(0.004, betas=(0.9,0.999), decay=.01 except LayerNorm bias)',smoothing=.1),checkpoint='first highest validation accuracy; fixed training epochs in both arms')
    write(out/'config.json',cfg)
    def loader(part,tx=WEAK,batch=a.feature_batch,shuffle_seed=None):
        return DataLoader(Images(rows[part],source,tx),batch_size=batch,shuffle=shuffle_seed is not None,generator=torch.Generator().manual_seed(SEED if shuffle_seed is None else shuffle_seed),num_workers=2,pin_memory=True)
    def amp(): return torch.autocast('cuda',dtype=torch.float16)
    jit=torch.jit.load(str(cpath),map_location='cpu'); model=clip.build_model(jit.state_dict()).float().eval().cuda().requires_grad_(False); del jit
    bank={}; cfeats={}
    with torch.inference_mode():
        for arm in ARMS:
            vectors=[]
            for name in names:
                tokens=clip.tokenize(texts[arm][name],truncate=False)
                z=[]
                for tok in tokens.split(16):
                    with amp(): z.append(model.encode_text(tok.cuda()).float())
                vectors.append(F.normalize(F.normalize(torch.cat(z),dim=-1).mean(0),dim=0))
            bank[arm]=torch.stack(vectors).cpu()
            event('text_encoded',arm=arm,counts=[len(texts[arm][c]) for c in names])
        for part in rows:
            z=[]
            for x,_ in loader(part):
                with amp(): z.append(F.normalize(model.encode_image(x.cuda(non_blocking=True)).float(),dim=-1).cpu())
            cfeats[part]=torch.cat(z); assert cfeats[part].isfinite().all()
            event('clip_cached',split=part,n=len(cfeats[part]))
    scale=model.logit_scale.detach().exp().item(); model.cpu(); torch.cuda.empty_cache()
    cde={arm:{part:(x@bank[arm].T).argmax(1) for part,x in cfeats.items()} for arm in ARMS}
    checkpoint(out/'cde.pt',dict(bank=bank,features=cfeats,pred=cde))
    dino=vit_base(patch_size=16,num_classes=0); dino.load_state_dict(torch.load(dpath,map_location='cpu',weights_only=True),strict=True); dino.eval().requires_grad_(False).cuda()
    features={}
    with torch.inference_mode():
        for part in rows:
            z=[]
            for x,_ in loader(part):
                with amp(): z.append(dino(x.cuda(non_blocking=True)).float().cpu())
            features[part]=torch.cat(z); assert features[part].isfinite().all()
            event('dino_cached',split=part,n=len(features[part]))
    del dino; torch.cuda.empty_cache()
    checkpoint(out/'dino_features.pt',features)
    teachers={}; targets={}; d=features['train'].cuda()
    for arm in ARMS:
        prob=(scale*(cfeats['train']@bank[arm].T)).softmax(1)
        ids,_,labels=top_k_indices_per_class2(dict(total_emb=prob,idxs=torch.arange(len(d)),labels=y['train']),k)
        assert len(ids.unique())==len(ids) and len(ids)>0
        seed(SEED); g=torch.Generator().manual_seed(SEED)
        head=nn.Sequential(nn.Linear(768,256),nn.ReLU(True),nn.Linear(256,10)).cuda(); opt=torch.optim.Adam(head.parameters(),lr=.001)
        updates=0
        for _ in range(50):
            for ix in torch.randperm(len(ids),generator=g).split(32):
                opt.zero_grad(set_to_none=True); loss=F.cross_entropy(head(d[ids[ix]]),labels[ix].cuda()); loss.backward(); opt.step(); updates+=1
        assert loss.isfinite()
        with torch.no_grad(): pred={p:head(f.cuda()).argmax(1).cpu() for p,f in features.items()}
        targets[arm]=pred['train']
        metrics=dict(anchors=accuracy(labels,y['train'][ids]),anchors_per_class=torch.bincount(labels,minlength=10).tolist(),updates=updates,val=accuracy(pred['val'],y['val']),test=accuracy(pred['test'],y['test']))
        teachers[arm]=dict(pred=pred,ids=ids,labels=labels,head={n:p.detach().cpu() for n,p in head.state_dict().items()},metrics=metrics)
        checkpoint(out/f'{arm}-teacher.pt',teachers[arm]); event('teacher_done',arm=arm,**metrics)
    del head,opt,d,loss
    model.cuda(); params={n:p for n,p in model.named_parameters() if n.startswith('visual.') and 'ln' in n}
    for p in params.values(): p.requires_grad_(True)
    initial={n:p.detach().cpu().clone() for n,p in params.items()}; strong=get_random_transform(224)
    @torch.inference_mode()
    def predict(part,p,w):
        ans=[]
        for x,_ in loader(part):
            with amp(): f=prompted(model,x.cuda(non_blocking=True),p)
            ans.append((f.float()@w.T).argmax(1).cpu())
        return torch.cat(ans)
    def initialize(arm):
        seed(SEED)
        with torch.no_grad():
            for n,p in params.items(): p.copy_(initial[n])
        width=model.visual.class_embedding.numel(); bound=math.sqrt(6/(3*16*16+width))
        p=nn.Parameter(torch.empty(1,16,width).uniform_(-bound,bound).cuda()); w=nn.Parameter(bank[arm].clone().cuda())
        opt=torch.optim.AdamW([dict(params=[p,w,*[v for n,v in params.items() if 'bias' not in n]],weight_decay=.01),dict(params=[v for n,v in params.items() if 'bias' in n],weight_decay=0.)],lr=.004,betas=(.9,.999))
        return p,w,opt,torch.amp.GradScaler('cuda',init_scale=512.)
    def restore(state,p,w):
        with torch.no_grad():
            p.copy_(state['prompt']); w.copy_(state['weight'])
            for n,v in params.items(): v.copy_(state['norm'][n])
    latest={}; best={}
    for epoch in range(1,a.epochs+1):
        for arm in ARMS:
            ep_t=time.monotonic(); p,w,opt,scaler=initialize(arm)
            if arm in latest:
                old=torch.load(latest[arm],map_location='cpu',weights_only=True); restore(old,p,w); opt.load_state_dict(old['opt']); scaler.load_state_dict(old['scaler']); del old
            n=updates=skips=0; losses=[]; opt.zero_grad(set_to_none=True)
            event('student_epoch_start',arm=arm,epoch=epoch)
            for x,ix in loader('train',strong,a.microbatch,SEED+epoch):
                group=min(a.effective_batch,len(y['train'])-(n//a.effective_batch)*a.effective_batch)
                with amp(): f=prompted(model,x.cuda(non_blocking=True),p)
                loss=F.cross_entropy(f.float()@w.T,targets[arm][ix].cuda(),label_smoothing=.1,reduction='sum')/group
                scaler.scale(loss).backward(); n+=len(ix); losses.append(loss.detach())
                if n%a.effective_batch==0 or n==len(y['train']):
                    before=scaler.get_scale(); scaler.step(opt); scaler.update(); skips+=int(scaler.get_scale()<before); updates+=1; opt.zero_grad(set_to_none=True)
                if n%128==0: event('student_progress',arm=arm,epoch=epoch,images=n,n_train=len(y['train']),updates=updates)
            assert n==len(y['train']) and updates==math.ceil(n/a.effective_batch) and skips<updates and torch.stack(losses).isfinite().all() and p.isfinite().all() and w.isfinite().all()
            val=predict('val',p,w); score=accuracy(val,y['val']); improved=arm not in best or score['correct']>best[arm][0]
            path=out/f'{arm}-epoch-{epoch:02d}.pt'
            state=dict(epoch=epoch,prompt=p.detach().cpu(),weight=w.detach().cpu(),norm={n:v.detach().cpu() for n,v in params.items()},val=val,opt=opt.state_dict(),scaler=scaler.state_dict(),metrics=score,updates=updates,amp_skips=skips)
            checkpoint(path,state); latest[arm]=path
            if improved: best[arm]=(score['correct'],path)
            event('student_epoch_done',arm=arm,epoch=epoch,val=score,updates=updates,amp_skips=skips,seconds=round(time.monotonic()-ep_t,1),peak_mib=round(torch.cuda.max_memory_allocated()/2**20))
            del p,w,opt,scaler,state
    final={}; preds={}
    for arm in ARMS:
        p,w,_,_=initialize(arm); st=torch.load(best[arm][1],map_location='cpu',weights_only=True); restore(st,p,w)
        val=predict('val',p,w); assert torch.equal(val,st['val']), 'Checkpoint validation replay mismatch'
        test=predict('test',p,w); preds[arm]=test
        final[arm]=dict(cde={part:accuracy(pr,y[part]) for part,pr in cde[arm].items()},teacher=teachers[arm]['metrics'],student=dict(best_epoch=st['epoch'],checkpoint=best[arm][1].name,val=accuracy(val,y['val']),test=accuracy(test,y['test'])))
    report=dict(arms=final,paired_test=dict(cde=paired(cde['baseline']['test'],cde['suffix']['test'],y['test']),teacher=paired(teachers['baseline']['pred']['test'],teachers['suffix']['pred']['test'],y['test']),student=paired(preds['baseline'],preds['suffix'],y['test'])))
    write(out/'predictions.json',[dict(id=r[0],label=r[1],**{f'{arm}_{stage}':int(z[i]) for arm in ARMS for stage,z in [('cde',cde[arm]['test']),('teacher',teachers[arm]['pred']['test']),('student',preds[arm])]}) for i,r in enumerate(rows['test'])]); write(out/'results.json',report); event('done',**report)

if __name__=='__main__': main()
