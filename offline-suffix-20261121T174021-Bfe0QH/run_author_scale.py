# 1. NoLA workspace and verified download helper — no training.
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, os, shutil, subprocess, sys, zipfile
import torch

ROOT = Path('/content/nola_research')
DATA, WEIGHTS, SOURCE = (ROOT / p for p in ('data', 'weights', 'source'))
for p in (DATA, WEIGHTS, SOURCE): p.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Runtime:', torch.cuda.get_device_name() if DEVICE.type == 'cuda' else 'CPU', '| PyTorch:', torch.__version__)

def digest(path, kind='sha256'):
    with path.open('rb') as f: return hashlib.file_digest(f, kind).hexdigest()

def fetch(url, dest, expected, kind='sha256'):
    if dest.is_file() and digest(dest, kind) == expected:
        return dest
    tmp = dest.with_suffix(dest.suffix + '.part')
    tmp.unlink(missing_ok=True)
    p = subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','2','--connect-timeout','15','--max-time','150','--output',str(tmp),url],capture_output=True,text=True,timeout=170)
    if p.returncode: raise RuntimeError(f'{dest.name}: {p.stderr}')
    if digest(tmp, kind) != expected:
        tmp.unlink(missing_ok=True)
        raise ValueError(f'Checksum mismatch: {dest.name}')
    tmp.replace(dest)
    return dest
print('Workspace:', ROOT)

# 2. Download the full EuroSAT RGB archive, then verify the published CoOp split.
from collections import Counter
from pathlib import PurePosixPath

EUROSAT_MD5 = 'f46e308c4d50d4bf32fedad2d3d62f3b'
SPLIT_SHA256 = '541503a77d0b26b735d85ec7a3e672804fe0aa466c2e5f07da772cc6c0af8ad2'
archive = fetch('https://zenodo.org/api/records/7711810/files/EuroSAT_RGB.zip/content', DATA/'EuroSAT_RGB.zip', EUROSAT_MD5, 'md5')
split_file = DATA/'split_zhou_EuroSAT.json'
if not split_file.is_file() or digest(split_file) != SPLIT_SHA256:
    import importlib.util
    if importlib.util.find_spec('gdown') is None:
        subprocess.run([sys.executable,'-m','pip','-q','install','gdown'],check=True,timeout=60)
    import gdown
    temp = split_file.with_suffix('.part')
    gdown.download(id='1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o',output=str(temp),quiet=True)
    assert temp.is_file() and digest(temp) == SPLIT_SHA256, 'Published split hash mismatch'
    temp.replace(split_file)
SPLITS = json.loads(split_file.read_text())
expected = {'train': 13500, 'val': 5400, 'test': 8100}
assert {k:len(v) for k,v in SPLITS.items()} == expected
paths = [x[0] for rows in SPLITS.values() for x in rows]
assert len(paths) == len(set(paths)) == 27000
class_map = {int(label):name for rows in SPLITS.values() for _,label,name in rows}
assert set(class_map) == set(range(10)) and all(class_map[int(label)] == name for rows in SPLITS.values() for _,label,name in rows)
CLASS_NAMES = [class_map[i] for i in range(10)]
with zipfile.ZipFile(archive) as z:
    members = [n for n in z.namelist() if n.lower().endswith('.jpg')]
    mapping = {'/'.join(PurePosixPath(n).parts[-2:]): n for n in members}
    assert len(mapping) == len(members) == 27000 and set(mapping) == set(paths)
    IMAGE_DIR = DATA/'images'
    for rel, member in mapping.items():
        relpath = PurePosixPath(rel)
        assert len(relpath.parts) == 2 and not relpath.is_absolute() and '..' not in relpath.parts
        dest = IMAGE_DIR/rel
        if not dest.is_file() or dest.stat().st_size != z.getinfo(member).file_size:
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(z.read(member))
assert {str(p.relative_to(IMAGE_DIR)) for p in IMAGE_DIR.glob('*/*.jpg')} == set(paths)
print('EuroSAT:',expected,'| 27,000 unique images across',len(CLASS_NAMES),'classes')
print('Images:',IMAGE_DIR,'| Split:',split_file)

# 3. Download and verify the two frozen backbone checkpoints.
CLIP_SHA256 = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
DINO_SHA256 = 'bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e'
weights = [
    (f'https://openaipublic.azureedge.net/clip/models/{CLIP_SHA256}/ViT-B-32.pt', WEIGHTS/'ViT-B-32.pt', CLIP_SHA256),
    ('https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth', WEIGHTS/'dino_vitbase16_pretrain.pth', DINO_SHA256),
]
with ThreadPoolExecutor(max_workers=2) as pool:
    CLIP_WEIGHTS, DINO_WEIGHTS = list(pool.map(lambda args: fetch(*args), weights))
for p in (CLIP_WEIGHTS,DINO_WEIGHTS): print(p.name, f'{p.stat().st_size:,} bytes', 'SHA-256 verified')

# 4. Smoke-test checkpoint loading and one real image on both backbones.
from PIL import Image
from torchvision import transforms as T

DINO_REV = '7c446df5b9f45747937fb0d72314eb9f7b66930a'
DINO_SOURCE = SOURCE/'dino'
DINO_SOURCE.mkdir(exist_ok=True)
for name in ('vision_transformer.py','utils.py'):
    p = DINO_SOURCE/name
    if not p.is_file():
        base = f'https://raw.githubusercontent.com/facebookresearch/dino/{DINO_REV}/'
        subprocess.run(['curl','-fsSL','--max-time','30','-o',str(p.with_suffix('.part')),base+name],check=True,timeout=35)
        p.with_suffix('.part').replace(p)
sys.path.insert(0,str(DINO_SOURCE))
from vision_transformer import vit_base

def image_tensor(path, mean, std):
    tx = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean,std)])
    with Image.open(path) as img:
        return tx(img.convert('RGB')).unsqueeze(0).to(DEVICE)

sample = IMAGE_DIR/SPLITS['train'][0][0]
# Load CLIP's official TorchScript checkpoint on its native device, then check the visual encoder.
clip_jit = torch.jit.load(str(CLIP_WEIGHTS),map_location=DEVICE).eval()
dino_model = vit_base(patch_size=16,num_classes=0)
dino_model.load_state_dict(torch.load(DINO_WEIGHTS,map_location='cpu',weights_only=True),strict=True)
dino_model = dino_model.eval().requires_grad_(False).to(DEVICE)
with torch.inference_mode():
    d = dino_model(image_tensor(sample,(.485,.456,.406),(.229,.224,.225)))
    if DEVICE.type == 'cuda':
        c = clip_jit.encode_image(image_tensor(sample,(.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711)))
        assert c.shape == (1,512) and torch.isfinite(c).all()
        print('CLIP forward:',tuple(c.shape),c.device)
    else:
        print('CLIP checkpoint loaded; TorchScript forward requires GPU here.')
    assert d.shape == (1,768) and torch.isfinite(d).all()
    print('DINO forward:',tuple(d.shape),d.device)
print('Setup complete. No hypotheses run; no test predictions computed.')

# Author-scale settings: released optimizer choices; locked 0.75 bank.
import io, math, random, importlib.util
import numpy as np
import pandas as pd
from PIL import ImageFilter
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
INPUT=ROOT/'NoLA-EuroSAT-Author-Inputs.zip'
assert digest(INPUT)=='1c2648a6cda77a0e634c3d1651650fbf01e8bba3d825aebd6ec0b625e0a3963c'
OUT=ROOT/'author_scale_075_v1'; OUT.mkdir(exist_ok=True)
SEED=7777; EPOCHS=50; EFFECTIVE=512; MICRO=32; K=270
TRAIN=SPLITS['train']; VAL=SPLITS['val']; TEST=SPLITS['test']
assert len(TRAIN)==13500 and K==math.floor(.2*len(TRAIN)/10)
with zipfile.ZipFile(INPUT) as z:
    broad=torch.load(io.BytesIO(z.read('embeddings.pt')),weights_only=True)['broad'].to(DEVICE)
    bank=torch.load(io.BytesIO(z.read('prototypes.pt')),weights_only=True)
assert bank['class_order']==CLASS_NAMES and broad.shape==(10,512)
proto=torch.cat(bank['vectors']).to(DEVICE); tau=bank['tau']; sizes=bank['sizes']
assert proto.shape==(23,512) and all(sum(s)==25 for s in sizes)
modpath=fetch('https://raw.githubusercontent.com/openai/CLIP/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/model.py', SOURCE/'openai_clip_model.py','9902cbe5ee90a1da2aa3e6f043e8a23dc1f8831193b963785c9af03d5c7bef2c')
spec=importlib.util.spec_from_file_location('author_clip',modpath); cm=importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)
clip=cm.build_model({k:v.detach().cpu() for k,v in clip_jit.state_dict().items()}).float().eval().to(DEVICE)
for n,p in clip.named_parameters():p.requires_grad_(n.startswith('visual.') and 'ln' in n)
params={n:p for n,p in clip.named_parameters() if p.requires_grad}; init={n:p.detach().cpu().clone() for n,p in params.items()}
mean=(.48145466,.4578275,.40821073); std=(.26862954,.26130258,.27577711)
weak=T.Compose([T.Resize(224,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize(mean,std)])
class Blur:
    def __call__(self,x):return x.filter(ImageFilter.GaussianBlur(random.uniform(.1,2)))
strong=T.Compose([T.RandomResizedCrop(224,scale=(.2,1.)),T.RandomHorizontalFlip(),T.RandomApply([T.ColorJitter(.4,.4,.4,.1)],p=.8),T.RandomPerspective(distortion_scale=.2,p=.5,interpolation=T.InterpolationMode.BICUBIC),T.ColorJitter(.4,.4,.4,.1),Blur(),T.ToTensor(),T.Normalize(mean,std)])
class Images(Dataset):
    def __init__(self,rows,tx):self.rows=rows; self.tx=tx
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(IMAGE_DIR/self.rows[i][0]) as im:return self.tx(im.convert('RGB')),i

def prompted(x,p=None):
    v=clip.visual; x=v.conv1(x).flatten(2).transpose(1,2)
    x=torch.cat((v.class_embedding.expand(len(x),1,-1).to(x.dtype),x),1)
    x=v.ln_pre(x+v.positional_embedding.to(x.dtype))
    if p is not None:x=torch.cat((x[:,:1],p.expand(len(x),-1,-1).to(x.dtype),x[:,1:]),1)
    x=v.transformer(x.transpose(0,1)).transpose(0,1)
    return v.ln_post(x[:,0])@v.proj

def scores(x,w,multi,raw=False):
    if not multi:return x@w.T
    norm=x.norm(dim=1,keepdim=True).clamp_min(1e-8) if raw else 1
    s=((x/norm)@w.T)/tau
    out=torch.stack([tau*(torch.logsumexp(a+torch.tensor(q,device=x.device).log(),dim=1)-math.log(25)) for a,q in zip(s.split([len(q) for q in sizes],dim=1),sizes)],dim=1)
    return out*norm

def loader(rows,tx,b,seed=None):
    g=torch.Generator().manual_seed(SEED if seed is None else seed)
    return DataLoader(Images(rows,tx),batch_size=b,shuffle=seed is not None,generator=g,num_workers=4,pin_memory=True)

def select(probs):
    conf,classes=probs.cpu().topk(3,dim=1); selected={}; ranks={}
    for c in range(10):
        for r in range(3):
            ix=torch.where(classes[:,r]==c)[0].numpy()
            if len(ix):
                ranks[c]=[r+1,len(ix)]
                order=pd.Series(conf[ix,r].numpy(),index=ix).sort_values(ascending=False).index[:K]
                for j in order:selected.setdefault(int(j),c)
                break
        else:raise RuntimeError(f'No top-3 candidate for {CLASS_NAMES[c]}')
    return torch.tensor(list(selected)),torch.tensor(list(selected.values())),ranks

with torch.no_grad():
    z=next(iter(loader(TRAIN[:4],weak,4)))[0].to(DEVICE)
    torch.testing.assert_close(prompted(z),clip.encode_image(z),rtol=1e-6,atol=1e-6)
print('Verified banks and prompted forward on real images:',len(TRAIN),len(VAL),len(TEST),K,tau)

# Author-scale runner: cache weak-view features, fit both teachers, then alternate the two students.
import time, traceback
from sklearn.metrics import confusion_matrix
Y={k:torch.tensor([r[1] for r in rows]) for k,rows in [('train',TRAIN),('val',VAL),('test',TEST)]}
H=CLASS_NAMES.index('Herbaceous Vegetation Land')

def atomic(obj,path):
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); tmp.replace(path)
def info(**d):
    print(json.dumps(d),flush=True)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')

@torch.no_grad()
def make_teachers():
    path=OUT/'teachers.pt'
    if path.exists():return torch.load(path,weights_only=True)
    all_c=[]; all_d=[]; ev_d=[]; t0=time.monotonic()
    for x,_ in loader(TRAIN,weak,64):
        x=x.to(DEVICE,non_blocking=True)
        all_c.append(F.normalize(clip_jit.encode_image(x).float(),dim=1).cpu())
        with torch.autocast('cuda',dtype=torch.float16):all_d.append(dino_model(x).float().cpu())
    for x,_ in loader(VAL,weak,64):
        with torch.autocast('cuda',dtype=torch.float16):ev_d.append(dino_model(x.to(DEVICE,non_blocking=True)).float().cpu())
    X=torch.cat(all_c).to(DEVICE); D=torch.cat(all_d).to(DEVICE); Dv=torch.cat(ev_d).to(DEVICE)
    assert X.shape==(13500,512) and D.shape==(13500,768) and Dv.shape==(5400,768) and D.isfinite().all()
    info(stage='cached_backbones',seconds=round(time.monotonic()-t0,1))
    result={}
    for name,multi,w in [('original',False,broad),('merged075',True,proto)]:
        prob=(clip_jit.logit_scale.exp().float()*scores(X,w,multi)).softmax(dim=1)
        ids,labs,ranks=select(prob); assert len(ids)==len(ids.unique()) and len(ids)<=2700
        torch.manual_seed(SEED); order=torch.Generator().manual_seed(SEED)
        head=nn.Sequential(nn.Linear(768,256),nn.ReLU(True),nn.Linear(256,10)).to(DEVICE)
        opt=torch.optim.Adam(head.parameters(),lr=.001)
        with torch.enable_grad():
            for e in range(50):
                for ix in torch.randperm(len(ids),generator=order).split(32):
                    opt.zero_grad(set_to_none=True); loss=F.cross_entropy(head(D[ids[ix]]),labs[ix].to(DEVICE)); loss.backward(); opt.step()
        targets=head(D).argmax(1).cpu(); vp=head(Dv).argmax(1).cpu(); correct=Y['train'][ids]==labs
        metrics=dict(anchors=len(ids),anchor_correct=int(correct.sum()),herb_anchors=int((labs==H).sum()),herb_anchors_correct=int((correct&(labs==H)).sum()),dino_val_correct=int((vp==Y['val']).sum()),dino_herb_correct=int(((vp==Y['val'])&(vp==H)).sum()))
        result[name]=dict(targets=targets,ids=ids,labels=labs,head={k:v.cpu() for k,v in head.state_dict().items()},metrics=metrics,ranks=ranks)
        info(stage='teacher_done',arm=name,**metrics)
    atomic(result,path)
    return result

def init_student(multi):
    torch.manual_seed(SEED)
    with torch.no_grad():
        for n,p in params.items():p.copy_(init[n])
    width=clip.visual.class_embedding.numel(); bound=math.sqrt(6/(3*16*16+width))
    p=nn.Parameter(torch.empty(1,16,width).uniform_(-bound,bound).to(DEVICE)); w=nn.Parameter((proto if multi else broad).clone())
    opt=torch.optim.AdamW([dict(params=[p,w,*[v for n,v in params.items() if 'bias' not in n]],weight_decay=.01),dict(params=[v for n,v in params.items() if 'bias' in n],weight_decay=0.)],lr=.004,betas=(.9,.999))
    scaler=torch.amp.GradScaler('cuda',init_scale=512.)
    return p,w,opt,scaler

def restore(state,p,w,opt=None,scaler=None):
    with torch.no_grad():
        p.copy_(state['prompt']); w.copy_(state['weight'])
        for n,v in params.items():v.copy_(state['norm'][n])
    if opt is not None:opt.load_state_dict(state['opt']); scaler.load_state_dict(state['scaler'])

@torch.inference_mode()
def evaluate(rows,p,w,multi):
    preds=[]
    for x,_ in loader(rows,weak,64):
        with torch.autocast('cuda',dtype=torch.float16): f=prompted(x.to(DEVICE,non_blocking=True),p)
        preds.append(scores(f.float(),w,multi,True).argmax(1).cpu())
    return torch.cat(preds)

def one_epoch(name,multi,e,targets,smoke=False):
    t0=time.monotonic(); p,w,opt,scaler=init_student(multi); current=OUT/f'{name}-latest.pt'
    old=torch.load(current,weights_only=True) if current.exists() and not smoke else None
    if old is not None:restore(old,p,w,opt,scaler)
    assert smoke or e==(old['epoch']+1 if old else 1)
    rows=TRAIN[:548] if smoke else TRAIN
    opt.zero_grad(set_to_none=True); n=0; updates=0; skips=0; losses=[]
    for x,ix in loader(rows,strong,MICRO,SEED+e):
        group_n=min(EFFECTIVE,len(rows)-(n//EFFECTIVE)*EFFECTIVE)
        with torch.autocast('cuda',dtype=torch.float16):f=prompted(x.to(DEVICE,non_blocking=True),p)
        loss=F.cross_entropy(scores(f.float(),w,multi,True),targets[ix].to(DEVICE),label_smoothing=.1,reduction='sum')/group_n
        scaler.scale(loss).backward(); n+=len(ix); losses.append(loss.detach())
        if n%EFFECTIVE==0 or n==len(rows):
            scale=scaler.get_scale(); scaler.step(opt); scaler.update(); skips+=int(scaler.get_scale()<scale)
            opt.zero_grad(set_to_none=True); updates+=1
    assert n==len(rows) and updates==math.ceil(n/EFFECTIVE) and skips<updates
    assert torch.isfinite(torch.stack(losses)).all() and torch.isfinite(w).all()
    v=evaluate(VAL[:64] if smoke else VAL,p,w,multi); truth=Y['val'][:len(v)]; correct=int((v==truth).sum())
    state=dict(epoch=e,prompt=p.detach().cpu(),weight=w.detach().cpu(),norm={k:v.detach().cpu() for k,v in params.items()},opt=opt.state_dict(),scaler=scaler.state_dict(),pred=v,best=max(correct,old['best'] if old else -1),stale=0 if old is None or correct>old['best'] else old['stale']+1)
    if not smoke:
        if state['stale']==0:atomic(state,OUT/f'{name}-best.pt')
        atomic(state,current)
    info(stage='smoke_done' if smoke else 'epoch_done',arm=name,epoch=e,n_train=n,updates=updates,amp_skips=skips,val_correct=correct,val_n=len(v),best_val_correct=state['best'],stale=state['stale'],seconds=round(time.monotonic()-t0,1))

def main():
    cfg=dict(seed=SEED,train=len(TRAIN),val=len(VAL),test=len(TEST),anchors_per_class=K,dino_epochs=50,dino_batch=32,clip_max_epochs=EPOCHS,effective_batch=EFFECTIVE,microbatch=MICRO,clip_lr=.004,prec='AMP FP16 T4',checkpoint='best val; early stop after 15 subsequent non-improvements',threshold=.75,tau=tau,split_sha256=SPLIT_SHA256,banks_sha256=digest(INPUT))
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    arms=[('original',False),('merged075',True)]
    smoke_targets=(torch.arange(len(TRAIN))%10).long()
    for name,m in arms:one_epoch(name,m,1,smoke_targets,True)
    teachers=make_teachers()
    for e in range(1,EPOCHS+1):
        for name,m in arms:
            pth=OUT/f'{name}-latest.pt'; old=torch.load(pth,weights_only=True) if pth.exists() else None
            if old is not None and (old['epoch']>=e or old['stale']>=15):continue
            one_epoch(name,m,e,teachers[name]['targets'])
    final={}
    for name,m in arms:
        p,w,_,_=init_student(m); st=torch.load(OUT/f'{name}-best.pt',weights_only=True); restore(st,p,w)
        val=evaluate(VAL,p,w,m); assert torch.equal(val,st['pred'])
        test=evaluate(TEST,p,w,m); assert len(test)==8100
        final[name]=dict(epoch=st['epoch'],val_correct=int((val==Y['val']).sum()),test_correct=int((test==Y['test']).sum()),herb_correct=int(((test==Y['test'])&(test==H)).sum()),herb_n=int((Y['test']==H).sum()),herb_predictions=int((test==H).sum()),confusion=confusion_matrix(Y['test'],test,labels=list(range(10))).tolist(),test_pred=test.tolist())
    (OUT/'results.json').write_text(json.dumps(final)+'\n')
    info(stage='done',test_n=8100,**{k:{n:v[n] for n in ('epoch','val_correct','test_correct','herb_correct','herb_n')} for k,v in final.items()})

if __name__ == "__main__":
    main()
