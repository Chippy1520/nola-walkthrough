"""CLIP and SigLIP 2 visual-prompt students on verified independent-nomination teachers."""
import argparse,csv,hashlib,itertools,json,math,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms as T
from factorial import ARMS, read
from run import HERE,CF,ROOT,REV,SHA,setup,prepare,compare,write,EPOCHS,SEED,BATCH_SIZE,LR,HIDDEN
from anchor_inputs import get_dataset,PINNED,_load_file_module
sys.path.insert(0,str(CF/'.siglip-deps'))
CLIPSHA='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
MEAN=(.48145466,.4578275,.40821073); STD=(.26862954,.26130258,.27577711)
SE=7777

def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def jsonsave(p,data):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)
def save(p,data):
    tmp=p.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(p)
def seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s)
def rank(i):return hashlib.sha256(f'{SE}:{i}'.encode()).digest()

def teacher(train,y,val,dt,c):
    # Same initialization, update order, architecture and math as archived _teacher.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED); h=nn.Sequential(nn.Linear(train.shape[1],HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,c)).cuda()
    opt=torch.optim.Adam(h.parameters(),lr=LR); gen=torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(train),generator=gen).split(BATCH_SIZE):
            idx=idx.cuda();opt.zero_grad(set_to_none=True);F.cross_entropy(h(train[idx]),y[idx]).backward();opt.step()
    h.eval()
    with torch.inference_mode():
        def pred(x):return torch.cat([h(b.cuda()).softmax(1).max(1).indices.cpu() for b in x.split(512)])
        return pred(val),pred(dt)

class Images(Dataset):
    def __init__(self,ds,ids,other,tf):self.src=ds._source;self.ids=list(ids);self.other=other;self.tf=tf
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):return self.tf(self.src.image(self.ids[i])),self.other[i]

class Student(nn.Module):
    def __init__(self,kind,ds):
        super().__init__(); self.kind=kind; self.sig=kind=='siglip2'
        if self.sig:
            from transformers import AutoTokenizer,SiglipModel
            path=CF/'models/siglip2-base-patch16-224';assert digest(path/'model.safetensors')==SHA
            tok=AutoTokenizer.from_pretrained(path,local_files_only=True); model=SiglipModel.from_pretrained(path,local_files_only=True,attn_implementation='eager').float().cuda().eval().requires_grad_(False)
            vectors=[]
            with torch.no_grad():
                for k in ds.classes:
                    vectors_c=[]; texts=ds.text_prompts[k]
                    for j in range(0,len(texts),32):
                        t=tok(list(texts[j:j+32]),padding='max_length',max_length=64,truncation=True,return_tensors='pt')['input_ids'].cuda(); vectors_c.append(F.normalize(model.get_text_features(input_ids=t).float(),dim=-1))
                    vectors.append(F.normalize(torch.cat(vectors_c).mean(0),dim=0))
                bank=torch.stack(vectors)
            self.visual=model.vision_model; width=self.visual.config.hidden_size; patch=16
        else:
            path=ROOT/'weights/ViT-B-32.pt';assert digest(path)==CLIPSHA; cm=_load_file_module('_student_clip',PINNED/'clip/model.py')
            model=cm.build_model(torch.jit.load(str(path),map_location='cpu').state_dict()).float().cuda().eval().requires_grad_(False);self.visual=model.visual
            bank=torch.as_tensor(ds.text_mean.copy(),device='cuda');width=self.visual.class_embedding.numel();patch=32
        del model
        for m in self.visual.modules():
            if isinstance(m,nn.LayerNorm): m.requires_grad_(True)
        seed(SE); bound=math.sqrt(6/(3*patch*patch+width));self.prompt=nn.Parameter(torch.empty(1,16,width).uniform_(-bound,bound).cuda());self.bank=nn.Parameter(bank.detach().clone());self.eval()
    def encode(self,x,prompt=True):
        v=self.visual
        if self.sig:
            x=v.embeddings(x)
            if prompt: x=torch.cat((self.prompt.expand(len(x),-1,-1).to(x.dtype),x),dim=1)
            return v.head(v.post_layernorm(v.encoder(inputs_embeds=x).last_hidden_state))
        x=v.conv1(x).flatten(2).transpose(1,2);x=torch.cat((v.class_embedding.expand(len(x),1,-1).to(x.dtype),x),dim=1);x=v.ln_pre(x+v.positional_embedding.to(x.dtype))
        if prompt: x=torch.cat((x[:,:1],self.prompt.expand(len(x),-1,-1).to(x.dtype),x[:,1:]),dim=1)
        return v.ln_post(v.transformer(x.transpose(0,1)).transpose(0,1)[:,0])@v.proj
    def forward(self,x,prompt=True):
        z=self.encode(x,prompt);return 100.*(F.normalize(z.float(),dim=-1)@F.normalize(self.bank.float(),dim=-1).T)
    def fitparams(self):return {k:v for k,v in self.named_parameters() if v.requires_grad}
    def check_native(self,x):
        with torch.inference_mode():
            y=self.visual(pixel_values=x).pooler_output if self.sig else self.visual(x)
            torch.testing.assert_close(self.encode(x,False),y,rtol=1e-5,atol=1e-6)

@torch.inference_mode()
def predict(net,loader,prompt=True):
    out=[]
    for x,_ in loader:
        with torch.autocast('cuda',dtype=torch.float16): z=net(x.cuda(),prompt)
        assert torch.isfinite(z).all();out.append(z.argmax(1).cpu())
    return torch.cat(out)

def main(a):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    a.out.mkdir(parents=True,exist_ok=a.resume); t0=time.monotonic(); src=HERE/'factorial'/a.dataset; cfg=json.loads((src/'verified.json').read_text()); tr,vl=(4096,1024) if a.dataset=='cifar100' else (None,None)
    data=prepare(a.dataset,**setup(a.dataset),weights_dir=ROOT/'weights',device='cuda',batch_size=8,workers=0,chunk_size=256,train_limit=tr,val_limit=vl);ds=get_dataset(a.dataset,**setup(a.dataset));assert data['metadata']['cache_signature']==cfg['feature_signature']
    tid=data['train_ids'];vid=data['val_ids'];vy=data['val_y']; tm={i:j for j,i in enumerate(tid)}; assert not np.intersect1d(tid,vid).size
    ti=np.array(sorted(sorted(range(len(tid)),key=lambda j:rank(tid[j]))[:a.train])) if a.train else np.arange(len(tid))
    vi=np.array(sorted(j for c in range(len(ds.classes)) for j in sorted(np.flatnonzero(vy==c),key=lambda j:rank(vid[j]))[:a.val_per_class])) if a.val_per_class else np.arange(len(vid))
    y=torch.from_numpy(vy[vi]); assert len(ti)==(a.train or len(tid)) and len(vi)==(len(ds.classes)*a.val_per_class if a.val_per_class else len(vid))
    meta=dict(dataset=a.dataset,smoke=a.smoke,epochs=a.epochs,train_n=len(ti),val_n=len(vi),train_full=len(tid),val_full=len(vid),steps=math.ceil(len(ti)/a.effective),micro=a.micro,effective=a.effective,seed=SE,prompts=16,logit_scale=100.,objective='CE on hard DINO targets, label smoothing .1',loss_lr=.004,teacher_cfg_sha=digest(src/'verified.json'),code_sha=digest(Path(__file__)),test_scored=False)
    split=dict(train=tid[ti].tolist(),val=vid[vi].tolist());cpath=a.out/'config.json';spath=a.out/'split.json'
    parent=None
    if a.extend_from:
        parent=json.loads((a.extend_from/'verified.json').read_text()); assert parent['epochs']<a.epochs
        assert json.loads((a.extend_from/'split.json').read_text())==split
        assert all(parent[k]==v for k,v in meta.items() if k not in ('epochs','code_sha')), 'continuation configuration mismatch'
        meta['continued_from']=str(a.extend_from.resolve());meta['parent_metrics_sha']=digest(a.extend_from/'verified.json')
        for arm in ARMS:
            assert digest(a.extend_from/parent['arms'][arm]['checkpoint'])==parent['arms'][arm]['sha256']
    if a.resume:
        assert json.loads(cpath.read_text())==meta and json.loads(spath.read_text())==split, 'resume configuration or split mismatch'
    else:jsonsave(cpath,meta);jsonsave(spath,split)
    ac=read(src/'anchors.csv');pv=read(src/'predictions.csv');dt=torch.from_numpy(data['train_dino'][ti]);val=torch.from_numpy(data['val_dino']); targets={}; baselines={}
    for arm in ARMS:
        rows=sorted((r for r in ac if r['arm']==arm),key=lambda r:(int(r['pseudo']),r['id'])); ti0=[tm[r['id']] for r in rows]
        v,targets[arm]=teacher(torch.from_numpy(data['train_dino'][ti0]).cuda(),torch.tensor([int(r['pseudo']) for r in rows],device='cuda'),val,dt,len(ds.classes)); exp={r['id']:int(r['pred']) for r in pv if r['arm']==arm}; assert np.array_equal(v.numpy(),[exp[i] for i in vid]),arm;baselines[arm]=v[vi]
    if parent:
        with np.load(a.extend_from/'teacher-targets.npz',allow_pickle=False) as ref:
            np.testing.assert_array_equal(ref['ids'],tid[ti]);assert all(np.array_equal(ref[arm],targets[arm].numpy()) for arm in ARMS)
    np.savez(a.out/'teacher-targets.npz',ids=tid[ti],**targets)
    print(a.dataset,'all original teachers reproduced exact predictions',flush=True);torch.cuda.empty_cache()
    results={};preds={}
    for arm in ARMS:
        kind=arm.split('+')[0]; seed(SE); net=Student(kind,ds);params=net.fitparams(); initial=net.prompt.detach().clone(); sig=net.sig; norm=T.Normalize((.5,)*3,(.5,)*3) if sig else T.Normalize(MEAN,STD)
        vtf=T.Compose([T.Resize((224,224),interpolation=T.InterpolationMode.BILINEAR) if sig else T.Resize(224,interpolation=T.InterpolationMode.BICUBIC),*([] if sig else [T.CenterCrop(224)]),T.ToTensor(),norm]); ttf=T.Compose([T.RandomResizedCrop(224,scale=(.5,1.),interpolation=T.InterpolationMode.BICUBIC),T.RandomHorizontalFlip(),T.RandAugment(num_ops=2,magnitude=9),T.ToTensor(),norm]); vload=DataLoader(Images(ds,vid[vi],y,vtf),batch_size=a.micro,num_workers=0)
        net.check_native(next(iter(vload))[0][:1].cuda()); frozen=predict(net,vload,False); before=predict(net,vload); hist=[]
        opt=torch.optim.AdamW([{'params':[v for k,v in params.items() if not k.endswith('bias')],'weight_decay':.01},{'params':[v for k,v in params.items() if k.endswith('bias')],'weight_decay':0.}],lr=.004); scaler=torch.amp.GradScaler('cuda',init_scale=128.); torch.cuda.reset_peak_memory_stats();t_arm=time.monotonic()
        start_epoch=1
        if parent:
            old=torch.load(a.extend_from/parent['arms'][arm]['checkpoint'],map_location='cpu',weights_only=True)
            with torch.no_grad():
                for k,v in params.items():v.copy_(old['trainable'][k])
            opt.load_state_dict(old['opt']);scaler.load_state_dict(old['scaler']);hist=list(parent['arms'][arm]['history']);start_epoch=parent['epochs']+1
            assert hist[-1]==old['metrics'] and torch.equal(predict(net,vload),old['val']),arm
            print(a.dataset,arm,'verified continuation from epoch',start_epoch-1,flush=True);del old
        for ep in range(start_epoch,a.epochs+1):
            path=a.out/f'{arm}-epoch-{ep}.pt'
            if a.resume and path.exists():
                old=torch.load(path,map_location='cpu',weights_only=True)
                with torch.no_grad():
                    for k,v in params.items():v.copy_(old['trainable'][k])
                opt.load_state_dict(old['opt']);scaler.load_state_dict(old['scaler']);hist.append(old['metrics']);del old
                print(a.dataset,arm,'resumed saved epoch',ep,flush=True);continue
            ep_t=time.monotonic();seed(SE+ep);g=torch.Generator().manual_seed(SE+ep);n=updates=skips=0;opt.zero_grad(set_to_none=True);losses=[]
            train=Images(ds,tid[ti],targets[arm],ttf)
            for x,p in DataLoader(train,batch_size=a.micro,shuffle=True,generator=g,num_workers=0):
                effective=min(a.effective,len(train)-n//a.effective*a.effective)
                with torch.autocast('cuda',dtype=torch.float16): z=net(x.cuda())
                loss=F.cross_entropy(z.float(),p.cuda(),label_smoothing=.1,reduction='sum')/effective;scaler.scale(loss).backward();n+=len(x);losses.append(loss.detach())
                if n%a.effective==0 or n==len(train):
                    old=scaler.get_scale();scaler.step(opt);scaler.update();skips+=int(scaler.get_scale()<old);updates+=1;opt.zero_grad(set_to_none=True)
                if len(train)>1000 and n%2048==0:print(a.dataset,arm,'epoch',ep,'fit images',n,'/',len(train),'elapsed',round(time.monotonic()-ep_t),flush=True)
            assert updates==meta['steps'] and skips<updates and torch.stack(losses).isfinite().all() and not torch.equal(net.prompt,initial);p=predict(net,vload)
            path=a.out/f'{arm}-epoch-{ep}.pt'; h=dict(epoch=ep,correct=int((p==y).sum()),updates=updates,skips=skips,seconds=round(time.monotonic()-ep_t,2)); hist.append(h)
            save(path,dict(trainable={k:v.detach().cpu() for k,v in params.items()},opt=opt.state_dict(),scaler=scaler.state_dict(),val=p,metrics=h)); print(a.dataset,arm,h,'peak MiB',round(torch.cuda.max_memory_allocated()/2**20),flush=True)
        st=torch.load(path,map_location='cpu',weights_only=True)
        with torch.no_grad():
            for k,v in params.items():v.copy_(st['trainable'][k])
        p=predict(net,vload);assert torch.equal(p,st['val']);preds[arm]=p
        results[arm]=dict(backbone=kind,teacher_correct=int((baselines[arm]==y).sum()),frozen_correct=int((frozen==y).sum()),untrained_prompt_correct=int((before==y).sum()),student_correct=int((p==y).sum()),history=hist,checkpoint=path.name,sha256=digest(path),trainable=sum(v.numel() for v in params.values()),seconds=round(time.monotonic()-t_arm,2));jsonsave(a.out/f'{arm}.json',results[arm]);del net,params,opt,scaler,st;torch.cuda.empty_cache()
    paired={f'{b} vs {a}':compare(preds[a].numpy(),preds[b].numpy(),y.numpy()) for a,b in itertools.combinations(ARMS,2)}
    rows=[dict(id=vid[j],true=int(y[i]),**{m:int(preds[m][i]) for m in ARMS},**{m+'_teacher':int(baselines[m][i]) for m in ARMS}) for i,j in enumerate(vi)];write(a.out/'predictions.csv',rows); reread=read(a.out/'predictions.csv'); assert len({r['id'] for r in reread})==len(y)
    for arm,r in results.items():assert sum(z[arm]==z['true'] for z in reread)==r['student_correct'] and sum(z[arm+'_teacher']==z['true'] for z in reread)==r['teacher_correct']
    jsonsave(a.out/'verified.json',dict(**meta,arms=results,paired=paired,seconds=round(time.monotonic()-t0,1))); print('VERIFIED STUDENTS',a.dataset,a.out,[(k,v['student_correct']) for k,v in results.items()],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['eurosat','cifar100']);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--extend-from',type=Path);p.add_argument('--train',type=int,default=320);p.add_argument('--val-per-class',type=int,default=20);p.add_argument('--epochs',type=int);p.add_argument('--micro',type=int,default=4);p.add_argument('--effective',type=int,default=32);a=p.parse_args();assert a.effective%a.micro==0
    a.epochs=a.epochs if a.epochs is not None else (1 if a.smoke else 2)
    if a.smoke:a.train,a.val_per_class,a.effective=11,1,8
    main(a)
