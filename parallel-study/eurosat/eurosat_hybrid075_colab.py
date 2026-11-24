# EuroSAT frozen-CLIP validation: fixed 0.75 cosine gate, never score test.
import csv, json, hashlib, zipfile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tx
from PIL import Image
from google.colab import files

def eurosat_hybrid075():
    dest=ROOT/'eurosat_hybrid075'; dest.mkdir(exist_ok=True)
    rows=VAL; assert len(rows)==5400 and len({r[0] for r in rows})==5400
    assert all(CLASS_NAMES[y]==name for _,y,name in rows)
    tf=tx.Compose([tx.Resize(224,interpolation=tx.InterpolationMode.BICUBIC),tx.CenterCrop(224),tx.ToTensor(),tx.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    class EvalImages(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self,i):
            with Image.open(IMAGE_DIR/rows[i][0]) as im: return tf(im.convert('RGB'))
    device=torch.device(DEVICE)
    with torch.inference_mode():
        e=torch.cat([F.normalize(clip_jit.encode_image(x.to(device,non_blocking=True)).float(),dim=1).cpu() for x in DataLoader(EvalImages(),batch_size=32,num_workers=2,pin_memory=device.type=='cuda')]); assert e.shape==(5400,512) and torch.isfinite(e).all()
        mean=F.normalize(B.cpu().mean(1),dim=1); aud=[]; all_scores=[]; hybrid_scores=[]
        tau=.05
        for i,(grp,size,raw_gpu) in enumerate(BBANK):
            raw=raw_gpu.cpu(); weight=torch.tensor(size).float(); m=mean[i:i+1]; torch.testing.assert_close(F.normalize((raw*weight[:,None]).sum(0),dim=0),m[0]); sim=F.cosine_similarity(raw,m,dim=1); keep=sim<.75
            aud.extend(dict(class_id=i,class_name=CLASS_NAMES[i],cluster=j,size=int(weight[j]),cosine=float(sim[j]),retained=bool(keep[j])) for j in range(len(grp)))
            def pool(v,w): return tau*(torch.logsumexp(e@v.T/tau+w.log(),dim=1)-w.sum().log())
            all_scores.append(pool(raw,weight)); hybrid_scores.append(pool(torch.cat((m,raw[keep])),torch.cat((weight.sum()[None],weight[keep]))))
        control=e@mean.T
        hybrid=torch.stack(hybrid_scores,1)
        untouched=[i for i in range(len(mean)) if not any(a['retained'] and a['class_id']==i for a in aud)]
        torch.testing.assert_close(control[:,untouched],hybrid[:,untouched],atol=2e-7,rtol=2e-5)
        logits={'wrapped_mean':control,'hybrid_075':hybrid,'cluster_only':torch.stack(all_scores,1)}
        if 'A' in globals() and 'C' in globals(): logits['original_cde']=e@F.normalize(torch.cat((A,C),dim=1).cpu().mean(1),dim=1).T
        preds={k:v.argmax(1) for k,v in logits.items()}; y=torch.tensor([r[1] for r in rows]); ok={k:p==y for k,p in preds.items()}; a,b=ok['wrapped_mean'],ok['hybrid_075']; fixed=~a&b; lost=a&~b
        def stats(mask):
            n=int(mask.sum()); return dict(n=n,**{k+'_correct':int((v&mask).sum()) for k,v in ok.items()},fixed=int((fixed&mask).sum()),lost=int((lost&mask).sum()))
        def table(name,data):
            with (dest/name).open('w') as fp:
                w=csv.DictWriter(fp,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
        table('candidate_audit.csv',aud)
        table('predictions.csv',[dict(path=r[0],true_id=r[1],**{k:int(v[i]) for k,v in preds.items()}) for i,r in enumerate(rows)])
        table('per_class.csv',[dict(class_id=i,class_name=n,**stats(y==i)) for i,n in enumerate(CLASS_NAMES)])
        result=dict(dataset='EuroSAT',split='CoOp validation only',stage='frozen CLIP ViT-B/32',threshold=.75,tau=tau,scoring='normalized weighted log-mean-exp; normalized mean weight 25; raw extra cluster means weighted by group size',retained=[v for v in aud if v['retained']],**stats(torch.ones_like(y,dtype=torch.bool)))
        (dest/'summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
        path=dest/'NoLA-EuroSAT-hybrid-075-report.zip'
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for f in sorted(dest.iterdir()):
                if f.suffix in ('.json','.csv'): z.write(f,f.name)
        print('SHA256',hashlib.sha256(path.read_bytes()).hexdigest());return path

EURO_HYBRID_REPORT=eurosat_hybrid075()
files.download(str(EURO_HYBRID_REPORT))
