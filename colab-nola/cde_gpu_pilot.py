"""Validation-only, same-prompt CDE grouping ablation. Run with a CUDA runtime."""
from pathlib import Path
import hashlib, json, math, sys, time, urllib.request, shutil, csv
import torch
from torch.nn.functional import normalize
from torchvision import transforms as T
from sklearn.cluster import KMeans
from PIL import Image

root = Path(__file__).resolve().parent
assert torch.cuda.is_available(), 'CUDA is required'
print(torch.cuda.get_device_name(), torch.__version__, flush=True)
sys.path.insert(0, str(root))
from clip import clip
url = clip._MODELS['ViT-B/32']
ckpt = root/'ViT-B-32.pt'
if not ckpt.exists():
    tmp = ckpt.with_suffix('.part')
    with urllib.request.urlopen(url, timeout=30) as s, tmp.open('wb') as d:
        shutil.copyfileobj(s,d)
    with tmp.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==url.split('/')[-2]
    tmp.rename(ckpt)
with ckpt.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()==url.split('/')[-2]
jit = torch.jit.load(str(ckpt),map_location='cpu')
model = clip.build_model(jit.state_dict()).cuda().eval()
del jit
preprocess = T.Compose([T.Resize(224,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
config=json.loads((root/'pilot_inputs.json').read_text())
names=config['class_order']; rows=config['rows']; prompts=config['texts']
assert len(rows)==240 and len({r[0] for r in rows})==240

def encode(model, rows):
    xs=[]
    for start in range(0,len(rows),32):
        batch=[]
        for file,_ in rows[start:start+32]:
            with Image.open(root/'images'/file) as im: batch.append(preprocess(im.convert('RGB')))
        xs.append(normalize(model.encode_image(torch.stack(batch).cuda()).float(),dim=-1))
    return torch.cat(xs)

started=time.perf_counter()
with torch.inference_mode():
    x=encode(model,rows)
    y=torch.tensor([names.index(c) for _,c in rows], device='cuda')
    text=[]; grouping={}; clusters=[]
    for name in names:
        strings=prompts[name]['descriptions']+prompts[name]['filled_templates']
        v=normalize(model.encode_text(clip.tokenize(strings).cuda()).float(),dim=-1)
        assert len(v)>=3
        g=KMeans(n_clusters=3,n_init=10,random_state=42).fit_predict(v.cpu().numpy())
        assert set(g)=={0,1,2}
        grouping[name]=[[s for s,k in zip(strings,g) if k==j] for j in range(3)]
        text.append(normalize(v.mean(dim=0),dim=0))
        clusters.append(torch.stack([normalize(v[g==j].mean(dim=0),dim=0) for j in range(3)]))
    baseline=x@torch.stack(text).T
    sub=torch.einsum('nd,ckd->nck', x, torch.stack(clusters))
    tau=.05
    variants={'cde':baseline,'multi_lme':tau*(torch.logsumexp(sub/tau,dim=-1)-math.log(3)),'multi_max':sub.max(dim=-1).values}
    chosen={k:v.argmax(dim=-1) for k,v in variants.items()}
    good={k:v==y for k,v in chosen.items()}
    torch.cuda.synchronize()
    counts={k:int(v.sum()) for k,v in good.items()}
    per_class={c:{'n':int((y==i).sum()),**{k:int((v&(y==i)).sum()) for k,v in good.items()}} for i,c in enumerate(names)}
    paired={k:{'fixed':int((v&~good['cde']).sum()),'regressed':int((~v&good['cde']).sum())} for k,v in good.items() if k!='cde'}
    assert all(counts[k]-counts['cde']==v['fixed']-v['regressed'] for k,v in paired.items())
    summary=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,n=len(rows),counts=counts,per_class=per_class,paired=paired,config=dict(clip='ViT-B/32',cluster_seed=42,k=3,tau=tau,split='existing EuroSAT case-study validation subset'),seconds=time.perf_counter()-started)
    (root/'cde_gpu_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'cde_gpu_prompt_groups.json').write_text(json.dumps(grouping,indent=2)+'\n')
    with (root/'cde_gpu_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['image','true_class',*chosen]); w.writerows((row[0],row[1],*(names[int(chosen[k][i])] for k in chosen)) for i,row in enumerate(rows))
    assert len(list(csv.DictReader((root/'cde_gpu_predictions.csv').open())))==len(rows)
    print(json.dumps(summary),flush=True)
