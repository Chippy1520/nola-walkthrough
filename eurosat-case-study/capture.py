from pathlib import Path
import json, sys, hashlib, subprocess, shutil
from datetime import datetime, timezone
from collections import Counter
from math import ceil

out = Path(__file__).resolve().parent
root = out.parent
J = dict(train_n=160, anchor_cap=4, head_batch=8, head_epochs=3, prompts=16, clip_batch=4, clip_lr=0.004)
PROTOCOL = dict(seed=42, val_seed=123, val_n=240, clip_epochs=[1,4], pool_n=320)
saved = next(e for e in map(json.loads, (root/'eurosat-full-sweep.jsonl').read_text().splitlines()) if e['job']==J and e['protocol']==PROTOCOL)
source = root/'NoLA-EuroSAT-Full-Sweep.ipynb'
cells = [''.join(c['source']) for c in json.loads(source.read_text())['cells'] if c['cell_type']=='code' and ''.join(c['source']).strip()]
import torch
torch.set_num_threads(5)
torch.set_num_interop_threads(1)
device = 'cpu'
def portable(s):
    return s.replace('.cuda()', '.to(device)').replace('device="cuda"', 'device=device')
exec(portable(cells[0]), globals())
pool = pool[:J['train_n']]
audit = audit[:len(pool)]
seen = {row[0] for row in random.Random(42).sample(split['test'],80)}
new_test = random.Random(137).sample([r for r in split['test'] if r[0] not in seen],240)
y_fresh = torch.tensor([r[1] for r in new_test])
assert len({r[0] for r in pool+val+new_test})==640

def write(name, data):
    (out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
write('provenance.json', dict(process='Instrumented CPU replay; not the original GPU run',device=device,torch=torch.__version__, utc=datetime.now(timezone.utc).isoformat(), job=J,protocol=PROTOCOL, fresh_test=dict(n=240,seed=137,excluded_teaching_test=80), upstream=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip(),dino_source=subprocess.check_output(['git','-C',str(root/'dino-source'),'rev-parse','HEAD'],text=True).strip(), hashes={p.name:sha(p) for p in [source,Path(__file__),image_dir.parent/'split_zhou_EuroSAT.json',root/'weights/ViT-B-32.pt',root/'weights/dino_vitbase16_pretrain.pth']}, logged_gpu=saved))
print('Stage 1: CLIP CDE and original selector',flush=True)
s=cells[1]; first=s.splitlines()[0]
s=s.replace(first,'checkpoint = torch.jit.load(str(root / "weights/ViT-B-32.pt"), map_location="cpu")\nmodel = clip.build_model(checkpoint.state_dict()).float().eval()\nwidth=model.text_projection.shape[1]\ndel checkpoint')
exec(portable(s),globals())
fresh_probs = clip_scores(new_test)
write('cde-text.json',{name:dict(descriptions=descriptions[name],templates=templates,filled_templates=[t.format(name) for t in templates]) for name in names})
torch.save(cde.detach().cpu(),out/'cde.pt')
ids, returned, labels = top_k_indices_per_class2(dict(total_emb=pool_probs,idxs=torch.arange(len(pool)),labels=audit),J['anchor_cap'])
top_values, top_ids = pool_probs.topk(3,dim=1)
get_rank=lambda row, cls: int((top_ids[row]==cls).nonzero()[0])+1
rows=[]; (out/'anchors').mkdir(exist_ok=True)
for i,pseudo,old_prob in zip(ids.tolist(),labels.tolist(),returned.tolist()):
    path,truth,_ = pool[i]
    photo = f'anchors/{pseudo:02d}_{i:03d}_{Path(path).name}'
    shutil.copyfile(image_dir/path,out/photo)
    rows.append(dict(row=i,path=path,image=photo,pseudo_class=names[pseudo],true_class=names[truth],correct=truth==pseudo,
                     assigned_rank=get_rank(i,pseudo),assigned_probability=float(pool_probs[i,pseudo]),selector_returned_top1_probability=old_prob))
pd.DataFrame(rows).to_csv(out/'anchors.csv',index=False)
candidate_rows=[]
for cls,name in enumerate(names):
    for rank in range(3):
        candidates=[i for i in range(len(pool)) if int(top_ids[i,rank])==cls]
        if candidates:
            candidates.sort(key=lambda i: float(pool_probs[i,cls]),reverse=True)
            for position,i in enumerate(candidates[:J['anchor_cap']],1):
                retained = any(r['row']==i and r['pseudo_class']==name for r in rows)
                path,truth,_=pool[i]
                photo=f'anchors/{cls:02d}_{i:03d}_{Path(path).name}'
                if not (out/photo).exists(): shutil.copyfile(image_dir/path,out/photo)
                candidate_rows.append(dict(class_name=name,within_class_rank=position,row=i,path=path,image=photo,true_class=names[truth],prediction_rank=rank+1,probability=float(pool_probs[i,cls]),selected=retained))
            break
pd.DataFrame(candidate_rows).to_csv(out/'top-k-candidates.csv',index=False)
print('Stage 2: frozen DINO features',flush=True)
exec(portable(cells[2]),globals())
# The prep cell drops dino after extracting; rerun its load only to score the fresh test subset.
exec(portable(cells[2].split('pool_features =')[0]),globals())
fresh_features=dino_features(new_test)
del dino
model.to(device)
print('Stage 3: DINO head and prompted CLIP',flush=True)
s=cells[3]
s=s.replace('    return targets, stats','''    with torch.no_grad():
        stage_scores['dino_head'] = [head(x).softmax(1).cpu() for x in (pool_features, val_features, fresh_features)]
    torch.save(head.state_dict(),out/'dino-head.pt')
    return targets, stats''')
s=s.replace('    for _ in range(epochs):','    for head_epoch in range(1, epochs+1):\n        losses = []')
s=s.replace('            F.cross_entropy(head(pool_features[ids[rows]]), labels[rows]).backward()','            head_loss = F.cross_entropy(head(pool_features[ids[rows]]), labels[rows])\n            losses.append((float(head_loss.detach()),len(rows)))\n            head_loss.backward()')
s=s.replace('            opt.step()\n    with torch.no_grad():','            opt.step()\n        history.append(dict(stage="dino_head",epoch=head_epoch,mean_loss=sum(x*b for x,b in losses)/sum(b for _,b in losses)))\n    with torch.no_grad():',1)
needle='    opt = torch.optim.AdamW(['
s=s.replace(needle,'''    @torch.no_grad()
    def capture_scores(rows):
        return torch.cat([logits(load_images(rows[i:i+inference_batch],te_transform)).softmax(1).detach().cpu()
                          for i in range(0,len(rows),inference_batch)])
    stage_scores['prompted_init'] = [capture_scores(val),capture_scores(new_test)]

'''+needle)
s=s.replace('        for ids in torch.randperm(n, generator=order).split(b):','        losses = []\n        for ids in torch.randperm(n, generator=order).split(b):')
s=s.replace('            opt.zero_grad()\n            F.cross_entropy(logits(images)', '            if epoch == 1 and len(augmented) < 6:\n                capture_augmentation(ids, images, targets)\n            opt.zero_grad()\n            F.cross_entropy(logits(images)')
augmented = []
def capture_augmentation(ids, images, targets):
    folder = out/'augmentations'; folder.mkdir(exist_ok=True)
    mean = torch.tensor([0.48145466,0.4578275,0.40821073])[:,None,None]
    std = torch.tensor([0.26862954,0.26130258,0.27577711])[:,None,None]
    def pixels(x): return (x.cpu()*std+mean).clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy()
    for i,image in zip(ids.tolist(),images):
        path, truth, _ = pool[i]
        if len(augmented)>=6 or any(r['true_class']==names[truth] for r in augmented): continue
        base = f'augmentations/{i:03d}'
        shutil.copyfile(image_dir/path,out/f'{base}-original.jpg')
        with Image.open(image_dir/path) as im:
            Image.fromarray(pixels(te_transform(im.convert('RGB')))).save(out/f'{base}-weak.png')
        Image.fromarray(pixels(image.detach())).save(out/f'{base}-strong.png')
        augmented.append(dict(row=i,path=path,true_class=names[truth],dino_label=names[targets[i]],original=f'{base}-original.jpg',weak=f'{base}-weak.png',strong=f'{base}-strong.png'))
    write('augmentations.json', augmented)
s=s.replace('            F.cross_entropy(logits(images), targets[ids].cuda(), label_smoothing=clip_label_smoothing).backward()','            loss = F.cross_entropy(logits(images), targets[ids].cuda(), label_smoothing=clip_label_smoothing)\n            losses.append((float(loss.detach()),len(ids)))\n            loss.backward()')
s=s.replace('        if epoch in clip_epochs:','''        history.append(dict(stage='prompted_clip',epoch=epoch,mean_loss=sum(l*b for l,b in losses)/sum(b for _,b in losses),updates=updates))
        print(f'CLIP epoch {epoch}: {updates} updates; mean loss {history[-1]["mean_loss"]:.3f}',flush=True)
        write('history.json',history)
        if epoch in clip_epochs:
            stage_scores[f'prompted_epoch_{epoch}'] = [capture_scores(val),capture_scores(new_test)]''')
s=s.replace('    return records','''    torch.save(dict(prompt=prompt.detach().cpu(),classifier=classifier.state_dict(),visual={k:p.detach().cpu() for k,p in trainable}),out/'prompted-adaptation.pt')
    return records''')
stage_scores={'cde':[pool_probs,val_probs,fresh_probs]}; history=[]
exec(portable(s),globals())
records=run_clip(J)
write('history.json',history)
summary=[]; predictions=[]
for stage,scores in stage_scores.items():
    partitions=[('adaptation',pool),('validation',val),('fresh_test',new_test)] if len(scores)==3 else [('validation',val),('fresh_test',new_test)]
    for (partition,parts),score in zip(partitions,scores):
        truth=torch.tensor([r[1] for r in parts]); pred=score.argmax(1); good=pred.eq(truth)
        per=[]; matrix=torch.zeros(len(names),len(names),dtype=torch.int64)
        for j,name in enumerate(names):
            mask=truth==j; per.append(dict(class_name=name,n=int(mask.sum()),correct=int(good[mask].sum()),predicted=int((pred==j).sum())))
            matrix[j]=pred[mask].bincount(minlength=len(names))
        summary.append(dict(stage=stage,split=partition,n=len(parts),correct=int(good.sum()),accuracy=float(good.float().mean()),per_class=per, confusion=matrix.tolist()))
        for i,(path,label,_) in enumerate(parts):
            predictions.append(dict(stage=stage,split=partition,row=i,image_path=path,true_class=names[label],predicted_class=names[pred[i]],correct=bool(good[i]),**{f'prob_{j}':float(x) for j,x in enumerate(score[i])}))
pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
write('metrics.json',dict(class_order=names,metrics=summary,clip_records=records,anchors=len(rows),anchor_correct=sum(r['correct'] for r in rows),candidate_count=len(candidate_rows)))
print('Verified measured outputs:',[(m['stage'],m['split'],m['correct'],m['n']) for m in summary],flush=True)
