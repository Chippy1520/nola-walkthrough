"""Produce a per-dataset scorecard from archived, row-aligned predictions."""
import csv,json
from pathlib import Path

LABELS={'clip+dino':'CLIP + DINO','siglip2+dino':'SigLIP 2 + DINO','clip+convnext':'CLIP + ConvNeXt','siglip2+convnext':'SigLIP 2 + ConvNeXt'}

def build(r,pair):
    def load(p): return json.loads(p.read_text())
    def rows(p):
        with p.open() as f:return list(csv.DictReader(f))
    def pct(a,n):return f'{a}/{n} ({100*a/n:.2f}%)'
    def interval(d,lohi):return f'{d:+.2f} [{lohi[0]:+.2f}, {lohi[1]:+.2f}]'
    out=[]
    for k,label in [('eurosat','EuroSAT'),('cifar100','CIFAR-100 coarse scarcity pilot')]:
        base=r/'siglip-replacement'; hp=base/'factorial'/k; sp=base/'students-epoch12'/k
        h=load(hp/'verified.json'); p=load(sp/'verified.json'); e2=load((base/'students'/k/'verified.json')); e4=load((base/'students-epoch4'/k/'verified.json'))
        vr=rows(sp/'predictions.csv'); n=p['val_n']; nh=h['val_n']; assert n==len(vr) and p['train_n']==320 and {p['epochs'],e4['epochs'],e2['epochs']}=={2,4,12}
        assert not p['test_scored'] and not h['test_scored']
        overview=[]; stage=[]; contrasts=[]; details=[]
        for a,l in LABELS.items():
            ar=h['arms'][a]; v=p['arms'][a]; hist=v['history']; peak=max(hist,key=lambda x:x['correct']); aa=a+'_teacher'
            d=pair(l,vr,aa,a,'true','id'); assert (d['base'],d['new'])==(v['teacher_correct'],v['student_correct']); assert (hist[1]['correct'],hist[3]['correct'])==(e2['arms'][a]['student_correct'],e4['arms'][a]['student_correct'])
            overview.append([l,pct(ar['correct'],nh),pct(ar['correct_anchors'],ar['anchors']),f"{ar['correct_extra']}/{sum(h['matched_extra'])}"])
            stage.append([l,pct(d['base'],n),pct(d['new'],n),f"{d['fix']} / {d['loss']}",interval(d['pp'],d['ci95'])]); details.append([l,f"{100*hist[1]['correct']/n:.2f}%",f"{100*hist[3]['correct']/n:.2f}%",f"{100*d['new']/n:.2f}%",f"{100*peak['correct']/n:.2f}% (e{peak['epoch']})"])
        for label2,key in [('SigLIP − CLIP (DINO)','siglip2+dino vs clip+dino'),('SigLIP − CLIP (ConvNeXt)','siglip2+convnext vs clip+convnext'),('ConvNeXt − DINO (CLIP)','clip+convnext vs clip+dino'),('ConvNeXt − DINO (SigLIP)','siglip2+convnext vs siglip2+dino')]:
            contrasts.append([label2,*[interval(t['paired'][key]['delta_pp'],t['paired'][key]['ci95_pp']) for t in (h,p)]])
        out.append(dict(label=label,head_n=nh,head_fit=h['train_n'],prompt_n=n,overview=overview,stage=stage,details=details,contrasts=contrasts))
    headings=[('Alignment head: original validation',['Arm','Top-1','Correct anchors','Correct additions'],'overview'),('Paired stage difference on the same images',['Arm','Alignment head','Visual-prompt classifier','Fixes / losses','Δ pp [95% CI]'],'stage'),('Visual-prompt classifier trajectory',['Arm','Epoch 2','Epoch 4','Epoch 12','Best observed validation'],'details'),('Matched comparisons',['Contrast','Alignment head Δ pp [95% CI]','Epoch-12 classifier Δ pp [95% CI]'],'contrasts')]
    prov='The alignment head is fitted on frozen original DINO ViT-B/16 features. The visual-prompt classifier is CLIP or SigLIP 2 with learned visual prompts. DINOv3 ConvNeXt-Tiny is used for retrieval only. Stage deltas compare identical pilot validation rows, not the full alignment-head score against the smaller classifier pilot. Classifier adaptation uses 320 images per dataset. Intervals are paired image-level bootstrap intervals, not across-seed intervals.'
    md=['# NoLA dataset-by-dataset statistics','',prov,'','## Coverage by dataset','','| Dataset | Frozen CDE | Alignment head | Visual-prompt classifier |','|---|---|---|---|','| EuroSAT | 5,400 validation | 5,400 validation; release-style and factorial protocols | Release-style: 8,100 test; factorial: 200 validation |','| CIFAR-100 coarse | 5,000 validation | 5,000 standard validation; separate 1,024-image scarcity pilot | 400 validation, scarcity pilot |','| RESISC45 | 6,255 validation | 6,255 validation for anchor-policy comparison | Not run |','| Oxford-IIIT Pet | 3,669 test, frozen template comparison | Not run for the template comparison | Not run for the template comparison |','','Baseline anchor-policy top-1 (%) with the frozen DINO alignment head (confidence / farthest-first / medoids / neighbor agreement):','']
    md+=['| Dataset | N | Confidence | Farthest-first | Medoids | Neighbor agreement |','|---|---:|---:|---:|---:|---:|']
    arc=rows(r/'parallel-study/dino-anchors/reports/combined.csv')
    for k,l in [('eurosat','EuroSAT'),('resisc45','RESISC45'),('cifar100','CIFAR-100 coarse')]:
        a=[v for v in arc if v['dataset']==k];md+=['| '+' | '.join([l,a[0]['val_n'],*[f"{float(x['val_accuracy_pct']):.2f}%" for x in a]])+' |']
    for s in out:
        md+=['',f"## {s['label']}",'',f"Independent-nomination factorial: {s['head_fit']:,} adaptation images, {s['head_n']:,} alignment-head validation images. On the smaller {s['prompt_n']:,}-image validation subset, both stages are scored on the exact same images."]
        for title,cols,key in headings:
            md+=['',f'### {title}','','| '+' | '.join(cols)+' |','|'+'---|'*len(cols),*['| '+' | '.join(v)+' |' for v in s[key]]]
    (r/'NoLA-Dataset-Statistics.md').write_text('\n'.join(md)+'\n');return out,headings,prov
