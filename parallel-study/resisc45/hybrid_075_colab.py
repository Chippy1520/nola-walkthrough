# RESISC45 frozen validation follow-up. Run after frozen_validation().
import csv, hashlib, zipfile
from pathlib import Path
import torch
from torch.nn import functional as F

@torch.inference_mode()
def hybrid_075_validation():
    threshold = .75
    assert CONFIG_ID == 'f67e0f157c474ccc'
    bank = load_tensor(CACHE/'text_bank.pt')
    groups = bank['sizes']
    parts = bank['prototypes'].split([len(g) for g in groups])
    counts = [torch.tensor(g) for g in groups]
    means = torch.stack([F.normalize((p * c[:,None]).sum(0),dim=0) for p,c in zip(parts,counts)])
    hybrid_vectors, hybrid_sizes, audit = [], [], []
    for i, (p, c, m) in enumerate(zip(parts, counts, means)):
        cosine = F.cosine_similarity(p,m[None],dim=1)
        keep = cosine < threshold
        hybrid_vectors.append(torch.cat([m[None],p[keep]]))
        hybrid_sizes.append([int(c.sum()),*c[keep].tolist()])
        audit.extend(dict(class_id=i,class_key=KEYS[i],cluster_id=j,multiplicity=int(c[j]),cosine_to_mean=float(v),retained=bool(keep[j])) for j,v in enumerate(cosine))
    scorer = BankScorer(dict(broad=means,prototypes=torch.cat(hybrid_vectors),sizes=hybrid_sizes))
    rows = SPLITS['val']
    features = load_tensor(CACHE/'clip_val.pt')
    assert features.shape==(6255,512) and len(rows)==6255
    old = frozen_scores(features)
    assert [int((old[k].argmax(1)==truth(rows)).sum()) for k in ARMS]==[3630,3506]
    with (OUT/'frozen_clip_val_predictions.csv').open() as f:
        prior=list(csv.DictReader(f))
    assert [(r['path'],int(r['truth_id'])) for r in prior]==[(r[0],r[1]) for r in rows]
    for col,k in zip(('original_pred','template_pred'),ARMS):
        assert old[k].argmax(1).tolist()==[int(r[col]) for r in prior]
    score = {k:torch.cat([scorer(x.to(DEVICE),arm).cpu() for x in features.split(1024)]) for k,arm in zip(('wrapped_mean','hybrid_075'),ARMS)}
    unchanged = [i for i,g in enumerate(hybrid_sizes) if len(g)==1]
    torch.testing.assert_close(score['hybrid_075'][:,unchanged],score['wrapped_mean'][:,unchanged],rtol=2e-5,atol=2e-7)
    pred={k:v.argmax(1) for k,v in (old|score).items()}
    dest=OUT/'hybrid_075';dest.mkdir(exist_ok=True)
    global BOOTSTRAP_REPS
    for a,b,label in [('wrapped_mean','hybrid_075','hybrid_vs_wrapped_mean'),(ARMS[0],'hybrid_075','hybrid_vs_original_cde')]:
        stats=paired_report(label,'val',rows,{ARMS[0]:pred[a],ARMS[1]:pred[b]})
        for p in OUT.glob(label+'_val_*'):
            p.replace(dest/p.name)
    y=truth(rows)
    metrics={k:dict(correct=int((v==y).sum()),accuracy_pct=float((v==y).sum())*100/len(y)) for k,v in pred.items()}
    summary=dict(config_id=CONFIG_ID,stage='frozen_clip',split='validation',n=len(y),threshold=threshold,tau=TAU,selection='cosine(raw cluster mean, normalized same-wrapped class mean) < threshold',aggregation='existing normalized weighted log-mean-exp; class mean weight = all descriptions; extra raw cluster means weighted by cluster size',retained=[a for a in audit if a['retained']],metrics=metrics)
    write_json(dest/'hybrid_summary.json',summary)
    csv_rows(dest/'candidate_audit.csv',audit)
    csv_rows(dest/'predictions.csv',[dict(path=r[0],truth_id=r[1],**{k:int(v[i]) for k,v in pred.items()}) for i,r in enumerate(rows)])
    output=WORK/'NoLA-RESISC45-hybrid-075-report.zip'
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.iterdir()): z.write(p,p.name)
    print(json.dumps(summary,indent=2))
    print(output,hashlib.sha256(output.read_bytes()).hexdigest())
    return output

HYBRID_REPORT=hybrid_075_validation()
from google.colab import files
files.download(str(HYBRID_REPORT))
