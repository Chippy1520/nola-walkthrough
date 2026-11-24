from pathlib import Path
import csv,json,re,textwrap
import numpy as np
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset_scorecard import build as build_scorecard
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

R=Path(__file__).resolve().parents[1]; A=R/'report-assets'; PDF=R/'NoLA-Experiment-Report.pdf'
INK='#142841'; MUTED='#4a5d72'; TEAL='#007a78'; BLUE='#3767a2'; RED='#a23c3c'; GRID='#dce5ec'; LITE='#edf4f8'
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'font.size':9,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,'text.color':INK})
J=lambda p:json.loads((R/p).read_text())
C=lambda p:list(csv.DictReader((R/p).open()))
D=['EuroSAT','RESISC45','CIFAR-100 coarse']; keys=['eurosat','resisc45','cifar100']
P='parallel-study/cluster-first/'

# Resample paired correctness differences within true class (fine subtype for CIFAR coarse).
# A multinomial draw over -1/0/+1 is equivalent to row bootstrap for a mean difference.
def pair(name, rows, a, b, truth, uid, groups=None):
    assert len({r[uid] for r in rows})==len(rows)
    y=np.array([r[truth] for r in rows]); n=len(y)
    x=np.array([r[a] for r in rows])==y; z=np.array([r[b] for r in rows])==y
    d=z.astype(int)-x.astype(int); f=int((d==1).sum()); l=int((d==-1).sum())
    groups=y if groups is None else groups
    rng=np.random.default_rng(2026); v=np.zeros(20000,dtype=int)
    for k in np.unique(groups):
        q=d[groups==k]; h=np.bincount(q+1,minlength=3)
        sample=rng.multinomial(len(q),h/len(q),size=len(v)); v+=sample[:,2]-sample[:,0]
    assert int(z.sum()-x.sum())==f-l
    return dict(name=name,n=n,base=int(x.sum()),new=int(z.sum()),fix=f,loss=l,pp=100*d.mean(),ci95=(100*np.quantile(v,[.025,.975])/n).tolist(),p=binomtest(f,f+l).pvalue if f+l else 1.)

suffix=[]; blend=[]
for k,label in zip(keys,D):
    p=f'cde-suffix-blend/{k}'; rows=C(p+'/predictions.csv'); s=J(p+'/verified.json'); assert s['verified']
    fine=None
    if k=='cifar100':
        with np.load(R/'parallel-study/cifar100/audit_manifest.npz',allow_pickle=False) as m:
            ids=np.array([int(r['row_id']) for r in rows]); np.testing.assert_array_equal(ids,m['validation_indices']); fine=m['train_fine'][ids]
    for arr,a,b in [(suffix,'legacy_mean' if k=='cifar100' else 'released_separate','suffix_mean'),(blend,'suffix_mean','suffix_blend090')]:
        q=pair(label,rows,a,b,'true_id','row_id',fine); assert (q['base'],q['new'])==(s['metrics'][a]['correct'],s['metrics'][b]['correct']); arr.append(q)
pilotfile='offline-suffix-20261121T174021-Bfe0QH/eurosat/predictions.json'
pilot=[pair(s.title(),J(pilotfile),'baseline_'+s,'suffix_'+s,'label','id') for s in ['cde','teacher','student']]
assert [(x['base'],x['new']) for x in pilot]==[(149,146),(192,193),(208,202)]
anchors=C('parallel-study/dino-anchors/reports/combined.csv')
rescue=J(P+'clustering_sweep_eurosat/verification.json'); sig=[J(P+f'siglip_factorial_{k}/verification.json') for k in ['eurosat','cifar100']]
nb=J('NoLA-EuroSAT-Run-It-Yourself.ipynb'); text='\n'.join(''.join(o.get('text',[])) for o in nb['cells'][16]['outputs']); ep=[(int(i),int(a)) for i,a in re.findall(r'Epoch (\d+)/50: val correct=(\d+)/5400',text)]; assert len(ep)==50 and max(ep,key=lambda e:e[1])==(3,4141)
(A/'paired-statistics.json').write_text(json.dumps(dict(suffix=suffix,blend=blend,eurosat_pilot=pilot),indent=2)+'\n')

page=0

def terms(s):
    pairs=[('DINO teacher','DINO alignment head'),('prompted CLIP student','CLIP visual-prompt classifier'),('Prompted student','Prompt-adapted classifier'),('Visual-prompt student','Visual-prompt classifier'),('visual-prompt student','visual-prompt classifier'),('Replacing the student too','Replacing the classifier backbone'),('trained teacher features','alignment-head input features')]
    for a,b in pairs:s=s.replace(a,b)
    for a,b in [('teacher','alignment head'),('student','prompt classifier')]:
        s=re.sub(r'\b'+a+r'(s?)\b',lambda m:(b.capitalize() if m[0][0].isupper() else b)+m[1],s,flags=re.I)
    return s

def txt(f,x,y,s,sz=10,col=INK,weight='normal',**kw):
    return f.text(x,y,terms(s),fontsize=sz,color=col,fontweight=weight,va='top',**kw)

def para(f,x,y,s,width=106,sz=10,col=INK):
    lines=textwrap.wrap(terms(s),width,break_long_words=False,break_on_hyphens=False)
    txt(f,x,y,'\n'.join(lines),sz,col,linespacing=1.36); return y-len(lines)*sz*.00174

def rect(f,x,y,w,h,col): f.patches.append(Rectangle((x,y),w,h,transform=f.transFigure,color=col,zorder=-1,linewidth=0))

def frame(tag,title,subtitle=None):
    global page
    page+=1; f=plt.figure(figsize=(8.27,11.69),facecolor='white')
    rect(f,.075,.963,.85,.004,TEAL); txt(f,.075,.946,tag.upper(),9,TEAL,'bold'); txt(f,.075,.912,title,16 if len(terms(title))>34 else 22,INK,'bold')
    if subtitle: para(f,.075,.865,subtitle,95,10,MUTED)
    rect(f,.075,.053,.85,.001,GRID); txt(f,.075,.042,'NOLA RESEARCH NOTES  /  EXPERIMENT REPORT',8,MUTED); txt(f,.925,.042,f'{page:02d}',8,MUTED,ha='right')
    return f

def ax(f,box):
    a=f.add_axes(box); a.spines[['top','right','left']].set_visible(False); a.spines['bottom'].set_color(GRID)
    a.tick_params(length=0); a.set_axisbelow(True); return a

def fs(f,box,items,xlabel):
    a=ax(f,box); pos=np.arange(len(items))[::-1]; a.axvline(0,color=MUTED,lw=.9,ls='--'); a.grid(axis='x',color=GRID)
    for y,r in zip(pos,items):
        x=r['pp']; lo,hi=r['ci95']; a.hlines(y,lo,hi,color=TEAL if x>=0 else RED,lw=1.8); a.plot(x,y,'o',ms=5,color=TEAL if x>=0 else RED)
    a.set(yticks=pos,yticklabels=[terms(r['name']) for r in items],ylim=(-.55,len(items)-.45),xlabel=terms(xlabel)); return a

def table(f,y,headers,rows,xs):
    h=.026; rect(f,.075,y-h+.005,.85,h,LITE)
    for i,(x,s) in enumerate(zip(xs,headers)): txt(f,x,y,str(s),8.3,TEAL,'bold',ha='left' if i==0 else 'right')
    for j,r in enumerate(rows,1):
        yy=y-j*h
        if j%2==0: rect(f,.075,yy-h+.005,.85,h,'#f5f8fa')
        for i,(x,s) in enumerate(zip(xs,r)): txt(f,x,yy,str(s),8.3,ha='left' if i==0 else 'right')
    return y-(len(rows)+1)*h

def rows(stats):
    return [[q['name'],f"{q['n']:,}",f"{100*q['base']/q['n']:.2f} → {100*q['new']/q['n']:.2f}",f"{q['fix']} / {q['loss']}",f"{q['pp']:+.2f}",f"[{q['ci95'][0]:+.2f}, {q['ci95'][1]:+.2f}]",f"{q['p']:.3g}"] for q in stats]

with PdfPages(PDF,metadata={'Title':'NoLA experiments: evidence and interpretation','Subject':'Paired comparisons, plots, and experiment ledger'}) as pdf:
    def save(f,name=None):
        from matplotlib.text import Text
        for node in f.findobj(Text): node.set_text(terms(node.get_text()))
        pdf.savefig(f)
        if name: f.savefig(A/(name+'.png'),dpi=140)
        plt.close(f)

    f=frame('01 / Baseline and training','NoLA experiments','An evidence review of prompt representations, anchor selection, and CLIP student adaptation.')
    for i,(v,l) in enumerate([('76.27%','EuroSAT student test'),('2 → 98','Herbaceous correct / 600'),('0 / 3','Datasets with an active 0.75 blend')]):
        x=.075+i*.288; rect(f,x,.713,.276,.098,LITE); txt(f,x+.016,.794,v,21,TEAL,'bold'); txt(f,x+.016,.742,l,8.5,MUTED)
    txt(f,.075,.682,'Full-split EuroSAT: CLIP student validation by epoch',13,INK,'bold')
    a=ax(f,[.14,.412,.75,.222]); a.grid(axis='y',color=GRID); xx,yy=np.array(ep).T
    a.plot(xx,100*yy/5400,color=TEAL,lw=1.8,label='Prompted student'); a.axhline(100*3623/5400,color=BLUE,ls='--',label='DINO teacher'); a.axhline(100*2759/5400,color=MUTED,ls=':',label='Frozen CDE')
    a.scatter(3,100*4141/5400,color=TEAL,zorder=3); a.annotate('76.69% at epoch 3',(3,100*4141/5400),xytext=(9,78.7),arrowprops={'arrowstyle':'-','color':MUTED},fontsize=8)
    a.annotate('71.13% at epoch 50',(50,100*3841/5400),xytext=(34,74.6),arrowprops={'arrowstyle':'-','color':MUTED},fontsize=8); a.legend(loc='lower right',frameon=False,fontsize=8); a.set(xlabel='Student epoch',ylabel='Validation top-1 (%)',ylim=(49,81),xlim=(0,53))
    y=para(f,.075,.347,'The selected student checkpoint scored 6,178/8,100 on test. The drop after the early peak is a checkpoint-selection finding, not proof of why training degraded.')
    y=para(f,.075,y-.025,'In the small replay, Herbaceous had 24 true train images, zero first-choice CLIP nominations and one incorrect fallback anchor. More training images or a bigger anchor budget alone did not prevent this failure.')
    para(f,.075,y-.026,'Scope: locally documented experiments; the older frozen Kaggle/DNC study was not re-audited. CDE = frozen CLIP text classifier; teacher = trained DINO head; student = visual-prompted CLIP.',104,9,MUTED)
    save(f,'baseline-training')

    f=frame('02 / Frozen CLIP','Prompt wording vs. vector blending','Effects are estimated on identical image rows within each comparison. Positive values favor the modified prompt or scorer.')
    txt(f,.075,.813,'Appending templates vs. the earlier mean baseline',12,INK,'bold')
    fs(f,[.27,.639,.61,.142],suffix,'Top-1 change (percentage points)')
    txt(f,.075,.560,'0.90 normalized blend vs. appended mean',12,INK,'bold')
    fs(f,[.27,.388,.61,.138],blend,'Top-1 change (percentage points)')
    xs=[.082,.30,.46,.56,.63,.80,.91]; headers=['Dataset','N','Top-1 (%)','Fix / loss','Δ pp','95% CI (pp)','p']
    txt(f,.075,.322,'Append comparison',9,TEAL,'bold'); table(f,.301,headers,rows(suffix),xs)
    txt(f,.075,.184,'Blend comparison',9,TEAL,'bold'); table(f,.163,headers,rows(blend),xs)
    txt(f,.075,.064,'At 0.75 no variants entered on any dataset. At 0.90: 2 EuroSAT, 10 RESISC45, 5 CIFAR clusters.',7.5,MUTED)
    save(f,'cde-effects')

    f=frame('03 / Stage-by-stage contrasts','Does the suffix help after training?','EuroSAT subset test (300 images) and separate full-split student validation milestones (5,400 images).')
    txt(f,.075,.811,'Matched test effects in the small end-to-end run',12,INK,'bold')
    fs(f,[.22,.62,.66,.154],pilot,'Appended minus separate (percentage points)')
    table(f,.548,['Stage','N','Top-1 (%)','Fix / loss','Δ pp','95% CI (pp)','p'],rows(pilot),xs)
    txt(f,.075,.414,'Full-split student validation: the ranking reversed',12,INK,'bold')
    a=ax(f,[.14,.232,.75,.14]); a.grid(axis='y',color=GRID); a.plot([1,2],np.array([4014,3950])/54,'o-',color=BLUE,label='Separate'); a.plot([1,2],np.array([3906,4036])/54,'s-',color=TEAL,label='Appended'); a.set(xlim=(.8,2.2),ylim=(71.9,75.1),xticks=[1,2],xlabel='Student epoch',ylabel='Validation top-1 (%)'); a.legend(loc='upper center',ncol=2,frameon=False,fontsize=8)
    y=para(f,.075,.17,'The full-split DINO teacher test score rose from 66.96% to 68.90%, and Herbaceous anchor count rose from 7 to 21. No final paired full-split student test export was found.',106,9)
    para(f,.075,y-.018,'Oxford Pets: 85.61% to 87.90% on 3,669 images with the one-template comparison (156 fixes, 72 losses). The CIFAR-100 suffix contrast also changes wording and class-name use; it does not isolate placement.',115,8.7,MUTED)
    save(f,'suffix-training')

    f=frame('04 / Anchor selection','Cleaner anchors, mixed accuracy','Matched anchor budgets. The panels show teacher validation, not the prompted CLIP student.')
    labels={'farthest_first':'Farthest-first','medoids':'Medoids','neighbor_agreement':'Neighbor agreement'}
    items=[]
    for k,l in zip(keys,D):
        sub=[z for z in anchors if z['dataset']==k]; base=float(sub[0]['anchor_precision_pct']); assert sub[0]['method']=='confidence'
        for z in sub[1:]:
            assert round(float(z['delta_pp']),7)==round(100*(int(z['fixed'])-int(z['regressions']))/int(z['val_n']),7)
            items.append(dict(name=l+' / '+labels[z['method']],pp=float(z['delta_pp']),ci95=[float(z['paired_ci_low_pp']),float(z['paired_ci_high_pp'])],precision=float(z['anchor_precision_pct'])-base))
    txt(f,.075,.815,'Teacher accuracy vs. confidence top-k (archived paired 95% intervals)',11,INK,'bold')
    fs(f,[.43,.553,.45,.222],items,'Teacher top-1 change (pp)')
    y=para(f,.075,.489,'Neighbor agreement improved anchor precision on all three datasets, but teacher gains were clear only on EuroSAT. Farthest-first and medoids lost teacher accuracy on all three.',109,9)
    txt(f,.075,.400,'EuroSAT Herbaceous: widening the candidate pool mattered',12,INK,'bold')
    trg=[q for q in rescue['target_classes'] if q['class_id']=='2']; order=['baseline','clip_expansion','spherical_kmeans','ward_pca32','dino_seed_retrieval','dbscan']; trg=sorted(trg,key=lambda q:order.index(q['arm']))
    a=ax(f,[.31,.163,.56,.196]); a.grid(axis='x',color=GRID); yy=np.arange(len(trg))[::-1]; vv=[int(q['val_correct']) for q in trg]; a.barh(yy,vv,color=[BLUE]*4+[TEAL,BLUE]); a.set(yticks=yy,yticklabels=['No rescue','CLIP expansion','Spherical k-means','Ward PCA','Direct DINO retrieval','DBSCAN'],xlabel='Correct Herbaceous predictions / 600',xlim=(0,max(vv)*1.20));
    for y,v in zip(yy,vv): a.text(v+1,y,str(v),va='center',fontsize=8)
    para(f,.075,.104,'Earlier cluster-first added zero Herbaceous anchors; all 113 additions went to Pasture. This later comparison used a broader pool and matched 64 additions per target class.',116,8.7,MUTED)
    save(f,'anchor-rescue')

    f=frame('05 / Reranking and methods','What SigLIP 2 actually tested','CLIP still supplied seeds and the candidate pool; SigLIP 2 was a reranker, not a full replacement.')
    si=[]; tab=[]
    for s,l in zip(sig,['EuroSAT','CIFAR-100 scarcity pilot']):
        for e in ['dino','convnext']:
            q=s['paired'][f'siglip2+{e} vs clip+{e}']; lab=l+' / '+('DINO' if e=='dino' else 'ConvNeXt'); si.append(dict(name=lab,pp=q['pp'],ci95=q['ci95_pp'])); x=s['results']['clip+'+e]; z=s['results']['siglip2+'+e]; n=s['val_n']; tab.append([lab,f"{x['correct_additions']} → {z['correct_additions']} / {s['added']}",f"{100*x['val_correct']/n:.2f} → {100*z['val_correct']/n:.2f}"])
    txt(f,.075,.811,'Swap only the reranker, keeping the retrieval encoder fixed',12,INK,'bold')
    fs(f,[.47,.596,.40,.169],si,'SigLIP 2 minus CLIP: teacher top-1 (pp)')
    table(f,.525,['Dataset / retriever','Correct added anchors','Teacher top-1 (%)'],tab,[.082,.685,.91])
    txt(f,.075,.344,'Statistical reading',13,INK,'bold')
    y=para(f,.075,.312,'New paired intervals: 20,000 percentile bootstrap draws of correctness differences, stratified by true class (fine subtype for CIFAR-100). Fixes and losses are computed on the same images; p is the two-sided exact McNemar test. These exploratory p-values are not adjusted for multiple comparisons.',110,9)
    y=para(f,.075,y-.023,'Anchor and reranking plots use the paired bootstrap intervals saved with their original reports. Image-level intervals do not capture variation from retraining seeds, prompt choices, split construction, or repeated development-set use.',110,9)
    para(f,.075,y-.023,'The next page tests whether SigLIP 2 can nominate anchors independently. Independent nomination is a distinct change from reranking a pool defined by CLIP.',110,9)
    save(f,'siglip-reranker')

    f=frame('06 / Full 2 × 2 comparison','SigLIP 2 × DINOv3 ConvNeXt','Four arms per dataset: CLIP or SigLIP 2 reranking crossed with original DINO or DINOv3 ConvNeXt-Tiny retrieval. The trained teacher features are original DINO in every arm.')
    arms=['clip+dino','siglip2+dino','clip+convnext','siglip2+convnext']
    arm_labels=['CLIP + DINO','SigLIP 2 + DINO','CLIP + DINOv3 ConvNeXt','SigLIP 2 + DINOv3 ConvNeXt']
    for s,l,top in zip(sig,['EuroSAT  /  5,400 validation images','CIFAR-100 coarse scarcity pilot  /  1,024 validation images'],[.799,.499]):
        txt(f,.075,top,l,12,INK,'bold')
        a=ax(f,[.425,top-.216,.46,.176]); a.grid(axis='x',color=GRID)
        y=np.arange(4)[::-1]; base=s['results'][arms[0]]['val_correct']; n=s['val_n']
        for i,k in enumerate(arms):
            q=s['results'][k]; d=100*(q['val_correct']-base)/n; color=TEAL if i%2 else BLUE
            if i:
                z=s['paired'][k+' vs '+arms[0]]; assert abs(d-z['pp'])<1e-10
                a.hlines(y[i],*z['ci95_pp'],color=color,lw=2.0)
            a.plot(d,y[i],'o',color=color,ms=6)
        a.axvline(0,color=MUTED,ls='--',lw=.8); a.set(xlim=(-.8,2.0),yticks=y,yticklabels=arm_labels,ylim=(-.55,3.55),xlabel='Teacher top-1 change vs. CLIP + DINO (pp)')
        vals=[]
        for k in arms:
            q=s['results'][k]; vals.append(f"{100*q['val_correct']/n:.2f}% ({q['correct_additions']}/{s['added']})")
        para(f,.075,top-.265,'Top-1 (correct additions), in plot order:  '+'    /    '.join(vals),117,8.2,MUTED)
    para(f,.075,.133,'CIFAR scarcity was simulated. EuroSAT had naturally underfilled classes. Intervals compare each arm with CLIP + original DINO within the same dataset. Better anchor precision did not consistently improve the teacher.',109,9)
    save(f,'siglip-convnext-factorial')

    f=frame('07 / Independent nomination','Can SigLIP 2 replace CLIP?','The replacement selects its own initial anchors and its own expansion candidates. Original DINO retrieval and teacher stay fixed; no visual-prompt student was trained.')
    arms=['clip_only','clip_pool_siglip_rank','siglip_only']; labels=['CLIP only','CLIP pool / SigLIP rerank','SigLIP 2 only']
    repl=[J(f'siglip-replacement/{k}/verified.json') for k in ['eurosat','cifar100']]
    fs(f,[.32,.627,.59,.167],[dict(name=l,pp=s['paired']['siglip_only vs clip_only']['delta_pp'],ci95=s['paired']['siglip_only vs clip_only']['ci95_pp']) for s,l in zip(repl,['EuroSAT','CIFAR-100 pilot'])],'Matched teacher top-1 change vs. CLIP-only (pp)')
    for i,(s,l) in enumerate(zip(repl,['EuroSAT','CIFAR-100 coarse pilot'])):
        y=.548-.155*i; n=s['val_n']; txt(f,.075,y,f'{l}  /  {n:,} validation images',11,INK,'bold')
        table(f,y-.028,['Arm','Correct anchors','Teacher top-1'],[[m,f"{s['arms'][k]['correct_anchors']:,} / {s['arms'][k]['anchors']:,}",f"{100*s['arms'][k]['val_correct']/n:.2f}%"] for k,m in zip(arms,labels)],[.082,.72,.91])
    y=para(f,.075,.231,'EuroSAT: SigLIP nominates 144 correct Herbaceous examples where CLIP nominates zero at top-1, but it selects only 2 correct Herbaceous anchors out of 71 at matched budgets, versus 38/71 for CLIP. With native budgets Herbaceous recall improves (98 to 345 out of 600), but overall teacher accuracy still drops from 68.02% to 60.98%.',108,9)
    para(f,.075,y-.018,'The CIFAR pilot did not need expansion, so the rerank control duplicates CLIP-only. These are single-seed development-validation results; the intervals quantify paired image variation, not training-seed variation. No end-to-end timing was measured from these cached runs.',108,9,MUTED)
    save(f,'siglip-independent')

    f=frame('08 / Independent nomination factorial','Replacement × retrieval','Each VLM independently selects seeds and expansion candidates. Per-class quotas and downstream original-DINO features are matched in all four arms.')
    labels=['CLIP + DINO','SigLIP 2 + DINO','CLIP + ConvNeXt','SigLIP 2 + ConvNeXt']; arms=['clip+dino','siglip2+dino','clip+convnext','siglip2+convnext']
    for i,(k,l) in enumerate(zip(['eurosat','cifar100'],['EuroSAT, full validation','CIFAR-100 coarse, simulated scarcity'])):
        s=J(f'siglip-replacement/factorial/{k}/verified.json'); n=s['val_n']; y=.805-.335*i
        txt(f,.075,y,f'{l} / {n:,} images',12,INK,'bold'); stats=[]; rr=[]
        for a,m in zip(arms,labels):
            v=s['arms'][a]; z=s['paired'].get(a+' vs clip+dino',dict(delta_pp=0.,ci95_pp=[0.,0.])); stats.append(dict(name=m,pp=z['delta_pp'],ci95=z['ci95_pp']));rr.append([m,f"{v['correct_anchors']:,}/{v['anchors']:,}",f"{v['correct_extra']}/{sum(s['matched_extra'])}",f"{100*v['correct']/n:.2f}%"])
        fs(f,[.36,y-.181,.54,.145],stats,'Teacher top-1 change vs. CLIP + DINO (pp)')
        table(f,y-.239,['Arm','Correct anchors','Correct additions','Top-1'],rr,[.082,.597,.76,.91])
    para(f,.075,.079,'95% within-class paired bootstrap intervals. Earlier CLIP controls reproduce exactly.',115,7.8,MUTED)
    save(f,'siglip-independent-factorial')

    # Attach the student pilot only after both datasets have independently verified outputs.
    paths=[f'siglip-replacement/students/{k}/verified.json' for k in ('eurosat','cifar100')]
    if all((R/p).is_file() for p in paths):
        f=frame('09 / Visual-prompt student pilot','Replacing the student too','The student uses the same backbone as its anchor nominator. Original-DINO teacher predictions replayed exactly before training.')
        arm_ids=['clip+dino','siglip2+dino','clip+convnext','siglip2+convnext']; labels=['CLIP + DINO','SigLIP 2 + DINO','CLIP + ConvNeXt','SigLIP 2 + ConvNeXt']
        for i,(p,l) in enumerate(zip(paths,['EuroSAT','CIFAR-100 coarse scarcity'])):
            s=J(p); n=s['val_n']; y=.805-i*.339; txt(f,.075,y,f"{l} / {s['train_n']} fit, {n} validation images",12,INK,'bold')
            a=ax(f,[.21,y-.196,.69,.155]); a.grid(axis='y',color=GRID); a.set(xticks=[0,1,2],xticklabels=['Frozen VLM','Prompt tuning: epoch 1','Prompt tuning: epoch 2'],ylabel='Top-1 (%)',ylim=(0,100))
            for j,(k,m) in enumerate(zip(arm_ids,labels)):
                z=s['arms'][k]; a.plot(range(3),np.array([z['frozen_correct'],*[q['correct'] for q in z['history']]])*100/n,marker=('o' if j<2 else 's'),ls=('-' if j<2 else '--'),color=(BLUE if j%2==0 else TEAL),label=m)
            if not i: a.legend(ncol=2,frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(.5,1.23))
            tab=[]
            for k,m in zip(arm_ids,labels):
                z=s['arms'][k]; tab.append([m,f"{100*z['teacher_correct']/n:.2f}%",f"{100*z['frozen_correct']/n:.2f}%",f"{100*z['student_correct']/n:.2f}%"])
            table(f,y-.238,['Arm','Teacher','Frozen VLM','Student, epoch 2'],tab,[.082,.57,.72,.91])
        save(f,'siglip-student-pilot')

    milestones=[ep for ep in (4,12) if all((R/f'siglip-replacement/students-epoch{ep}/{k}/verified.json').is_file() for k in ('eurosat','cifar100'))]
    if milestones:
        ep=milestones[-1]; f=frame('10 / Matched longer training','Does the ranking hold?','Continuation restores the same optimizer and prompts. Same 320 fit images per dataset; both datasets use the original pilot validation rows.')
        ids=['clip+dino','siglip2+dino','clip+convnext','siglip2+convnext']; labels=['CLIP + DINO','SigLIP 2 + DINO','CLIP + ConvNeXt','SigLIP 2 + ConvNeXt']
        for i,(k,label) in enumerate(zip(('eurosat','cifar100'),('EuroSAT','CIFAR-100 coarse scarcity'))):
            y=.805-i*.340; s=J(f'siglip-replacement/students-epoch{ep}/{k}/verified.json'); n=s['val_n']; assert s['epochs']==ep
            txt(f,.075,y,f'{label} / {n} validation images',12,INK,'bold')
            a=ax(f,[.20,y-.205,.70,.163]); a.grid(axis='y',color=GRID); a.set(xlabel='Student epoch',ylabel='Top-1 (%)',xticks=[1,2,3,4] if ep==4 else [2,4,6,8,10,12]); a.axvline(2,color=MUTED,ls=':',lw=.8)
            tab=[]
            for j,(m,l) in enumerate(zip(ids,labels)):
                h=s['arms'][m]['history'];assert [r['epoch'] for r in h]==list(range(1,ep+1));a.plot([r['epoch'] for r in h],[100*r['correct']/n for r in h],marker=('o' if j<2 else 's'),ms=4,ls=('-' if j<2 else '--'),color=(BLUE if j%2==0 else TEAL),label=l)
                tab.append([l,*[f"{100*h[t-1]['correct']/n:.2f}%" for t in (2,4)],f"{100*h[-1]['correct']/n:.2f}%" if ep>4 else '—'])
            if i==0: a.legend(ncol=2,frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(.52,1.22))
            table(f,y-.247,['Arm','Epoch 2','Epoch 4','Epoch 12'],tab,[.082,.62,.77,.91])
        save(f,'siglip-student-continuation')

    cards,card_sections,stage_note=build_scorecard(R,pair)
    f=frame('Dataset scorecard / coverage','Which stages were measured?', 'A missing result is not a zero. Full alignment-head validation and small visual-prompt pilots are listed separately.')
    y=para(f,.075,.806,stage_note,103,9,MUTED)
    table(f,y-.035,['Dataset','Frozen CDE','Alignment head','Visual-prompt classifier'],[
        ['EuroSAT','5,400 val','5,400 val','200 val pilot; 8,100 baseline test'],
        ['CIFAR-100 coarse','5,000 val','5,000 standard; 1,024 scarcity val','400 val scarcity pilot'],
        ['RESISC45','6,255 val','6,255 val','Not run'],
        ['Oxford-IIIT Pet','3,669 test','Not run in template comparison','Not run in template comparison']], [.082,.39,.66,.912])
    txt(f,.075,.518,'DINO alignment head: anchor-policy study',12,INK,'bold')
    anchor_rows=[]
    for k,l in zip(keys,D):
        z=[q for q in anchors if q['dataset']==k]; anchor_rows.append([l,z[0]['val_n'],*[f"{float(q['val_accuracy_pct']):.2f}%" for q in z]])
    table(f,.484,['Dataset','N','Confidence','Farthest-first','Medoids','Neighbor agreement'],anchor_rows,[.082,.34,.47,.60,.72,.91])
    para(f,.075,.34,'The four-arm replacement study and the twelve-epoch visual-prompt run cover EuroSAT and CIFAR-100 only. RESISC45 contributes frozen CDE and anchor-policy results; Oxford-IIIT Pet contributes the frozen template comparison.',107,9)
    save(f)
    for card in cards:
        f=frame('Dataset scorecard / independent nomination',card['label'],f"{card['head_fit']:,} adaptation images for anchors. Prompt tuning uses 320 fit images. Stage differences use the same {card['prompt_n']} validation rows.")
        layouts=[([.082,.52,.74,.91],.822),([.082,.428,.612,.747,.91],.637),([.082,.47,.59,.71,.91],.452),([.082,.65,.91],.267)]
        for (head,cols,key),(xx,y) in zip(card_sections,layouts):
            if key=='overview': head=f"DINO alignment head / {card['head_n']:,} validation images"
            if key=='stage': cols=['Arm','Alignment head','Prompt classifier','Fix / loss','Δ pp [95% CI]']
            if key=='details': cols=['Arm','Epoch 2','Epoch 4','Epoch 12','Best val (epoch)']
            if key=='contrasts': cols=['Contrast','Alignment head (pp)','Epoch-12 classifier (pp)']
            txt(f,.075,y,head,10,INK,'bold'); table(f,y-.023,cols,card[key],xx)
        para(f,.075,.093,'Δ = later minus earlier or named contrast. Intervals quantify paired image-level variation at one seed. Best observed validation is descriptive, not held-out test accuracy.',110,8,MUTED)
        save(f)

    # Include the narrative draft and the complete inventory, with page breaks per record.
    sections=[]
    for file,tag in [('NoLA-Overall-Report-Draft.md','Overall report / draft'),('EXPERIMENT-SUMMARY.md','Appendix / experiment inventory')]:
        md=(R/file).read_text(); sections.extend((tag,s) for s in re.split(r'^## ',md,flags=re.M)[1:])
    fieldlabels=['HYPOTHESIS','RESULT','INTERPRETATION']
    clean=lambda s:s.replace('**','').replace('`','')
    for tag,sec in sections:
        lines=sec.splitlines(); heading=lines[0]; items=[]; headings=None
        for line in lines[1:]:
            if not line.strip() or line.startswith('|---') or line.startswith('| Experiment'): continue
            if line.startswith('|'):
                vals=[clean(v.strip()) for v in line.strip('|').split('|')]
                if tag.startswith('Overall'):
                    if headings is None: headings=vals; continue
                    assert len(headings)==len(vals)
                    items.append([(vals[0],'  /  '.join(f'{k}: {v}' for k,v in zip(headings[1:],vals[1:])))])
                else:
                    assert len(vals)==3; items.append(list(zip(fieldlabels,vals)))
            else: items.append([('',clean(line))])
        def appendix(continued=False): return frame(tag,heading.split('. ',1)[-1], 'Continued from the previous page.' if continued else ('Hypotheses, measured results, and failure modes; stages and splits are kept separate.' if tag.startswith('Appendix') else None))
        f=appendix(); y=.815
        for item in items:
            wrapped=[(l,textwrap.wrap(terms(v),98,break_long_words=True,break_on_hyphens=False)) for l,v in item]
            need=sum(len(w)*.0147+(.016 if l else 0)+.008 for l,w in wrapped)+.013
            if y-need<.078: save(f); f=appendix(True); y=.815
            for l,w in wrapped:
                if l: txt(f,.085,y,l,7.9,TEAL,'bold'); y-=.016
                txt(f,.085,y,'\n'.join(w),8.5,INK,linespacing=1.34); y-=len(w)*.0147+.008
            if len(item)>1: rect(f,.085,y,.815,.0008,GRID)
            y-=.013
        save(f)
print(PDF,page,'pages')
