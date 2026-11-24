from pathlib import Path
import json, re, textwrap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

root = Path(__file__).resolve().parent
read = lambda name: json.loads((root/name).read_text())
p = read('provenance.json'); data = read('metrics.json'); history = read('history.json')
texts = read('cde-text.json'); aug = read('augmentations.json')
anchors = pd.read_csv(root/'anchors.csv'); cand = pd.read_csv(root/'top-k-candidates.csv')
pred = pd.read_csv(root/'predictions.csv')
classes = data['class_order']; j = p['job']
lookup = {(m['stage'], m['split']):m for m in data['metrics']}
stages = [('cde','Unprompted CLIP CDE'), ('dino_head','Fitted DINO head'), ('prompted_init','Prompted CLIP before training'), ('prompted_epoch_1','Prompted CLIP, epoch 1'), ('prompted_epoch_4','Prompted CLIP, epoch 4')]
for m in data['metrics']:
    frame = pred[(pred.stage==m['stage']) & (pred.split==m['split'])]; assert len(frame)==m['n'] and frame.correct.sum()==m['correct'] and np.asarray(m['confusion']).sum()==m['n']
assert len(anchors)==data['anchors'] and int(anchors.correct.sum())==data['anchor_correct'] and len(cand)==data['candidate_count']
assert anchors.row.is_unique and int(cand.selected.sum())==len(anchors)
assert len(aug)==6 and set(texts)==set(classes)
for image in list(cand.image) + [a[key] for a in aug for key in ('original','weak','strong')]:
    with Image.open(root/image) as im: im.verify()

def score(key,part):
    m=lookup.get((key,part)); return f"{m['correct']}/{m['n']} ({100*m['correct']/m['n']:.2f}%)" if m else 'Not measured'

gpu = next(r for r in p['logged_gpu']['records'] if r['clip_epoch']==4)
fig_dir = root/'figures'; fig_dir.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
navy='#153047'; green='#177865'; red='#aa4937'; gray='#4c6070'

with PdfPages(root/'NoLA-EuroSAT-Case-Study.pdf', metadata={'Title':'NoLA on EuroSAT: a small case study','Subject':'Audited anchor selection, augmentation and stage results'}) as pdf:
 def page(title, subtitle=None):
    f=plt.figure(figsize=(8.27,11.69),facecolor='white')
    f.text(.075,.958,title,fontsize=22,fontweight='bold',color=navy)
    if subtitle: f.text(.075,.928,subtitle,fontsize=9,color=gray)
    return f
 def para(f,y,value,size=10,width=94):
    text='\n'.join(textwrap.wrap(value,width,break_long_words=False)); f.text(.075,y,text,fontsize=size,va='top',color=navy,linespacing=1.45)
 def save(f,name=None):
    if name: f.savefig(fig_dir/f'{name}.pdf')
    pdf.savefig(f); plt.close(f)
 f=page('NoLA on EuroSAT','CASE STUDY  /  160 adaptation images  /  one selected configuration')
 para(f,.882,f"The saved GPU grid found a setting where prompted CLIP beat both baselines: {gpu['nola']}/240 versus {gpu['cde']}/240 unprompted CDE and {gpu['dino_head']}/240 with the fitted DINO head. This report separately replays that setting on CPU to capture the evidence. Results can change with device, numerical precision and random-number implementation; the two runs must not be conflated.")
 para(f,.779,'Adaptation: 160 images from the published CoOp training split, seed 42. Validation: 240 images, seed 123. Fresh held-out audit: 240 test images, seed 137, excluding the 80 test images previously used in the teaching notebook. The epoch-4 recipe was fixed from the saved validation sweep before computing the held-out audit. Neither validation nor test labels were training targets.')
 cfg=[('CLIP / DINO backbones','ViT-B/32 / ViT-B/16 (frozen DINO)'), ('CLIP candidate scores','25 released descriptions + 6 fixed templates per class'), ('Anchor cap / returned',f"{j['anchor_cap']} per predicted class / {len(anchors)} anchors total"), ('DINO head',f"768 → 256 → 10; Adam 0.001; batch {j['head_batch']}; {j['head_epochs']} epochs"), ('Prompted CLIP',f"{j['prompts']} visual prompts; AdamW {j['clip_lr']}; batch {j['clip_batch']}; 4 epochs"), ('CLIP loss/weight decay','Label smoothing 0.1; regular group decay 0.01; biases 0')]
 ax=f.add_axes([.075,.465,.85,.215]); ax.axis('off'); t=ax.table(cellText=cfg,colLabels=['Setting','Case-study recipe'],colWidths=[.34,.66],loc='center',cellLoc='left'); t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,1.6)
 ax=f.add_axes([.075,.217,.85,.19]); ax.axis('off'); table=[[label,score(key,'validation'),score(key,'fresh_test')] for key,label in stages]; t=ax.table(cellText=table,colLabels=['CPU replay stage','Validation','Fresh test audit'],colWidths=[.42,.29,.29],loc='center',cellLoc='left');t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,1.55)
 d=lookup['prompted_epoch_4','fresh_test']['correct']; h=lookup['dino_head','fresh_test']['correct']; c=lookup['cde','fresh_test']['correct']
 para(f,.16,f"On the fresh test audit, final NoLA changed the correct count by {d-c:+d} relative to CDE and {d-h:+d} relative to the DINO head. This is one selected small run, not a reproduction of the paper’s 73.5% result and not evidence of a universal improvement.")
 save(f)
 f=page('What changed at each stage','All entries come from the instrumented CPU replay.')
 ax=f.add_axes([.10,.648,.80,.23]); x=np.arange(len(stages)); colors=[navy,green];
 for i,part in enumerate(('validation','fresh_test')): ax.bar(x+(i-.5)*.35,[100*lookup[key,part]['correct']/lookup[key,part]['n'] for key,_ in stages],.35,label=part.replace('_',' '),color=colors[i]);
 ax.set_xticks(x,['CDE','DINO head','Prompt init','Prompt E1','Prompt E4']); ax.set_ylabel('Top-1 accuracy (%)'); ax.legend(frameon=False); ax.set_ylim(0,100)
 ax=f.add_axes([.10,.390,.35,.16]); hh=[r for r in history if r['stage']=='dino_head']; ax.plot([r['epoch'] for r in hh],[r['mean_loss'] for r in hh],'o-',color=green); ax.set(title='DINO head: anchor loss',xlabel='Epoch',ylabel='Mean loss')
 ax=f.add_axes([.55,.390,.35,.16]); cc=[r for r in history if r['stage']=='prompted_clip']; ax.plot([r['epoch'] for r in cc],[r['mean_loss'] for r in cc],'o-',color=navy); ax.set(title='Prompted CLIP: pseudo-label loss',xlabel='Epoch',ylabel='Mean loss')
 para(f,.325,f"Adaptation accuracy (audit only): CDE {score('cde','adaptation')}; DINO labels {score('dino_head','adaptation')}. Anchor label precision: {int(anchors.correct.sum())}/{len(anchors)}. DINO fits a small head to CLIP pseudo-labels, then supplies hard targets for all adaptation images. Prompted CLIP changes visual prompts, visual normalization parameters and the class weights initialized from CDE.")
 para(f,.218,'Loss is measured against pseudo-labels. It may decrease without an increase in accuracy on either held-out split. The randomly prompted initialization is not the unprompted baseline: extra tokens can change attention before the first optimizer step.')
 para(f,.128,'The bundle contains class-by-class counts, confusion matrices and image-level probabilities, including intermediate stages that are not shown in every figure.')
 save(f,'stage-results')
 f=page('Class-level diagnostics','Anchor count is after deduplication. Per-class accuracy uses only the true-class denominator.')
 records=[]
 for i,name in enumerate(classes):
    full=anchors[anchors.pseudo_class==name]; den=len(full); hit=int(full.correct.sum());
    acc=[]
    for key in ('cde','dino_head','prompted_epoch_4'):
        m=lookup[key,'validation']['per_class'][i]; acc.append(f"{m['correct']}/{m['n']}")
    records.append([name,f'{hit}/{den}',*acc])
 ax=f.add_axes([.067,.57,.86,.32]); ax.axis('off'); t=ax.table(cellText=records,colLabels=['Class','Anchors correct','CDE','DINO','NoLA E4'],colWidths=[.33,.21,.14,.14,.18],cellLoc='left',loc='center'); t.auto_set_font_size(False);t.set_fontsize(8);t.scale(1,1.42)
 ax=f.add_axes([.21,.145,.59,.35]); ax.imshow(lookup['prompted_epoch_4','validation']['confusion'],cmap='Blues'); ax.set_xticks(range(10),range(10)); ax.set_yticks(range(10),range(10)); ax.set(xlabel='Predicted class ID',ylabel='True class ID',title='NoLA epoch 4: validation confusion matrix'); ax.set_aspect('auto')
 f.text(.075,.086,'IDs 0–9 follow the order in the table; counts are not row-normalized.',fontsize=9,color=gray)
 save(f,'per-class')
 for page_i, classes_part in enumerate((classes[:5],classes[5:]),1):
    f=page(f'Top-k candidate images · {page_i}/2','Every pre-deduplication top-k candidate. Green = retained; red = removed as a duplicate.')
    for i,name in enumerate(classes_part):
      y=.855-i*.159; f.text(.075,y,name,fontsize=10,fontweight='bold',color=navy)
      for col,(_,row) in enumerate(cand[cand.class_name==name].iterrows()):
        xx=.075+col*.221; ax=f.add_axes([xx,y-.102,.16,.091]); ax.imshow(Image.open(root/row.image)); ax.axis('off')
        text=f"#{row.within_class_rank} · rank {row.prediction_rank} · p={row.probability:.3f}\ntrue: {row.true_class}"
        f.text(xx,y-.113,text,fontsize=7,va='top',color=green if row.selected else red)
    save(f,f'anchors-{page_i}')
 f=page('From original to training view','These strong views were captured from the actual first-epoch tensors used for CLIP updates.')
 for col,name in enumerate(('Original RGB image','Weak / deterministic view','Strong / actual CLIP input')): f.text(.075+col*.29,.882,name,fontsize=9,color=navy,fontweight='bold')
 for i,a in enumerate(aug):
    y=.748-i*.115
    for col,key in enumerate(('original','weak','strong')):
        ax=f.add_axes([.075+col*.29,y,.18,.092]); ax.imshow(Image.open(root/a[key])); ax.axis('off')
    f.text(.075,y-.010,f"True: {a['true_class']} · DINO target: {a['dino_label']}",fontsize=7,color=gray)
 para(f,.044,'Weak: resize + center crop. Strong: random crop/resize, flip, jitter, perspective and blur. Views shown after undoing channel normalization.',8,113)
 save(f,'augmentations')
 f=page('DINO feature-space audit','PCA is fitted on adaptation images only; label colors are for interpretation.')
 ax=f.add_axes([.04,.11,.92,.77]); ax.imshow(Image.open(fig_dir/'dino-pca.png')); ax.axis('off')
 para(f,.105,'PCA keeps the two directions with the most variance, not necessarily the most discriminative directions. It is a diagnostic, not an input to the DINO head. Match the annotated row IDs to anchors.csv.',8,113)
 save(f)
 for name,entry in texts.items():
    f=page(name,'TEXT APPENDIX  /  exact text supplied to the CLIP encoder')
    body=[]
    for i,s in enumerate(entry['descriptions'],1): body.extend(textwrap.wrap(f'{i:02d}. {s}',102,subsequent_indent='    '))
    body.extend(['','Filled CLIP templates:']); body.extend(f'T{i}. {v}' for i,v in enumerate(entry['filled_templates'],1))
    fs=min(9,510/max(len(body),1)); f.text(.075,.89,'\n'.join(body),fontsize=fs,va='top',color=navy,linespacing=1.27)
    save(f)

special = {'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','\\':r'\textbackslash{}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','→':' to ','–':'--','’':"'",'“':'``','”':"''",'×':r'\(\times\)'}
def esc(s): return ''.join(special.get(c,c if ord(c)<128 else '?') for c in str(s))
tex=[r'\documentclass[11pt]{article}',r'\usepackage[a4paper,margin=23mm]{geometry}',r'\usepackage{graphicx,booktabs,longtable}',r'\title{NoLA on EuroSAT: a small case study}',r'\date{}',r'\begin{document}',r'\maketitle',r'\section{Design and provenance}',esc(f"Selected GPU validation run: NoLA {gpu['nola']}/240, CDE {gpu['cde']}/240, DINO head {gpu['dino_head']}/240. The tables below are a separate CPU replay, not the original GPU measurements. Data: 160 adaptation, 240 validation, and a 240-image fixed test audit excluding the 80 earlier teaching test images. The recipe was selected on validation, not on this test audit."),r'\begin{itemize}']
tex += [r'\item '+esc(k+': '+str(v)) for k,v in j.items()]
tex += [r'\end{itemize}',r'\section{Replay results}',r'\begin{tabular}{lll}\toprule Stage & Validation & Fresh test\\\midrule']
tex += [' & '.join(map(esc,(label,score(key,'validation'),score(key,'fresh_test'))))+r'\\' for key,label in stages]
tex += [r'\bottomrule\end{tabular}',r'\clearpage']
for name in ('stage-results','per-class','anchors-1','anchors-2','augmentations','dino-pca'):
    tex.append(r'\noindent\includegraphics[width=\textwidth,height=0.93\textheight,keepaspectratio]{figures/'+name+r'.pdf}\clearpage')
tex += [r'\appendix\section{Exact CDE inputs}']
for name,entry in texts.items():
    tex += [r'\subsection{'+esc(name)+'}',r'\begin{enumerate}']+[r'\item '+esc(t) for t in entry['descriptions']]+[r'\end{enumerate}',r'\paragraph{Filled CLIP templates}\begin{itemize}']+[r'\item '+esc(t) for t in entry['filled_templates']]+[r'\end{itemize}']
tex += [r'\end{document}']; (root/'NoLA-EuroSAT-Case-Study.tex').write_text('\n'.join(tex)+'\n')
(root/'README.md').write_text('''# NoLA EuroSAT case study\n\nRead `NoLA-EuroSAT-Case-Study.pdf` first. The editable LaTeX source references the PDFs in `figures/`; compile it from this directory with `pdflatex` if available. The delivered PDF was rendered with Matplotlib, not LaTeX.\n\n- `provenance.json`: selected original GPU log entry, replay configuration, identities and hashes.\n- `metrics.json`, `history.json`, `predictions.csv`: correct counts, per-class counts, confusion matrices, epoch losses and per-image class scores. Probability columns follow `metrics.json` class order. No per-image adaptation scores were collected for prompted CLIP, only for CDE and DINO.\n- `cde-text.json`, `cde.pt`: original saved descriptions, filled templates and computed matrix.\n- `top-k-candidates.csv`, `anchors.csv`, `anchors/`: pre-deduplication candidates, the retained anchors and original JPEGs. `prediction_rank` on fallback means rank among CLIP's class guesses; `within_class_rank` is image rank within the chosen fallback tier.\n- `augmentations.json`, `augmentations/`: original, weak and actual strong training views plus labels.\n- `capture.py` and `build_report.py`: run with the Robotics Research Lab Python environment. Capture reads the original sweep notebook code and data under the parent directory; it writes a CPU replay. The checkpoint files are compact evaluation weights, not full optimizer snapshots.\n\nAccuracy conclusions are descriptive for a single configuration and seed, not paper reproduction or deployment claims.\n''')
print('Verified and rendered:',len(stages),'stages,',len(cand),'candidate entries,',len(aug),'augmentation triplets,',len(pred),'prediction rows')
