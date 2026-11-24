from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent
root = out.parent
source = root/'NoLA-EuroSAT-Full-Sweep.ipynb'
cells = [''.join(c['source']) for c in json.loads(source.read_text())['cells'] if c['cell_type']=='code' and ''.join(c['source']).strip()]
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
exec(cells[0].replace('.cuda()', '.cpu()'), globals())
pool=pool[:160]
# Only load the frozen DINO encoder; do not fit or invoke CLIP.
s = cells[2].split('pool_features =')[0]
s = s.replace('model.cpu()','').replace('.cuda()', '.cpu()')
exec(s,globals())
train = dino_features(pool).numpy().astype('float64')
valid = dino_features(val).numpy().astype('float64')
mu = train.mean(0)
_, singular, v = np.linalg.svd(train-mu,full_matrices=False)
axes = v[:2]
rates = (singular[:2]**2)/(singular**2).sum()
x = (train-mu) @ axes.T
z = (valid-mu) @ axes.T
np.savez_compressed(out/'dino-pca.npz',train_features=train.astype('float32'),val_features=valid.astype('float32'),mean=mu,components=axes,train_xy=x,val_xy=z,explained_variance_ratio=rates)
rows = []
for part, images, xy in [('adaptation',pool,x),('validation',val,z)]:
    for idx,(row,q) in enumerate(zip(images,xy)):
        rows.append(dict(split=part,row=idx,image_path=row[0],true_class=row[2],pc1=q[0],pc2=q[1]))
pd.DataFrame(rows).to_csv(out/'dino-pca-coordinates.csv',index=False)
a=pd.read_csv(out/'anchors.csv')
colors=plt.get_cmap('tab10')
fig, panels=plt.subplots(2,2,figsize=(8.27,8.8),sharex=True,sharey=True)
fig.subplots_adjust(left=.10,right=.96,top=.82,bottom=.18,wspace=.12,hspace=.27)
fig.text(.08,.95,'DINO features in two dimensions',fontsize=21,fontweight='bold')
fig.text(.08,.88,f'PCA fit on 160 adaptation vectors only. Validation and anchors share these PCA axes.\nPC1 + PC2 explain {100*rates.sum():.1f}% of adaptation variance; overlap in 2D is not proof of overlap in 768D.',fontsize=9)
for ax,xy,samples,title in [(panels[0,0],x,pool,'Adaptation: true class (audit)'),(panels[0,1],z,val,'Validation: true class (audit)')]:
 for i,name in enumerate(names):
    mask=np.array([r[1]==i for r in samples]); ax.scatter(xy[mask,0],xy[mask,1],s=15,alpha=.74,color=colors(i),label=name)
 ax.set_title(title,fontsize=9)
for ax in panels[1]: ax.scatter(x[:,0],x[:,1],color='#cccccc',s=10)
for i,name in enumerate(names):
 group=a[a.pseudo_class==name]; q=x[group.row.to_numpy()]; panels[1,0].scatter(q[:,0],q[:,1],s=35,c=[colors(i)],edgecolors=np.where(group.correct,'none','black'),linewidths=1.3)
panels[1,0].set_title('Anchors: assigned CLIP label',fontsize=9)
from matplotlib.lines import Line2D
panels[1,0].legend(handles=[Line2D([],[],marker='o',color='black',markerfacecolor='none',linestyle='none',label='Wrong pseudo-label')],fontsize=7,frameon=False,loc='lower right')
wrong=a[~a.correct]; q=x[wrong.row.to_numpy()]
panels[1,1].scatter(q[:,0],q[:,1],s=45,facecolors='none',edgecolors='#b5422c',linewidths=1.3)
for (_,r),pt in zip(wrong.iterrows(),q): panels[1,1].annotate(str(int(r.row)),pt,fontsize=7,xytext=(4,3),textcoords='offset points')
panels[1,1].set_title('Wrong anchors only (adaptation row IDs)',fontsize=9)
for ax in panels[1]: ax.set_xlabel(f'PC1 ({rates[0]*100:.1f}%)',fontsize=9)
for ax in panels[:,0]: ax.set_ylabel(f'PC2 ({rates[1]*100:.1f}%)',fontsize=9)
for ax in panels.flat: ax.grid(alpha=.15); ax.tick_params(labelsize=8)
handles,labels=panels[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.04),ncol=3,fontsize=8,frameon=False)
(out/'figures').mkdir(exist_ok=True)
fig.savefig(out/'figures/dino-pca.pdf'); fig.savefig(out/'figures/dino-pca.png',dpi=170); plt.close(fig)
(out/'dino-pca.json').write_text(json.dumps(dict(method='Centered PCA by SVD; fit only on raw 160 adaptation DINO vectors; no labels in fit',explained_variance_ratio=rates.tolist(),train_shape=train.shape,val_shape=valid.shape),indent=2)+'\n')
print('PCA verified:',train.shape,valid.shape,rates.tolist())
