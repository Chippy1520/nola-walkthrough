"""Frozen validation rerun; does not write to any prior experiment."""
import argparse, csv, hashlib, io, json, time, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.special import logsumexp
import torch
from clip.clip import _transform, build_model, tokenize
from utils.data_utils import ds_specific_templates

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAR = ROOT/'parallel-study'
WEIGHT = ROOT/'weights/ViT-B-32.pt'
CLIP_SHA = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
CIFAR_WRAPS = ['Photo: {}','A photo: {}','The photo: {}','A centered photo: {}','This is a photograph: {}','A photograph: {}']
SAT_WRAPS = ['Satellite photo: {}','A satellite photo: {}','The satellite photo: {}','A centered satellite photo: {}','This is an aerial image: {}','An aerial image: {}']


def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def unit(x):
    n = np.linalg.norm(x,axis=-1,keepdims=True)
    assert np.isfinite(n).all() and (n > 1e-10).all()
    return x/n


def table(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


def blend(m, raw, sizes, keep):
    if not keep.any(): return m.copy()
    # Inherit the previous class/cluster weights, but mix directions, not logits.
    return unit(sizes.sum()*m + sizes[keep] @ unit(raw[keep]))


def banks(v):
    m=unit(v.mean(0)); groups=fcluster(linkage(pdist(v.astype('float64'),'cosine'),method='complete'),.25,criterion='distance')
    inds=[np.flatnonzero(groups==j) for j in sorted(set(groups))]
    sizes=np.array([len(g) for g in inds]); raw=np.stack([v[g].mean(0) for g in inds]); cos=unit(raw)@m
    np.testing.assert_array_equal(np.sort(np.concatenate(inds)),np.arange(len(v)))
    # Verify the partition identity in float64: different FP32 reduction orders
    # can exceed 1e-7 on concentrated coordinates without a grouping error.
    check=v.astype('float64')
    np.testing.assert_allclose(sizes @ np.stack([check[g].mean(0) for g in inds]) / len(v),check.mean(0),atol=1e-14)
    d={'mean':m,'blend075':blend(m,raw,sizes,cos<.75),'blend090':blend(m,raw,sizes,cos<.90)}
    return d,raw,sizes,cos,[g.tolist() for g in inds]


def smoke():
    m=np.array([1.,0.]); raw=np.array([[0.,.25],[1.,0.]]); sizes=np.array([2,6]); k=np.array([True,False]);
    np.testing.assert_allclose(blend(m,raw,sizes,k),unit(np.array([8.,2.])))
    np.testing.assert_array_equal(blend(m,raw,sizes,k&False),m)


@torch.inference_mode()
def run(args):
    out=HERE/args.dataset; out.mkdir(exist_ok=True)
    d=args.dataset; cuda=torch.device('cuda'); assert sha(WEIGHT)==CLIP_SHA
    state=torch.jit.load(str(WEIGHT),map_location='cpu').state_dict()
    model=build_model(state).eval().requires_grad_(False).to(cuda); del state
    print(d,torch.cuda.get_device_name(),flush=True)
    sources={}; historical={}
    if d=='cifar100':
        base=PAR/'cifar100'; descfile=base/'coarse_descriptions.json'; old=base/'frozen_hybrid_075'
        manifest=base/'audit_manifest.npz'; feats=old/'validation_features.npz'
        with np.load(manifest,allow_pickle=False) as a, np.load(feats,allow_pickle=False) as b:
            keys=a['coarse_names'].tolist(); ids=a['validation_indices']; y=a['train_coarse'][ids]; fine=a['train_fine'][ids]; x=b['features']
            np.testing.assert_array_equal(ids,b['row_ids']); assert len(ids)==5000
            assert hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()=='3704fd5869ef075ed9e9ef2949506bcf4cfda2a08761683ba3ea78620dbf74b0'
        with np.load(old/'validation_scores.npz',allow_pickle=False) as a, (base/'frozen_hybrid_gate_0.90'/'predictions.csv').open() as f:
            b=list(csv.DictReader(f)); np.testing.assert_array_equal(ids,a['row_ids']); np.testing.assert_array_equal(ids,[int(r['row_id']) for r in b]); np.testing.assert_array_equal(y,[int(r['truth_coarse_id']) for r in b]);
            scores={'legacy_mean':a['same_prompt_mean'],'legacy_hybrid075':a['hybrid_075']}
            historical['legacy_hybrid090']=np.array([int(r['hybrid']) for r in b]); np.testing.assert_array_equal(scores['legacy_mean'].argmax(1),[int(r['same_prompt_mean']) for r in b])
        templates=[w.replace(': {}',' of {}.') for w in CIFAR_WRAPS]; wrappers=CIFAR_WRAPS
        sources={p.name:sha(p) for p in (manifest,feats)}
    else:
        tag={'eurosat':'EuroSAT','resisc45':'RESISC45'}[d]; descfile=ROOT/f'upstream/descriptions/generic/{tag}.json'
        if d=='eurosat':
            split=ROOT/'data/eurosat/split_zhou_EuroSAT.json'; rows=json.loads(split.read_text())['val']; sources[split.name]=sha(split)
            image_root=ROOT/'data/eurosat/EuroSAT_RGB'; expected=5400
            archive=None
        else:
            report=next((PAR/'resisc45').glob('*report.zip'))
            with zipfile.ZipFile(report) as z:
                cfg=json.loads(z.read('config.json')); rows=json.loads(z.read('split_train_val.json'))['val']
            assert sha(args.archive)==cfg['image_archive_sha256']; sources={report.name:sha(report),args.archive.name:sha(args.archive)}
            archive=zipfile.ZipFile(args.archive); members={ '/'.join(s.split('/')[-2:]):s for s in archive.namelist() if s.lower().endswith('.jpg') }
            assert len(members)==31500; expected=6255
        mapping={i:k for _,i,k in rows}; keys=[mapping[i] for i in range(len(mapping))]; ids=np.array([p for p,_,_ in rows]); y=np.array([i for _,i,_ in rows]); fine=None
        if d=='resisc45':
            ids_to_desc={i:p.split('/')[0] for p,i,_ in rows}
            assert all(p.split('/')[0]==ids_to_desc[i] and cfg['class_name_map'][ids_to_desc[i]]==k for p,i,k in rows)
        assert len(set(ids))==len(ids)==expected and all(keys[i]==k for _,i,k in rows)
        templates=ds_specific_templates[tag]; wrappers=SAT_WRAPS; scores={}
        featfile=out/'features.npz'; tf=_transform(model.visual.input_resolution)
        if featfile.exists():
            with np.load(featfile,allow_pickle=False) as f:
                np.testing.assert_array_equal(ids,f['row_ids']); assert str(f['provenance'])==json.dumps(sources,sort_keys=True); x=f['features']
        else:
            chunks=[]; start=time.monotonic()
            for i in range(0,len(ids),48):
                images=[]
                for name in ids[i:i+48]:
                    path=image_root/name if archive is None else io.BytesIO(archive.read(members[name]))
                    with Image.open(path) as im: images.append(tf(im.convert('RGB')))
                chunks.append(unit(model.encode_image(torch.stack(images).to(cuda)).float().cpu().numpy()))
                if i%960==0: print(f'{d} images {i}/{len(ids)} in {time.monotonic()-start:.1f}s',flush=True)
            x=np.concatenate(chunks); np.savez(out/'features.tmp.npz',row_ids=ids,features=x,provenance=json.dumps(sources,sort_keys=True)); (out/'features.tmp.npz').replace(featfile)
        if archive: archive.close()
        sources['encoded_features']=sha(featfile)
    np.testing.assert_allclose(np.linalg.norm(x,axis=1),1,atol=2e-6)
    source=json.loads(descfile.read_text()); sources[descfile.name]=sha(descfile)
    if d=='resisc45':
        assert set(ids_to_desc.values())==set(source)
        source={k:source[ids_to_desc[i]] for i,k in enumerate(keys)}
    assert set(keys)==set(source)
    def encode(strings):
        tokens=tokenize(strings,truncate=False)
        return unit(np.concatenate([model.encode_text(t.to(cuda)).float().cpu().numpy() for t in tokens.split(128)]))
    allvec={}; audits=[]; rendered={}; extra={}
    for i,k in enumerate(keys):
        desc=source[k]; filled=[t.format(k) for t in templates]
        suffix=[f'{s.rstrip()} {t}' for s in desc for t in filled]
        assert len(suffix)==len(desc)*len(templates)
        v=unit(encode(suffix).reshape(len(desc),len(templates),512).mean(1))
        b,raw,n,c,groups=banks(v)
        for arm,vec in b.items(): allvec.setdefault('suffix_'+arm,[]).append(vec)
        audits.extend(dict(class_id=i,class_name=k,size=int(s),cosine_to_mean=float(z),retained075=bool(z<.75),retained090=bool(z<.90),description_ids=json.dumps(g)) for s,z,g in zip(n,c,groups))
        # Historic nonlinear pooling on the suffix bank, to isolate pooling from wording.
        for gate, suffix_id in ((.75,'075'),(.90,'090')):
            keep=c<gate; w=np.r_[len(v),n[keep]]; vectors=np.concatenate([b['mean'][None],raw[keep]])
            if keep.any(): logits=.05*(logsumexp(x @ vectors.T/.05+np.log(w),axis=1)-np.log(w.sum()))
            else: logits=None
            extra.setdefault(f'suffix_lme{suffix_id}',[]).append(logits)
        rendered[k]={'suffix':suffix}
        if d!='cifar100':
            separate=desc+filled; wrapped=[w.format(s) for s in desc for w in wrappers]
            e=unit(encode(wrapped).reshape(len(desc),len(wrappers),512).mean(1)); old,_,_,_,_=banks(e)
            for arm,vec in [('released_separate',unit(encode(separate).mean(0))),('legacy_mean',old['mean'])]: allvec.setdefault(arm,[]).append(vec)
            rendered[k].update(separate=separate,wrapped=wrapped)
        print(f'{d} text {i+1}/{len(keys)}',flush=True)
    allvec={k:np.stack(v).astype('float32') for k,v in allvec.items()}
    for k,v in allvec.items():
        assert v.shape==(len(keys),512); np.testing.assert_allclose(np.linalg.norm(v,axis=1),1,atol=2e-6); scores[k]=x@v.T
    for k,cols in extra.items(): scores[k]=np.stack([scores['suffix_mean'][:,i] if v is None else v for i,v in enumerate(cols)],1)
    for q in ('075','090'):
        untouched=[i for i in range(len(keys)) if not any(r['class_id']==i and r['retained'+q] for r in audits)]
        np.testing.assert_array_equal(scores['suffix_mean'][:,untouched],scores['suffix_blend'+q][:,untouched])
    pred={**{k:s.argmax(1) for k,s in scores.items()},**historical}; ref=pred['suffix_mean']==y
    def report(mask):
        a=ref[mask]; n=int(mask.sum()); result={}
        for k,p in pred.items():
            b=p[mask]==y[mask]; fixed=int((~a&b).sum()); lost=int((a&~b).sum())
            nc=int(b.sum()); assert nc-int(a.sum())==fixed-lost
            result[k]=dict(correct=nc,n=n,acc_pct=100*nc/n,fixed_vs_suffix_mean=fixed,lost_vs_suffix_mean=lost)
        return result
    stats=report(np.ones(len(y),dtype=bool)); by_class={k:report(y==i) for i,k in enumerate(keys)}
    payload=dict(dataset=d,stage='frozen CLIP ViT-B/32 validation only',torch=torch.__version__,device=torch.cuda.get_device_name(),clip_sha256=CLIP_SHA,run_sha256=sha(Path(__file__)),source_hashes=sources,templates=templates,classes=keys,clustering_cosine=.75,gates=[.75,.90],retained={q:sum(row['retained'+q] for row in audits) for q in ('075','090')},clusters=len(audits),metrics=stats,by_class=by_class)
    if fine is not None: payload['by_fine_id']={int(i):report(fine==i) for i in np.unique(fine)}
    np.savez(out/'text_vectors.npz',**allvec); np.savez(out/'scores.npz',row_ids=ids,**scores)
    table(out/'predictions.csv',[dict(row_id=idx,true_id=int(y[i]),**{k:int(p[i]) for k,p in pred.items()}) for i,idx in enumerate(ids)])
    table(out/'clusters.csv',audits); (out/'rendered_prompts.json').write_text(json.dumps(rendered,indent=2)+'\n')
    (out/'summary.tmp').write_text(json.dumps(payload,indent=2)+'\n'); (out/'summary.tmp').replace(out/'summary.json')
    print(json.dumps(dict(metrics=stats,retained=payload['retained'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('dataset',choices=['eurosat','cifar100','resisc45','smoke']); p.add_argument('--archive',type=Path); a=p.parse_args()
    torch.set_num_threads(2); smoke()
    if a.dataset!='smoke': run(a)
    else: print('Blend arithmetic and exact empty-gate parity passed')
