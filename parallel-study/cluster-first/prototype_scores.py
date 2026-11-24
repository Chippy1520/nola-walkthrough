"""Cache aligned train-only CLIP image features for the saved EuroSAT prototype bank."""
import io
import sys
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch
from torch.nn.functional import normalize
from torch.utils.data import DataLoader,Subset
from torchvision import transforms as T
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE.parent/'dino-anchors'))
from anchor_inputs import get_dataset,_asset,_load_file_module,_atomic_npz,_read_chunk,PINNED
BANK=ROOT/'colab-nola/NoLA-EuroSAT-Template-Aware-075.zip'

def bank():
    with ZipFile(BANK) as outer, ZipFile(io.BytesIO(outer.read('template_aware_075/NoLA-EuroSAT-Author-Inputs.zip'))) as inner:
        return torch.load(io.BytesIO(inner.read('prototypes.pt')),weights_only=True)

def extract(limit=None):
    scope='smoke' if limit else 'full'
    ds=get_dataset('eurosat',work_dir=HERE/'cache',archive=ROOT/'data/EuroSAT_RGB.zip')
    t=T.Compose([T.Resize(224,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    images=ds.dataset('train',t,limit)
    ids=np.array([r[0] for r in images.rows]); model=None
    for j in range(0,len(ids),256):
        end=min(j+256,len(ids)); p=HERE/'clip_vectors'/scope/f'{j:06d}.npz'
        if _read_chunk(p,ids[j:end],{'clip':(end-j,512)}) is not None: continue
        if model is None:
            w=_asset('clip',HERE/'cache',ROOT/'weights/ViT-B-32.pt')
            cm=_load_file_module('_proto_clip',PINNED/'clip/model.py')
            model=cm.build_model(torch.jit.load(str(w),map_location='cpu').state_dict()).float().eval().requires_grad_(False).cuda()
        out=[]; pos=j
        with torch.inference_mode():
            for im, names, _ in DataLoader(Subset(images,range(j,end)),batch_size=8,num_workers=0):
                assert np.array_equal(names,ids[pos:pos+len(im)]); pos+=len(im)
                out.append(normalize(model.encode_image(im.cuda()).float(),dim=1).cpu().numpy())
        assert pos==end
        x=np.concatenate(out);assert x.shape==(end-j,512) and np.isfinite(x).all()
        _atomic_npz(p,ids=ids[j:end],clip=x)
        if j==0 or j//256%5==0 or end==len(ids): print(scope,end,len(ids),flush=True)
    print('COMPLETE',scope,len(ids),flush=True)

if __name__=='__main__':
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    if '--smoke' in sys.argv:extract(64)
    else:
        extract(64);extract()
