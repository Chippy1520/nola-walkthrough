"""Build a self-contained-source Colab notebook; public data and weights fetched at runtime."""
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import textwrap
import zipfile

ROOT=Path(__file__).resolve().parent

def build(inputs, extras, dataset):
    inputs=Path(inputs).resolve()
    sources={p.name:p for p in (ROOT/'anchor_eval.py', ROOT/'anchor_driver.py', inputs)}
    sources.update(extras)
    assert len(sources)==len(extras)+3
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,p in sorted(sources.items()):
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            if p.suffix=='.py': compile(p.read_text(),str(p),'exec')
            z.write(p,name)
    payload=buffer.getvalue(); digest=hashlib.sha256(payload).hexdigest()
    b64='\n'.join(repr(x) for x in textwrap.wrap(base64.b64encode(payload).decode(),104))
    cells=[]
    def add(s,kind='code'):
        cell=dict(cell_type=kind,metadata={},source=s.splitlines(keepends=True))
        if kind=='code':
            compile(s,'<notebook cell>','exec');cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    name=f'NoLA-DINO-Anchors-{dataset}'
    add(f'# {name}\n\nFull-pool selector ablation. Four matched arms, validation only. The final CLIP student is not trained.\n', 'markdown')
    add(f"DATASET={dataset!r}\nUSE_DRIVE=False  # mount Drive only with authorization; always download verified reports before closing\nWORK='/content/nola_dino_anchors'\n")
    add("import os, sys, zipfile, io, hashlib, base64, subprocess\nfrom pathlib import Path\nimport torch\nassert torch.cuda.is_available(), 'GPU runtime required: Runtime → Change runtime type'\nprint(torch.cuda.get_device_name(),torch.__version__,flush=True)\nif USE_DRIVE:\n    from google.colab import drive\n    drive.mount('/content/drive')\n    WORK='/content/drive/MyDrive/NoLA-DINO-Anchors'\nROOT=Path(WORK)\nROOT.mkdir(parents=True,exist_ok=True)\nsubprocess.run([sys.executable,'-m','pip','-q','install','ftfy==6.3.1'],check=True)\nembedded=base64.b64decode((\n"+b64+f"\n))\nassert hashlib.sha256(embedded).hexdigest()=={digest!r}\nwith zipfile.ZipFile(io.BytesIO(embedded)) as z:\n    z.extractall(ROOT/'code')\n")
    add("import os, subprocess, sys\nos.environ['NOLA_ANCHOR_ROOT']=str(ROOT)\nproc=subprocess.run([sys.executable,'-u',str(ROOT/'code'/'anchor_driver.py'),DATASET],cwd=ROOT/'code',check=True)\n")
    add("from google.colab import files\nfrom pathlib import Path\nfor report in sorted((ROOT/'reports').glob('*.zip')):\n    if DATASET=='all' or DATASET in report.name:\n        print(report,hashlib.sha256(report.read_bytes()).hexdigest())\n        files.download(str(report))\n")
    nb=dict(nbformat=4,nbformat_minor=2,metadata=dict(accelerator='GPU',colab=dict(name=name+'.ipynb',provenance=[]),kernelspec=dict(display_name='Python 3',language='python',name='python3')),cells=cells)
    out=ROOT/(name+'.ipynb');out.write_text(json.dumps(nb,indent=1)+'\n');print(out, digest, len(payload))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',default=ROOT/'anchor_inputs.py');p.add_argument('--extra',action='append',default=[]);p.add_argument('--dataset',choices=['all','eurosat','resisc45','cifar100'],default='all');a=p.parse_args(); extras={k:Path(v) for k,v in (x.split('=',1) for x in a.extra)};build(a.inputs,extras,a.dataset)
