"""Run the full-pool selector ablations, packaging each dataset as it finishes."""
import os
import sys
from pathlib import Path
import gc
import hashlib
import json
import time
import traceback
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

DATASETS=('eurosat','resisc45','cifar100')


def archive(src, target):
    stage=target.with_suffix('.part')
    with zipfile.ZipFile(stage,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for file in sorted(src.rglob('*')):
            if file.is_file() and not file.name.endswith(('.tmp','.part')):
                z.write(file,file.relative_to(src))
        for script in Path(__file__).resolve().parent.glob('anchor_*.py'):
            z.write(script,'code/'+script.name)
    with zipfile.ZipFile(stage) as z:
        assert z.testzip() is None and len(z.namelist())>3
    stage.replace(target)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def main(which):
    from anchor_inputs import prepare
    from anchor_eval import evaluate_dataset
    root=Path(os.environ['NOLA_ANCHOR_ROOT'])
    reports=root/'reports'; reports.mkdir(parents=True,exist_ok=True)
    names=DATASETS if which=='all' else (which,)
    assert all(n in DATASETS for n in names)
    assert torch.cuda.is_available(), 'Full Colab comparison requires a GPU'
    print('DEVICE',torch.cuda.get_device_name(),torch.__version__,flush=True)
    failures={}
    for name in names:
        start=time.monotonic(); out=root/'results'/name;out.mkdir(parents=True,exist_ok=True)
        try:
            print('DATASET',name,'started',flush=True)
            prepared=prepare(name,work_dir=root,batch_size=32)
            evaluate_dataset(name=name,output=out,**prepared)
            del prepared
            target=reports/f'NoLA-DINO-Anchors-{name}-report.zip'
            dig=archive(out,target)
            print('DATASET',name,'finished in seconds',round(time.monotonic()-start,1),target,dig,flush=True)
        except Exception as e:
            failures[name]=repr(e)
            traceback.print_exc()
        finally:
            gc.collect();torch.cuda.empty_cache()
    (reports/'run_status.json').write_text(json.dumps(dict(requested=names,failed=failures),indent=2))
    if failures: raise RuntimeError(f'Dataset(s) failed: {failures}')

if __name__=='__main__': main(sys.argv[1])
