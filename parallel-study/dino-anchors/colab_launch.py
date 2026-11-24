from pathlib import Path
import hashlib, io, os, subprocess, sys, zipfile
import torch
from google.colab import files
assert torch.cuda.is_available(), 'Expected a GPU runtime'
print('GPU:',torch.cuda.get_device_name(),flush=True)
root=Path('/content/nola_dino_anchors');root.mkdir(exist_ok=True)
code=root/'code'
if not (code/'anchor_driver.py').is_file():
    uploads=files.upload()
    data=uploads['NoLA-DINO-Anchors-inputs.zip']
    assert hashlib.sha256(data).hexdigest()=='c8569118e327371faef0cd5c22b3337de83850f6856a0bdd986f40c35b73c2b0'
    with zipfile.ZipFile(io.BytesIO(data)) as z:z.extractall(code)
env=dict(os.environ,NOLA_ANCHOR_ROOT=str(root))
if 'anchor_job' not in globals() or anchor_job.poll() is not None:
    log=(root/'run.log').open('ab',buffering=0)
    anchor_job=subprocess.Popen([sys.executable,'-u',str(code/'anchor_driver.py'),'all'],cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT)
    log.close()
print('Worker:',anchor_job.pid,root/'run.log',flush=True)
