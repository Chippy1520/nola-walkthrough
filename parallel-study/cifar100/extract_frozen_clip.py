"""Resume CPU-only OpenAI CLIP ViT-B/32 inference on the audited validation split."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
from pathlib import Path
import pickle
import tarfile
import time

from clip.clip import _transform, build_model
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'frozen_hybrid_075'
WEIGHTS = ROOT / '../../weights/ViT-B-32.pt'
ARCHIVE = ROOT / '.cache/cifar-100-python.tar.gz'
CHECKS = {WEIGHTS: '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af',
          ARCHIVE: '85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7'}
SPLIT = '3704fd5869ef075ed9e9ef2949506bcf4cfda2a08761683ba3ea78620dbf74b0'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_npz(path, **arrays):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def valid(chunk, ids):
    with np.load(chunk, allow_pickle=False) as z:
        assert str(z['encoder_id']) == sha(Path(__file__))
        assert np.array_equal(z['row_ids'], ids)
        features = z['features']
        assert features.shape == (len(ids), 512) and features.dtype == np.float32
        assert np.isfinite(features).all()
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=2e-6)
        return features


def load_cpu():
    # The released fork's eager clip.load unconditionally calls .cuda().
    state = torch.jit.load(str(WEIGHTS), map_location='cpu').state_dict()
    model = build_model(state).float().eval().requires_grad_(False)
    assert all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters())
    return model, _transform(model.visual.input_resolution)


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    t0 = time.monotonic()
    print(f'CPU extraction start: pid={os.getpid()} affinity={sorted(os.sched_getaffinity(0))} torch={torch.__version__} threads={torch.get_num_threads()}', flush=True)
    for path, expected in CHECKS.items():
        assert sha(path) == expected, path
    with np.load(ROOT / 'audit_manifest.npz', allow_pickle=False) as m:
        ids = m['validation_indices'].astype('<i8')
        assert len(ids) == len(np.unique(ids)) == 5000
        assert hashlib.sha256(ids.tobytes()).hexdigest() == SPLIT
        with tarfile.open(ARCHIVE, 'r:gz') as tf:
            train = pickle.load(tf.extractfile('cifar-100-python/train'), encoding='bytes')
        for key in ['fine', 'coarse']:
            assert np.array_equal(m[f'train_{key}'], train[f'{key}_labels'.encode()])
    pixels = train[b'data'][ids].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    del train
    model, transform = load_cpu()
    chunks = OUT / 'feature_chunks'
    chunks.mkdir(parents=True, exist_ok=True)
    print(f'Inputs verified; model CPU-only; preprocessing={transform}', flush=True)
    outputs = []
    new = 0
    start = time.monotonic()
    for i in range(0, len(ids), 100):
        idx = ids[i:i+100]
        file = chunks / f'{i:04d}.npz'
        if file.exists():
            outputs.append(valid(file, idx))
            continue
        feats = []
        for batch in np.array_split(pixels[i:i+100], 7):  # 14–15 images, bounded activation memory
            x = torch.stack([transform(Image.fromarray(pix)) for pix in batch])
            feats.append(torch.nn.functional.normalize(model.encode_image(x).float(), dim=1).numpy())
        atomic_npz(file, row_ids=idx, features=np.concatenate(feats), encoder_id=np.array(sha(Path(__file__))))
        outputs.append(valid(file, idx))
        new += len(idx)
        rate = new / (time.monotonic() - start)
        done = i+len(idx)
        print(f'{done}/{len(ids)} verified; {rate:.2f} images/s this process, ETA {(len(ids)-done)/rate:.0f}s', flush=True)
    atomic_npz(OUT / 'validation_features.npz', row_ids=ids, features=np.concatenate(outputs), encoder_id=np.array(sha(Path(__file__))))
    valid(OUT / 'validation_features.npz', ids)
    print(f'Extraction complete, {time.monotonic()-t0:.1f}s in this process', flush=True)


if __name__ == '__main__':
    main()
