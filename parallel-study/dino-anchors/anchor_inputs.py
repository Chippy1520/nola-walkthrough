"""Pinned train/validation inputs for the NoLA DINO-anchor study (no training)."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import pickle
import random
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image

PINNED = Path(__file__).resolve().parent / 'pinned'
DEFAULT_WORK = Path('/content/nola-dino-anchors')
SIZES = {'eurosat': (13500, 5400, 8100), 'resisc45': (24930, 6255, 315),
         'cifar100': (45000, 5000, 10000)}
CLIP_SHA = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
CLIP_LOGIT_SCALE = 100.00000762939453  # exp(float32(logit_scale)) in the pinned checkpoint.
DINO_SHA = 'bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e'
ASSETS = {
    'eurosat': ('EuroSAT_RGB.zip', 'https://zenodo.org/api/records/7711810/files/EuroSAT_RGB.zip/content', 'md5', 'f46e308c4d50d4bf32fedad2d3d62f3b'),
    'resisc45': ('NWPU-RESISC45.zip', 'https://huggingface.co/datasets/isaaccorley/resisc45/resolve/883edc0eee77b2c84225472f10f126e3ed83fa6e/NWPU-RESISC45.zip', 'sha256', 'beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53'),
    'cifar100': ('cifar-100-python.tar.gz', 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', 'sha256', '85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7'),
    'clip': ('ViT-B-32.pt', f'https://openaipublic.azureedge.net/clip/models/{CLIP_SHA}/ViT-B-32.pt', 'sha256', CLIP_SHA),
    'dino': ('dino_vitbase16_pretrain.pth', 'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth', 'sha256', DINO_SHA),
}
ORIGINS = {'eurosat': 'Published CoOp split_zhou_EuroSAT.json',
           'resisc45': 'Pinned release DTD adapter (seed 7777); validation carved with seed 7778; not an author split manifest',
           'cifar100': 'Local audit manifest (seed 42, fine-stratified holdout from official train; official test untouched)'}


def digest(path, kind='sha256'):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, kind).hexdigest()


def verify_pins():
    hashes = {}
    for line in (PINNED/'SHA256SUMS').read_text().splitlines():
        sha, name = line.split('  ', 1)
        if digest(PINNED/name) != sha:
            raise ValueError(f'Pinned file checksum mismatch: {name}')
        hashes[name] = sha
    return hashes


def _asset(name, work, override=None):
    filename, url, kind, sha = ASSETS[name]
    p = Path(override).expanduser() if override is not None else work/'downloads'/filename
    if p.is_file():
        if digest(p, kind) != sha:
            raise ValueError(f'{kind} mismatch: {p}; replace the file or remove it to re-download')
        return p.resolve()
    if override is not None:
        raise FileNotFoundError(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=p.parent, suffix='.part', delete=False) as tmp:
        t = Path(tmp.name)
    try:
        subprocess.run(['curl', '-fLsS', '--retry', '2', '--connect-timeout', '20',
                        '--max-time', '1200', '-o', str(t), url], check=True)
        if digest(t, kind) != sha:
            raise ValueError(f'{kind} mismatch for downloaded {filename}')
        t.replace(p)
    finally:
        t.unlink(missing_ok=True)
    return p.resolve()


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _name(name):
    key = name.lower().strip().replace('-', '').replace('_', '')
    if key == 'cifar100coarse':
        key = 'cifar100'
    if key not in SIZES:
        raise ValueError(f'Unknown dataset {name!r}; choose from {tuple(SIZES)}')
    return key


@dataclass
class DatasetInputs:
    name: str
    classes: tuple[str, ...]
    train_rows: tuple[tuple[str, int], ...]
    val_rows: tuple[tuple[str, int], ...]
    text_prompts: dict
    text_mean: np.ndarray
    metadata: dict
    _source: object

    def dataset(self, split, transform=None, limit=None):
        """Indexable dataset returning (PIL RGB or transformed image, stable ID, label)."""
        if split not in ('train', 'val'):
            raise ValueError('Only train and val are exposed; test is not decoded')
        rows = self.train_rows if split == 'train' else self.val_rows
        return _Images(rows[:limit], self._source, transform)

    @property
    def train(self):
        return self.dataset('train')

    @property
    def val(self):
        return self.dataset('val')


class _Images:
    def __init__(self, rows, source, transform):
        self.rows, self.source, self.transform = rows, source, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row_id, label = self.rows[i]
        image = self.source.image(row_id)
        return (self.transform(image) if self.transform else image), row_id, label


@lru_cache(maxsize=6)
def _zip_for_process(path, pid):
    return zipfile.ZipFile(path)


@dataclass
class _JPEG:
    path: Path
    members: dict | None = None

    def image(self, row_id):
        if self.members is None:
            with Image.open(self.path/row_id) as im:
                return im.convert('RGB')
        with _zip_for_process(self.path, os.getpid()).open(self.members[row_id]) as data:
            with Image.open(io.BytesIO(data.read())) as im:
                return im.convert('RGB')


@dataclass
class _CIFAR:
    pixels: np.ndarray
    fine: np.ndarray
    fine_names: np.ndarray

    def image(self, row_id):
        return Image.fromarray(self.pixels[int(row_id.split(':')[1])])


def _resisc_split(keys, names):
    # Sorted JPEGs + NoLA DTD's round(.99*N) cutoff; preserve the parent notebook's row order.
    rng, val_rng = random.Random(7777), random.Random(7778)
    train, val, test = [], [], []
    for y, key in enumerate(keys):
        rows = [(f'{key}/{key}_{i:03d}.jpg', y, names[key]) for i in range(1, 701)]
        rng.shuffle(rows)
        fit, held = rows[:round(.99*len(rows))], rows[round(.99*len(rows)):]
        vi = set(val_rng.sample(range(len(fit)), round(.2*len(fit))))
        train.extend(row for i, row in enumerate(fit) if i not in vi)
        val.extend(row for i, row in enumerate(fit) if i in vi)
        test.extend(held)
    return {k: [list(r) for r in rows] for k, rows in zip(('train','val','test'), (train,val,test))}


def _jpeg_input(name, work, archive, image_dir, config):
    if name == 'eurosat':
        splits = _json(PINNED/name/'split_zhou_EuroSAT.json')
        classes = tuple(config['class_order'])
    else:
        classes = tuple(config['class_order'])
        splits = _resisc_split(classes, config['class_name_map'])
        prior = _json(PINNED/name/'split_train_val.json')
        _check(all(splits[k] == prior[k] for k in ('train','val')), 'RESISC45 split differs from frozen report')
    _check(tuple(len(splits[k]) for k in ('train','val','test')) == SIZES[name], 'Unexpected split sizes')
    all_rows = splits['train'] + splits['val'] + splits['test']
    expected = {p for p, _, _ in all_rows}
    _check(len(expected) == len(all_rows), 'Repeated image in split')
    folders = {}
    for row_id, y, label in all_rows:
        p = PurePosixPath(row_id)
        _check(len(p.parts) == 2 and not p.is_absolute() and '..' not in p.parts and p.suffix == '.jpg', 'Invalid image path')
        _check(0 <= y < len(classes) and label == config.get('class_name_map', {}).get(classes[y], classes[y])
               and (name != 'resisc45' or p.parts[0] == classes[y]), 'Invalid class mapping')
        _check(folders.setdefault(p.parts[0], y) == y, 'Inconsistent folder label')
    _check(set(folders.values()) == set(range(len(classes))) and len(folders) == len(classes), 'Incomplete class mapping')
    if image_dir is not None:
        _check(archive is None, 'Use either archive or image_dir')
        p = Path(image_dir).expanduser().resolve()
        members = {q.relative_to(p).as_posix() for q in p.glob('*/*') if q.is_file() and q.suffix.lower() == '.jpg'}
        _check(members == expected, f'{name}: extracted JPEG inventory mismatch at {p}')
        source = _JPEG(p)
        asset_id = {'image_dir': str(p), 'archive_verified': False}
    else:
        p = _asset(name, work, archive)
        members = {}
        with zipfile.ZipFile(p) as z:
            for obj in z.infolist():
                rel = PurePosixPath(obj.filename)
                if not obj.is_dir() and rel.suffix.lower() == '.jpg':
                    _check(not rel.is_absolute() and '..' not in rel.parts and len(rel.parts) >= 2, 'Invalid ZIP member')
                    key = '/'.join(rel.parts[-2:])
                    _check(key not in members, f'Duplicate JPEG: {key}')
                    members[key] = obj.filename
        _check(set(members) == expected, f'{name}: archive JPEG inventory mismatch')
        source = _JPEG(p, members)
        asset_id = {'archive_checksum': ASSETS[name][2:], 'archive_verified': True}
    return (tuple((p, y) for p, y, _ in splits['train']), tuple((p, y) for p, y, _ in splits['val']), source, asset_id)


def _cifar_input(work, archive, classes):
    p = _asset('cifar100', work, archive)
    with np.load(PINNED/'cifar100/audit_manifest.npz', allow_pickle=False) as m:
        a = dict(m)
    fit, val = a['fit_indices'], a['validation_indices']
    _check(tuple(map(len, (fit, val, a['test_fine']))) == SIZES['cifar100'] and
           np.array_equal(np.sort(np.r_[fit, val]), np.arange(50000)), 'CIFAR split coverage mismatch')
    _check(hashlib.sha256(val.astype('<i8').tobytes()).hexdigest() ==
           '3704fd5869ef075ed9e9ef2949506bcf4cfda2a08761683ba3ea78620dbf74b0', 'CIFAR validation ID checksum mismatch')
    _check(tuple(a['coarse_names']) == classes, 'CIFAR class order mismatch')
    with tarfile.open(p, 'r:gz') as tf:
        # Read only the official train member. The SHA-256 above is required before unpickling.
        for member in tf:
            if member.name == 'cifar-100-python/train' and member.isfile():
                with tf.extractfile(member) as stream:
                    data = pickle.load(stream, encoding='bytes')
                break
        else:
            raise ValueError('Missing official CIFAR train member')
    for key in ('fine','coarse'):
        _check(np.array_equal(data[f'{key}_labels'.encode()], a[f'train_{key}']), f'CIFAR {key} labels differ from pinned audit')
    _check(np.array_equal(a['fine_to_coarse'][a['train_fine']], a['train_coarse']), 'CIFAR fine-to-coarse mismatch')
    pixels = data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    def rows(indices):
        return tuple((f'train:{i:05d}', int(a['train_coarse'][i])) for i in indices)
    source = _CIFAR(pixels, a['train_fine'], a['fine_names'])
    return rows(fit), rows(val), source, {'archive_checksum': ASSETS['cifar100'][2:], 'archive_verified': True}


def get_dataset(name, *, work_dir=DEFAULT_WORK, archive=None, image_dir=None):
    """Load pinned split/prompt inputs; optional local verified archive or JPEG directory."""
    name = _name(name)
    work = Path(work_dir).expanduser()
    pins = verify_pins()
    root = PINNED/name
    config = _json(root/'text_config.json')
    classes = tuple(config['class_order'])
    descriptions = _json(root/'descriptions.json')
    _check(set(classes) == set(descriptions) and len(classes) == len(descriptions), 'Description class mismatch')
    if name == 'cifar100':
        _check(image_dir is None, 'CIFAR-100 requires the official tar archive')
        train, val, source, asset_id = _cifar_input(work, archive, classes)
        prompts = {c: tuple(w.format(d) for d in descriptions[c] for w in config['wrappers']) for c in classes}
    else:
        train, val, source, asset_id = _jpeg_input(name, work, archive, image_dir, config)
        prompts = {c: tuple(descriptions[c]) + tuple(t.format(c.replace('_',' ')) for t in config.get('class_templates', ())) for c in classes}
    _check(all({y for _, y in rows} == set(range(len(classes))) for rows in (train, val)), 'Train/validation class coverage mismatch')
    mean = np.load(root/'clip_mean.npy', allow_pickle=False)
    _check(mean.shape == (len(classes),512) and mean.dtype == np.float32 and np.isfinite(mean).all() and
           np.allclose(np.linalg.norm(mean, axis=1), 1, atol=2e-3), 'Invalid pinned CLIP means')
    return DatasetInputs(name, classes, train, val, prompts, mean, dict(
        dataset=name, split_origin=ORIGINS[name], train_count=len(train), val_count=len(val),
        test_count=SIZES[name][2], test_decoded=False, text_origin=config['mean_origin'],
        text_kind=config['kind'], input_sha256={k:v for k,v in pins.items() if k.startswith(name+'/')},
        **asset_id), source)


def _load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _backbones(work, device, weights_dir):
    import torch
    override = Path(weights_dir).expanduser() if weights_dir else None
    clip_path, dino_path = (_asset(key, work, override/ASSETS[key][0] if override else None)
                            for key in ('clip','dino'))
    cm = _load_file_module('_anchor_openai_clip', PINNED/'clip/model.py')
    with torch.no_grad():
        clip = cm.build_model(torch.jit.load(str(clip_path), map_location='cpu').state_dict()).float().eval().requires_grad_(False).to(device)
    # DINO's upstream single-file model uses an absolute `utils` import. Bind only while loading it.
    old = sys.modules.get('utils')
    try:
        sys.modules['utils'] = _load_file_module('_anchor_dino_utils', PINNED/'dino/utils.py')
        dm = _load_file_module('_anchor_dino_vit', PINNED/'dino/vision_transformer.py')
    finally:
        if old is None:
            sys.modules.pop('utils', None)
        else:
            sys.modules['utils'] = old
    dino = dm.vit_base(patch_size=16, num_classes=0)
    dino.load_state_dict(torch.load(dino_path, map_location='cpu', weights_only=True), strict=True)
    return clip, dino.float().eval().requires_grad_(False).to(device)


def _atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix='.part', delete=False) as tmp:
        t = Path(tmp.name)
        try:
            np.savez(tmp, **arrays)
            tmp.close()
            t.replace(path)
        finally:
            t.unlink(missing_ok=True)


def _schema(split, n, c):
    return (dict(dino=(n,768), clip_scores=(n,c)) if split == 'train' else dict(dino=(n,768)))


def _read_chunk(path, ids, schema):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z['ids'], ids):
                return None
            arrays = {k:z[k] for k in schema}
        if all(a.shape == schema[k] and a.dtype == np.float32 and np.isfinite(a).all() for k,a in arrays.items()):
            return arrays
    except (ValueError, OSError, EOFError, KeyError, zipfile.BadZipFile):
        return None
    return None


def prepare(name, *, work_dir=DEFAULT_WORK, cache_dir=None, archive=None, image_dir=None,
            weights_dir=None, device=None, batch_size=32, workers=2, chunk_size=512,
            train_limit=None, val_limit=None):
    """Return keyword arguments for evaluate_dataset; limits are for smoke checks only.

    Train CLIP scores are cosine similarities to the pinned means. DINO CLS features
    are raw (not L2-normalized). No train labels are used in extraction/scoring.
    """
    import torch
    import torchvision
    from torch.nn.functional import normalize
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms as T

    work = Path(work_dir).expanduser()
    ds = get_dataset(name, work_dir=work, archive=archive, image_dir=image_dir)
    _check(batch_size > 0 and chunk_size > 0 and workers >= 0, 'Invalid extraction settings')
    _check(all(v is None or v > 0 for v in (train_limit,val_limit)), 'Limits must be positive')
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    tx = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize((.48145466,.4578275,.40821073), (.26862954,.26130258,.27577711))])
    runtime = (torch.__version__, torchvision.__version__, dev.type,
               torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu', Image.__version__,
               torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
               torch.backends.cudnn.allow_tf32)
    limits = {'train':train_limit, 'val':val_limit}
    metadata = dict(ds.metadata, clip_weights_sha256=CLIP_SHA, dino_weights_sha256=DINO_SHA,
                    clip_scores='cosine (unscaled)', clip_logit_scale=CLIP_LOGIT_SCALE,
                    dino_features='raw CLS', dtype='float32',
                    preprocess='bicubic short side 224 / center crop 224 / CLIP RGB mean and std for both backbones',
                    runtime=runtime, limits=limits, batch_size=batch_size, chunk_size=chunk_size,
                    source_sha256=digest(__file__), pins_sha256=digest(PINNED/'SHA256SUMS'))
    if isinstance(ds._source, _JPEG) and ds._source.members is None:
        # For an unverified local directory, bind the cache to actual train/val bytes, not its path alone.
        h = hashlib.sha256()
        for rows in (ds.train_rows[:train_limit], ds.val_rows[:val_limit]):
            for row_id, _ in rows:
                h.update(bytes.fromhex(digest(ds._source.path/row_id)))
        metadata['local_train_val_jpeg_sha256'] = h.hexdigest()
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    metadata['cache_signature'] = signature
    base = Path(cache_dir).expanduser() if cache_dir is not None else work/'features'
    root = base/ds.name/signature
    models = None
    banks = None
    result = dict(classes=ds.classes, metadata=metadata)
    for split in ('train','val'):
        images = ds.dataset(split, tx, limits[split])
        rows = images.rows
        ids = np.array([p for p, _ in rows])
        y = np.array([y for _, y in rows], dtype=np.int64)
        output = {k:np.empty(s, dtype=np.float32) for k,s in _schema(split, len(rows), len(ds.classes)).items()}
        for start in range(0, len(rows), chunk_size):
            end = min(start+chunk_size, len(rows))
            path = root/split/f'{start:06d}.npz'
            schema = _schema(split, end-start, len(ds.classes))
            chunk = _read_chunk(path, ids[start:end], schema)
            if chunk is None:
                if models is None:
                    models = _backbones(work, dev, weights_dir)
                    banks = torch.from_numpy(ds.text_mean).to(dev)
                clip, dino = models
                loader = DataLoader(Subset(images, range(start,end)), batch_size=batch_size,
                                    shuffle=False, num_workers=workers, pin_memory=dev.type=='cuda')
                records = {k:[] for k in schema}
                with torch.inference_mode():
                    offset = start
                    for x, batch_ids, _ in loader:
                        _check(np.array_equal(batch_ids, ids[offset:offset+len(x)]), 'DataLoader row order mismatch')
                        offset += len(x)
                        x = x.to(dev, non_blocking=dev.type=='cuda')
                        records['dino'].append(dino(x).float().cpu().numpy())
                        if split == 'train':
                            records['clip_scores'].append((normalize(clip.encode_image(x).float(), dim=1) @ banks.T).cpu().numpy())
                chunk = {k:np.concatenate(v) for k,v in records.items()}
                _check(all(v.shape == schema[k] and v.dtype == np.float32 and np.isfinite(v).all() for k,v in chunk.items()), 'Invalid feature chunk')
                _atomic_npz(path, ids=ids[start:end], **chunk)
            for k, array in chunk.items():
                output[k][start:end] = array
        result[f'{split}_ids'] = ids
        result['audit_train_y' if split=='train' else 'val_y'] = y
        if isinstance(ds._source, _CIFAR):
            fine = ds._source.fine_names[ds._source.fine[[int(p.split(':')[1]) for p in ids]]]
            metadata['audit_train_fine_y' if split == 'train' else 'val_fine_y'] = fine.tolist()
        result[f'{split}_dino'] = output['dino']
        if split == 'train':
            result['train_clip_scores'] = output['clip_scores']
    return result
