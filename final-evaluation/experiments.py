"""Numerical helpers for notebook 03; no model loading or experiments on import."""
import csv
import hashlib
import math

import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from torch.nn.functional import normalize

SIGLIP_REV = '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'
SIGLIP_SHA = '612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b'
V3_SHA = 'a7637753a2e85d709d1a5474e13382d39e7bc581a35868b477585c970f36847f'
V3_REV = '39c7bb32f83c11bbb13cb23058f8852a4dac88ed'
V3_MODEL = 'convnext_tiny.dinov3_lvd1689m'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify(path, expected):
    if not path.is_file():
        return f'missing {path.name}'
    if sha(path) != expected:
        return f'checksum mismatch: {path.name}'
    return ''


@torch.no_grad()
def siglip_embeddings(path, rows, source, prompts, loader, batch):
    from transformers import AutoTokenizer, SiglipModel
    from torchvision import transforms as T

    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = SiglipModel.from_pretrained(path, local_files_only=True).float().eval().cuda()
    transform = T.Compose([T.Resize((224, 224), T.InterpolationMode.BILINEAR),
                           T.ToTensor(), T.Normalize((.5,) * 3, (.5,) * 3)])

    def projected(out):
        # Older transformers return the projected tensor directly; newer ones expose it as pooled output.
        return out if isinstance(out, torch.Tensor) else out.pooler_output

    def text_vector(texts):
        vectors = []
        for start in range(0, len(texts), 32):
            pieces = texts[start:start + 32]
            ids = tok(pieces, padding='max_length', max_length=64, truncation=True,
                      return_tensors='pt')['input_ids'].cuda()
            vectors.append(projected(model.get_text_features(input_ids=ids)).float().cpu())
        return normalize(torch.cat(vectors), dim=1)

    bank = torch.stack([normalize(text_vector(p).mean(0), dim=0) for p in prompts])
    images = {}
    for split, part in rows.items():
        vecs = [projected(model.get_image_features(pixel_values=x.cuda())).float().cpu()
                for x, _ in loader(part, source, transform, batch=batch)]
        images[split] = normalize(torch.cat(vecs), dim=1)
        assert images[split].shape == (len(part), bank.shape[1])
    return images, bank


@torch.no_grad()
def dinov3_features(path, rows, source, loader, batch):
    import timm
    from safetensors.torch import load_file

    model = timm.create_model(V3_MODEL, pretrained=False, num_classes=0)
    model.load_state_dict(load_file(str(path)), strict=True)
    model.eval().cuda()
    config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**config, is_training=False)
    vectors = {s: torch.cat([model(x.cuda()).float().cpu()
                            for x, _ in loader(part, source, transform, batch=batch)])
               for s, part in rows.items()}
    return vectors, config


def clusters(vectors, min_cos):
    if len(vectors) == 1:
        return [torch.tensor([0])]
    cosine_distance = pdist(vectors.double().numpy(), metric='cosine')
    ids = torch.from_numpy(fcluster(linkage(cosine_distance, method='complete'),
                                   1. - min_cos, criterion='distance'))
    return [torch.where(ids == c)[0] for c in ids.unique()]


def centroid(vectors):
    return normalize(vectors.mean(0), dim=0)


def score(images, bank, tau=.05):
    # [image, prototype] → one class score; each surviving prototype has one vote.
    cols = []
    for prototypes in bank:
        cosine = images @ prototypes.T
        cols.append(cosine[:, 0] if len(prototypes) == 1 else
                    tau * (torch.logsumexp(cosine / tau, dim=1) - math.log(len(prototypes))))
    return torch.stack(cols, dim=1)


def candidates(logits, cap):
    p = logits.softmax(1)
    owner = p.argmax(1)
    rows = [torch.where(owner == c)[0] for c in range(p.shape[1])]
    return [r[p[r, c].argsort(descending=True, stable=True)][:cap] for c, r in enumerate(rows)]


def pack(pools, quota):
    indices = torch.cat([r[:q].sort().values for r, q in zip(pools, quota)])
    labels = torch.arange(len(pools)).repeat_interleave(torch.tensor(quota))
    assert len(indices) == len(labels) == len(indices.unique()) > 0, 'No shared anchors or duplicate anchors'
    return indices, labels


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
