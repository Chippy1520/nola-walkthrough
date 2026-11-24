"""Small release-style NoLA runner. Run setup_inputs.setup before importing."""
import csv
import hashlib
import io
import json
import random
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import CIFAR10

from clip import clip
from utils.data_utils import ds_specific_templates
from utils.model_utils import te_transform, tr_transforms, top_k_indices_per_class2
from vision_transformer import vit_base

DATASETS = {
    'eurosat': ('EuroSAT', 'EuroSAT_RGB', 73.5),
    'oxford_pets': ('OxfordPets', 'images', 89.3),
    'cifar10': ('CIFAR10_local', None, 94.9),
    'dtd': ('DescribableTextures', 'images', 56.1),
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(tag, data, seed=7777):
    bank, folder, _ = DATASETS[tag]
    data = Path(data)/tag
    if tag != 'cifar10':
        rows = json.loads((data/f'split_zhou_{bank}.json').read_text())
        names = dict((y, name) for _, y, name in rows['train'])
        names = [names[i] for i in range(len(names))]
        source = data/folder
        assert all((source/key).is_file() and names[y] == name
                   for part in rows.values() for key, y, name in part)
    else:
        source = {part: CIFAR10(data, train=part == 'train', download=False)
                  for part in ('train', 'test')}
        names = source['train'].classes
        rows = {'train': [], 'val': [], 'test': []}
        for y, name in enumerate(names):
            ids = [i for i, label in enumerate(source['train'].targets) if label == y]
            random.Random(seed+y).shuffle(ids)
            for part, group in (('val', ids[:500]), ('train', ids[500:])):
                rows[part].extend((f'train/{i}', y, name) for i in group)
        rows['test'] = [(f'test/{i}', y, names[y]) for i, y in enumerate(source['test'].targets)]
    keys = [key for part in rows.values() for key, _, _ in part]
    assert len(keys) == len(set(keys))
    return rows, source, names, bank


class Images(Dataset):
    def __init__(self, rows, source, transform):
        self.rows, self.source, self.transform = rows, source, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        key = self.rows[i][0]
        if isinstance(self.source, Path):
            with Image.open(self.source/key) as file:
                image = file.convert('RGB')
        else:
            part, index = key.split('/')
            image = Image.fromarray(self.source[part].data[int(index)])
        return self.transform(image), i


def loader(rows, source, transform, batch=32, shuffle=False):
    return DataLoader(Images(rows, source, transform), batch_size=batch,
                      shuffle=shuffle, num_workers=2, pin_memory=True)


def load_backbones(root):
    weights = Path(root)/'weights'
    original = torch.jit.load(str(weights/'ViT-B-32.pt'), map_location='cpu')
    model = clip.build_model(original.state_dict()).float().eval().cuda().requires_grad_(False)
    dino = vit_base(patch_size=16, num_classes=0)
    dino.load_state_dict(torch.load(weights/'dino_vitbase16_pretrain.pth',
                                    map_location='cpu', weights_only=True))
    return model, dino.eval().cuda().requires_grad_(False)


def get_prompts(root, bank):
    path = Path(root)/'upstream'/'descriptions'/'generic'/f'{bank}.json'
    return json.loads(path.read_text()), ds_specific_templates[bank]


@torch.no_grad()
def encode_texts(model, texts):
    parts = [model.encode_text(clip.tokenize(batch).cuda()).float().cpu()
             for batch in (texts[i:i+32] for i in range(0, len(texts), 32))]
    return F.normalize(torch.cat(parts), dim=1)


def build_cde_from_prompts(model, prompts):
    return torch.stack([F.normalize(encode_texts(model, texts).mean(0), dim=0)
                        for texts in prompts])


def build_cde(model, names, descriptions, templates):
    prompts = [descriptions[name]+[t.format(name) for t in templates] for name in names]
    return build_cde_from_prompts(model, prompts)


@torch.no_grad()
def embed_images(encode, rows, source, normalize=False):
    features = []
    for images, _ in loader(rows, source, te_transform, batch=64):
        with torch.autocast('cuda', dtype=torch.float16):
            z = encode(images.cuda()).float().cpu()
        features.append(F.normalize(z, dim=1) if normalize else z)
    return torch.cat(features)


def clip_scores(model, rows, source):
    return embed_images(model.encode_image, rows, source, normalize=True)


def dino_features(model, rows, source):
    return embed_images(model, rows, source)


def extract_features(encoder, model, splits, source):
    return {part: encoder(model, rows, source) for part, rows in splits.items()}


def class_scores(features, cde, scale=1.0):
    return {part: scale*(x @ cde.T) for part, x in features.items()}


def select_anchors(logits, cap):
    # The release uses labels for class IDs and debug prints only. Supply known class IDs, not image truth.
    n, classes = logits.shape
    with redirect_stdout(io.StringIO()):
        ids, _, pseudo = top_k_indices_per_class2(
            {'total_emb': logits.softmax(1), 'idxs': torch.arange(n),
             'labels': torch.arange(n) % classes}, cap)
    return ids, pseudo


def fit_alignment(features, ids, pseudo, classes, epochs=50):
    head = nn.Sequential(nn.Linear(features.shape[1], 256), nn.ReLU(), nn.Linear(256, classes)).cuda()
    optimizer = torch.optim.Adam(head.parameters())
    data = DataLoader(TensorDataset(features[ids], pseudo), batch_size=32, shuffle=True)
    for _ in range(epochs):
        for x, y in data:
            optimizer.zero_grad()
            F.cross_entropy(head(x.cuda()), y.cuda()).backward()
            optimizer.step()
    return head.cpu().eval()


@torch.no_grad()
def alignment_predictions(head, features):
    return {part: head(x).argmax(1) for part, x in features.items()}


def top1_counts(predictions, truth):
    return int((predictions == truth).sum()), len(predictions)


class PromptedCLIP(nn.Module):
    def __init__(self, clip_model, cde):
        super().__init__()
        self.visual = clip_model.visual
        for name, p in self.visual.named_parameters():
            p.requires_grad_('ln' in name or 'bn' in name)
        width = self.visual.class_embedding.numel()
        bound = (6/(3*16*16+width))**.5
        self.prompt = nn.Parameter(torch.empty(1, 16, width).uniform_(-bound, bound))
        self.classifier = nn.Linear(cde.shape[1], cde.shape[0], bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(cde)

    def tokens(self, x):
        patches = self.visual.embeddings_patch(x)
        prompt = self.prompt.expand(len(x), -1, -1)
        return torch.cat((patches[:, :1], prompt.to(patches.dtype), patches[:, 1:]), dim=1)

    def forward(self, images):
        return self.classifier(self.visual.forward_after_patch_embeddings(self.tokens(images)))


def strong(image):
    return tr_transforms.random_tranform(image)


@torch.no_grad()
def predict(model, rows, source, batch):
    return torch.cat([model(x.cuda()).argmax(1).cpu()
                      for x, _ in loader(rows, source, te_transform, batch)])


def write_csv(path, records):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def save(path, obj):
    temp = path.with_suffix('.part')
    torch.save(obj, temp)
    temp.replace(path)


def fit_prompt(model, rows, source, labels, truth, out, epochs, batch, micro, seed):
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in parameters if 'bias' not in n], 'weight_decay': .01},
        {'params': [p for n, p in parameters if 'bias' in n], 'weight_decay': 0.},
    ], lr=.004)
    scaler = torch.amp.GradScaler('cuda')
    latest = out/'latest.pt'
    state = dict(epoch=0, best_epoch=0, best=-1, curve=[])
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=True)
        for name, p in parameters:
            p.data.copy_(state['parameters'][name])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
    for epoch in range(state['epoch']+1, epochs+1):
        seed_all(seed+epoch)
        data = loader(rows['train'], source, strong, micro, shuffle=True)
        update_every = batch//micro
        loss_total = 0.
        optimizer.zero_grad()
        for step, (images, ids) in enumerate(data):
            start = step//update_every*batch
            effective_n = min(batch, len(rows['train'])-start)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(images.cuda())
                loss = F.cross_entropy(logits, labels[ids].cuda(), label_smoothing=.1, reduction='sum')
            scaler.scale(loss/effective_n).backward()
            loss_total += loss.item()
            if (step+1) % update_every == 0 or step+1 == len(data):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        pred = predict(model, rows['val'], source, micro)
        correct = int((pred == truth['val']).sum())
        weights = {name: p.detach().cpu() for name, p in parameters}
        if correct > state['best']:
            state.update(best=correct, best_epoch=epoch, best_parameters=weights)
        state['curve'].append(dict(epoch=epoch, train_loss=loss_total/len(rows['train']),
                                  val_correct=correct, val_n=len(pred)))
        state.update(epoch=epoch, parameters=weights, optimizer=optimizer.state_dict(), scaler=scaler.state_dict())
        save(latest, state)

    selected = dict(epoch=state['best_epoch'], parameters=state['best_parameters'])
    save(out/'best.pt', selected)
    for name, p in parameters:
        p.data.copy_(selected['parameters'][name])
    return selected['epoch'], state['curve']


def frozen_baseline(clip_model, dino, rows, source, names, descriptions, templates, cache):
    if cache.exists():
        frozen = torch.load(cache, weights_only=True)
        cde, image, features = frozen['cde'], frozen['image'], frozen['dino']
    else:
        cde = build_cde(clip_model, names, descriptions, templates)
        image = extract_features(clip_scores, clip_model, rows, source)
        features = extract_features(dino_features, dino, rows, source)
        save(cache, dict(cde=cde, image=image, dino=features))
    return cde, image, features


def export_baseline(out, key, rows, source, truth, names, logits, aligned, model,
                    ids, pseudo, selected, curve, head_epochs, micro):
    preds, metrics = [], []
    for part in rows:
        if part == 'train':
            continue
        methods = [('cde', 0, logits[part].argmax(1)), ('alignment', head_epochs, aligned[part]),
                   ('prompted_clip', selected, predict(model, rows[part], source, micro))]
        for stage, epoch, pred in methods:
            correct, count = top1_counts(pred, truth[part])
            metrics.append(dict(**key, split=part, stage=stage, epoch=epoch, n=count,
                                correct=correct, top1_pct=100*correct/count))
            for (image_id, y, _), p in zip(rows[part], pred.tolist()):
                preds.append(dict(**key, split=part, stage=stage, epoch=epoch, image_id=image_id, truth=y, prediction=p))
    write_csv(out/'predictions.csv', preds)
    write_csv(out/'metrics.csv', metrics)
    write_csv(out/'learning_curve.csv', [dict(**key, **row) for row in curve])
    anchor_truth = truth['train'][ids]
    anchor_hits = pseudo == anchor_truth
    selected_by_class = pseudo.bincount(minlength=len(names)).tolist()
    correct_by_class = pseudo[anchor_hits].bincount(minlength=len(names)).tolist()
    write_csv(out/'anchor_summary.csv', [dict(**key, selected=len(ids), correct=int(anchor_hits.sum()),
                                               precision_pct=100*anchor_hits.float().mean().item())])
    write_csv(out/'anchors_by_class.csv', [
        dict(**key, class_id=i, class_name=name, selected=n, correct=hits,
             precision_pct=100*hits/n if n else '')
        for i, (name, n, hits) in enumerate(zip(names, selected_by_class, correct_by_class))])
    write_csv(out/'anchors.csv', [dict(**key, image_id=rows['train'][i][0], truth=int(truth['train'][i]), pseudo=y)
                                 for i, y in zip(ids.tolist(), pseudo.tolist())])


def train_alignment_stage(features, anchors, classes, epochs, seed):
    ids, pseudo = anchors
    seed_all(seed)
    return fit_alignment(features['train'], ids, pseudo, classes, epochs)


def train_visual_prompt_stage(clip, cde, rows, source, labels, truth, out,
                              epochs, batch, micro, seed):
    model = PromptedCLIP(clip, cde).cuda().eval()
    selected, curve = fit_prompt(model, rows, source, labels, truth, out,
                                 epochs, batch, micro, seed)
    return model, selected, curve


def run_dataset(tag, root, data, out, *, seed=7777, epochs=50, batch=512, micro=32,
                head_epochs=50, smoke=False):
    assert batch % micro == 0 and torch.cuda.is_available()
    seed_all(seed)
    rows, source, names, bank = load_data(tag, data, seed)
    if smoke:
        rows = {part: random.Random(seed).sample(group, min(320, len(group)))
                for part, group in rows.items() if part != 'test'}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    truth = {part: torch.tensor([y for _, y, _ in group]) for part, group in rows.items()}
    source_hash = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    descriptions, templates = get_prompts(root, bank)
    manifest = dict(dataset=tag, run_id=out.parent.name, seed=seed, split_sha256=source_hash,
                    counts={part: len(group) for part, group in rows.items()}, classes=names,
                    protocol='smoke' if smoke else 'release_style',
                    split='45k/5k deterministic train/val + official test' if tag == 'cifar10' else 'CoOp Zhou',
                    anchor_rule='released top-1, top-2/3 fallback, dedup; known class IDs',
                    cde='released normalized prompt mean', clip='OpenAI ViT-B/32', dino='DINO ViT-B/16',
                    source_archive={'nola': 'caec2722bba84bd4807256645c5cb75d96d7ed12',
                                    'dino': '7c446df5b9f45747937fb0d72314eb9f7b66930a'},
                    wrapper_sha256=sha(Path(__file__)),
                    bank_sha256=hashlib.sha256(json.dumps([descriptions, templates], sort_keys=True).encode()).hexdigest(),
                    epochs=epochs, head_epochs=head_epochs, effective_batch=batch, microbatch=micro,
                    checkpoint='first best validation top-1; test once', precision='FP16 autocast',
                    paper_reference_status='context only: exact protocol parity unverified',
                    torch=str(torch.__version__), gpu=torch.cuda.get_device_name())
    manifest_path = out/'manifest.json'
    if manifest_path.exists():
        assert manifest == json.loads(manifest_path.read_text()), 'Use a new run ID for changed settings.'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (out/'split.json').write_text(json.dumps(rows))
    clip_model, dino = load_backbones(root)
    cde, image, features = frozen_baseline(clip_model, dino, rows, source, names,
                                           descriptions, templates, out/'frozen.pt')
    del dino
    logits = class_scores(image, cde, clip_model.logit_scale.detach().exp().cpu())
    cap = max(16, int(.2 * len(rows['train'])/len(names)))
    ids, pseudo = select_anchors(logits['train'], cap)
    head = train_alignment_stage(features, (ids, pseudo), len(names), head_epochs, seed)
    aligned = alignment_predictions(head, features)
    model, selected, curve = train_visual_prompt_stage(
        clip_model, cde, rows, source, aligned['train'], truth, out, epochs, batch, micro, seed)
    key = dict(run_id=out.parent.name, dataset=tag, seed=seed, split_id=source_hash, variant=manifest['protocol'])
    export_baseline(out, key, rows, source, truth, names, logits, aligned, model,
                    ids, pseudo, selected, curve, head_epochs, micro)
    marker = out/'complete.tmp'
    marker.write_text(json.dumps(dict(dataset=tag, selected_epoch=selected, epochs=epochs)))
    marker.replace(out/'complete.json')
    print('Exported artifacts:', out)
    return out
