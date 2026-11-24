"""Matched selector-only comparison on frozen CLIP and DINO features.

No image loading, encoder fitting, student training, or test-set evaluation.
Input rows must already be aligned; class labels index the supplied class list.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ARMS = ('confidence', 'farthest_first', 'medoids', 'neighbor_agreement')
EPOCHS = 50
SEED = 0
BATCH_SIZE = 32
HIDDEN = 256
LR = 1e-3
NEIGHBORS = 20
MEDOID_ITERATIONS = 10


def _array(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _matrix(x, name):
    x = x.detach().to(device='cpu', dtype=torch.float32).numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError(f'{name} must be a nonempty, finite matrix')
    return x


def _ids(x, n, name):
    x = _array(x)
    if x.shape != (n,):
        raise ValueError(f'{name} must contain one ID per row')
    ids = np.asarray([str(v) for v in x])
    if np.any(ids == ''):
        raise ValueError(f'{name} must not contain empty IDs')
    return ids


def _labels(x, n, c, name):
    x = _array(x)
    if x.shape != (n,) or x.dtype.kind not in 'iu' or np.any(x < 0) or np.any(x >= c):
        raise ValueError(f'{name} must be an integer vector of length {n} in [0, {c})')
    return x.astype(np.int64, copy=False)


def _allocate(scores, k, scale):
    """Return confidence-ordered, disjoint pools and their reserved budgets.

    Reserve top-1 picks for every represented class, then try top-2 and top-3
    for originally unrepresented classes (first rank with available rows).
    A fallback may use only unreserved rows. After reserving all baseline
    anchors, offer remaining rows to fallback pools at their chosen rank,
    then to top-1 pools. Fallback ties in ownership go to lower class IDs.
    Rows arrive in canonical ID order; stable sorts break confidence ties.
    """
    n, c = scores.shape
    ranks = np.argsort(-scores, axis=1, kind='stable')[:, :min(3, c)]
    # Float64 log-softmax ranks probabilities without saturation at probability 1.
    scores = scores.astype(np.float64) * scale
    scores -= scores.max(axis=1, keepdims=True)
    logp = scores - np.log(np.exp(scores).sum(axis=1, keepdims=True))
    ranked = []
    for r in range(ranks.shape[1]):
        by_class = []
        for cls in range(c):
            rows = np.flatnonzero(ranks[:, r] == cls)
            by_class.append(rows[np.argsort(-logp[rows, cls], kind='stable')])
        ranked.append(by_class)

    owned = np.zeros(n, dtype=bool)
    pools = [np.empty(0, dtype=np.int64) for _ in range(c)]
    used_rank = np.full(c, -1, dtype=np.int64)
    for r, candidates in enumerate(ranked):
        for cls, rows in enumerate(candidates):
            if used_rank[cls] >= 0:
                continue
            picked = rows[~owned[rows]][:k]
            if picked.size:
                pools[cls] = picked
                used_rank[cls] = r
                owned[picked] = True
    budgets = np.array([len(rows) for rows in pools])
    extra_priority = sorted(np.flatnonzero(used_rank >= 0), key=lambda cls: (used_rank[cls] == 0, used_rank[cls], cls))
    for cls in extra_priority:
        rows = ranked[used_rank[cls]][cls]
        extra = rows[~owned[rows]][:2 * budgets[cls]]
        owned[extra] = True
        pools[cls] = np.concatenate((pools[cls], extra))
    return pools, budgets, used_rank, logp, ranks[:, 0].copy()


def _farthest(x, k):
    """Cosine maximin, seeded with the highest-confidence pool member."""
    idx = torch.zeros((), dtype=torch.long, device=x.device)
    chosen = torch.empty(k, dtype=torch.long, device=x.device)
    available = torch.ones(len(x), dtype=torch.bool, device=x.device)
    nearest = torch.full((len(x),), -torch.inf, device=x.device)
    for i in range(k):
        chosen[i] = idx
        # index_select avoids scalar reads back from the GPU.
        anchor = x.index_select(0, idx.reshape(1)).squeeze(0)
        available.scatter_(0, idx.reshape(1), False)
        nearest = torch.maximum(nearest, x @ anchor)
        idx = nearest.masked_fill(~available, torch.inf).argmin()
    return chosen


def _medoids(x, initial):
    """Alternate cosine assignment and exact within-cluster medoid updates.

    For unit vectors, minimizing summed cosine distances equals maximizing
    the dot product with the cluster sum. Clustering is locally optimized;
    it does not solve the global k-medoids objective exactly.
    """
    if len(initial) == len(x):
        return initial
    medoids = initial
    cluster_ids = torch.arange(len(initial), device=x.device)
    for _ in range(MEDOID_ITERATIONS):
        assignment = (x @ x.index_select(0, medoids).T).argmax(dim=1)
        # Own medoid wins identical-vector ties, so no cluster is empty.
        assignment[medoids] = cluster_ids
        membership = F.one_hot(assignment, len(initial)).T.to(x.dtype)
        sums = membership @ x
        objectives = (x * sums.index_select(0, assignment)).sum(dim=1)
        next_medoids = objectives.unsqueeze(0).expand(len(initial), -1).masked_fill(
            membership == 0, -torch.inf
        ).argmax(dim=1)
        if torch.equal(next_medoids, medoids):
            break
        medoids = next_medoids
    return medoids


def _agreement(x, top1, query, target):
    """Exact cosine kNN over all unique training IDs, excluding the query.

    Neighbor votes are original CLIP top-1 labels, including for a fallback
    candidate: the vote is compared with the class it provisionally owns.
    Chunking bounds memory. At the cutoff, ties prefer lower canonical IDs.
    """
    if len(x) == 1:
        return np.zeros(len(query), dtype=np.float32)
    k = min(NEIGHBORS, len(x) - 1)
    values = []
    block = max(1, min(256, 8_000_000 // len(x)))
    for start in range(0, len(query), block):
        q = query[start:start + block]
        similarity = x.index_select(0, q) @ x.T
        similarity.scatter_(1, q[:, None], -torch.inf)
        scores, nearest = similarity.topk(k + 1, dim=1)
        tie = scores[:, k - 1] == scores[:, k]
        nearest = nearest[:, :k].clone()
        if tie.any():
            nearest[tie] = torch.argsort(similarity[tie], dim=1, descending=True, stable=True)[:, :k]
        values.append((top1[nearest] == target[start:start + block, None]).float().mean(dim=1))
    return torch.cat(values).cpu().numpy()


def _select(scores, dino, k, scale, device):
    pools, budgets, used_rank, logp, top1 = _allocate(scores, k, scale)
    normalized = torch.from_numpy(dino).to(device)
    norms = torch.linalg.vector_norm(normalized, dim=1, keepdim=True)
    if not torch.isfinite(norms).all() or torch.any(norms == 0):
        raise ValueError('unique train_dino rows must have finite, nonzero norms')
    normalized = normalized / norms
    query = np.concatenate(pools)
    targets = np.repeat(np.arange(len(pools)), [len(p) for p in pools])
    agreement = np.full(len(dino), np.nan, dtype=np.float32)
    agreement[query] = _agreement(normalized, torch.as_tensor(top1, device=device),
                                  torch.as_tensor(query, device=device), torch.as_tensor(targets, device=device))
    selected = {arm: [] for arm in ARMS}
    for cls, (pool, count) in enumerate(zip(pools, budgets)):
        if len(pool) == count:
            choices = (pool,) * len(ARMS)
        else:
            features = normalized[torch.as_tensor(pool, device=device)]
            far = _farthest(features, count)
            med = _medoids(features, far)
            # Stable agreement sort inherits confidence/ID order for ties.
            near = np.argsort(-agreement[pool], kind='stable')[:count]
            choices = (pool[:count], pool[far.cpu().numpy()], pool[med.cpu().numpy()], pool[near])
        for arm, rows in zip(ARMS, choices):
            if len(rows) != count or len(np.unique(rows)) != count:
                raise RuntimeError(f'unmatched or duplicate selection in {arm}, class {cls}')
            selected[arm].append(rows)
    if len(np.unique(query)) != len(query):
        raise RuntimeError('candidate ownership is not disjoint')
    return selected, pools, budgets, used_rank, logp, top1, agreement


def _teacher(train, pseudo, val, c, device):
    # Only input adapter features and pseudo-labels enter training. Seed CPU
    # initialization and shuffles independently of caller RNG and arm order.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        head = nn.Sequential(nn.Linear(train.shape[1], HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, c))
    head.to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        for cpu_idx in torch.randperm(len(train), generator=generator).split(BATCH_SIZE):
            idx = cpu_idx.to(device)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(head(train[idx]), pseudo[idx]).backward()
            optimizer.step()
    head.eval()
    with torch.inference_mode():
        predictions = [head(batch.to(device)).softmax(dim=1).max(dim=1) for batch in val.split(512)]
        return (torch.cat([p.indices for p in predictions]).cpu().numpy(),
                torch.cat([p.values for p in predictions]).cpu().numpy())


def _accuracy(hits):
    return float(hits.mean()) if len(hits) else None


def _csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def _json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f'Cannot serialize metadata object of type {type(x).__name__}')


def evaluate_dataset(name, train_clip_scores, train_dino, val_dino, audit_train_y, val_y,
                     train_ids, val_ids, classes, output, metadata=None):
    """Run all four arms, write CSVs and summary.json into output, return summary.

    CLIP scores are cosine similarities (softmax confidence at scale 100 by
    default; metadata['clip_logit_scale'] may supply the encoder's scale, or
    1.0 for already scaled logits). First train occurrence wins duplicate
    IDs. Duplicate validation IDs and train/validation ID overlap are errors.
    Optional metadata['audit_train_fine_y'] and metadata['val_fine_y'] hold
    row-aligned subtype IDs/names for reporting only. Accuracy is a fraction;
    unsupported class metrics are null. One fixed seed and final epoch only.
    """
    meta = dict(metadata or {})
    train_fine = meta.pop('audit_train_fine_y', None)
    val_fine = meta.pop('val_fine_y', None)
    scale = float(meta.get('clip_logit_scale', 100.0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('clip_logit_scale must be finite and positive')
    classes = list(map(str, classes))
    c = len(classes)
    if not c or len(set(classes)) != c:
        raise ValueError('classes must contain unique class names')
    scores = _matrix(train_clip_scores, 'train_clip_scores')
    train = _matrix(train_dino, 'train_dino')
    val = _matrix(val_dino, 'val_dino')
    n = len(train)
    if scores.shape != (n, c) or train.shape[1] != val.shape[1]:
        raise ValueError('CLIP scores must align with training rows and classes; DINO dimensions must match')
    train_ids = _ids(train_ids, n, 'train_ids')
    val_ids = _ids(val_ids, len(val), 'val_ids')
    unique_ids, first = np.unique(train_ids, return_index=True)
    if len(np.unique(val_ids)) != len(val_ids) or np.intersect1d(unique_ids, val_ids).size:
        raise ValueError('validation IDs must be unique and disjoint from training IDs')
    train, scores = train[first], scores[first]
    k = max(16, len(train) // (5 * c))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        selections, pools, budgets, used_rank, logp, top1, agreement = _select(scores, train, k, scale, device)
    pseudo = torch.as_tensor(np.repeat(np.arange(c), budgets), device=device)
    val_tensor = torch.from_numpy(val)
    frozen = {}
    for arm in ARMS:
        # Sort within class by canonical ID so ranking order cannot affect
        # the optimizer when two arms choose the same set.
        rows = np.concatenate([np.sort(group) for group in selections[arm]])
        pred, conf = _teacher(torch.from_numpy(train[rows]).to(device), pseudo, val_tensor, c, device)
        frozen[arm] = (rows, pred, conf)

    # Audit data are not accessed until every selection and fit is complete.
    y_train = _labels(audit_train_y, n, c, 'audit_train_y')[first]
    y_val = _labels(val_y, len(val), c, 'val_y')
    train_fine = None if train_fine is None else _ids(train_fine, n, 'audit_train_fine_y')[first]
    val_fine = None if val_fine is None else _ids(val_fine, len(val), 'val_fine_y')
    pseudo = pseudo.cpu().numpy()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        'name': str(name), 'classes': classes, 'device': str(device), 'evaluation_split': 'validation',
        'n_train_original': n, 'n_train_unique': len(train), 'n_val': len(val),
        'target_per_class': k, 'realized_per_class': budgets.tolist(),
        'candidate_counts': [len(pool) for pool in pools],
        'candidate_label_rank': [int(r + 1) if r >= 0 else None for r in used_rank],
        'clip_logit_scale': scale, 'neighbor_k': min(NEIGHBORS, len(train) - 1),
        'medoid_iterations': MEDOID_ITERATIONS,
        'teacher': {'seed': SEED, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'optimizer': 'Adam',
                    'lr': LR, 'hidden_dim': HIDDEN, 'features': 'raw frozen DINO'},
        'ownership': 'reserve top-1 baseline anchors; then top-2/top-3 fallback for empty classes; '
                     'fill fallback then top-1 pools up to 3x realized count using only unowned IDs',
        'metadata': meta, 'arms': {},
    }
    baseline_correct = frozen['confidence'][1] == y_val
    anchor_fields = ('arm', 'train_id', 'train_row', 'pseudo_class', 'true_class', 'correct', 'clip_top1',
                     'clip_prob', 'candidate_label_rank', 'neighbor_agreement', 'true_subtype')
    prediction_fields = ('arm', 'val_id', 'val_row', 'pred_class', 'true_class', 'correct', 'prob', 'true_subtype')
    anchors, predictions, by_class, by_subtype = [], [], [], []
    for arm, (rows, pred, conf) in frozen.items():
        anchor_correct = pseudo == y_train[rows]
        correct = pred == y_val
        summary['arms'][arm] = {
            'anchor_count': len(rows), 'anchor_precision': _accuracy(anchor_correct),
            'val_accuracy': _accuracy(correct),
            'val_accuracy_delta_vs_confidence': _accuracy(correct) - _accuracy(baseline_correct),
            'val_fixed_vs_confidence': int((correct & ~baseline_correct).sum()),
            'val_broken_vs_confidence': int((~correct & baseline_correct).sum()),
        }
        anchors.extend((arm, unique_ids[i], first[i], cls, y_train[i], int(ok), top1[i],
                        float(np.exp(logp[i, cls])), used_rank[cls] + 1, float(agreement[i]),
                        '' if train_fine is None else train_fine[i])
                       for i, cls, ok in zip(rows, pseudo, anchor_correct))
        predictions.extend((arm, val_ids[i], i, pred[i], y_val[i], int(correct[i]), float(conf[i]),
                            '' if val_fine is None else val_fine[i]) for i in range(len(val)))
        for cls, label in enumerate(classes):
            truth = y_val == cls
            ap = anchor_correct[pseudo == cls]
            by_class.append((arm, cls, label, budgets[cls], len(pools[cls]), _accuracy(ap),
                             int(truth.sum()), _accuracy(correct[truth])))
        for split, fine, labels, hits in (('train_anchors', None if train_fine is None else train_fine[rows],
                                           y_train[rows], anchor_correct), ('validation', val_fine, y_val, correct)):
            if fine is not None:
                for cls in np.unique(labels):
                    for subtype in np.unique(fine[labels == cls]):
                        group = (labels == cls) & (fine == subtype)
                        by_subtype.append((arm, split, cls, classes[cls], subtype, int(group.sum()), _accuracy(hits[group])))
    _csv(output / 'anchor_audit.csv', anchor_fields, anchors)
    _csv(output / 'predictions.csv', prediction_fields, predictions)
    _csv(output / 'by_class.csv', ('arm', 'class_id', 'class', 'anchors', 'candidates', 'anchor_precision',
                                   'val_count', 'val_accuracy'), by_class)
    if train_fine is not None or val_fine is not None:
        _csv(output / 'by_subtype.csv', ('arm', 'split', 'true_class_id', 'true_class', 'subtype', 'count', 'accuracy'), by_subtype)
    with (output / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, default=_json_default, allow_nan=False)
        handle.write('\n')
    return summary
