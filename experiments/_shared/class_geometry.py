"""Export descriptive validation geometry. Labels are used only for analysis."""
import csv

import torch
from torch.nn.functional import normalize


def export_geometry(out, key, space, features, truth, rows, names):
    # L2-normalize rows so all distances below are cosine distances.
    x = normalize(features.detach().cpu().float(), dim=1)
    y = truth.cpu().long()
    num_classes = len(names)
    if len(x) != len(y) or len(y) != len(rows) or any(row[1] != label for row, label in zip(rows, y.tolist())):
        raise ValueError('Feature, label and validation row order must agree')
    sizes = y.bincount(minlength=num_classes)
    sums = torch.zeros(num_classes, x.shape[1]).index_add_(0, y, x)
    centroids = normalize(sums, dim=1)
    similarities = x @ centroids.T
    similarities[:, sizes == 0] = -torch.inf
    similarities[torch.arange(len(y)), y] = -torch.inf
    rival_cos, rival = similarities.max(dim=1)
    own = (x * normalize(sums[y] - x, dim=1)).sum(dim=1).clamp(-1, 1)

    def save(name, records):
        records = list(records)
        with (out / f'{space}_{name}.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    save('image_geometry', (
        dict(**key, space=space, split='val', image_id=row[0], class_id=c,
             class_name=names[c], within_distance=1 - float(own[i]) if sizes[c] > 1 else '',
             rival_class=int(rival[i]) if torch.isfinite(rival_cos[i]) else '',
             rival_similarity=float(rival_cos[i]) if torch.isfinite(rival_cos[i]) else '',
             margin=float(own[i] - rival_cos[i]) if sizes[c] > 1 and torch.isfinite(rival_cos[i]) else '')
        for i, (row, c) in enumerate(zip(rows, y.tolist()))
    ))
    pair_cos = centroids @ centroids.T
    save('class_similarity', (
        dict(**key, space=space, split='val', class_id=a, class_name=names[a],
             other_id=b, other_name=names[b], centroid_similarity=float(pair_cos[a,b]),
             n=int(sizes[a]), other_n=int(sizes[b]))
        for a in range(num_classes) for b in range(num_classes) if sizes[a] and sizes[b]
    ))
