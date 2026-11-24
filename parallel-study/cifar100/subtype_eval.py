"""CPU-only coarse-classifier diagnostics by true CIFAR-100 fine subtype."""
from pathlib import Path

import numpy as np


def _integer_vector(values, name, length, upper):
    values = np.asarray(values)
    if values.shape != (length,) or values.dtype.kind not in 'iu':
        raise ValueError(f'{name} must be a 1-D integer array of length {length}')
    if np.any(values < 0) or np.any(values >= upper):
        raise ValueError(f'{name} values must be in [0, {upper})')
    return values.astype(np.int64, copy=False)


def evaluate_subtypes(predictions, *, split, row_ids, manifest=None):
    """Evaluate IDs [N] or finite scores [N,20] in official coarse-ID order.

    Pass one original 0-based row ID per prediction, in prediction order.
    For validation use official *train* row IDs; for test use official test row IDs.
    Requires exactly the full named split, but supports arbitrary row order.
    Accuracy is the fraction predicting the subtype's true *coarse* parent.
    Tied scores select the lowest coarse ID. No fine predictions are required.
    """
    if split not in ('validation', 'test'):
        raise ValueError('split must be validation or test')
    if manifest is None:
        base = Path(__file__).resolve().parent if '__file__' in globals() else OUT
        manifest = base / 'audit_manifest.npz'
    with np.load(manifest, allow_pickle=False) as data:
        source = 'train' if split == 'validation' else 'test'
        all_fine, all_coarse = data[f'{source}_fine'], data[f'{source}_coarse']
        expected = data['validation_indices'] if split == 'validation' else np.arange(len(all_fine))
        names, parents, coarse_names = (data[k] for k in ('fine_names', 'fine_to_coarse', 'coarse_names'))
    n = len(expected)
    row_ids = _integer_vector(row_ids, 'row_ids', n, len(all_fine))
    if not np.array_equal(np.sort(row_ids), expected):
        raise ValueError(f'row_ids must cover {split} exactly once (no duplicates or foreign rows)')
    pred = np.asarray(predictions)
    if pred.ndim == 2:
        if pred.shape != (n, 20) or pred.dtype.kind not in 'fiu' or not np.isfinite(pred).all():
            raise ValueError(f'scores must be a finite numeric ({n}, 20) array')
        pred = pred.argmax(axis=1)
    pred = _integer_vector(pred, 'predicted coarse IDs', n, 20)
    fine, coarse = all_fine[row_ids], all_coarse[row_ids]
    matrix = np.zeros((100, 20), dtype=np.int64)
    np.add.at(matrix, (fine, pred), 1)
    rows = []
    for fid, parent in enumerate(parents):
        counts = matrix[fid]
        support, correct = int(counts.sum()), int(counts[parent])
        rows.append(dict(fine_id=fid, fine_name=str(names[fid]), coarse_id=int(parent),
                         coarse_name=str(coarse_names[parent]), support=support, correct=correct,
                         accuracy=correct / support if support else None,
                         wrong_coarse_counts={int(cid): int(counts[cid]) for cid in np.flatnonzero(counts) if cid != parent}))
    return dict(split=split, n=n, correct=int(np.count_nonzero(pred == coarse)),
                coarse_accuracy=float(np.mean(pred == coarse)),
                macro_subtype_accuracy=float(np.mean([r['accuracy'] for r in rows if r['support']])),
                by_subtype=rows)
