"""Independently reconstruct metrics and check all saved CIFAR validation artifacts."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'frozen_hybrid_075'
SPLIT_SHA = '3704fd5869ef075ed9e9ef2949506bcf4cfda2a08761683ba3ea78620dbf74b0'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_csv(name):
    with (OUT/name).open() as f:
        return list(csv.DictReader(f))


def check_group(rows, field, group, y, a, b):
    assert [int(r[field]) for r in rows] == list(range(len(rows)))
    for i, r in enumerate(rows):
        m = group == i
        ca, cb = a[m] == y[m], b[m] == y[m]
        assert int(r['n']) == m.sum()
        assert int(r['mean_correct']) == ca.sum() and int(r['hybrid_correct']) == cb.sum()
        assert int(r['fixed']) == (~ca&cb).sum() and int(r['lost']) == (ca&~cb).sum()


def main():
    manifest = ROOT/'audit_manifest.npz'
    with np.load(manifest, allow_pickle=False) as d:
        ids = d['validation_indices']
        y, fine = d['train_coarse'][ids], d['train_fine'][ids]
        assert not np.intersect1d(ids, d['fit_indices']).size
        np.testing.assert_array_equal(np.bincount(y), np.full(20, 250))
        np.testing.assert_array_equal(np.bincount(fine), np.full(100, 50))
    assert hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest() == SPLIT_SHA
    files = sorted((OUT/'feature_chunks').glob('*.npz'))
    assert [p.name for p in files] == [f'{i:04d}.npz' for i in range(0, 5000, 100)]
    with np.load(OUT/'validation_features.npz', allow_pickle=False) as f:
        np.testing.assert_array_equal(f['row_ids'], ids)
        x = f['features']
        assert str(f['encoder_id']) == sha(ROOT/'extract_frozen_clip.py')
    assert x.shape == (5000,512) and np.isfinite(x).all()
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=2e-6)
    for i,path in enumerate(files):
        j = slice(i*100,(i+1)*100)
        with np.load(path, allow_pickle=False) as f:
            np.testing.assert_array_equal(f['row_ids'], ids[j])
            np.testing.assert_array_equal(f['features'], x[j])
    audit = read_csv('candidate_audit.csv')
    assert all((float(r['cosine_to_mean']) < .75) == (r['retained'] == 'True') for r in audit)
    assert all(float(r['min_pair_cosine']) >= .75 - 1e-10 for r in audit)
    counts = np.bincount([int(r['coarse_id']) for r in audit if r['retained']=='True'], minlength=20)
    with np.load(OUT/'text_bank.npz', allow_pickle=False) as b:
        np.testing.assert_array_equal(b['sizes'], counts+1)
        ref_mean = x @ b['means'].T
        p = np.split(b['prototypes'], np.cumsum(b['sizes'])[:-1])
        w = np.split(b['weights'], np.cumsum(b['sizes'])[:-1])
        bank = json.loads((OUT/'prompt_bank.json').read_text())
        assert bank['descriptions_sha256'] == str(b['bank_sha']) == sha(ROOT/'coarse_descriptions.json')
        assert str(b['scorer_sha']) == sha(ROOT/'score_frozen_clip.py')
        descriptions = [bank['descriptions'][k] for k in bank['class_order']]
        assert bank['exact_wrapped_prompts'] == [w.format(d) for ds in descriptions for d in ds for w in bank['wrappers']]
        vecs = np.split(b['wrapped_description_vectors'].astype('float64'), np.cumsum(list(map(len,descriptions)))[:-1])
        ref_hybrid = np.column_stack([.05*np.log(np.average(np.exp(x.astype('float64') @ v.T/.05), weights=z, axis=1)) for v,z in zip(p,w)])
        for i, (v,z) in enumerate(zip(p,w)):
            all_c = [r for r in audit if int(r['coarse_id']) == i]
            assert int(z[0]) == sum(int(r['size']) for r in all_c)
            assert z[1:].tolist() == [int(r['size']) for r in all_c if r['retained']=='True']
            np.testing.assert_array_equal(v[0], b['means'][i])
            d = vecs[i]
            mean = d.mean(0)
            mean /= np.linalg.norm(mean)
            np.testing.assert_allclose(v[0], mean, atol=1e-7)
            groups = [json.loads(r['description_ids']) for r in all_c]
            assert sorted(j for g in groups for j in g) == list(range(len(d)))
            retained_vectors = []
            for r, g in zip(all_c, groups):
                raw = d[g].mean(0)
                cosine = (raw @ mean) / np.linalg.norm(raw)
                assert len(g) == int(r['size'])
                assert json.loads(r['descriptions']) == [descriptions[i][j] for j in g]
                np.testing.assert_allclose(cosine, float(r['cosine_to_mean']), atol=1e-7)
                if r['retained'] == 'True':
                    retained_vectors.append(raw)
            if retained_vectors:
                np.testing.assert_allclose(v[1:], retained_vectors, atol=1e-7)
    with np.load(OUT/'validation_scores.npz', allow_pickle=False) as s:
        np.testing.assert_array_equal(s['row_ids'], ids)
        am, bh = s['same_prompt_mean'], s['hybrid_075']
        assert am.shape == bh.shape == (5000,20) and np.isfinite(am).all() and np.isfinite(bh).all()
        np.testing.assert_allclose(am, ref_mean, atol=1e-6)
        np.testing.assert_allclose(bh, ref_hybrid, atol=1e-6)
        np.testing.assert_array_equal(bh[:, counts==0], am[:, counts==0])
        a, b = am.argmax(1), bh.argmax(1)
    preds = read_csv('predictions.csv')
    for key, expected in zip(['row_id','fine_id','truth_coarse_id','same_prompt_mean','hybrid_075'], [ids,fine,y,a,b]):
        np.testing.assert_array_equal([int(r[key]) for r in preds], expected)
    check_group(read_csv('by_coarse.csv'), 'coarse_id', y, y, a, b)
    check_group(read_csv('by_fine_subtype.csv'), 'fine_id', fine, y, a, b)
    summary = json.loads((OUT/'summary.json').read_text())
    ca, cb = a==y, b==y
    assert summary['mean_correct'] == ca.sum() and summary['hybrid_correct'] == cb.sum()
    assert summary['fixed'] == (~ca&cb).sum() and summary['lost'] == (ca&~cb).sum()
    assert summary['prediction_changes'] == (a!=b).sum() and summary['n'] == len(preds) == len(ids) == 5000
    assert summary['delta_pp'] == 100*(int(cb.sum())-int(ca.sum()))/5000
    assert summary['split_indices_sha256'] == SPLIT_SHA and summary['manifest_sha256'] == sha(manifest)
    assert summary['feature_sha256'] == sha(OUT/'validation_features.npz') and summary['prediction_sha256'] == sha(OUT/'predictions.csv')
    assert summary['n_clusters'] == len(audit) and summary['n_retained'] == counts.sum() and summary['test_scored'] is False
    names = ['summary.json','predictions.csv','validation_scores.npz','validation_features.npz','candidate_audit.csv',
             'by_coarse.csv','by_fine_subtype.csv','prompt_bank.json','text_bank.npz']
    tmp = OUT/'checksums.tmp'
    tmp.write_text(''.join(f'{sha(OUT/name)}  {name}\n' for name in names))
    tmp.replace(OUT/'SHA256SUMS')
    print(f'Verified exact audited row order and 50/50 chunks, score replay, predictions, breakdown counts and summaries: {ca.sum()}/5000 mean; {cb.sum()}/5000 hybrid.')


if __name__ == '__main__':
    main()
