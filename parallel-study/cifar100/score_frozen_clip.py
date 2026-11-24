"""Same-prompt frozen mean vs .75 hybrid; CPU only, validation only."""
import argparse
import csv
import json
from pathlib import Path
import platform

from extract_frozen_clip import OUT, ROOT, WEIGHTS, CHECKS, SPLIT, atomic_npz, load_cpu, sha, valid
import clip
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.special import logsumexp
from scipy.stats import binomtest
import torch
from torch.nn.functional import normalize

PROMPTS = ROOT / 'coarse_descriptions.json'
WRAPPERS = ('Photo: {}', 'A photo: {}', 'The photo: {}', 'A centered photo: {}',
            'This is a photograph: {}', 'A photograph: {}')
THRESHOLD, TAU = .75, .05
MANIFEST = ROOT / 'audit_manifest.npz'
ARMS = ('same_prompt_mean', 'hybrid_075')


def write_json(name, data):
    path = OUT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_csv(name, rows):
    path = OUT / name
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


@torch.inference_mode()
def bank():
    assert sha(WEIGHTS) == CHECKS[WEIGHTS]
    with np.load(MANIFEST, allow_pickle=False) as m:
        keys = m['coarse_names'].tolist()
    descriptions = json.loads(PROMPTS.read_text())
    assert len(keys) == 20 and set(keys) == set(descriptions)
    descs = [s for k in keys for s in descriptions[k]]
    assert len(descs) == len(set(descs)) and all(len(descriptions[k]) >= 2 for k in keys)
    prompts = [w.format(s) for s in descs for w in WRAPPERS]
    model, _ = load_cpu()
    enc = torch.cat([normalize(model.encode_text(tokens), dim=1)
                     for tokens in clip.tokenize(prompts, truncate=False).split(32)])
    vecs = normalize(enc.reshape(len(descs), len(WRAPPERS), 512).mean(1), dim=1).numpy()
    means, protos, counts, audit = [], [], [], []
    for i, (k, v) in enumerate(zip(keys, np.split(vecs, np.cumsum([len(descriptions[k]) for k in keys])[:-1]))):
        distances = pdist(v.astype(np.float64), metric='cosine')
        labels = fcluster(linkage(distances, method='complete'), 1-THRESHOLD, criterion='distance')
        groups = sorted([np.flatnonzero(labels == j).tolist() for j in np.unique(labels)])
        assert sorted(j for group in groups for j in group) == list(range(len(v)))
        dm = squareform(distances)
        raw = np.stack([v[g].mean(0) for g in groups])
        n = np.array([len(g) for g in groups], dtype=np.int64)
        np.testing.assert_allclose(n @ raw / len(v), v.mean(0), atol=2e-8)
        m = v.mean(0)
        m /= np.linalg.norm(m)
        cos = raw.astype(np.float64) @ m.astype(np.float64) / np.linalg.norm(raw.astype(np.float64), axis=1) / np.linalg.norm(m.astype(np.float64))
        keep = cos < THRESHOLD
        means.append(m)
        protos.append(np.concatenate([m[None], raw[keep]]))
        counts.append(np.r_[len(v), n[keep]])
        for j, g in enumerate(groups):
            min_pair = 1 - float(dm[np.ix_(g, g)].max())
            assert min_pair >= THRESHOLD - 1e-10
            audit.append(dict(coarse_id=i, coarse_name=k, cluster_id=j, size=int(n[j]),
                              min_pair_cosine=min_pair, cosine_to_mean=float(cos[j]),
                              raw_norm=float(np.linalg.norm(raw[j])), retained=bool(keep[j]),
                              description_ids=g, descriptions=[descriptions[k][t] for t in g]))
    code_id = sha(Path(__file__))
    atomic_npz(OUT / 'text_bank.npz', means=np.stack(means), prototypes=np.concatenate(protos),
               weights=np.concatenate(counts), sizes=np.array(list(map(len, counts))),
               wrapped_description_vectors=vecs, bank_sha=np.array(sha(PROMPTS)), scorer_sha=np.array(code_id),
               weights_sha=np.array(CHECKS[WEIGHTS]))
    write_json('prompt_bank.json', dict(class_order=keys, descriptions=descriptions, wrappers=list(WRAPPERS),
              exact_wrapped_prompts=prompts, descriptions_sha256=sha(PROMPTS), threshold=THRESHOLD, tau=TAU))
    write_csv('candidate_audit.csv', [{**r, 'description_ids':json.dumps(r['description_ids']),
                                      'descriptions':json.dumps(r['descriptions'])} for r in audit])
    print(f'Text bank verified: {len(prompts)} prompts / {len(descs)} descriptions / {len(audit)} clusters / {sum(r["retained"] for r in audit)} retained', flush=True)


def scores(x):
    with np.load(OUT / 'text_bank.npz', allow_pickle=False) as b:
        assert str(b['bank_sha']) == sha(PROMPTS) and str(b['scorer_sha']) == sha(Path(__file__))
        assert str(b['weights_sha']) == CHECKS[WEIGHTS]
        mean = x @ b['means'].T
        hybrid = np.empty_like(mean)
        cursor = 0
        for i, n in enumerate(b['sizes']):
            p, w = b['prototypes'][cursor:cursor+n], b['weights'][cursor:cursor+n]
            cursor += n
            np.testing.assert_array_equal(p[0], b['means'][i])
            if n == 1:
                # Weighted log-mean-exp of a single component is the component itself.
                # Reuse the exact dot product, avoiding distinct GEMV/GEMM rounding.
                hybrid[:, i] = mean[:, i]
                continue
            logits = x @ p.T / TAU
            hybrid[:, i] = TAU * (logsumexp(logits + np.log(w), axis=1) - np.log(w.sum()))
            independent = TAU * np.log(np.average(np.exp(logits.astype(np.float64)), weights=w, axis=1))
            np.testing.assert_allclose(hybrid[:, i], independent, atol=1e-7)
        assert cursor == len(b['weights']) and np.isfinite(mean).all() and np.isfinite(hybrid).all()
    return mean, hybrid


def row(mask, y, a, b):
    z, h = a[mask] == y[mask], b[mask] == y[mask]
    n, na, nb = len(z), int(z.sum()), int(h.sum())
    fixed, lost = int((~z & h).sum()), int((z & ~h).sum())
    assert nb - na == fixed - lost
    return dict(n=n, mean_correct=na, hybrid_correct=nb, mean_accuracy_pct=100*na/n,
                hybrid_accuracy_pct=100*nb/n, delta_pp=100*(nb-na)/n, fixed=fixed, lost=lost,
                prediction_changes=int(np.count_nonzero(a[mask] != b[mask])))


def evaluate(smoke=False):
    from subtype_eval import evaluate_subtypes
    with np.load(MANIFEST, allow_pickle=False) as m:
        ids = m['validation_indices']
        if smoke:
            x = valid(OUT / 'feature_chunks' / '0000.npz', ids[:100])
            assert all(z.shape == (100, 20) for z in scores(x))
            print('Real-image scoring smoke verified: 100 rows, 20 coarse classes, both arms (not validation metrics)', flush=True)
            return
        x = valid(OUT / 'validation_features.npz', ids)
        y, f = m['train_coarse'][ids], m['train_fine'][ids]
        keys, fine_names, parents = m['coarse_names'], m['fine_names'], m['fine_to_coarse']
    assert len(ids) == 5000 and np.array_equal(parents[f], y)
    mean, hybrid = scores(x)
    a, b = mean.argmax(1), hybrid.argmax(1)
    stats = row(np.ones(len(ids), dtype=bool), y, a, b)
    by_coarse = [dict(coarse_id=i, coarse_name=str(k), **row(y==i, y, a, b)) for i,k in enumerate(keys)]
    by_fine = [dict(fine_id=i, fine_name=str(k), coarse_id=int(parents[i]), coarse_name=str(keys[parents[i]]),
                    **row(f==i, y, a, b)) for i,k in enumerate(fine_names)]
    assert all(r['n'] == 250 for r in by_coarse) and all(r['n'] == 50 for r in by_fine)
    for s, field in zip((mean,hybrid), ('mean_correct','hybrid_correct')):
        q = evaluate_subtypes(s, split='validation', row_ids=ids)
        assert q['correct'] == stats[field] and [r['correct'] for r in q['by_subtype']] == [r[field] for r in by_fine]
    rng = np.random.default_rng(42)
    boot = np.zeros(10000, dtype=np.int64)
    delta = (b==y).astype(np.int8)-(a==y).astype(np.int8)
    for fid in range(100):
        d = delta[f==fid]
        draws = rng.multinomial(len(d), np.bincount(d+1,minlength=3)/len(d), size=len(boot))
        boot += draws[:,2]-draws[:,0]
    stats['delta_pp_ci95'] = (100*np.quantile(boot,[.025,.975])/len(ids)).tolist()
    n_disc = stats['fixed'] + stats['lost']
    stats['mcnemar_exact_p'] = float(binomtest(stats['fixed'], n_disc).pvalue) if n_disc else 1.0
    stats['mean_macro_coarse_pct'] = float(np.mean([r['mean_accuracy_pct'] for r in by_coarse]))
    stats['hybrid_macro_coarse_pct'] = float(np.mean([r['hybrid_accuracy_pct'] for r in by_coarse]))
    atomic_npz(OUT / 'validation_scores.npz', row_ids=ids, same_prompt_mean=mean, hybrid_075=hybrid)
    write_csv('predictions.csv', [dict(row_id=int(i), fine_id=int(s), truth_coarse_id=int(t),
                                      same_prompt_mean=int(u), hybrid_075=int(v)) for i,s,t,u,v in zip(ids,f,y,a,b)])
    write_csv('by_coarse.csv', by_coarse)
    write_csv('by_fine_subtype.csv', by_fine)
    with (OUT / 'candidate_audit.csv').open() as fp:
        candidates = list(csv.DictReader(fp))
    reten = [r for r in candidates if r['retained'] == 'True']
    stats.update(stage='frozen_CLIP_ViT-B/32', device='cpu', precision='float32', split='validation', test_scored=False,
                 comparison=list(ARMS), cluster_min_cosine=THRESHOLD, keep_if_cosine_to_mean_below=THRESHOLD, tau=TAU,
                 aggregation='normalized weighted log-mean-exp; normalized class mean weighted by total descriptions, retained raw cluster means weighted by size',
                 n_clusters=len(candidates), n_retained=len(reten), classes_with_retained=sorted({int(r['coarse_id']) for r in reten}),
                 bootstrap=dict(seed=42,reps=len(boot),stratified_by='fine subtype, preserving the audited split composition'),
                 python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, split_indices_sha256=SPLIT,
                 manifest_sha256=sha(MANIFEST), descriptions_sha256=sha(PROMPTS), weights_sha256=CHECKS[WEIGHTS],
                 archive_sha256=next(v for k,v in CHECKS.items() if k!=WEIGHTS), scorer_sha256=sha(Path(__file__)),
                 feature_sha256=sha(OUT/'validation_features.npz'), prediction_sha256=sha(OUT/'predictions.csv'))
    write_json('summary.json', stats)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['bank', 'smoke', 'evaluate'])
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.mode == 'bank':
        bank()
    else:
        evaluate(args.mode == 'smoke')
