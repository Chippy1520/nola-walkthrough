import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

for target in sys.argv[1:]:
    d=Path(__file__).resolve().parent/target
    s=json.loads((d/'summary.json').read_text())
    with (d/'predictions.csv').open() as f: rows=list(csv.DictReader(f))
    ids=[r['row_id'] for r in rows]; assert len(ids)==len(set(ids))
    y=np.array([int(r['true_id']) for r in rows]); ref=np.array([int(r['suffix_mean']) for r in rows])==y
    with np.load(d/'scores.npz',allow_pickle=False) as scores, np.load(d/'text_vectors.npz',allow_pickle=False) as vectors:
        assert list(map(str,scores['row_ids']))==ids
        for k in vectors.files:
            assert vectors[k].shape==(len(s['classes']),512)
            np.testing.assert_allclose(np.linalg.norm(vectors[k],axis=1),1,atol=2e-6)
        results={}
        for key,m in s['metrics'].items():
            p=np.array([int(r[key]) for r in rows]); z=p==y
            if key in scores: np.testing.assert_array_equal(scores[key].argmax(1),p)
            else: assert key=='legacy_hybrid090' and (z.sum()==2918)
            assert (len(z),int(z.sum()),int((z&~ref).sum()),int((~z&ref).sum()))==(m['n'],m['correct'],m['fixed_vs_suffix_mean'],m['lost_vs_suffix_mean'])
            for i,name in enumerate(s['classes']): assert s['by_class'][name][key]['correct']==int((z&(y==i)).sum())
            fixed,lost=m['fixed_vs_suffix_mean'],m['lost_vs_suffix_mean']
            results[key]=dict(**m, delta_pp_vs_suffix_mean=100*(fixed-lost)/len(y),paired_p=binomtest(fixed,fixed+lost).pvalue if fixed+lost else 1.)
    with (d/'clusters.csv').open() as f: groups=list(csv.DictReader(f))
    for q,c in s['retained'].items():
        assert sum(r['retained'+q]=='True' for r in groups)==c
        if c==0: assert all(r['suffix_mean']==r['suffix_blend'+q] for r in rows)
    checks=dict(rows=len(y),verified=True,retained=s['retained'],metrics=results)
    (d/'verified.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(target,len(y),'retained',s['retained'],[(k,round(v['acc_pct'],2),round(v['delta_pp_vs_suffix_mean'],2)) for k,v in results.items()])
