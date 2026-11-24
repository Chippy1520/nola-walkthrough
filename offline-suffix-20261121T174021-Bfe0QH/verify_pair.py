"""CPU-only artifact verification. No model load, network access or GPU use."""
import hashlib
import json
import math
import sys
from pathlib import Path

out=Path(sys.argv[1]).resolve()
def load(name): return json.loads((out/name).read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
c=load('config.json'); m=load('manifest.json'); bank=load('prompts.json'); p=load('predictions.json'); r=load('results.json')
assert c['manifest_sha256']==sha(out/'manifest.json') and c['prompts_sha256']==sha(out/'prompts.json') and c['source_sha256']==sha(Path(c['argv'][0]))
assert c['counts']=={k:len(v) for k,v in m.items()}
ids=[row[0] for rows in m.values() for row in rows]; assert len(ids)==len(set(ids))
assert [(q['id'],q['label']) for q in p]==[(key,y) for key,y,_ in m['test']]
for name in c['classes']:
    t=[x.format(name) for x in c['templates']]; b=bank['baseline'][name]; d=b[:-len(t)]
    assert b[-len(t):]==t and bank['suffix'][name]==[f'{x} {suffix}' for x in d for suffix in t]
events=[json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
epoch_results=[e for e in events if e['stage']=='student_epoch_done']
for arm in ('baseline','suffix'):
    es=[z for z in epoch_results if z['arm']==arm]; assert [z['epoch'] for z in es]==list(range(1,c['epochs']+1))
    assert all(z['updates']==math.ceil(c['counts']['train']/c['effective_batch']) for z in es)
    match=max(es,key=lambda z:z['val']['correct']); sr=r['arms'][arm]['student']
    assert sr['best_epoch']==match['epoch'] and sr['val']==match['val'] and (out/sr['checkpoint']).stat().st_size>0
for stage in ('cde','teacher','student'):
    b=[q[f'baseline_{stage}']==q['label'] for q in p]; s=[q[f'suffix_{stage}']==q['label'] for q in p]
    fixed=sum(y and not x for x,y in zip(b,s)); lost=sum(x and not y for x,y in zip(b,s)); delta=100*(sum(s)-sum(b))/len(p)
    for arm, flags in [('baseline',b),('suffix',s)]:
        report=r['arms'][arm][stage]['test']; assert report['correct']==sum(flags) and report['n']==len(p) and math.isclose(report['accuracy'],sum(flags)/len(p))
    changed=sum(q[f'baseline_{stage}']!=q[f'suffix_{stage}'] for q in p)
    assert r['paired_test'][stage]==dict(fixed=fixed,lost=lost,changed=changed,delta_pp=delta) and fixed-lost==sum(s)-sum(b)
    print(f'{stage:7} baseline {sum(b)}/{len(p)} ({sum(b)/len(p):.2%}) | suffix {sum(s)}/{len(p)} ({sum(s)/len(p):.2%}) | {delta:+.2f} pp, fixed {fixed}, lost {lost}, changed {changed}')
assert events[-1]['stage']=='done'
print('Verified: provenance, prompt construction, disjoint row IDs, checkpoint selection, update counts, and paired held-out metrics.')
