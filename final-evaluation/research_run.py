import hashlib
import json
import random
import torch
import torchvision
import experiments as ex
import nola_eval as shared


def prepare_run(base, root, data, tag, run_name, seed, train_limit, val_limit):
    assert torch.cuda.is_available(), 'Colab GPU required'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    rows, source, names, bank = shared.load_data(tag, data, seed=seed)
    rows = {s: rows[s] for s in ('train', 'val')}
    if train_limit is not None or val_limit is not None:
        rng = random.Random(seed)
        for s, limit in [('train', train_limit), ('val', val_limit)]:
            if limit is not None:
                rows[s] = [rows[s][i] for i in sorted(rng.sample(range(len(rows[s])), limit))]
    assert all(rows.values())
    row_ids = {s: [str(r[0]) for r in part] for s, part in rows.items()}
    assert all(len(set(ids)) == len(ids) for ids in row_ids.values())
    overlap = len(set(row_ids['train']) & set(row_ids['val']))
    assert overlap == 0, 'Train/validation row IDs overlap'
    for part in rows.values():
        assert all(0 <= y < len(names) and name == names[y] for _, y, name in part)
    split_blob = json.dumps(rows, ensure_ascii=False, default=str).encode()
    split_id = hashlib.sha256(split_blob).hexdigest()
    desc, templates = shared.get_prompts(root, bank)
    prompt_parts = [(list(desc[name]), [t.format(name) for t in templates]) for name in names]
    out = base / 'runs' / '03-experiments' / run_name
    out.mkdir(parents=True, exist_ok=False)
    ctx = dict(run_id=run_name, dataset=tag, split_id=split_id, seed=seed,
               scope='full' if train_limit is val_limit is None else 'pilot')
    (out / 'split.json').write_bytes(split_blob)
    (out / 'prompts.json').write_text(json.dumps(dict(classes=names, parts=prompt_parts), indent=2))
    return rows, source, names, prompt_parts, row_ids, out, ctx


def export_arm_results(base, out, ctx, rows, row_ids, names, arms, contrasts,
                  prediction_sets, cap, clip_scale, replacement, problems, epochs):
    manifest = dict(**ctx, epochs=epochs, anchor_target=cap, stage='development_validation',
                    clip_scale=clip_scale, replacements=replacement, missing=problems,
                    train_n=len(rows['train']), val_n=len(rows['val']), overlap=0,
                    torch=str(torch.__version__), torchvision=torchvision.__version__,
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                    source_hashes={p.name: ex.sha(p) for p in
                                   [base/'nola_eval.py', base/'experiments.py', base/'research_run.py']})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    split_y = {s: torch.tensor([r[1] for r in part]) for s, part in rows.items()}
    statuses, metrics = [], []
    metric_fields = list(ctx) + ['arm', 'family', 'text', 'geometry', 'selection',
                                 'checkpoint', 'split', 'stage', 'metric', 'value', 'unit', 'correct', 'n']
    for arm, config in arms.items():
        family, text, geometry, selection = (config[k] for k in ('family', 'text', 'geometry', 'selection'))
        meta = dict(**ctx, arm=arm, family=family, text=text, geometry=geometry, selection=selection)
        target = config['anchor_ids']
        checkpoint = '' if target is None else f'{arm}/alignment.pt'
        status = dict(**meta, checkpoint=checkpoint, status='not_run', reason=config['reason'])
        if target is not None:
            idx, pseudo = target
            work = out / arm
            pred = prediction_sets[arm]
            truth = split_y['val']
            for stage, p in pred.items():
                hit, _ = shared.top1_counts(p, truth)
                metrics.append(dict(**meta, checkpoint=checkpoint if stage == 'alignment' else '',
                                    split='val', stage=stage, metric='top1', value=100 * hit/len(truth),
                                    unit='percent', correct=hit, n=len(truth)))
            truth_a = split_y['train'][idx]
            hit, _ = shared.top1_counts(pseudo, truth_a)
            metrics.append(dict(**meta, checkpoint='', split='train', stage='anchors',
                                metric='pseudo_label_precision', value=100 * hit/len(idx),
                                unit='percent', correct=hit, n=len(idx)))
            ex.write_csv(work / 'anchors.csv', [dict(**meta, split='train', row_id=row_ids['train'][i],
                                                       pseudo=int(p), truth=int(y))
                                                for i, p, y in zip(idx.tolist(), pseudo, truth_a)])
            ex.write_csv(work / 'predictions.csv', [dict(**meta, split='val', row_id=row_id, stage=stage,
                          checkpoint=checkpoint if stage == 'alignment' else '', truth=int(y), pred=int(v))
                          for stage, p in pred.items() for row_id, y, v in zip(row_ids['val'], truth, p)])
            status.update(status='complete', reason='')
        statuses.append(status)
        ex.write_csv(out / 'arms.csv', statuses)
        ex.write_csv(out / 'metrics.csv', metrics, fields=metric_fields)

    pairs = []
    truth = split_y['val']
    for variant, control, factor in contrasts:
        for stage in ('cde', 'alignment'):
            done = variant in prediction_sets and control in prediction_sets
            old_hits, new = (prediction_sets[k][stage] == truth for k in (control, variant)) if done else (None, None)
            fixes = int((~old_hits & new).sum()) if done else ''
            losses = int((old_hits & ~new).sum()) if done else ''
            pairs.append(dict(**ctx, split='val', variant=variant, control=control, factor=factor,
                              stage=stage, status='complete' if done else 'not_run',
                              n=len(truth) if done else '', fixes=fixes, losses=losses,
                              delta=100 * (fixes - losses)/len(truth) if done else '', unit='percentage_points'))
    ex.write_csv(out / 'comparisons.csv', pairs, fields=list(ctx) + [
        'split', 'variant', 'control', 'factor', 'stage', 'status', 'n', 'fixes', 'losses', 'delta', 'unit'])
    return prediction_sets
