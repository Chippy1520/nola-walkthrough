# Keep the CLIP seed pool fixed; change the ranking for extra anchors.
cosine = release_cosine['train']
features = dino_features['train']
seeds = seed_pools
x = normalize(features, dim=1)
used = torch.zeros(len(x), dtype=torch.bool)
used[torch.cat(seeds)] = True
under = [c for c, ids in enumerate(seeds) if 0 < len(ids) < cap]
clip_only = clip_dino = None
if under:
    under = torch.tensor(under)
    owner = under[cosine[:, under].argmax(1)]
    gap = cosine.max(1).values - cosine[torch.arange(len(x)), owner]
    log_p = (clip_scale*cosine).log_softmax(1)
    baseline, retrieved, added = [], [], []
    for c, ids in enumerate(seeds):
        pool = torch.where((owner == c) & (~used) & (gap <= .06))[0]
        budget = min(cap-len(ids), min(64, max(8, cap//2)), len(pool))
        if budget:
            confidence = log_p[pool, c]
            nearest = (x[pool] @ x[ids].T).topk(min(5, len(ids)), dim=1).values.mean(1)
            base = pool[confidence.argsort(descending=True, stable=True)[:budget]]
            new = pool[(confidence + .5*nearest).argsort(descending=True, stable=True)[:budget]]
        else:
            base = new = pool[:0]
        added.append(budget)
        baseline.append(torch.cat((ids, base)))
        retrieved.append(torch.cat((ids, new)))
    if sum(added):
        quota = [len(ids) for ids in baseline]
        clip_only = ex.pack(baseline, quota)
        clip_dino = ex.pack(retrieved, quota)
if clip_only is not None:
    anchor_arms.update(clip_expansion=clip_only, dino_retrieval=clip_dino)
else:
    print('Seed-retrieval comparison skipped: no eligible additions.')
