text_banks = {}
for descriptions, templates in prompt_parts:
    d = nola_eval.encode_texts(clip, descriptions)
    t = nola_eval.encode_texts(clip, templates)
    # Historical suffix: each formatted template follows each description, before tokenization.
    suffix = [f'{p.rstrip()} {q}' for p in descriptions for q in templates]
    # Literal inverse order: append each description to each formatted template.
    template_suffix = [f'{q.rstrip()} {p}' for p in descriptions for q in templates]
    sx = nola_eval.encode_texts(clip, suffix)
    tx = nola_eval.encode_texts(clip, template_suffix)
    # Group near-duplicate prompts before merging or blending.
    sx_by_desc = normalize(sx.reshape(len(d), len(t), -1).mean(1), dim=1)
    mean = ex.centroid(sx_by_desc)
    groups = ex.clusters(sx_by_desc, .75)
    directions = torch.stack([ex.centroid(sx_by_desc[g]) for g in groups])
    weights = torch.tensor([len(g) for g in groups], dtype=mean.dtype)
    banks = dict(descriptions=ex.centroid(d)[None],
                 template_add=normalize(ex.centroid(d) + ex.centroid(t), dim=0)[None],
                 suffix=ex.centroid(sx)[None], template_suffix=ex.centroid(tx)[None],
                 unmerged=d, suffix_mean=mean[None])
    for cutoff, name in [(.75, 'merge075'), (.80, 'merge080')]:
        banks[name] = torch.stack([ex.centroid(d[g]) for g in ex.clusters(d, cutoff)])
    for gate, name in [(.75, 'blend075'), (.90, 'blend090')]:
        keep = directions @ mean < gate
        v = normalize(len(d) * mean + weights[keep] @ directions[keep], dim=0) if keep.any() else mean
        banks[name] = v[None]
    for kind, prototypes in banks.items():
        text_banks.setdefault(kind, []).append(prototypes)
