cap = max(1, len(rows['train']) // (5*len(names)))
seed_pools = ex.candidates(release_logits['train'], cap)
anchor_arms = {}
# Rank CLIP nominees by agreement among frozen DINO neighbors.
x = normalize(dino_features['train'].float().cuda(), dim=1)
p = release_logits['train'].softmax(1).cuda()
label = p.argmax(1)
k, block = min(20, len(x)-1), 256
neighbors, distances = [], []
for start in range(0, len(x), block):
    cosine = x[start:start + block] @ x.T
    r = torch.arange(len(cosine), device=x.device)
    cosine[r, start + r] = -torch.inf
    vals, idx = cosine.topk(k, dim=1)
    neighbors.append(idx)
    distances.append((1. - vals).clamp_min(1e-6))
idx, dist = torch.cat(neighbors), torch.cat(distances)
scale = dist[:, -1]
weight = torch.exp(-dist.square() / (scale[:, None] * scale[idx]))
same = (label[idx] == label[:, None]).float()
overlap = (p.sqrt()[:, None, :] * p.sqrt()[idx]).sum(2).clamp(0, 1)
# Change only the graph edge weights.
modified = weight * (.25 + .75 * overlap)
agreement = [((w * same).sum(1) / w.sum(1).clamp_min(1e-12)).cpu() for w in (weight, modified)]
ranks = []
for values in agreement:
    class_pools = [torch.where(label.cpu() == c)[0] for c in range(p.shape[1])]
    ranks.append([r[values[r].argsort(descending=True, stable=True)] for r in class_pools])
first = idx[0, :min(5, k)]
graph_trace = dict(neighbors=first.cpu().tolist(), dino_weights=weight[0, :len(first)].cpu().tolist(),
             reweighted=modified[0, :len(first)].cpu().tolist())
rankings = ranks
quota = [len(p) for p in seed_pools]
for mode, rank in zip(('dino_only', 'dino_clip'), rankings):
    anchor_arms[mode] = ex.pack(rank, quota)
print('One train image and its DINO neighbors:', rows['train'][0][0],
      [rows['train'][i][0] for i in graph_trace['neighbors']])
print('Same edges: original → reweighted', graph_trace['dino_weights'], graph_trace['reweighted'])
