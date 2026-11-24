# NoLA dataset-by-dataset statistics

The alignment head is fitted on frozen original DINO ViT-B/16 features. The visual-prompt classifier is CLIP or SigLIP 2 with learned visual prompts. DINOv3 ConvNeXt-Tiny is used for retrieval only. Stage deltas compare identical pilot validation rows, not the full alignment-head score against the smaller classifier pilot. Classifier adaptation uses 320 images per dataset. Intervals are paired image-level bootstrap intervals, not across-seed intervals.

## Coverage by dataset

| Dataset | Frozen CDE | Alignment head | Visual-prompt classifier |
|---|---|---|---|
| EuroSAT | 5,400 validation | 5,400 validation; release-style and factorial protocols | Release-style: 8,100 test; factorial: 200 validation |
| CIFAR-100 coarse | 5,000 validation | 5,000 standard validation; separate 1,024-image scarcity pilot | 400 validation, scarcity pilot |
| RESISC45 | 6,255 validation | 6,255 validation for anchor-policy comparison | Not run |
| Oxford-IIIT Pet | 3,669 test, frozen template comparison | Not run for the template comparison | Not run for the template comparison |

Baseline anchor-policy top-1 (%) with the frozen DINO alignment head (confidence / farthest-first / medoids / neighbor agreement):

| Dataset | N | Confidence | Farthest-first | Medoids | Neighbor agreement |
|---|---:|---:|---:|---:|---:|
| EuroSAT | 5400 | 67.07% | 61.48% | 62.07% | 69.74% |
| RESISC45 | 6255 | 69.32% | 66.38% | 67.16% | 69.59% |
| CIFAR-100 coarse | 5000 | 63.98% | 62.20% | 62.38% | 62.80% |

## EuroSAT

Independent-nomination factorial: 13,500 adaptation images, 5,400 alignment-head validation images. On the smaller 200-image validation subset, both stages are scored on the exact same images.

### Alignment head: original validation

| Arm | Top-1 | Correct anchors | Correct additions |
|---|---|---|---|
| CLIP + DINO | 3673/5400 (68.02%) | 1834/2452 (74.80%) | 96/128 |
| SigLIP 2 + DINO | 3139/5400 (58.13%) | 1648/2452 (67.21%) | 66/128 |
| CLIP + ConvNeXt | 3695/5400 (68.43%) | 1832/2452 (74.71%) | 94/128 |
| SigLIP 2 + ConvNeXt | 3182/5400 (58.93%) | 1650/2452 (67.29%) | 68/128 |

### Paired stage difference on the same images

| Arm | Alignment head | Visual-prompt classifier | Fixes / losses | Δ pp [95% CI] |
|---|---|---|---|---|
| CLIP + DINO | 143/200 (71.50%) | 128/200 (64.00%) | 19 / 34 | -7.50 [-13.50, -1.50] |
| SigLIP 2 + DINO | 113/200 (56.50%) | 120/200 (60.00%) | 24 / 17 | +3.50 [-2.50, +9.50] |
| CLIP + ConvNeXt | 140/200 (70.00%) | 139/200 (69.50%) | 19 / 20 | -0.50 [-6.00, +5.00] |
| SigLIP 2 + ConvNeXt | 112/200 (56.00%) | 124/200 (62.00%) | 30 / 18 | +6.00 [-0.50, +12.50] |

### Visual-prompt classifier trajectory

| Arm | Epoch 2 | Epoch 4 | Epoch 12 | Best observed validation |
|---|---|---|---|---|
| CLIP + DINO | 64.50% | 68.50% | 64.00% | 76.50% (e9) |
| SigLIP 2 + DINO | 56.00% | 60.00% | 60.00% | 64.50% (e7) |
| CLIP + ConvNeXt | 60.50% | 73.50% | 69.50% | 78.50% (e11) |
| SigLIP 2 + ConvNeXt | 68.50% | 58.50% | 62.00% | 68.50% (e2) |

### Matched comparisons

| Contrast | Alignment head Δ pp [95% CI] | Epoch-12 classifier Δ pp [95% CI] |
|---|---|---|
| SigLIP − CLIP (DINO) | -9.89 [-11.17, -8.61] | -4.00 [-11.00, +3.00] |
| SigLIP − CLIP (ConvNeXt) | -9.50 [-10.81, -8.26] | -7.50 [-14.00, -1.00] |
| ConvNeXt − DINO (CLIP) | +0.41 [-0.11, +0.94] | +5.50 [+1.00, +10.00] |
| ConvNeXt − DINO (SigLIP) | +0.80 [+0.26, +1.35] | +2.00 [-3.00, +7.50] |

## CIFAR-100 coarse scarcity pilot

Independent-nomination factorial: 4,096 adaptation images, 1,024 alignment-head validation images. On the smaller 400-image validation subset, both stages are scored on the exact same images.

### Alignment head: original validation

| Arm | Top-1 | Correct anchors | Correct additions |
|---|---|---|---|
| CLIP + DINO | 656/1024 (64.06%) | 666/768 (86.72%) | 38/40 |
| SigLIP 2 + DINO | 647/1024 (63.18%) | 697/768 (90.76%) | 39/40 |
| CLIP + ConvNeXt | 664/1024 (64.84%) | 666/768 (86.72%) | 38/40 |
| SigLIP 2 + ConvNeXt | 653/1024 (63.77%) | 695/768 (90.49%) | 37/40 |

### Paired stage difference on the same images

| Arm | Alignment head | Visual-prompt classifier | Fixes / losses | Δ pp [95% CI] |
|---|---|---|---|---|
| CLIP + DINO | 246/400 (61.50%) | 222/400 (55.50%) | 28 / 52 | -6.00 [-10.25, -1.75] |
| SigLIP 2 + DINO | 237/400 (59.25%) | 227/400 (56.75%) | 37 / 47 | -2.50 [-6.50, +1.75] |
| CLIP + ConvNeXt | 252/400 (63.00%) | 212/400 (53.00%) | 22 / 62 | -10.00 [-14.25, -5.75] |
| SigLIP 2 + ConvNeXt | 241/400 (60.25%) | 219/400 (54.75%) | 27 / 49 | -5.50 [-9.50, -1.50] |

### Visual-prompt classifier trajectory

| Arm | Epoch 2 | Epoch 4 | Epoch 12 | Best observed validation |
|---|---|---|---|---|
| CLIP + DINO | 60.50% | 59.50% | 55.50% | 60.50% (e2) |
| SigLIP 2 + DINO | 58.25% | 58.00% | 56.75% | 59.25% (e9) |
| CLIP + ConvNeXt | 56.25% | 56.25% | 53.00% | 56.25% (e2) |
| SigLIP 2 + ConvNeXt | 58.75% | 56.25% | 54.75% | 58.75% (e2) |

### Matched comparisons

| Contrast | Alignment head Δ pp [95% CI] | Epoch-12 classifier Δ pp [95% CI] |
|---|---|---|
| SigLIP − CLIP (DINO) | -0.88 [-3.03, +1.27] | +1.25 [-3.50, +6.00] |
| SigLIP − CLIP (ConvNeXt) | -1.07 [-3.42, +1.27] | +1.75 [-2.75, +6.25] |
| ConvNeXt − DINO (CLIP) | +0.78 [-0.20, +1.76] | -2.50 [-4.75, -0.25] |
| ConvNeXt − DINO (SigLIP) | +0.59 [-0.39, +1.56] | -2.00 [-5.25, +1.25] |
