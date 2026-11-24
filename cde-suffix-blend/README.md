# NoLA CDE suffix + normalized blend

Frozen CLIP ViT-B/32 validation replay. Previous studies, training checkpoints, test splits and notebooks were not modified. EuroSAT uses all 5,400 official CoOp validation rows; CIFAR-100 uses the same 5,000 fine-stratified validation rows and image features as the earlier study. RESISC45 uses the archived 6,255-row adapter-derived validation manifest (not the authors' exact split).

For each released description, append each filled class template separately before encoding. Normalize each CLIP embedding, average those six vectors and normalize to get the description vector. Use the released six satellite templates for EuroSAT/RESISC45; for coarse CIFAR-100, use six class-name suffixes adapted from the earlier custom photo wording. Exact texts are in `rendered_prompts.json`. Complete linkage groups description vectors at cosine 0.75, as before. The historical variation gates are 0.75 and 0.90 (strictly below cosine to class mean), not tuned here. The new scorer mixes normalized directions: `unit(N * unit(class_mean) + sum(n_j * unit(cluster_mean_j)))`, where `N` is the description count and `n_j` each retained cluster size. Outputs exactly one vector per original class. Zero passing clusters reproduces suffix-mean predictions exactly. As a pooling control, score the identical suffix bank with the prior weighted log-mean-exp rule (separate score components, same labels, tau 0.05).

| Dataset | Original separate | Previous wrapped mean | Suffix mean | Suffix blend 0.75 | Suffix blend 0.90 | Suffix old pooling 0.90 |
|---|---:|---:|---:|---:|---:|---:|
| EuroSAT | 51.06% | 51.59% | 51.48% | 51.48% | 51.20% | 51.59% |
| CIFAR-100 coarse | — | 58.50% | 50.48% | 50.48% | 51.10% | 51.58% |
| RESISC45 | 58.07% | 56.48% | 58.86% | 58.86% | 58.90% | 58.96% |

EuroSAT: original separate and wrapped mean were rerun locally on the same image embeddings as the new arms, not copied from old Colab FP16 outputs. At 0.90 two clusters passed, 19 fixes/34 losses. CIFAR-100: wrapped mean reused exact prior CPU predictions (2925/5000); suffix text encoding was on the local GPU, scoring reused the CPU image vectors. Five clusters passed at 0.90, 59 fixes/28 losses. At 0.75 no cluster passed in either dataset. These are validation CDE measurements, not retrained DINO/CLIP student runs.

`verify.py eurosat cifar100 resisc45` independently checks row IDs, top-1 predictions, normalized vector shapes, paired counts and per-class metrics. Outputs: `{dataset}/verified.json`, `predictions.csv`, `scores.npz`, `text_vectors.npz`, `clusters.csv`, `rendered_prompts.json`. Each summary also identifies input/runner hashes and exact template lists.

RESISC45: verified on the previous adapter-derived split. No cluster passed at 0.75; ten passed at 0.90, yielding 12 fixes and 10 losses over the suffix mean. The previous wrapped mean is a newly run mean-only control, not the historical cluster-only 56.05% arm. Bounded range-resume matched the archived SHA-256. All 6,255 image vectors had been cached when the first text pass stopped at a FP32 partition-consistency assertion (maximum observed reduction-order difference 1.69e-7). `run.py` now verifies exact membership and the partition identity in FP64, with the classifier math unchanged. The restart from cached features completed and verified.
