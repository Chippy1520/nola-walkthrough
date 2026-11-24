# Independent nomination × retrieval

Four arms: CLIP-only or SigLIP 2-only fit-pool nomination, initial seeds, expansion candidate ownership and ranking, crossed with original DINO or DINOv3 ConvNeXt-Tiny retrieval for expansion. Downstream teacher always uses original DINO features and the earlier fixed head/training. Match per-class seed counts and additions by minimum feasible budget. Compare on full EuroSAT 13,500/5,400 and the earlier CIFAR-100 4,096/1,024 scarcity pilot (replay stress class IDs and cap, hold each model's removed natural seeds out of its own expansion pool). No image-level truth enters selection or training. No test, student training or runtime benchmark. Use row-aligned score and feature caches with checksums.

Run `factorial.py eurosat --smoke` then `factorial.py eurosat` and `factorial.py cifar100` in the desktop research-lab venv with the project GPU lock. Logs `factorial-*.log`; verified artifacts under `factorial/{eurosat,cifar100}/`. Prior CLIP controls are checked for anchor/validation prediction parity, but heads are refit in this run. Two-sided exact McNemar and within-class paired validation bootstrap describe image variation under the fixed seed, not seed variation. Do not conflate this with previous same-CLIP-pool reranking.

Verified CLIP seed/extra selections and validation predictions exactly match the earlier four-arm study for both retrievers and both datasets. The first smoke fixture had no feasible matched extras with the production gate; the smoke-only fixture was changed to exercise 8 additions. No production setting changed.

| Arm | EuroSAT teacher | CIFAR-100 scarcity teacher |
|---|---:|---:|
| CLIP + DINO | 68.02% | 64.06% |
| SigLIP 2 + DINO | 58.13% | 63.18% |
| CLIP + DINOv3 ConvNeXt | 68.43% | 64.84% |
| SigLIP 2 + DINOv3 ConvNeXt | 58.93% | 63.77% |

EuroSAT: replacing CLIP with SigLIP lost 9.89 pp with DINO (95% CI −11.17 to −8.61) and 9.50 pp with ConvNeXt (−10.81 to −8.26). On scarcity-stressed CIFAR-100, the losses were 0.88 pp (−3.03 to +1.27) and 1.07 pp (−3.42 to +1.27), respectively. ConvNeXt recovers 0.80 pp over DINO within SigLIP on EuroSAT, not enough to close the nomination gap. See `factorial/{eurosat,cifar100}/verified.json` for all paired comparisons.
