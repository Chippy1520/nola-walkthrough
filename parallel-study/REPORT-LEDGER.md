# Multi-prototype NoLA: experiment ledger

Goal: compare the original NoLA CDE with the fixed 0.75 complete-linkage prototype bank built after applying photo templates, and test whether distinct visual subtypes benefit. Keep datasets, stages, validation/test splits, seeds, and configurations distinct. No held-out test data for selection.

## EuroSAT (corrected photo-template run)

- Snapshot (UTC 2026-11-20 20:38:21): `eurosat/EuroSAT-template-report-snapshot.zip`; matching records in `eurosat/milestones.csv`. No final test score at snapshot time.
- Dataset: official CoOp 13,500 train / 5,400 validation / 8,100 test. Original CDE vs 16 prototypes at cutoff 0.75. Seed 7777. DINO 50 epochs, CLIP max 50 epochs, effective CLIP batch 512 (AMP, microbatch 32).
- Original / templated: DINO validation 3,624 / 3,679 correct, CLIP epoch 1 validation 4,052 / 4,114 correct (each over 5,400). Correct anchors: 1,735 of 2,324 / 1,808 of 2,371; Herbaceous anchors: 4 of 7 / 3 of 7. Herbaceous DINO validation correct: 3 / 4. Preliminary; no test inference from these values.
- Updated snapshot: `eurosat/EuroSAT-template-report-snapshot-epoch2.zip`; `eurosat/matched_validation.csv` now includes both epoch 1 (+1.15 pp) and epoch 2 (−0.30 pp). The gain is not yet consistent over epochs.
- Epoch 1 paired: 264 fixes, 202 regressions, +62 net. Neither arm correctly predicted any of the 600 Herbaceous validation examples at this checkpoint; see `eurosat/epoch1_per_class_paired.csv`.
- Scratchpad error fixed: an obsolete unwrapped-bank parity assertion was removed from the corrected-bank build cell. The rebuilt 16 prototype vectors matched the saved training bank; no worker restart or experiment change.
- Local recovery bundle: `eurosat/EuroSAT-template-checkpoints-20261120T203912.zip` (verified ZIP and SHA-256); code/inputs in `../colab-nola/NoLA-EuroSAT-Template-Aware-075.zip`. The existing GPU worker was not stopped or restarted for this snapshot.

### EuroSAT 0.75 mean-plus-variants follow-up

- Frozen CLIP ViT-B/32, same 5,400 CoOp validation images, all test images unscored. A class keeps its normalized mean over the wrapped descriptions; add a raw cluster mean only if its cosine to that mean is strictly below 0.75. Fixed normalized weighted log-mean-exp at temperature 0.05, class mean weight 25, extras weighted by cluster size.
- 0 of 16 candidate clusters retained (lowest cosine 0.78312, River). Same-prompt mean and hybrid matched on every prediction: 2,784/5,400 (51.56%) each. Cluster-only 2,785/5,400 (51.57%); original CDE 2,753/5,400 (50.98%). This is not a test of whether active extras improve EuroSAT.
- Verified paired predictions, candidate audit, and per-class counts in `eurosat/NoLA-EuroSAT-hybrid-075-report.zip` (SHA-256 `3e79bab5fab0e6438623cf5bc3b5e3c21b08bd43140d6108ba4a3e00212b8bbc`). Executed code: `eurosat/eurosat_hybrid075_colab.py`. This frozen check is separate from the training run.

## RESISC45

- Independent Colab T4 run completed frozen-CLIP validation. Notebook: `resisc45/NoLA-RESISC45.ipynb`; report archive: `resisc45/NoLA-RESISC45-f67e0f157c474ccc-report.zip` (SHA-256 `3b12591b766d95c0075ef8a8d68ee664c74447a3c35375f2c2195cc3bb2219b9`).
- Adapter-derived deterministic split (not a verified copy of the authors' exact partition): 24,930 train / 6,255 validation / 315 test. The test split was not scored. 2,250 released descriptions became 123 template-aware prototypes.
- Frozen CLIP validation: 3,630/6,255 (58.03%) original vs 3,506/6,255 (56.05%) templated 0.75, −1.98 pp; 206 fixes and 330 regressions. Paired bootstrap 95% interval: −2.64 to −1.28 pp. Archive contains predictions, per-class results, confusion matrices, config, split metadata and prototype vectors. Full adaptation was not run.

## DINO-assisted anchor selection (three-dataset Colab T4 study)

Full fit pools, fixed mean CDE and CLIP pseudo-labels, matched realized anchors per class, frozen DINO features, four 50-epoch teacher fits per dataset (single seed). No test scoring or final CLIP student. See `dino-anchors/RUN-LEDGER.md` and `dino-anchors/reports/combined.csv` (paired bootstrap intervals, precision and per-dataset archive hashes).

| Dataset | Fit pool | Anchors per arm | Confidence | Farthest-first | Medoids | Neighbor agreement |
|---|---:|---:|---:|---:|---:|---:|
| EuroSAT | 13,500 | 2,324 | 67.07% | 61.48% | 62.07% | 69.74% |
| RESISC45 | 24,930 | 4,791 | 69.32% | 66.38% | 67.16% | 69.59% |
| CIFAR-100 coarse | 45,000 | 9,000 | 63.98% | 62.20% | 62.38% | 62.80% |

Neighbor agreement improved audited anchor precision on all three datasets, but teacher accuracy clearly improved only on EuroSAT (+2.67 pp; paired 95% CI +1.80 to +3.54). RESISC45 was inconclusive (+0.27 pp; −0.59 to +1.12), and CIFAR declined (−1.18 pp; −2.20 to −0.18). Farthest-first and medoids reduced teacher accuracy across all three fixed comparisons. EuroSAT Herbaceous still had just seven eligible anchors (four correctly pseudo-labeled in each arm); neighbor agreement matched baseline at 2/600 correct. RESISC45 uses the previously documented deterministic adapter-derived split.

## CIFAR-100 coarse

- CPU audit executed on the original data bytes. Verified 60,000 label pairs, exactly five fine subtypes for each of 20 coarse classes. Fixed split: 45,000 fit / 5,000 validation / 10,000 official test; each subtype contributes 450 / 50 / 100 rows. Seed 42. No model trained.
- `cifar100/cifar100_coarse_audit.ipynb`, `cifar100/audit_report.md`, `cifar100/audit_summary.json`, and per-class count CSVs. Fit/predict coarse IDs only; fine labels used for split stratification and result breakdown, never fitting or selection of examples by model scores. `cifar100/subtype_eval.py` supports paired coarse prediction audits.
- Frozen CLIP ViT-B/32, CPU float32: same-prompt mean vs. mean plus clusters with cosine strictly below 0.75. 0/27 clusters retained (lowest cosine 0.84085, flowers); both scored 2,925/5,000 (58.50%), with identical predictions. No test scoring. Newly authored coarse prompts; fine IDs not used to build the bank. Verified summary, candidate audit, paired predictions, and per-subtype counts under `cifar100/frozen_hybrid_075/`.
- Higher selection cutoff replay (0.90; clustering stays at 0.75): 7/27 retained across 6 classes. Same-prompt mean 2,925/5,000 (58.50%); hybrid 2,918/5,000 (58.36%, −0.14 pp). 257 predictions changed: 55 fixes and 62 regressions. Paired subtype-stratified bootstrap (10,000 samples, seed 42) 95% CI −0.54 to +0.26 pp; exact McNemar p=0.579. Results and per-subtype breakdown in `cifar100/frozen_hybrid_gate_0.90/`, replay in `cifar100/replay_variant_gate.py`. Validation only; unchanged prompts, image features and pooling.
