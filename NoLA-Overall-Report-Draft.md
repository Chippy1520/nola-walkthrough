# NoLA experiment report — draft

## Abstract

We tested three parts of a NoLA-style adaptation pipeline: CLIP class-description embeddings (CDE), pseudo-labeled anchor selection and visual retrieval, and visual-prompt prompt classifier training. The strongest reproducible-within-run effect of a modification was targeted rescue of an underfilled EuroSAT class: direct DINO retrieval raised Herbaceous Vegetation recall from 2/600 without rescue to 98/600, compared with 72/600 for a matched CLIP expansion. Overall gains were smaller and did not consistently transfer. Replacing CLIP with SigLIP 2 for independent anchor selection clearly degraded the EuroSAT DINO alignment head under both retrievers. An early EuroSAT win for the SigLIP 2 + ConvNeXt prompt classifier disappeared at longer fixed training budgets. We have not established that any modification improves on the release-style CLIP baseline in a matched, full-split prompt classifier test, or across datasets.

The dataset-by-dataset scorecard (counts, denominators, and same-image stage differences) is in `NoLA-Dataset-Statistics.md` and in the PDF immediately after the learning curves.

## 1. Protocol and baseline

CDE denotes the frozen text-based classifier; the DINO alignment head is a classifier fitted on frozen original DINO ViT-B/16 features; the visual-prompt classifier is CLIP or SigLIP 2 with learned visual tokens. DINOv3 ConvNeXt-Tiny (`convnext_tiny.dinov3_lvd1689m`) was a frozen retrieval encoder only, never the alignment head or prompt classifier. Most follow-ups use development validation, with one training seed per major comparison. Coarse CIFAR-100 includes separate 5,000-image validation analyses and a 1,024-image simulated-scarcity alignment head pilot; these are not interchangeable. RESISC45 uses an adapter-derived split. The older frozen Kaggle/DNC study was not re-audited and is outside this draft.

The local release-style EuroSAT run achieved 51.09% CDE, 67.09% DINO alignment head, and 76.69% best prompt classifier on the same 5,400 validation images. The validation-selected epoch-3 checkpoint scored 76.27% on 8,100 test images (6,178 correct). The same run finished at 71.13% validation at epoch 50. A small sweep with 160 and 320 adaptation images showed that a larger anchor cap could improve the alignment head while hurting the prompt classifier; only 10 of 128 jobs from the larger hyperparameter grid were saved. These are reasons to record class coverage and checkpoint trajectories, not evidence that longer training or larger anchor budgets always hurt.

## 2. Prompt and prototype hypotheses

A 240-image EuroSAT pilot favored three prototypes with log-mean-exp (127 correct) over one mean (118); max pooling scored 103. The larger 5,160-image analysis showed that clustering the authored bank could underperform both the single mean (53.59%) and the unmerged bank (54.28%): merged variants scored 48.39–49.42%. Broad-plus-variant prompts improved the weak broad-only bank (47.67% to 50.21% at best) but did not beat those controls. For the released prompt bank, one merge reached 52.29% versus 51.10% for released CDE, yet Herbaceous stayed at zero correct. The SVD/cosine audit showed that dimensionality reduction could make prompt pairs appear close that did not pass the full-space test. Photo-wrapped prototypes on EuroSAT briefly improved the prompt classifier by 1.15 pp at epoch 1 and lost 0.30 pp at epoch 2; on RESISC45 the frozen score declined from 58.03% to 56.05%. Conclusion: neither prompt grouping nor a better raw text score guarantees useful anchors or a better prompt classifier.

| Frozen CLIP experiment | N | Baseline → appended | Appending effect, 95% CI (pp) | Extra effect of the 0.90 normalized blend (pp) |
|---|---:|---:|---:|---:|
| Oxford-IIIT Pet, one-template comparison | 3,669 | 85.61% → 87.90% | +2.29; 156 fixes, 72 losses | Not tested |
| EuroSAT validation | 5,400 | 51.06% → 51.48% | +0.43 [−0.11, +0.96] | −0.28 [−0.54, −0.02] |
| RESISC45 validation | 6,255 | 58.07% → 58.86% | +0.80 [+0.11, +1.47] | +0.03 [−0.11, +0.18] |
| CIFAR-100 coarse validation | 5,000 | 58.50% → 50.48% | −8.02 [−9.00, −7.04] | +0.62 [+0.26, +0.98] |

CIFAR changed wording and included explicit class names, so its contrast does not isolate suffix placement. The 0.75 variant cutoff retained nothing in the final appended-bank experiment; its predictions were identical to the mean. At 0.90, one normalized weighted vector performed numerically worse than soft pooling of the same components on every dataset. The earlier 0.90 wrapped-bank CIFAR LME experiment was also inconclusive (−0.14 pp). These results do not support a universal variation recipe. In a 300-image paired EuroSAT prompt classifier test, the suffix changed top-1 by −2.00 pp [−5.00, +1.00]. The separate full-split suffix run improved alignment head test accuracy (66.96% to 68.90%), but the prompt classifier validation ordering reversed between epoch 1 and 2; no verified final paired prompt classifier test was found for that run. CIFAR-10 suffix training has no verified endpoint.

## 3. Anchor and retrieval hypotheses

The baseline sometimes nominated no correct candidates for a class. For example, a small EuroSAT audit found 24 true Herbaceous fit images, zero first-choice CLIP nominations, and one incorrect fallback. More anchors helped only if the additional images were useful. Farthest-first and medoids lowered alignment head accuracy on all three datasets. Neighbor agreement improved EuroSAT by +2.67 pp [1.80, 3.54], was inconclusive on RESISC45 (+0.27 pp [−0.59, 1.12]), and hurt coarse CIFAR-100 (−1.18 pp [−2.20, −0.18]). Forced cluster balance lost roughly 9.50–11.24 pp on EuroSAT; mapping text prototypes onto clusters did not recover the loss.

In the first cluster-first rescue, all 113 additions went to Pasture and none to Herbaceous. Opening the pool, while matching 64 additions per target class, changed that: no-rescue, CLIP expansion, and direct DINO retrieval gave 2/600, 72/600, and 98/600 Herbaceous recall. Direct retrieval raised overall alignment head top-1 from 67.28% to 68.02% versus matched CLIP expansion (+0.74 pp [0.15, 1.35]) and selected 96/128 correct additions versus 79/128. DBSCAN reached 68.28%, but one dominant cluster and thousands of noise points do not establish meaningful subtypes. Ward and spherical k-means did not surpass DBSCAN in this run; spectral clustering and CLIP-reweighted edges did not consistently beat direct retrieval. In the CIFAR scarcity pilot, direct DINO had 38/40 correct additions versus CLIP's 28/40, yet lowered alignment head accuracy, showing that anchor purity alone is an insufficient criterion.

## 4. SigLIP 2 and DINOv3 ConvNeXt-Tiny

The original four-way reranking study retained CLIP-defined seeds and candidates. SigLIP 2 made more chosen additions correct in every matched swap. It raised EuroSAT DINO-retrieval alignment head accuracy from 68.02% to 68.57%, but did not help every pairing on CIFAR. The independent-nomination experiment then gave each model its own pool and matched per-class counts. The CLIP controls reproduced earlier predictions exactly.

| Nominator + retriever | EuroSAT alignment head, n=5,400 | CIFAR-100 scarcity alignment head, n=1,024 | EuroSAT prompt classifier, epoch 12, n=200 | CIFAR-100 prompt classifier, epoch 12, n=400 |
|---|---:|---:|---:|---:|
| CLIP + original DINO | 68.02% | 64.06% | 64.00% | 55.50% |
| SigLIP 2 + original DINO | 58.13% | 63.18% | 60.00% | 56.75% |
| CLIP + DINOv3 ConvNeXt-Tiny | 68.43% | 64.84% | 69.50% | 53.00% |
| SigLIP 2 + DINOv3 ConvNeXt-Tiny | 58.93% | 63.77% | 62.00% | 54.75% |

Against CLIP with the same retriever, SigLIP-only EuroSAT alignment head effects were −9.89 pp [−11.17, −8.61] with DINO and −9.50 pp [−10.81, −8.26] with ConvNeXt. Both CIFAR intervals included zero. SigLIP independently nominated 144 correct Herbaceous examples at top-1 where CLIP nominated none, but the matched selection retained only 2 correct out of 71 versus 38/71 with CLIP and DINO expansion. Under native budgets SigLIP improved Herbaceous alignment head recall from 98/600 to 345/600, but aggregate accuracy fell to 60.98% versus 68.02%. The ability to propose useful examples is therefore distinct from ranking and selecting them under a tight budget.

The prompt classifier experiments changed both the nominator and prompt classifier backbone in the SigLIP arms; they do not isolate which caused the difference. All prompt classifiers saw the same 320 fit images per dataset and used fixed checkpoints at epochs 2, 4, and 12. The early EuroSAT SigLIP 2 + ConvNeXt result (68.50% at epoch 2) did not improve later. CLIP + ConvNeXt reached 78.50% at epoch 11 and ended 5.50 pp [1.00, 10.00] ahead of CLIP + DINO at epoch 12 on the same pilot rows. On CIFAR that same retrieval swap ended −2.50 pp [−4.75, −0.25]. The best observed CIFAR run was CLIP + DINO at epoch 2 (60.50%). Best-observed validation scores are descriptive and cannot be compared directly with the 76.27% full-split EuroSAT test baseline.

## 5. Conclusions and next comparison

Keep released CLIP CDE and the simple confidence selector as the baseline. Test targeted expansion for genuinely underfilled classes, using CLIP-only expansion as the control that distinguishes the added budget from the retrieval method. Compare against direct DINO retrieval and optionally ConvNeXt with the same per-class additions. Use EuroSAT and a second dataset with measured natural scarcity; keep the prompt classifier architecture and checkpoint-selection rule fixed, report multiple seeds, and lock a held-out test before looking at outcomes. Report overall prompt classifier top-1, correctness and coverage of rescued anchors, rare-class recall, and any regression on non-target classes. The current runs support a rescue hypothesis, not a universal new pipeline.

Statistical notes: differences are in percentage points; quoted 95% intervals are paired image-level bootstrap intervals from the report builders or archived studies. The recent frozen comparisons stratify by true class (fine subtype for CIFAR), and retain exact McNemar p-values and fix/loss counts in the PDF or machine-readable results. Repeated development sets, multiple comparisons, training seeds, and checkpoint selection limit generalization; the intervals do not quantify all of those sources of uncertainty. No new experiments were run to write this draft.

Sources and full experiment register: `EXPERIMENT-SUMMARY.md`; `NoLA-Experiment-Report.pdf` (plots and appendix); `report-assets/paired-statistics.json`; `parallel-study/dino-anchors/reports/combined.csv`; `parallel-study/cluster-first/clustering_sweep_eurosat/verification.json`; `cde-suffix-blend/`; `siglip-replacement/{factorial,students,students-epoch4,students-epoch12}/`; and the saved EuroSAT walkthrough notebook outputs.
