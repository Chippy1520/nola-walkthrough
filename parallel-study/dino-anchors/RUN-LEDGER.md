# DINO anchor selection study

Requested: run all four matched arms (CLIP confidence, DINO farthest-first, DINO medoids, DINO neighborhood agreement) on full EuroSAT, RESISC45 and CIFAR-100 coarse training pools in Colab; save statistics and close each finished notebook afterward.

Protocol: one mean per class; keep CLIP pseudo-labels fixed; use DINO only for selection in the three intervention arms and for the teacher in every arm. Same candidate pool, same per-class realized anchor budget, dedup and matched teacher hyperparameters. Validation only. Ground truth used after selection for audit; CIFAR fine labels for breakdown only. Preserve existing splits and report RESISC45 adapter partition caveat. The final CLIP student is not part of this selector-only trial.

All three datasets completed on a Colab Tesla T4. The CIFAR publisher download timed out; the CIFAR-only restart used the byte-identical TensorFlow/Keras mirror. All three report archives were downloaded, checked against the remote SHA-256 values, and independently verified against their paired prediction and anchor audit rows. The completed experiment scratchpad was closed after local verification. See `reports/combined.csv`, the three report ZIPs, and `../REPORT-LEDGER.md`. Reproducible notebook: `NoLA-DINO-Anchors-all.ipynb`.
