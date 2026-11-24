# CIFAR-100 coarse: frozen CLIP ViT-B/32, same-prompt mean vs .75 hybrid

Validation only: all 5,000 audited rows; no fit or official test inference. Coarse appearance descriptions were hand-written at category level (no fine-name prompt expansion or validation-based tuning); fine labels are used only for split audit, diagnostics and composition-preserving bootstrap. Both arms use the exact same 80 descriptions and 480 photo-wrapped prompts. Normalize each wrapper embedding, average wrappers per description and normalize, then compute the normalized class mean. Within each class apply complete linkage at cosine ≥ .75 and take raw cluster means. Keep only clusters with cosine to the class mean strictly below .75. The hybrid combines the normalized class mean (weight 4) with retained raw cluster means (weight equal to cluster size) by normalized weighted log-mean-exp at τ=.05. If a class has no retained clusters, its hybrid score equals its mean score. This is a single fixed prompt-bank comparison, not a causal test of all possible variants.

Results: `summary.json` (only created after scoring the complete split), `predictions.csv`, `by_coarse.csv`, `by_fine_subtype.csv`, `candidate_audit.csv`, `prompt_bank.json`, `text_bank.npz`, `validation_features.npz`, `validation_scores.npz`. Independent read-back verification writes `SHA256SUMS` and logs to `verification.log`. `feature_chunks/` holds validated atomic 100-row CPU checkpoints. `extraction.log` includes the first failed startup: the local release fork calls `.cuda()` unconditionally in its eager loader, and CUDA was hidden. The successful path builds the eager model from checkpoint tensors loaded on CPU; no upstream file was modified.

## Reproduce or resume

From `<project-root>/parallel-study/cifar100`, after any current workers have exited, run:

```bash
set -e -o pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../../upstream:../../.deps OPENBLAS_NUM_THREADS=1
PY=~/Desktop/Robotics/research-lab/.venv/bin/python
MKL_NUM_THREADS=3 OMP_NUM_THREADS=3 nice -n 10 taskset -c 3-5 "$PY" -u extract_frozen_clip.py 2>&1 | tee -a frozen_hybrid_075/extraction.log
for mode in bank smoke evaluate; do
  MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 nice -n 15 taskset -c 4-5 "$PY" -u score_frozen_clip.py "$mode" 2>&1 | tee -a frozen_hybrid_075/scoring.log
done
nice -n 15 taskset -c 5 python3 -u verify_frozen_clip.py 2>&1 | tee -a frozen_hybrid_075/verification.log
(cd frozen_hybrid_075 && sha256sum -c SHA256SUMS)
```

`extract_frozen_clip.py` checks the exact pinned model, CIFAR archive, split membership, and per-image archive/manifest label alignment. Resume checks contiguous checkpoint identity, row order, shape, dtype, finiteness, and unit norms. Do not change the extractor when reusing chunks: its source digest is part of the checkpoint identity.
