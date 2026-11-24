# NoLA experiments

Each current experiment has its own Python entry point and results folder. No aggregated notebook is needed. Historical experiments are archived separately, with their original code and exports.

## Run a current experiment

With Python 3.12 and CUDA-enabled PyTorch/torchvision, install `gdown ftfy wget timm transformers sentencepiece matplotlib pandas` in your environment. Run `python3 scripts/fetch_sources.py` from the repository root to fetch the pinned code. Then, for example:

```bash
python3 experiments/cde-unmerged/run.py --datasets eurosat
python3 experiments/anchors-dino-only/run.py --datasets eurosat --micro 4
```

With no dataset argument, each entry processes EuroSAT, Oxford Pets, CIFAR-10 and DTD. Lower microbatches leave effective batch size at 512. Public inputs download separately. Runtime files go to the ignored `_shared/runs/` directory by default; use `--output` to change the output root. Code paths overlap where needed for matched anchor generation, but only the selected arm is evaluated/trained.

CDE results are frozen validation only. Downstream results distinguish training anchor precision/coverage, validation DINO alignment, and validation visual-prompt CLIP. No historical pilot is used as a full-run result or as a CIFAR-10 result.

## Current grid

| Experiment | Change |
|---|---|
| [cde-release](cde-release/) | Original released description and template mean. |
| [cde-descriptions](cde-descriptions/) | Mean of released descriptions without separate templates. |
| [cde-template-add](cde-template-add/) | Add the template centroid to the description centroid and normalize. |
| [cde-suffix](cde-suffix/) | Append each class template after each description before encoding. |
| [cde-template-suffix](cde-template-suffix/) | Place each class template before each description before encoding. |
| [cde-unmerged](cde-unmerged/) | Keep each description vector and score with the shared prototype scorer. |
| [cde-suffix-mean](cde-suffix-mean/) | Average template encodings per description, then form one class centroid. |
| [cde-merge075](cde-merge075/) | Merge description prototypes by complete-link cosine at 0.75. |
| [cde-merge080](cde-merge080/) | Merge description prototypes by complete-link cosine at 0.80. |
| [cde-blend075](cde-blend075/) | Blend distinct normalized directions into the suffix mean with the 0.75 gate. |
| [cde-blend090](cde-blend090/) | Blend distinct normalized directions into the suffix mean with the 0.90 gate. |
| [anchors-dino-only](anchors-dino-only/) | Rank CLIP nominees using agreement among frozen DINO neighbors. |
| [anchors-dino-clip](anchors-dino-clip/) | Reweight the same DINO graph with CLIP posterior overlap. |
| [anchors-clip-expansion](anchors-clip-expansion/) | Expand the same seed pool using CLIP confidence; matched control for DINO retrieval. |
| [anchors-dino-retrieval](anchors-dino-retrieval/) | Expand the same seed pool using CLIP confidence and DINO neighbor similarity. |
| [encoders-clip-dino](encoders-clip-dino/) | CLIP nomination with original DINO alignment features; matched control. |
| [encoders-clip-dinov3](encoders-clip-dinov3/) | CLIP nomination with DINOv3 ConvNeXt-Tiny features. |
| [encoders-siglip2-dino](encoders-siglip2-dino/) | Independent SigLIP 2 nomination with original DINO alignment features. |
| [encoders-siglip2-dinov3](encoders-siglip2-dinov3/) | Independent SigLIP 2 nomination with DINOv3 ConvNeXt-Tiny features. |

## Historical experiments

| Experiment | Scope |
|---|---|
| [notebook02-baseline](notebook02-baseline/) | Native four-dataset baseline; existing metrics, reuse without retraining. |
| [eurosat-scale-sweep](eurosat-scale-sweep/) | EuroSAT small/full scale and optimizer sweeps; partial grid remains partial. |
| [eurosat-diagnostics](eurosat-diagnostics/) | Instrumented EuroSAT class/anchor diagnostics. |
| [legacy-cde-prototypes](legacy-cde-prototypes/) | Early EuroSAT CDE clustering and prompt-bank diagnostics. |
| [legacy-suffix-blend](legacy-suffix-blend/) | Frozen suffix/mean/variation study on EuroSAT, CIFAR-100 coarse and RESISC45. |
| [legacy-suffix-training](legacy-suffix-training/) | EuroSAT suffix-only transfer through training; full split is an interim trajectory. |
| [legacy-eurosat-hybrid](legacy-eurosat-hybrid/) | Template-aware merge at 0.75; frozen and interim training comparisons. |
| [legacy-resisc45-hybrid](legacy-resisc45-hybrid/) | Template-aware 0.75 merging on RESISC45; frozen validation results. |
| [legacy-pets-suffix](legacy-pets-suffix/) | Oxford Pets suffix formatting; separate original comparison, not the v2 validation split. |
| [legacy-cifar100-frozen](legacy-cifar100-frozen/) | CIFAR-100 coarse frozen prototype and variation-gate studies. |
| [legacy-dino-anchor-ranking](legacy-dino-anchor-ranking/) | Confidence, diversity and visual-neighbor alignment-head experiments. |
| [legacy-cluster-rescue](legacy-cluster-rescue/) | Cluster-first anchor rescue versus expansion. |
| [legacy-clustering-sweep](legacy-clustering-sweep/) | Matched rescue with direct retrieval and clustering variants. |
| [legacy-spectral-rescue](legacy-spectral-rescue/) | Frozen DINO-neighbor graph with and without CLIP edge reweighting; EuroSAT full and CIFAR-100 pilot. |
| [legacy-text-prototype-mapping](legacy-text-prototype-mapping/) | DINO visual-cluster assignment and text prototypes; alignment head only. |
| [legacy-convnext-retrieval](legacy-convnext-retrieval/) | Replace retrieval features while keeping CLIP nominations. |
| [legacy-siglip-reranking](legacy-siglip-reranking/) | SigLIP 2 reranking inside a CLIP candidate pool crossed with two retrieval encoders. |
| [legacy-independent-siglip](legacy-independent-siglip/) | Independent SigLIP 2 selection versus CLIP, original matched budgets. |
| [legacy-encoder-factorial](legacy-encoder-factorial/) | Independent nomination × retrieval factorial. EuroSAT full validation; CIFAR-100 pilot. Alignment head only. |
| [legacy-prompt-classifier-pilot](legacy-prompt-classifier-pilot/) | Fine-tuned CLIP pilot through 12 epochs; EuroSAT 200 and CIFAR-100 coarse 400 validation images, not full split. |
