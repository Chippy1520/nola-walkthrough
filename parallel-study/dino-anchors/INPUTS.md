# DINO-anchor Colab inputs

Keep `anchor_inputs.py` and the complete `pinned/` tree together (or unzip `NoLA-DINO-Anchor-Inputs.zip`). `pinned/SHA256SUMS` verifies the packaged inputs. Python 3.11+, NumPy, Pillow, PyTorch, torchvision and curl are required; Colab already supplies these. No tokenizer, dependency install or browser is required by the adapter. Dataset archives and model weights are downloaded separately and verified on first use.

```python
from anchor_inputs import get_dataset, prepare

inputs = get_dataset('eurosat')
image, row_id, coarse_or_class_label = inputs.train[0]  # PIL RGB
prepared = prepare('eurosat')
# evaluate_dataset(name='eurosat', output=output_path, **prepared)
```

`prepare` returns NumPy float32 train cosine scores and raw DINO CLS features for train and validation, plus stable IDs, audit labels, class order and metadata. It uses one deterministic weak view with CLIP RGB normalization for both encoders, matching the prior matched adapters. CLIP ViT-B/32 and DINO ViT-B/16 are frozen; CLIP confidence scale comes from the pinned checkpoint. CIFAR fine names are carried only as aligned audit metadata. Test is inaccessible through the dataset API and no test image is decoded.

| Dataset key | Train / val / excluded test | Pinned text classifier |
| --- | ---: | --- |
| `eurosat` | 13,500 / 5,400 / 8,100 | Released broad CDE from `NoLA-EuroSAT-Author-Inputs.zip` |
| `resisc45` | 24,930 / 6,255 / 315 | Original CDE from the frozen RESISC45 report; adapter-generated seed-7777 split with seed-7778 validation carve, **not** an author manifest |
| `cifar100` | 45,000 / 5,000 / 10,000 | Seed-42 audit manifest; same-prompt mean from the frozen coarse study, without variants |

Defaults keep downloads and resumable, row-ID-checked feature chunks under `/content/nola-dino-anchors/`. Override `work_dir` or `cache_dir` for persistence. For offline use, pass `archive=...` for the pinned archive, `weights_dir=...` containing both named checkpoints, or `image_dir=...` for a canonical extracted EuroSAT/RESISC45 JPEG tree. The last option verifies membership, not archive provenance; the feature cache is then bound to train/validation bytes. For a CPU smoke test use `device='cpu', workers=0, train_limit=2, val_limit=2, batch_size=2`; limits create a separate cache and do not substitute for a full evaluation.
