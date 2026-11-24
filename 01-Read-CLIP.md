# 01 — Read and run CLIP from upstream NoLA

Upstream source: `upstream/` at `caec2722bba84bd4807256645c5cb75d96d7ed12`. The only lesson code is `01_clip.py`; upstream is unmodified.

```bash
cd ~/nola-walkthrough
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/upstream:$PWD/.deps" \
  "$HOME/Desktop/Robotics/research-lab/.venv/bin/python" \
  01_clip.py dog.jpg cuda weights
```

`dog.jpg` is the [PyTorch Hub example](https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg). The checkpoint is the original NoLA CLIP loader's ViT-B/32 download, checked against the embedded SHA-256. No dataset is needed for this first exercise.

Read in this order:

1. `01_clip.py`: give CLIP one photo and three text prompts.
2. `upstream/clip/clip.py:108`: why the image is cropped and normalized; `:123`: how the weights are loaded. Here the fork returns `(model, embedding_width, preprocess)` and its default non-JIT branch calls `.cuda()` even if asked for CPU.
3. `upstream/clip/model.py:372`: `encode_image`, `encode_text`, L2 normalization, then scaled cosine similarity.
4. `upstream/trainers/NoLA.py:149`: NoLA replaces one prompt per class with a normalized average over the authors' descriptions and dataset templates. Same image-to-text scoring idea, richer class vectors. This function is read-only study here; it has not been run on a benchmark yet.

Observed on GTX 1050 Ti: image `(1,3,224,224)`, text tokens `(3,77)`, both encoders project to width 512. Logits in dog/wolf/cat order: `28.0`, `24.1875`, `22.4375`; softmax over those prompts: `97.46%`, `2.15%`, `0.37%`. Peak PyTorch tensor allocation `355 MiB` (not total system GPU use). Scores depend on the choices offered. Try replacing `wolf` with `golden retriever` and predict how CLIP's ranking or probabilities might change before running it. This is one inference example, not benchmark accuracy.

## What comes after this lesson

- **Text classifier on EuroSAT:** trace `gen_emb` into the shipped description JSON and dataset templates; compare plain CLIP and CDE on the same verified split and class order.
- **Original selector and teacher:** `select_top_k` → `utils/model_utils.py:top_k_indices_per_class2` → `Taal`/`AdapterMLP` → `train_taal` and `test_taal`. The official selector falls back to second/third predictions for an empty class; don't confuse it with the project's frozen comparator.
- **Student:** `incorporate_prompt` → `forward_backward` → `setup_optim` → `after_epoch`; see which parameters update, which image view each model sees, and how the checkpoint is chosen. Check loss, teacher vs student accuracy, and per-class mistakes.
- **Reproduction:** start with one small pilot (memory/timing only), then an official split and settings; report any differences explicitly. Evaluating an author checkpoint is not retraining. The paper reports EuroSAT top-1 of 40.6% for CLIP and 73.5% for NoLA in Table 1; these are references, not results from this walkthrough.

No DNC changes, frozen-study changes, or Kaggle jobs are part of this walkthrough.
