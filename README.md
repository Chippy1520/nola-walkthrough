# NoLA walkthrough and experiments

An independent walkthrough, release-style reimplementation, and set of follow-up experiments for **NoLA (No Labels Attached)**. Start with the notebooks to follow the method, then see the reports and result files for what changed and what did not.

## Credit

NoLA was introduced by Mohamed Fazli Mohamed Imam, Rufael Fekadu Marew, Jameel Hassan Abdul Samadh, Mustansar Fiaz, Alham Fikri Aji, and Hisham Cholakkal in *CLIP meets DINO for Tuning Zero-Shot Classifier using Unlabeled Image Collections* (BMVC 2025). See the [official NoLA code](https://github.com/fazliimam/NoLA), [paper and citation](https://bmvc2025.bmva.org/proceedings/281/), and [CITATION.bib](CITATION.bib). The upstream NoLA commit used here is [`caec272`](https://github.com/fazliimam/NoLA/tree/caec2722bba84bd4807256645c5cb75d96d7ed12). This is not the authors' repository or a claim of exact paper reproduction.

Other dependencies and prior work: [OpenAI CLIP](https://github.com/openai/CLIP), [Meta DINO](https://github.com/facebookresearch/dino), [LaFTer](https://github.com/jmiemirza/LaFter), [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch), [CoOp](https://github.com/KaiyangZhou/CoOp), and [SigLIP 2](https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/image_text). Upstream source trees, datasets, weights and checkpoints are not redistributed here; their own terms apply. For the companion article, see [CLIP Meets DINO](https://github.com/Chippy1520/clip-meets-dino-nola). The earlier reliability/DNC study is separate.

## Start here

| Entry point | What it covers |
|---|---|
| [01 · Teaching EuroSAT](final-evaluation/01-NoLA-Teaching-EuroSAT.ipynb) | Trace images and text through CLIP, build class-description embeddings, select anchors, fit the DINO alignment head, and tune visual prompts. |
| [02 · Full evaluator](final-evaluation/02-NoLA-Full-Evaluator.ipynb) | Release-style four-dataset baseline: EuroSAT, Oxford Pets, CIFAR-10 and DTD. |
| [03 · Experiments](final-evaluation/03-NoLA-Experiments.ipynb) | Compare text-bank changes and anchor-ranking modifications using named pipeline stages. |
| [Experiment summary](EXPERIMENT-SUMMARY.md) and [PDF report](NoLA-Experiment-Report.pdf) | Findings, negative results, scope, and provenance. |
| [Results index](results/README.md) | Baseline outputs, exploratory comparisons, and archived CSV/JSON exports. |

## Run the evaluator

1. Clone this repository and run `python3 scripts/fetch_sources.py` from its root. This downloads the pinned NoLA and DINO repositories and creates `final-evaluation/source.zip` locally. It is intentionally ignored by Git.
2. In your Google Drive create `MyDrive/NoLA-Final-Evaluation/`. Upload the generated `source.zip` plus `final-evaluation/setup_inputs.py`; for notebook 03 also upload `nola_eval.py`, `experiments.py`, and `research_run.py` from the same directory.
3. Import notebook 02 into a GPU-backed Colab runtime. Review the Settings and setup cell before execution. Set `SMOKE = True` for a short check using a separate run folder. Full mode uses seed 7777, 50 DINO alignment epochs and 50 visual-prompt epochs, effective batch 512, and microbatch 32. It downloads the public data and weights and writes outputs under `MyDrive/NoLA-Final-Evaluation/runs/`.

The main evaluator is the supported entry point. Older exploratory notebooks and launchers are retained as research records and may need path or environment adjustments. Notebook outputs are cleared in Git; use the checked-in CSV/JSON and report files for measured results. CIFAR-10 reserves 5,000 official training images for validation. Oxford Pets test was previously examined. Single-seed pilots, reused validation sets, different split definitions, and partial runs should not be compared as if they were independent test results.
