# NoLA EuroSAT case study

Read `NoLA-EuroSAT-Case-Study.pdf` first. The editable LaTeX source references the PDFs in `figures/`; compile it from this directory with `pdflatex` if available. The delivered PDF was rendered with Matplotlib, not LaTeX.

- `provenance.json`: selected original GPU log entry, replay configuration, identities and hashes.
- `metrics.json`, `history.json`, `predictions.csv`: correct counts, per-class counts, confusion matrices, epoch losses and per-image class scores. Probability columns follow `metrics.json` class order. No per-image adaptation scores were collected for prompted CLIP, only for CDE and DINO.
- `cde-text.json`, `cde.pt`: original saved descriptions, filled templates and computed matrix.
- `top-k-candidates.csv`, `anchors.csv`, `anchors/`: pre-deduplication candidates, the retained anchors and original JPEGs. `prediction_rank` on fallback means rank among CLIP's class guesses; `within_class_rank` is image rank within the chosen fallback tier.
- `augmentations.json`, `augmentations/`: original, weak and actual strong training views plus labels.
- `capture.py` and `build_report.py`: run with the Robotics Research Lab Python environment. Capture reads the original sweep notebook code and data under the parent directory; it writes a CPU replay. The checkpoint files are compact evaluation weights, not full optimizer snapshots.

Accuracy conclusions are descriptive for a single configuration and seed, not paper reproduction or deployment claims.
