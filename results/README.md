# Results index

## Completed four-dataset baseline

[`baseline-full-v4/metrics.csv`](baseline-full-v4/metrics.csv) is the aggregate export from the completed Colab run (24 rows, seed 7777). Test top-1 is shown below; the prompted CLIP checkpoint was selected on validation. This run is separate from the earlier local EuroSAT run and is not an exact Table 1 reproduction.

| Dataset | Frozen CDE | DINO alignment head | Visual-prompt classifier | Selected epoch |
|---|---:|---:|---:|---:|
| EuroSAT | 51.60% | 67.16% | 76.25% | 3 |
| Oxford Pets | 85.88% | 89.81% | 88.63% | 16 |
| CIFAR-10 | 89.50% | 92.65% | 95.07% | 12 |
| DTD | 50.53% | 52.19% | 55.61% | 4 |

The same export includes [50-epoch learning curves](baseline-full-v4/learning_curve.csv), [anchor summary](baseline-full-v4/anchor_summary.csv), [anchors by class](baseline-full-v4/anchors_by_class.csv), and [paper reference values](baseline-full-v4/paper_reference.csv) (split comparability not verified).

## Exploratory studies and controls

- [Experiment summary](../EXPERIMENT-SUMMARY.md) brings together experiments, controls and limitations.
- [EuroSAT case study](../eurosat-case-study/README.md) and [failure audit](../eurosat-case-study/weakness-audit/README.md): local baseline, class coverage and alignment-to-prompt failures.
- [Prompt suffix and prototype blend](../cde-suffix-blend/README.md): same-bank and multi-dataset frozen CDE tests.
- [Anchor and retrieval ledger](../parallel-study/REPORT-LEDGER.md), [DINO anchor results](../parallel-study/dino-anchors/RUN-LEDGER.md), and [cluster-first runs](../parallel-study/cluster-first/RUN-LEDGER.md): controlled swaps, coverage, and failure cases.
- [SigLIP replacement](../siglip-replacement/README.md): matched comparisons, not evidence that SigLIP universally replaces CLIP.
- [Archive exports](archive-exports/): readable historical CSV/JSON extracted from locally saved report archives. These can include partial runs and multiple snapshots; scope is documented in each source ledger.
