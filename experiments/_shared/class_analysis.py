"""Offline readout of Notebook 03 validation exports. Usage: python class_analysis.py RUN_DIR."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

IDS = ['dataset', 'split', 'image_id']
GROUP = ['dataset', 'family', 'arm', 'stage']


def analyze(root):
    output = root / 'class_analysis'
    output.mkdir(exist_ok=True)
    scores, confusions, geometry, associations = [], [], [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and p != output):
        geo = {p.name.removesuffix('_image_geometry.csv'): pd.read_csv(p)
               for p in folder.glob('*_image_geometry.csv')}
        paths = [('cde', folder / 'cde_predictions.csv')]
        paths += [(p.parent.parent.name, p) for p in sorted(folder.glob('*/*/predictions.csv'))
                  if (p.parent / 'complete.json').is_file()]
        for family, path in paths:
            if not path.is_file() or (family == 'cde' and not (folder / 'CDE-complete.json').is_file()):
                continue
            pred = pd.read_csv(path).query("split == 'val'").assign(family=family)
            pred['correct'] = pred.prediction.eq(pred.truth)
            if pred.duplicated([*GROUP, 'image_id']).any():
                raise ValueError(f'Duplicate predictions: {path}')
            scores.append(pred.groupby([*GROUP, 'truth']).correct.agg(n='size', correct='sum').reset_index())
            c = pred.groupby([*GROUP, 'truth', 'prediction']).size().rename('count').reset_index()
            confusions.append(c)
            for space, frame in geo.items():
                merged = pred.merge(frame, on=[*IDS, 'split_id'], validate='many_to_one')
                if len(merged) != len(pred) or not merged.truth.eq(merged.class_id).all():
                    raise ValueError(f'Geometry and prediction rows disagree: {path}, {space}')
                geometry.append(merged.groupby([*GROUP, 'space', 'class_id', 'class_name']).agg(
                    n=('correct', 'size'), accuracy=('correct', 'mean'),
                    within_distance=('within_distance', 'mean'), rival_similarity=('rival_similarity', 'mean'),
                    margin=('margin', 'mean')).reset_index())
                # Correct-vs-wrong geometry is descriptive; class-specific rows avoid pooling class difficulty.
                associations.append(merged.groupby([*GROUP, 'space', 'class_id', 'correct']).agg(
                    n=('image_id', 'size'), within_distance=('within_distance', 'mean'),
                    rival_similarity=('rival_similarity', 'mean'), margin=('margin', 'mean')).reset_index())
    if not scores:
        raise SystemExit('No completed validation exports found.')
    acc = pd.concat(scores, ignore_index=True)
    acc['accuracy_pct'] = 100 * acc.correct / acc.n
    acc.to_csv(output / 'per_class_accuracy.csv', index=False)
    acc.groupby(GROUP).accuracy_pct.mean().rename('macro_accuracy_pct').to_csv(output / 'macro_accuracy.csv')
    counts = pd.concat(confusions, ignore_index=True)
    counts['row_pct'] = 100 * counts['count'] / counts.groupby([*GROUP,'truth'])['count'].transform('sum')
    counts.to_csv(output / 'confusions.csv', index=False)
    if not geometry:
        return
    geo = pd.concat(geometry, ignore_index=True)
    geo.to_csv(output / 'accuracy_vs_geometry.csv', index=False)
    pd.concat(associations, ignore_index=True).to_csv(output / 'correctness_vs_geometry.csv', index=False)
    links = []
    with PdfPages(output / 'class_geometry.pdf') as pdf:
        for (dataset, space), points in geo.groupby(['dataset', 'space']):
            ref = points.query("family == 'cde' and arm == 'release'")
            similarity = pd.read_csv(root / dataset / f'{space}_class_similarity.csv')
            mat = similarity.pivot(index='class_name', columns='other_name', values='centroid_similarity')
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            im = axes[0].imshow(mat, vmin=-1, vmax=1, cmap='coolwarm'); fig.colorbar(im, ax=axes[0], shrink=.7)
            axes[0].set_title('Class centroid cosine similarity')
            if len(mat) <= 12:
                axes[0].set_xticks(range(len(mat)), mat.columns, rotation=90, fontsize=7)
                axes[0].set_yticks(range(len(mat)), mat.index, fontsize=7)
            for ax, col, label in zip(axes[1:], ['within_distance', 'rival_similarity'], ['Within-class distance', 'Nearest rival similarity']):
                ax.scatter(ref[col], 100 * ref.accuracy)
                ax.set(xlabel=label, ylabel='Frozen CLIP per-class accuracy (%)')
            fig.suptitle(f'{dataset} | {space} geometry | validation (descriptive)')
            fig.tight_layout(); pdf.savefig(fig); fig.savefig(output / f'{dataset}_{space}.png', dpi=180); plt.close(fig)
            # Include zero-confusion pairs, not just the observed mistakes.
            sim = similarity.loc[similarity.class_id.ne(similarity.other_id)].rename(columns={'class_id':'truth', 'other_id':'prediction'})
            for key, group in counts[counts.dataset.eq(dataset)].groupby(GROUP):
                pairs = sim.merge(group[['truth','prediction','row_pct']], on=['truth','prediction'], how='left')
                pairs['confusion_pct'] = pairs.row_pct.fillna(0)
                links.append(pairs.assign(**dict(zip(GROUP, key))))
    if links:
        pairs = pd.concat(links, ignore_index=True)
        pairs.to_csv(output / 'pair_similarity_vs_confusion.csv', index=False)
        # Report rank associations separately within each dataset and experiment arm.
        summaries = []
        for keys, frame in geo.groupby([*GROUP, 'space']):
            for column in ('within_distance', 'rival_similarity', 'margin'):
                valid = frame[[column, 'accuracy']].dropna().rank()
                rho = valid[column].corr(valid.accuracy) if len(valid) >= 3 and (valid.nunique() > 1).all() else None
                summaries.append(dict(zip([*GROUP, 'space'], keys), metric=column,
                                      n_classes=len(valid), spearman_rho=rho))
        pd.DataFrame(summaries).to_csv(output / 'geometry_associations.csv', index=False)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    analyze(parser.parse_args().run)
