"""Render completed Notebook 03 exports; never train models or impute missing arms."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

STAGES = ['cde', 'alignment', 'prompted_clip']
STAGE_NAMES = ['Frozen classifier', 'DINO alignment', 'Prompted CLIP']
GROUPS = {
    'encoders': [('clip_dino', 'siglip2_dino'), ('clip_dinov3', 'siglip2_dinov3'),
                 ('clip_dino', 'clip_dinov3'), ('siglip2_dino', 'siglip2_dinov3')],
    'anchors': [('dino_only', 'dino_clip'), ('clip_expansion', 'dino_retrieval')],
}


def paired(before, after):
    identity = ['run_id', 'split_id', 'image_id']
    before = before.set_index(identity, verify_integrity=True).sort_index()
    after = after.set_index(identity, verify_integrity=True).sort_index()
    if before.empty or not before['truth'].equals(after['truth']):
        raise ValueError('Pair does not use the same nonempty split and labels')
    good0 = before.prediction.eq(before.truth)
    good1 = after.prediction.eq(after.truth)
    fixes, regressions = int((~good0 & good1).sum()), int((good0 & ~good1).sum())
    delta = 100 * (fixes - regressions) / len(before)
    examples = []
    for kind, mask in [('correction', ~good0 & good1), ('regression', good0 & ~good1)]:
        for row in mask[mask].index[:3]:
            examples.append(dict(kind=kind, image_id=row[-1], truth=before.at[row, 'truth'],
                                 before=before.at[row, 'prediction'], after=after.at[row, 'prediction']))
    return dict(n=len(before), control_pct=100 * good0.mean(), variant_pct=100 * good1.mean(),
                delta_pp=delta, corrections=fixes, regressions=regressions), examples


def render(run):
    out = run / 'presentation'
    out.mkdir(exist_ok=True)
    panels, comparisons, examples, stages, anchor_counts, curves = [], [], [], [], [], []
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

    def record(group, arm0, arm1, stage, before, after):
        fields = dict(group=group, control=arm0, variant=arm1, stage=stage)
        stats, samples = paired(before, after)
        comparisons.append(dict(**fields, **stats))
        examples.extend(dict(**fields, **sample) for sample in samples)

    def graph_deltas(group, title):
        table = pd.DataFrame([r for r in comparisons if r['group'] == group])
        if table.empty:
            return
        table['pair'] = table.control + ' → ' + table.variant
        values = table.pivot(index='pair', columns='stage', values='delta_pp')
        values = values.reindex(columns=[s for s in STAGES if s in values])
        fig, ax = plt.subplots(figsize=(11, max(3.5, len(values) * .45 + 1.7)))
        values.plot.barh(ax=ax, color=['#168aad', '#f4a261', '#4257a1'][:len(values)], rot=0)
        ax.axvline(0, color='#222', linewidth=1)
        ax.set(xlabel='Validation top-1 change (percentage points)', ylabel='', title=title)
        ax.legend(frameon=False, title='Stage')
        panels.append((group + '_ablation', fig))

    if (run / 'cde_contrasts.csv').is_file():
        pred = pd.read_csv(run / 'cde_predictions.csv')
        for pair in pd.read_csv(run / 'cde_contrasts.csv').itertuples():
            record('cde', pair.control, pair.arm, 'cde', pred[pred.arm.eq(pair.control)], pred[pred.arm.eq(pair.arm)])
        graph_deltas('cde', 'Text-bank ablations | frozen CLIP only')

    for group, pairs in GROUPS.items():
        for arm0, arm1 in pairs:
            files = [run / group / arm / 'predictions.csv' for arm in (arm0, arm1)]
            if not all(path.is_file() and (path.parent / 'metrics.csv').is_file() for path in files):
                continue
            before, after = (pd.read_csv(path) for path in files)
            for stage in STAGES:
                record(group, arm0, arm1, stage, before[before.stage.eq(stage)], after[after.stage.eq(stage)])
        graph_deltas(group, f'{group.title()} | matched ablations')

    for path in sorted(run.glob('**/metrics.csv')):
        folder = path.parent
        if not (folder / 'anchors_by_class.csv').is_file() or not (folder / 'learning_curve.csv').is_file():
            continue
        stages.append(pd.read_csv(path))
        anchor_counts.append(pd.read_csv(folder / 'anchors_by_class.csv'))
        curves.append(pd.read_csv(folder / 'learning_curve.csv'))

    if stages:
        accuracy = pd.concat(stages, ignore_index=True)
        counts = pd.concat(anchor_counts, ignore_index=True)
        learning = pd.concat(curves, ignore_index=True)
        accuracy.to_csv(out / 'stage_accuracy.csv', index=False)
        counts.to_csv(out / 'anchors_by_class.csv', index=False)
        learning.to_csv(out / 'learning_curves.csv', index=False)
        purity = counts.groupby('arm')[['selected', 'correct']].sum()
        purity['purity_pct'] = 100 * purity.correct / purity.selected
        purity.to_csv(out / 'anchor_purity.csv')
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        for arm, group in accuracy.groupby('arm'):
            a = group.set_index('stage').reindex(STAGES)
            axes[0, 0].plot(STAGE_NAMES, a.top1_pct, marker='o', label=arm)
        for arm, group in learning.groupby('arm'):
            axes[0, 1].plot(group.epoch, 100 * group.val_correct / group.val_n, label=arm)
            axes[1, 1].plot(group.epoch, group.train_loss, label=arm)
        purity.purity_pct.plot.barh(ax=axes[1, 0], color='#168aad')
        axes[0, 0].set(title='Accuracy by stage', ylabel='Validation top-1 (%)')
        axes[0, 1].set(title='Prompt learning curve', xlabel='Epoch', ylabel='Validation top-1 (%)')
        axes[1, 0].set(title='Audit of training anchors', xlabel='Pseudo-label purity (%)', ylabel='')
        axes[1, 1].set(title='Visual-prompt training loss', xlabel='Epoch', ylabel='Loss')
        axes[0, 0].legend(frameon=False, fontsize=8)
        panels.append(('pipeline_summary', fig))

    if not panels:
        raise SystemExit('No completed experiment exports to plot')
    pd.DataFrame(comparisons).to_csv(out / 'paired_ablations.csv', index=False)
    pd.DataFrame(examples).to_csv(out / 'example_tiles.csv', index=False)
    with PdfPages(out / 'NoLA-ablation-figures.pdf') as pdf:
        for name, fig in panels:
            fig.text(.98, .015, 'EuroSAT development validation · single seed · completed arms only', ha='right', fontsize=9)
            fig.tight_layout(rect=(0, .035, 1, 1))
            fig.savefig(out / f'{name}.png', dpi=220, bbox_inches='tight')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f'{out}: {len(comparisons)} matched contrasts, {len(panels)} figures')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    render(parser.parse_args().run)
