"""Plot Experiment 7's saved, unsampled W&B training history."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build(root=ROOT):
    evidence = root / 'outputs/handoff/report-evidence'
    history = json.loads((evidence / 'e7-wandb-history.json').read_text())
    train = [{'step': row['optimizer_step'], 'loss': row['train/loss']}
             for row in history['rows'] if row.get('train/loss') is not None]
    validation = [{'step': row['optimizer_step'], 'accuracy': row['development/accuracy']}
                  for row in history['rows'] if row.get('development/accuracy') is not None]
    train.sort(key=lambda row: row['step'])
    validation.sort(key=lambda row: row['step'])
    if [row['step'] for row in train] != list(range(1, 81)):
        raise ValueError('Require all 80 optimizer updates')
    if [row['step'] for row in validation] != [0, 50, 80]:
        raise ValueError('Expected the three original validation measurements')
    result = json.loads((root / 'outputs/handoff/experiment-7/pplx-9b-result.json').read_text())
    accepted = result['training_summary']['best_step']
    receipt = {'run': history['run'], 'experiment': 7, 'x_axis': 'Optimizer update',
               'training': train, 'validation': validation, 'accepted_step': accepted,
               'validation_step_zero': 'Pre-training diagnostic; not a selected checkpoint',
               'method': 'Raw logged training loss and validation accuracy; no smoothing or imputed evaluations'}
    (evidence / 'e7-training-curve.json').write_text(json.dumps(receipt, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    axes[0].plot([row['step'] for row in train], [row['loss'] for row in train], color='#477baf', lw=1.4)
    axes[0].set(title='Training loss', ylabel='Training objective loss', ylim=(0, .7))
    axes[1].plot([row['step'] for row in validation],
                 [100 * row['accuracy'] for row in validation], 'o-', color='#a34439')
    axes[1].set(title='Validation accuracy', ylabel='Accuracy (%)', ylim=(75, 90))
    axes[1].axvline(accepted, linestyle=':', color='#485461', label=f'Accepted update {accepted}')
    axes[1].legend(loc='lower right', frameon=False, fontsize=9)
    for ax in axes:
        ax.set(xlabel='Optimizer update', xlim=(0, 82), xticks=[0, 20, 40, 60, 80])
        ax.grid(alpha=.15)
    fig.suptitle('E7: PPLX recipe replication attempt', fontsize=13)
    fig.tight_layout()
    for extension in ['png', 'svg']:
        fig.savefig(root / f'outputs/handoff/figures/e7-wandb-loss.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Plotted {len(train)} training updates and {len(validation)} validation measurements; accepted update {accepted}')


if __name__ == '__main__':
    build()
