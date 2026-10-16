"""Plot the completed ten-epoch E8 run from its exported epoch records."""
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build(root=ROOT):
    folder = root / 'outputs/handoff/experiment-8'
    rows = json.loads((folder / 'learning-curves.json').read_text())
    best = json.loads((folder / 'best.json').read_text())['epoch']
    if [row['epoch'] for row in rows] != list(range(1, 11)):
        raise ValueError('Require exactly ten recorded epochs')
    if best != max(rows, key=lambda row: (row['dev']['accuracy'], -row['epoch']))['epoch']:
        raise ValueError('Accepted epoch disagrees with validation selection')
    points = [{'epoch': row['epoch'], 'train_loss': row['train_nll'],
               'validation_accuracy': row['dev']['accuracy']} for row in rows]
    (folder / 'plotted-points.json').write_text(json.dumps({
        'source': 'Exported learning-curves.json; same epoch metrics synced to W&B',
        'scope': 'Ten-epoch run only', 'accepted_epoch': best,
        'loss': 'Online epoch training cross entropy, not full-train checkpoint NLL',
        'validation': 'Accuracy, not validation NLL', 'points': points}, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    epochs = [row['epoch'] for row in points]
    axes[0].plot(epochs, [row['train_loss'] for row in points], 'o-', color='#477baf')
    axes[0].set(title='Training loss', ylabel='Cross-entropy loss', ylim=(0, .6))
    axes[1].plot(epochs, [100 * row['validation_accuracy'] for row in points], 'o-', color='#a34439')
    axes[1].set(title='Validation accuracy', ylabel='Accuracy (%)', ylim=(78, 88))
    axes[1].axvline(best, linestyle=':', color='#485461', label=f'Accepted epoch {best}')
    axes[1].legend(loc='lower right', frameon=False, fontsize=9)
    for ax in axes:
        ax.set(xlabel='Training epoch', xticks=epochs, xlim=(.7, 10.3))
        ax.grid(alpha=.15)
    fig.suptitle('E8: zero-initialized residual output, ten epochs', fontsize=13)
    fig.tight_layout()
    for extension in ['png', 'svg']:
        fig.savefig(root / f'outputs/handoff/figures/e8-wandb-loss.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'E8: plotted ten epochs; accepted epoch {best}')


if __name__ == '__main__':
    build()
