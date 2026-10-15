"""Plot E5 fixed-order and E6 v3.1 epoch metrics from saved W&B histories."""
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
    figures = root / 'outputs/handoff/figures'
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    for number, prefix, title in [(5, 'fixed/', 'E5: fixed-order residual MLP'),
                                   (6, '', 'E6: v3.1 option orders')]:
        history = json.loads((evidence / f'e{number}-wandb-history.json').read_text())
        points = []
        for row in history['rows']:
            epoch = row.get(prefix + 'epoch')
            if epoch is not None:
                points.append({'epoch': epoch, 'train_loss': row[prefix + 'train_nll'],
                               'validation_accuracy': row[prefix + 'dev_accuracy'],
                               'wandb_step': row['_step']})
        points.sort(key=lambda p: p['epoch'])
        if [p['epoch'] for p in points] != [1, 2, 3, 4, 5]:
            raise ValueError(f'E{number}: require exactly five complete epoch records')
        selected = max(points, key=lambda p: (p['validation_accuracy'], -p['epoch']))['epoch']
        receipt = {'run': history['run'], 'experiment': number,
                   'scope': 'completed fixed-order branch only' if number == 5 else 'v3.1 cached-permutation run only',
                   'x_axis': 'Actual training epoch; W&B logging steps are not epochs',
                   'train_loss': 'Logged epoch training cross entropy (train_nll); no smoothing',
                   'validation_metric': 'Accuracy, not validation NLL',
                   'accepted_epoch': selected, 'points': points}
        (evidence / f'e{number}-training-curve.json').write_text(json.dumps(receipt, indent=2) + '\n')
        epochs = [p['epoch'] for p in points]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
        axes[0].plot(epochs, [p['train_loss'] for p in points], 'o-', color='#477baf')
        axes[0].set(title='Training loss', ylabel='Cross-entropy loss', ylim=(0, .6))
        axes[1].plot(epochs, [100 * p['validation_accuracy'] for p in points], 'o-', color='#a34439')
        axes[1].set(title='Validation accuracy', ylabel='Accuracy (%)', ylim=(78, 88))
        axes[1].axvline(selected, linestyle=':', color='#485461', label=f'Accepted epoch {selected}')
        axes[1].legend(loc='lower right', frameon=False, fontsize=9)
        for ax in axes:
            ax.set(xlabel='Training epoch', xticks=epochs, xlim=(.8, 5.2))
            ax.grid(alpha=.15)
        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
        for extension in ['png', 'svg']:
            fig.savefig(figures / f'e{number}-wandb-loss.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f'E{number}: plotted {len(points)} epochs; accepted epoch {selected}')


if __name__ == '__main__':
    build()
