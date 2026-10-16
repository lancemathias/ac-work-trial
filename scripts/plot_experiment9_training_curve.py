"""Separate E9 training and validation charts for lambda 0 and coefficients ending in 3."""
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build(root=ROOT):
    folder = root / 'outputs/handoff/experiment-9'
    subset = ['0', '0.003', '0.03', '0.3', '3']
    colors = ['#333333', '#0072B2', '#E69F00', '#009E73', '#CC79A7']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    loss_fig, loss = plt.subplots(figsize=(10, 4.8))
    accuracy_fig, accuracy = plt.subplots(figsize=(10, 4.8))
    points = {}
    for coefficient, color in zip(subset, colors):
        rows = json.loads((folder / f'lambda-{coefficient}-curves.json').read_text())
        if [row['epoch'] for row in rows] != list(range(1, 11)):
            raise ValueError('Require ten original epochs per coefficient')
        epochs = [row['epoch'] for row in rows]
        ce = [row['online_train']['nll'] for row in rows]
        acc = [100 * row['dev']['accuracy'] for row in rows]
        width = 2.4 if coefficient == '0.3' else 1.5
        loss.plot(epochs, ce, color=color, lw=width, label='λ = ' + coefficient)
        accuracy.plot(epochs, acc, 'o-', color=color, lw=width, markersize=4, label='λ = ' + coefficient)
        points[coefficient] = [{'epoch': e, 'train_cross_entropy': x, 'validation_accuracy': y / 100}
                               for e, x, y in zip(epochs, ce, acc)]
    winner = json.loads((folder / 'summary.json').read_text())['winner']
    accuracy.scatter([winner['best_epoch']], [100 * winner['dev']['accuracy']],
                     marker='*', s=190, color='#009E73', edgecolor='black', zorder=10)
    accuracy.annotate('Selected: λ=0.3, epoch 8', (8, 100 * winner['dev']['accuracy']),
                      xytext=(6, 19), textcoords='offset points', fontsize=10)
    loss.set(ylabel='Training cross-entropy loss', ylim=(.30, .50),
             title='E9: residual L2 sweep — training loss')
    accuracy.set(ylabel='Validation accuracy (%)', ylim=(80, 88),
                 title='E9: residual L2 sweep — validation accuracy')
    for fig, ax, filename in [(loss_fig, loss, 'e9-training-loss'),
                               (accuracy_fig, accuracy, 'e9-validation-accuracy')]:
        ax.set(xlabel='Training epoch', xticks=range(1, 11), xlim=(.8, 10.2))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=.15)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=5,
                  frameon=False, fontsize=10)
        ax.set_title(ax.get_title(), pad=40)
        fig.tight_layout()
        for extension in ['png', 'svg']:
            fig.savefig(root / f'outputs/handoff/figures/{filename}.{extension}', dpi=180)
        plt.close(fig)
    (folder / 'plotted-points.json').write_text(json.dumps({
        'lambdas': subset, 'loss': 'Online train cross entropy; excludes lambda penalty',
        'validation': 'Accuracy; no test curves or imputed points', 'points': points,
        'selected_lambda': winner['lambda'], 'selected_epoch': winner['best_epoch']}, indent=2) + '\n')
    print('Created separate training-loss and validation-accuracy charts for lambda 0, 0.003, 0.03, 0.3, 3.')


if __name__ == '__main__':
    build()
