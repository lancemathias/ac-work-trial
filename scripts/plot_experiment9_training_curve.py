"""Combined E9 chart for lambda 0 and coefficients ending in 3."""
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def build(root=ROOT):
    folder = root / 'outputs/handoff/experiment-9'
    subset = ['0', '0.003', '0.03', '0.3', '3']
    colors = ['#333333', '#0072B2', '#E69F00', '#009E73', '#CC79A7']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, loss = plt.subplots(figsize=(11.5, 6.2))
    accuracy = loss.twinx()
    points = {}
    for coefficient, color in zip(subset, colors):
        rows = json.loads((folder / f'lambda-{coefficient}-curves.json').read_text())
        if [row['epoch'] for row in rows] != list(range(1, 11)):
            raise ValueError('Require ten original epochs per coefficient')
        epochs = [row['epoch'] for row in rows]
        ce = [row['online_train']['nll'] for row in rows]
        acc = [100 * row['dev']['accuracy'] for row in rows]
        width = 2.4 if coefficient == '0.3' else 1.5
        loss.plot(epochs, ce, color=color, lw=width)
        accuracy.plot(epochs, acc, 'o--', color=color, lw=width, markersize=4)
        points[coefficient] = [{'epoch': e, 'train_cross_entropy': x, 'validation_accuracy': y / 100}
                               for e, x, y in zip(epochs, ce, acc)]
    winner = json.loads((folder / 'summary.json').read_text())['winner']
    accuracy.scatter([winner['best_epoch']], [100 * winner['dev']['accuracy']],
                     marker='*', s=190, color='#009E73', edgecolor='black', zorder=10)
    accuracy.annotate('Selected: λ=0.3, epoch 8', (8, 100 * winner['dev']['accuracy']),
                      xytext=(6, 19), textcoords='offset points', fontsize=10)
    loss.set(xlabel='Training epoch', ylabel='Training cross-entropy loss', xticks=range(1, 11),
             xlim=(.8, 10.2), ylim=(.30, .50))
    accuracy.set(ylabel='Validation accuracy (%)', ylim=(80, 88))
    loss.spines['top'].set_visible(False)
    accuracy.spines['top'].set_visible(False)
    loss.grid(alpha=.15)
    fig.suptitle('E9: residual L2 sweep — training loss and validation accuracy', fontsize=14, y=.98)
    fig.legend([Line2D([], [], color=c, lw=2) for c in colors],
               ['λ = ' + c for c in subset], loc='upper center', bbox_to_anchor=(.5, .93),
               ncol=5, frameon=False)
    fig.legend([Line2D([], [], color='#555555', lw=2),
                Line2D([], [], color='#555555', marker='o', ls='--')],
               ['Training loss (left axis)', 'Validation accuracy (right axis)'],
               loc='lower center', bbox_to_anchor=(.5, .01), ncol=2, frameon=False)
    fig.subplots_adjust(top=.81, bottom=.18, left=.09, right=.91)
    for extension in ['png', 'svg']:
        fig.savefig(root / f'outputs/handoff/figures/e9-training-curves.{extension}', dpi=180)
    plt.close(fig)
    (folder / 'plotted-points.json').write_text(json.dumps({
        'lambdas': subset, 'loss': 'Online train cross entropy; excludes lambda penalty',
        'validation': 'Accuracy; no test curves or imputed points', 'points': points,
        'selected_lambda': winner['lambda'], 'selected_epoch': winner['best_epoch']}, indent=2) + '\n')
    print('Combined chart: exactly lambda 0, 0.003, 0.03, 0.3, 3; train loss and validation accuracy.')


if __name__ == '__main__':
    build()
