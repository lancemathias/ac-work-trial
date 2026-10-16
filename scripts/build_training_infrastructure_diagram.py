"""Hand-drawn training infrastructure overview; PNG, SVG and editable Excalidraw."""
import hashlib
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager


def build(root=ROOT):
    width, height = 1800, 1290
    out = root / 'outputs/handoff/figures'
    out.mkdir(parents=True, exist_ok=True)
    font_path = Path('/System/Library/Fonts/Supplemental/Comic Sans MS.ttf')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        family = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(18, 12.9))
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis('off')
    fig.patch.set_facecolor('#fffefa')
    elements = []
    colors = {'ink': '#243247', 'blue': '#e9f2ff', 'green': '#e8f5ed',
              'yellow': '#fff5d6', 'purple': '#f0eafd', 'gray': '#f5f5f3'}

    def base(kind, x, y, w, h, stroke=colors['ink'], fill='transparent'):
        identifier = f'infra-{len(elements)}'
        return dict(id=identifier, type=kind, x=x, y=y, width=w, height=h,
                    angle=0, strokeColor=stroke, backgroundColor=fill,
                    fillStyle='solid', strokeWidth=1.7, strokeStyle='solid',
                    roughness=1, opacity=100, groupIds=[], frameId=None,
                    roundness={'type': 3} if kind == 'rectangle' else None,
                    seed=1000 + len(elements), version=1, versionNonce=1,
                    isDeleted=False, boundElements=None, updated=0, link=None,
                    locked=False)

    def text(x, y, value, size=17, color=colors['ink'], weight='normal'):
        ax.text(x, y, value, ha='left', va='top', fontsize=size * .72,
                fontfamily=[family, 'DejaVu Sans'], color=color, weight=weight, linespacing=1.45, zorder=4)
        lines = value.splitlines()
        e = base('text', x, y, max(len(t) for t in lines) * size * .57,
                 len(lines) * size * 1.25, stroke=color)
        e.update(text=value, originalText=value, fontSize=size, fontFamily=1,
                 textAlign='left', verticalAlign='top', containerId=None,
                 autoResize=True, lineHeight=1.25)
        elements.append(e)

    def box(x, y, w, h, title, body, fill='blue', size=17):
        patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=11',
                              facecolor=colors[fill], edgecolor=colors['ink'], lw=1.5, zorder=2)
        patch.set_sketch_params(scale=.7, length=120, randomness=1.8)
        ax.add_patch(patch)
        elements.append(base('rectangle', x, y, w, h, fill=colors[fill]))
        text(x + 16, y + 15, title, size + 2, weight='bold')
        text(x + 16, y + 53, body, size)

    def arrow(points, color='#526b89', dashed=False):
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.55,
                    ls='--' if dashed else '-', zorder=1)
        patch = FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                mutation_scale=14, color=color, lw=1.55,
                                linestyle='--' if dashed else '-', zorder=1)
        ax.add_patch(patch)
        x, y = points[0]
        e = base('arrow', x, y, max(p[0] for p in points) - min(p[0] for p in points),
                 max(p[1] for p in points) - min(p[1] for p in points), stroke=color)
        e.update(points=[[px - x, py - y] for px, py in points],
                 startBinding=None, endBinding=None, lastCommittedPoint=None,
                 startArrowhead=None, endArrowhead='arrow', elbowed=False)
        if dashed:
            e['strokeStyle'] = 'dashed'
        elements.append(e)

    text(40, 28, 'Training infrastructure: inputs → Modal jobs → evidence', 30, weight='bold')
    text(42, 78, 'Applied Compute work trial  •  research data flow  •  October 14–15, 2026', 17)
    # Modal boundary is drawn first, behind cards and connectors.
    boundary = FancyBboxPatch((400, 120), 940, 1030, boxstyle='round,pad=0,rounding_size=15',
                               facecolor='#fcfcff', edgecolor='#8492a6', lw=1.5, linestyle='--', zorder=0)
    ax.add_patch(boundary)
    e = base('rectangle', 400, 120, 940, 1030, stroke='#8492a6', fill='#fcfcff')
    e['strokeStyle'] = 'dashed'
    elements.append(e)
    text(425, 137, 'MODAL  workspace: ac-worktrials  |  profile: ac-worktrial  |  environment: worktrial-lance', 17, weight='bold')
    box(40, 150, 320, 215, 'Local Git repository',
        'scripts/ + configs/ + manifests\nPinned source SHA, seeds, schemas\nDataset IDs and split checksums\nLaunchers build a Modal image\nDurable reports + small results', 'gray', 16)
    box(40, 410, 320, 275, 'Public inputs',
        'Task datasets → typed decisions\nChoice / Noul / Score contract\nANLI, RACE, ReClor, MultiRC,\nHelpSteer2, JEEBench; later\nImplicatures, CICERO, CaseHOLD\n\nHugging Face: pinned Qwen\nweights + tokenizers', 'yellow', 16)
    box(40, 740, 320, 390, 'Shared test panel',
        '2,111 decisions / 11 benchmarks\nBBH, Belebele, Circa,\nContractNLI, FinancialPhraseBank,\nJevBench public-hard, JudgeBench,\nRAGTruth, TabFact, TruthfulQA,\nWinoGrande\n\nPanel IDs + conversions in Git\nTest labels used for scoring only\nRepeatedly observed: exploratory\nNo test-based checkpoint selection', 'yellow', 16)
    box(440, 192, 860, 95, 'Modal image  /project',
        'Source, configs, manifests + required evaluation inputs; verify per-run source/data hashes.', 'gray', 17)
    box(440, 325, 400, 235, 'VOLUME: ac-worktrial-training-data',
        'Mounted as /data where needed\nJSONL + provenance + grouped splits\nv1: 5,000; v2: 2,553; v2.1: 5,053\nv3: 7,928 decisions\nEarly hidden-state caches: /hidden-cache\nE7: raw reconstruction + 10k sample\nTrain / validation / calibration distinct', 'blue', 16)
    box(440, 600, 400, 125, 'VOLUME: ac-worktrial-models',
        'Mounted as /models\n/hub: pinned base weights + tokenizers\nPretrained inputs; not run checkpoints', 'blue', 16)
    box(440, 765, 400, 340, 'VOLUME: ac-worktrial-training-results',
        'Mounted as /results\nRun configs, seeds, hashes, split receipts\nResidual features + frozen code logits\nv3.1: 11,755 cached training views\n(6,262 native + 5,493 Choice views)\nInitial / epoch / accepted checkpoints\nOptimizer states, histories, calibration\nPredictions, metrics + completion seals\nLegacy paths retained; aliases resolve\nvolume.commit() persists job output', 'blue', 16)
    box(900, 325, 400, 205, '1. Tokenize + extract / infer',
        'A100-80GB; pinned 9B / 27B model\nValidate schema, option tokens, length\nFrozen passes → hidden states / logits\nHash prompt + model + tokenizer + split\nUse cached features for later head fits', 'green', 17)
    box(900, 585, 400, 235, '2. Fit + select on validation',
        'A100-80GB; CPU helpers as needed\nE1–E6, E8–E9: train heads only\nE7 exception: full-weight 9B training\nFixed seeds + per-run optimizer config\nValidation chooses checkpoint / λ\nSave epoch states; promote accepted one\nE8/E9 reuse cache: no backbone load', 'green', 16)
    box(900, 870, 400, 235, '3. Calibrate + evaluate',
        'Fit scalar temperature on calibration\nLock checkpoint / λ before test access\nEvaluate on shared panel / saved logits\nAccuracy, composite, NLL, Brier, ECE\nSave predictions and checksum receipts\nOther saved checkpoints stay reachable\nTest outputs do not feed optimization', 'green', 16)
    box(1410, 325, 340, 315, 'W&B: sole experiment tracker',
        'lmathias / ac-worktrial-decisions\nPer-run configs, seed + source identity\nTrain loss + validation accuracy\nSelection and calibrated test summaries\nGroups / λ sweeps / canonical labels\n\nAuth injected from Modal secret:\nac-worktrial-wandb\nMetrics/config-only for E8/E9;\ncheckpoints stay local / in Modal', 'purple', 16)
    box(1410, 785, 340, 320, 'Durable local exports',
        'outputs/model-artifacts/\nTensor / optimizer archives + SHA256\n\noutputs/traces/ + outputs/handoff/\nPredictions, metrics, configs, seeds\nFigures + report-evidence receipts\n\nGit / SSH → GitHub delivery branches\nCode, small results, reports versioned\nVerify exports before Modal cleanup', 'gray', 16)
    # Inputs and runtime flow.
    arrow([(360, 245), (440, 245)])
    arrow([(360, 460), (440, 460)])
    arrow([(360, 635), (440, 655)])
    arrow([(840, 415), (900, 415)])
    arrow([(840, 640), (870, 640), (870, 490), (900, 490)])
    arrow([(840, 685), (900, 685)])  # E7 can load weights directly.
    arrow([(1100, 530), (1100, 585)])
    arrow([(900, 510), (858, 510), (858, 850), (840, 850)])
    arrow([(840, 790), (900, 790)])
    arrow([(900, 810), (865, 810), (865, 945), (840, 945)])
    arrow([(1100, 820), (1100, 870)])
    arrow([(840, 1000), (900, 1000)])
    arrow([(900, 1065), (840, 1065)])
    # Evaluation-only panel route stays outside training boxes.
    arrow([(200, 1130), (200, 1175), (1200, 1175), (1200, 1105)], '#b18a28')
    text(430, 1147, 'evaluation-only panel', 16, '#8b6a19')
    # Observability is outward-only; auth is injected, never exported.
    arrow([(1300, 670), (1370, 670), (1370, 485), (1410, 485)], '#7c62a3', True)
    arrow([(1300, 965), (1385, 965), (1385, 660), (1730, 660), (1730, 640)], '#7c62a3', True)
    arrow([(1580, 640), (1580, 785)], '#7c62a3', True)
    text(1428, 705, 'unsampled history → plots', 16, '#705691')
    arrow([(650, 1105), (650, 1220), (1580, 1220), (1580, 1105)], '#526b89')
    text(875, 1196, 'download artifacts → verify SHA256 → retain locally', 16)
    text(44, 1250, 'Blue = persistent volumes   Green = compute   Purple dashed = telemetry   Arrows = data movement; mounts vary by job.', 16)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for extension in ['png', 'svg']:
        fig.savefig(out / f'training-infrastructure.{extension}', dpi=160, facecolor=fig.get_facecolor())
    svg = out / 'training-infrastructure.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    scene = {'type': 'excalidraw', 'version': 2, 'source': 'ac-worktrial',
             'elements': elements,
             'appState': {'viewBackgroundColor': '#fffefa', 'gridSize': None}, 'files': {}}
    (out / 'training-infrastructure.excalidraw').write_text(json.dumps(scene, indent=2) + '\n')
    print('Created hand-drawn PNG/SVG and editable Excalidraw infrastructure overview.')


if __name__ == '__main__':
    build()
