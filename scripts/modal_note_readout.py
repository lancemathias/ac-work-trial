"""User-requested explanatory note on the existing W&B run; CPU only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if Path('/project/scripts').is_dir():
    sys.path.insert(0, '/project/scripts')
import modal
from modal_residual_readout import image, secret, ENV

app = modal.App('ac-worktrial-readout-wandb-note')
NOTE = '''## Experiment intent and run history (2026-10-15)

This job was intended to compare fixed versus shuffled option-order training on the v3 dataset, using a strictly frozen Qwen3.8-27B backbone. The head adds a 1,024-wide MLP correction (LayerNorm -> Linear -> GELU -> Linear, 255 outputs) to the frozen answer-code LM readout. The prompt maps options to 255 tokenizer-verified single-token codes. MLP weights use Xavier uniform initialization (zero-mean distribution, gain 1); linear biases start at zero. Both conditions began with identical head weights and the same data splits.

We initially trained BOTH the fixed-order and shuffled-order conditions in this one W&B run, logging their metrics under `fixed/*` and `shuffled/*`. Fixed-order features were cached once; the original shuffled condition extracted a fresh permutation every epoch, which dominated runtime.

We then stopped the paired job, discontinued shuffled training, and resumed ONLY fixed-order training from its saved epoch-4 head and optimizer state. The continuation reused the original feature cache, completed epoch 5, and resumed this same W&B run ID (`c09mctl9`) with `resume="must"`.

The displayed W&B step reaches 7 because it is a shared, zero-based logging counter across the initial fixed AND shuffled updates and the resumed fixed update. It does NOT mean that fixed-order training ran for seven epochs. Use `fixed/epoch` and `shuffled/epoch` to interpret each condition's actual training progress. The intended fixed-order training length was five epochs.

The fixed-only completion selects the checkpoint by dev accuracy and fits temperature on the separate calibration split. Training and native-order benchmark inference completed; report generation subsequently failed because cached predictions lacked a `latency_ms` field. Reporting and artifact recovery are handled on CPU from saved checkpoints and predictions, without repeating training or GPU inference.
'''


@app.function(image=image, cpu=1, memory=2048, timeout=180, secrets=[secret])
def update_note():
    import wandb
    run = wandb.Api().run('lmathias/ac-worktrial-decisions/c09mctl9')
    marker = '## Experiment intent and run history (2026-10-15)'
    prior = run.notes or ''
    if marker in prior:
        prior = prior.split(marker, 1)[0].rstrip()
    run.notes = (prior + '\n\n' if prior else '') + NOTE
    run.update()
    reread = wandb.Api().run('lmathias/ac-worktrial-decisions/c09mctl9')
    if NOTE.strip() not in (reread.notes or ''):
        raise RuntimeError('W&B note verification failed')
    return {'status': 'verified', 'url': reread.url, 'notes': reread.notes}


@app.local_entrypoint()
def main():
    from modal.config import _profile, config
    assert _profile == 'ac-worktrial' and config.get('environment') == ENV
    import json
    result = update_note.remote()
    Path('outputs/autoresearch/readout-wandb-note.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
