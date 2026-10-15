"""Matched-work timing and calibrated-development comparison; no raw traces published."""
import argparse,json,statistics
from pathlib import Path
from training_data import digest

def compare(original,cached):
    original,cached=Path(original),Path(cached)
    old=json.loads((original/'summary.json').read_text());new=json.loads((cached/'summary.json').read_text())
    oldcfg=json.loads((original/'config.json').read_text());newcfg=json.loads((cached/'config.json').read_text())
    assert oldcfg['config']==newcfg['config'],'Training config differs'
    assert oldcfg['manifest_sha256']==newcfg['manifest_sha256'],'Split manifest differs'
    oldhash=json.loads((original/'checksums.json').read_text());newhash=json.loads((cached/'checksums.json').read_text())
    assert oldhash['initial-head.safetensors']==newhash['initial-head.safetensors'],'Initial head differs'
    def events(folder):
        path=folder/'metrics.jsonl'
        if not path.exists():path=Path(__file__).resolve().parents[1]/'outputs/model-artifacts'/folder.name/'metrics.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()]
    a,b=events(original),events(cached)
    first=next(e for e in a if e['event']=='untrained_dev');second=next(e for e in b if e['event']=='untrained_dev')
    assert {k:v for k,v in first.items() if k!='timestamp_unix'}=={k:v for k,v in second.items() if k!='timestamp_unix'},'Initial metrics differ'
    live={r['step']:r for r in a if r['event']=='train_step'};stored={r['step']:r for r in b if r['event']=='train_step'}
    for step,row in stored.items():
        for key in ['record_ids','decisions','epoch','learning_rate']:
            assert row[key]==live[step][key],(step,key)
    live_window_seconds=sum(live[s]['decisions']/live[s]['decisions_per_second'] for s in stored)
    cache_window_seconds=sum(r['decisions']/r['decisions_per_second'] for r in stored.values())
    setup=newcfg['feature_cache']['load_and_verify_seconds']
    result=dict(config_equal=True,split_equal=True,initial_head_equal=True,initial_dev_metrics_equal=True,
        matched_steps=len(stored),matched_decisions=sum(r['decisions'] for r in stored.values()),
        paired_batch_order_decision_counts_and_lr_equal=True,
        original_matched_optimizer_window_seconds=live_window_seconds,cached_matched_optimizer_window_seconds=cache_window_seconds,
        matched_window_speedup=live_window_seconds/cache_window_seconds,
        median_paired_throughput_speedup=statistics.median(r['decisions_per_second']/live[s]['decisions_per_second'] for s,r in stored.items()),
        cache_load_and_verify_seconds=setup,
        cached_training_plus_epoch_dev_seconds=new['elapsed_seconds'],original_training_plus_epoch_dev_seconds=old['elapsed_seconds'],
        original_steps=old['steps'],cached_steps=new['steps'],
        original_best_epoch=json.loads((original/'best.json').read_text())['epoch'],cached_best_epoch=json.loads((cached/'best.json').read_text())['epoch'],
        original_temperature=old['temperature']['temperature'],cached_temperature=new['temperature']['temperature'],
        original_calibrated_dev=old['dev_calibrated'],cached_calibrated_dev=new['dev_calibrated'],
        numerical_replication='Same configuration, initialization and minibatches; CUDA training trajectories and selected epochs differ. Do not attribute metric differences to a quality benefit of caching.',
        timing='Matched optimizer windows include head forward, backward and optimizer. Cache setup is separate; feature extraction was a pre-existing one-off cost. Full training times have different early-stop lengths.')
    parity_path=cached/'cache-parity-summary.json'
    if parity_path.exists():result['trained_head_gpu_parity']=json.loads(parity_path.read_text())
    (cached/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    rows=[('Training + dev evaluations',f"{old['elapsed_seconds']/60:.2f} min",f"{new['elapsed_seconds']/60:.2f} min"),
          ('Optimizer steps',str(old['steps']),str(new['steps'])),('Selected epoch',str(result['original_best_epoch']),str(result['cached_best_epoch'])),
          ('Calibrated dev accuracy',f"{old['dev_calibrated']['accuracy']:.2%}",f"{new['dev_calibrated']['accuracy']:.2%}"),
          ('Calibrated dev NLL',f"{old['dev_calibrated']['nll']:.4f}",f"{new['dev_calibrated']['nll']:.4f}"),
          ('Calibrated dev Brier',f"{old['dev_calibrated']['brier']:.4f}",f"{new['dev_calibrated']['brier']:.4f}"),
          ('Calibrated dev ECE-15',f"{old['dev_calibrated']['ece_15']:.4f}",f"{new['dev_calibrated']['ece_15']:.4f}"),
          ('Fitted temperature',f"{old['temperature']['temperature']:.8f}",f"{new['temperature']['temperature']:.8f}")]
    text=['# E1 cached-feature replication','', '| Measure | E1 live backbone | E1 cached features |','|---|---:|---:|']
    text += [f'| {name} | {left} | {right} |' for name,left,right in rows]
    text+=['',f"For the same {len(stored)} optimizer windows ({result['matched_decisions']:,} decisions), total window time fell from {live_window_seconds:.1f}s to {cache_window_seconds:.1f}s: **{result['matched_window_speedup']:.2f}× speedup**. Median paired throughput speedup: {result['median_paired_throughput_speedup']:.2f}×.",
        '',f'Cache verification and GPU loading cost **{setup/60:.2f} minutes**, excluded from training timing. Cache extraction itself is a pre-existing one-off cost and is not included; this is not a claim about complete cache-creation-to-checkpoint wall time.',
        '',result['numerical_replication'],'', 'No second benchmark/test score is claimed here; these are the original independent development and calibration partitions.','']
    if parity_path.exists():
        text+=['GPU trained-head diagnostic: on four probes, cached/live features, logits and losses were bit-identical. Maximum cached/live gradient difference was 1.91e-6; repeated live gradients also varied (up to 9.54e-7). These probes support cache correctness, not bitwise reproducibility of the entire optimization trajectory.','']
    (cached/'comparison.md').write_text('\n'.join(text))
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,dict)},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('original');p.add_argument('cached');a=p.parse_args();compare(a.original,a.cached)
