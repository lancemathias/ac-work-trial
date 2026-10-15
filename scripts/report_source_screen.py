"""Rebuild the source-screen ledger from recorded hypotheses and verified outputs."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BATCH='ab1-math-accuracy-seed20261014'
NAMES=['ruletaker','aegis2','civilcomments','snli-multinli','massive','banking77']
def build():
 evidence={};rows=[]
 def read(relative):
  path=ROOT/relative;raw=path.read_bytes();actual=hashlib.sha256(raw).hexdigest();evidence[str(relative)]=actual
  if str(relative).startswith(('outputs/training/','outputs/test/')):
   manifest=path.parent/'checksums.json';hashes=json.loads(manifest.read_text());evidence[str(manifest.relative_to(ROOT))]=hashlib.sha256(manifest.read_bytes()).hexdigest()
   if hashes[path.name]!=actual:raise ValueError('Exported metadata checksum changed: '+str(relative))
  return json.loads(raw)
 cards={c['id']:c for c in read('outputs/autoresearch/experiment-cards.json')['cards']}
 for name in NAMES:
  decision=ROOT/'outputs/autoresearch'/(name+'-decision.json')
  if not decision.exists():continue
  card=cards['source-addback-'+name];paired=read('outputs/autoresearch/'+name+'-vs-control-paired.json');protocol=read('outputs/autoresearch/'+name+'-protocol-verification.json')
  if protocol['status']!='passed' or not all(protocol['checks'].values()):raise ValueError('Unverified protocol: '+name)
  training=Path('outputs/training')/(BATCH+'-'+name);summary=read(training/'summary.json');best=read(training/'best.json');wandb=read(training/'wandb-run.json');metrics=read(Path('outputs/test')/(BATCH+'-'+name+'-test')/'metrics.json')
  if summary['status']!='complete' or summary['selection_metric']!='accuracy' or summary['steps']!=510:raise ValueError('Wrong training recipe: '+name)
  if abs(metrics['macro_coverage_adjusted_accuracy']-paired['candidate_composite'])>1e-12:raise ValueError('Rescore and stored test metric differ: '+name)
  targeted=card['success_criterion']['targeted_suites'];target_gain=sum(paired['per_suite'][s]['delta'] for s in targeted)/len(targeted)
  primary=paired['composite_delta']>=.01;mechanism=target_gain>=.05
  rows.append({'condition':name,'hypothesis':card['hypothesis'],'training_seconds':summary['elapsed_seconds'],'selected_epoch':best['epoch'],'dev_accuracy':summary['dev_raw']['accuracy'],'temperature':summary['temperature']['temperature'],'test_composite':paired['candidate_composite'],'delta':paired['composite_delta'],'paired_interval':paired['paired_cluster_bootstrap_95_interval'],'targeted_suites':targeted,'targeted_mean_delta':target_gain,'primary_pass':primary,'mechanism_pass':mechanism,'decision':'advance for confirmation' if primary or mechanism else 'no demonstrated gain','wandb':wandb['url']})
 return {'status':'verified_completed_conditions_only','batch':BATCH,'reference_composite':.6092810003679568,'conditions':rows,'evidence_sha256':evidence,'qualification':'Same-seed 9B source screens; 6,512 training, 809 dev and 857 calibration decisions; 510 updates, math head SDPA, accuracy/earliest-tie selection and calibration-only temperature. Repeatedly observed test panel; intervals omit seed variability and selection uncertainty.'}
def render(value):
 lines=['# Matched source add-back ledger','',value['qualification'],'','Control composite: 60.9281%. Each intervention adds 250 decisions; its comparison control adds 250 unused examples from existing sources.','','| Condition | Train seconds | Epoch | Dev accuracy | T | Composite | Delta pp | Target delta pp | Decision |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
 for r in value['conditions']:lines.append('| [{condition}]({wandb}) | {training_seconds:.1f} | {selected_epoch} | {dev_accuracy:.2%} | {temperature:.3f} | {test_composite:.2%} | {delta:+.3f} | {target:+.2f} | {decision} |'.format(**dict(r,delta=100*r['delta'],target=100*r['targeted_mean_delta'])))
 lines+=['','Primary advance threshold: +1 pp composite. Mechanism threshold: +5 pp mean on the prespecified targeted suites. Passing either warrants further investigation; neither establishes a robust source effect.','']
 for r in value['conditions']:
  low,high=r['paired_interval'];lines+=['- **'+r['condition']+'**: '+r['hypothesis']+' Target: '+', '.join(r['targeted_suites'])+'. Paired conditional 95% interval: '+f'[{100*low:+.3f}, {100*high:+.3f}] pp.']
 lines+=['','All listed sources currently retain collapse on at least one diagnostic suite. See individual decision receipts for class distributions. RuleTaker has a separate preregistered instruction-semantics repair; its failed original result remains in this table.','','Frozen 27B pretrained readout is a separate candidate at 77.1359%, not a matched source-screen control. Jev1.13 target: 85.82%.','','Machine-readable results and input hashes: `outputs/autoresearch/source-screen-ledger.json`.']
 return '\n'.join(lines)+'\n'
if __name__=='__main__':
 value=build();(ROOT/'outputs/autoresearch/source-screen-ledger.json').write_text(json.dumps(value,indent=2)+'\n');(ROOT/'docs/source-screen-ledger.md').write_text(render(value));print(json.dumps({'status':value['status'],'completed_conditions':len(value['conditions'])}))
