"""Plot verified completed autoresearch evaluations in original run-start order."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
out=ROOT/'outputs/handoff';scores=json.loads((ROOT/'outputs/autoresearch/overnight-benchmark-scores.json').read_text())['scores']
# Original W&B run IDs, not the timestamps of retrospective imports.
runs=[('E4 optimized','2ouo1e3l','Optimize feature-extraction kernels'),('Bidirectional full attention','dyjus5wh','Let options see later evidence'),('Auto head control repeat','vned4p6n','Repeat original training'),('Auto head control 1','vbk46ez0','Original training control'),('Native 27B','rahruwds','Use pretrained answer weights'),('Source control','ud99ptl5','Matched data control'),('Math head control repeat','6bwp0vab','Repeat deterministic training'),('Joint-head + native mixture','6spap1d0','Mix trained and pretrained outputs'),('RuleTaker','df2fil2b','Add rule reasoning'),('Aegis2','4acn895m','Add safety labels'),('CivilComments','nastxy6m','Add toxicity labels'),('SNLI + MultiNLI','l4nad696','Add entailment labels'),('MASSIVE','hqwshgo4','Add intent labels'),('BANKING77','1npt61ie','Add banking intents'),('RuleTaker repaired','62h35cer','Clarify closed-world rules'),('Native + reverse','6djekmw2','Average native and reversed options')]
previous=json.loads((out/'autoresearch-progress.json').read_text());byid={x['run_id']:{'created_at':x['run_started_utc'],'url':x['wandb']} for x in previous['points']};byid.setdefault('2ouo1e3l',{'created_at':'2026-10-15T08:27:25Z','url':'https://wandb.ai/lmathias/ac-worktrial-decisions/runs/2ouo1e3l'})
points=[{'label':label,'run_id':rid,'test_composite_percent':100*scores[label]['Composite'],'run_started_utc':byid[rid]['created_at'],'description':description,'wandb':byid[rid]['url']} for label,rid,description in runs]
points.sort(key=lambda x:x['run_started_utc'])
base=100*scores['E4 fallback']['Composite'];best=base
for i,x in enumerate(points,1):x.update(index=i,new_observed_best=x['test_composite_percent']>best);best=max(best,x['test_composite_percent']);x['running_best_percent']=best
receipt={'metric':'Fixed-panel coverage-adjusted 11-suite test composite (%)','ordering':'Original W&B run creation/start time; completed scored conditions only, not completion order','baseline':{'label':'E4 fallback (user-directed reference)','test_composite_percent':base},'caveat':'Exploratory reused panel; heterogeneous branches and 9B/27B conditions; running max is descriptive, not a validation-selected promotion rule. Prepared/unscored conditions omitted.','points':points}
(out/'autoresearch-progress.json').write_text(json.dumps(receipt,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(13,5.7));xs=np.arange(len(points)+1);ys=[base]+[x['test_composite_percent'] for x in points];bestline=np.maximum.accumulate(ys)
ax.step(xs,bestline,where='post',color='#57b887',lw=2.5,label='Running best observed',zorder=2)
losers=[x for x in points if not x['new_observed_best']];winners=[x for x in points if x['new_observed_best']]
ax.scatter([x['index'] for x in losers],[x['test_composite_percent'] for x in losers],color='#a7afb7',alpha=.7,s=45,label='Other completed evaluation',zorder=3)
ax.scatter([0]+[x['index'] for x in winners],[base]+[x['test_composite_percent'] for x in winners],color='#33bf80',edgecolor='#285b43',s=65,label='Reference / new observed best',zorder=4)
for i,y,label in [(0,base,'E4 reference')]+[(x['index'],x['test_composite_percent'],x['description']) for x in winners]:
 ax.annotate(f'{label}\n{y:.2f}%',(i,y),xytext=(8,13),textcoords='offset points',fontsize=9,color='#277549',ha='left')
ax.set(title=f'Autoresearch progress: {len(points)} completed evaluations, {len(winners)} new observed bests',xlabel='Completed evaluation index (ordered by original run start)',ylabel='Test composite accuracy (%) · higher is better',xticks=xs,ylim=(55,88),xlim=(-.5,len(points)+.7));ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper left',bbox_to_anchor=(0,.92),fontsize=9,frameon=False)
fig.tight_layout();fig.savefig(out/'figures/autoresearch-progress.png',dpi=180,bbox_inches='tight');plt.close(fig)
print(f'Plotted {len(points)} completed evaluations and {len(winners)} new observed bests.')
