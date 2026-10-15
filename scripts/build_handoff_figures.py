"""Generate handoff figures strictly from the committed measurement ledger."""
import json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'work/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/handoff/figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
scores=json.loads((ROOT/'outputs/autoresearch/overnight-benchmark-scores.json').read_text())['scores']
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
labels=['E1','E2','E3','E4 optimized','E5 bidirectional','Native 27B','Native + reverse']
fig,ax=plt.subplots(figsize=(11,5.5));values=[100*scores[k]['Composite'] for k in labels];bars=ax.barh(labels[::-1],values[::-1],color=['#197a6b']*2+['#477baf']*5)
for b,v in zip(bars,values[::-1]):ax.text(v+.6,b.get_y()+b.get_height()/2,f'{v:.2f}%',va='center')
ax.axvline(100*scores['Jev 1.13']['Composite'],color='#a34439',ls='--',label='Jev 1.13: 85.82%');ax.set_xlim(0,100);ax.set_xlabel('Equal-weight 11-suite macro accuracy (%)');ax.set_title('Completed recipes on the same 2,111-decision panel',loc='left',pad=16);ax.legend(loc='upper right');fig.text(.01,.01,'Historical comparisons change data, recipe, prompt and/or backbone. Single-seed exploratory results; not isolated causal effects.',fontsize=8);fig.tight_layout(rect=[0,.05,1,1]);save(fig,'completed-recipes')
suites=[k for k in scores['Jev 1.13'] if k!='Composite'];conds=['E1','E4 optimized','E5 bidirectional','Native + reverse','Jev 1.13'];matrix=np.array([[100*scores[c][s] for s in suites] for c in conds])
fig,ax=plt.subplots(figsize=(13,4.5));im=ax.imshow(matrix,vmin=30,vmax=100,cmap='YlGnBu',aspect='auto');ax.set_xticks(range(len(suites)),['BBH','Belebele','Circa','ContractNLI','Finance','Jev hard','JudgeBench','RAGTruth','TabFact','TruthfulQA','WinoGrande'],rotation=35,ha='right');ax.set_yticks(range(len(conds)),conds)
for i in range(len(conds)):
 for j in range(len(suites)):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white' if matrix[i,j]>72 else '#172132',fontsize=9)
ax.set_title('Aggregate improvements hide different task failures',loc='left',pad=15);fig.colorbar(im,ax=ax,label='Accuracy (%)');fig.tight_layout();save(fig,'task-transfer')
fig,axes=plt.subplots(1,2,figsize=(11,4.4));epochs=[1,2,3,4];axes[0].plot(epochs,[1.128574,.872580,.821533,.736250],'o-',label='Mean train loss');axes[0].plot(epochs,[.964205,.861869,.872634,.887704],'o-',label='Dev NLL');axes[0].axvline(2,color='gray',ls=':');axes[0].set(xlabel='Epoch',ylabel='Loss / NLL',title='E1: learning followed by weaker dev NLL');axes[0].legend();axes[1].bar(['Judge: B','RAG: no hallucination','Finance: positive','Contract: entailment'],[98.5,100,61.5,87.44],label='Prediction share',color='#477baf');axes[1].plot(range(4),[43,66,20.5,46.23],'o',color='#a34439',label='Gold share on valid examples');axes[1].set_ylim(0,110);axes[1].set_ylabel('Share (%)');axes[1].tick_params(axis='x',rotation=30);axes[1].set_title('Collapse exceeds the label imbalance');axes[1].legend(fontsize=9);fig.tight_layout();save(fig,'learning-and-collapse')
conds=['RuleTaker','Aegis2','CivilComments','SNLI + MultiNLI','MASSIVE','BANKING77'];delta=[100*(scores[c]['Composite']-scores['Source control']['Composite']) for c in conds];fig,ax=plt.subplots(figsize=(10,4.7));bars=ax.barh(conds[::-1],delta[::-1],color=['#197a6b' if v>0 else '#a34439' for v in delta[::-1]])
for b,v in zip(bars,delta[::-1]):ax.text(v+(.035 if v>=0 else -.035),b.get_y()+b.get_height()/2,f'{v:+.2f}',va='center',ha='left' if v>=0 else 'right')
ax.axvline(0,color='gray',lw=.8);ax.set_xlim(-1.1,1.5);ax.set_xlabel('Composite difference from matched existing-source control (percentage points)');ax.set_title('Source additions: limited evidence of general transfer',loc='left',pad=15);fig.text(.01,.01,'Frozen 9B, Medium head, 250 additional decisions per condition, fixed heldouts and 510 updates. One seed; no cross-seed confirmation.',fontsize=8);fig.tight_layout(rect=[0,.05,1,1]);save(fig,'source-additions')
print('Generated four PNG/SVG figure pairs')
