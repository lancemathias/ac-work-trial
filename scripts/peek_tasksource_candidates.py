"""Collect only the first two stored train rows per specified source, in shard order."""
import json
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
SOURCES=['CONDAQA','help-nli','nan-nli','lonli','doc-nli','wice','vitaminc','ruletaker','babi_nli/simple-negation','procedural-typed-decisions/evidence_sufficiency','procedural-typed-decisions/partial_observation_calibration','procedural-typed-decisions/needle_retrieval','procedural-typed-decisions/policy_applicability','procedural-typed-decisions/policy_under_uncertainty','procedural-typed-decisions/record_aggregation']
def main():
    found={s:[] for s in SOURCES}
    shards=sorted((ROOT/'work/deliverable-audit/untracked-data/cacc/outputs/tasksource-source-audit/filtered-parquet').glob('*.parquet'))
    for path in shards:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                if row['source'] in found and row['split']=='train' and len(found[row['source']])<2:
                    found[row['source']].append(row)
        if all(len(v)==2 for v in found.values()):break
    assert all(len(v)==2 for v in found.values())
    out=ROOT/'outputs/analysis/tasksource-candidate-peek'
    out.mkdir(parents=True,exist_ok=True)
    (out/'first-two-train.json').write_text(json.dumps(found,indent=2)+'\n')
    print(json.dumps({s:[{'id':r['id'],'kind':r['kind'],'state_chars':len(r['state'])} for r in rows] for s,rows in found.items()},indent=2))
if __name__=='__main__':main()
