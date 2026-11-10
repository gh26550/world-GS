"""Run a fixed multi-prompt planning comparison; retain every failure."""
import hashlib,json,time,traceback
from pathlib import Path


def run(suite,output):
    from .semantic_plan import plan_with_llm
    import torch
    src=Path(suite);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    spec=json.loads(src.read_text(encoding='utf-8-sig'))
    files=['semantic_plan.py','planning_benchmark.py','window_geometry.py']
    code={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in files}
    frozen=dict(suite=spec,code=code)
    lock=out/'protocol.json'
    if lock.exists() and json.loads(lock.read_text())!=frozen:
        raise ValueError('Protocol/code changed: create a new comparison output')
    lock.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    rows=[]
    for profile in spec['models']:
        for case in spec['cases']:
            name=profile['id'].rsplit('/',1)[-1]+'_'+profile['quantization']
            folder=out/name/case['id'];record=folder/'benchmark_result.json'
            if record.exists():rows.append(json.loads(record.read_text()));continue
            folder.mkdir(parents=True,exist_ok=True);started=time.monotonic()
            if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
            row=dict(case=case['id'],model=profile,scope=spec['scope'])
            try:
                plan=plan_with_llm(case['prompt'],folder,model_id=profile['id'],quantization=profile['quantization'])
                row.update(passed=True,objects=len(plan['objects']),model_revision=plan.get('planner_revision'))
            except Exception as error:
                row.update(passed=False,error=type(error).__name__+': '+str(error))
                (folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
            row.update(elapsed_seconds=time.monotonic()-started,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
            record.write_text(json.dumps(row,indent=2),encoding='utf-8');rows.append(row)
            (out/'results.json').write_text(json.dumps(dict(complete=False,rows=rows),indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)
    summary=[dict(model=p,passed=sum(r['passed'] for r in rows if r['model']==p),total=sum(r['model']==p for r in rows)) for p in spec['models']]
    report=dict(complete=True,rows=rows,summary=summary,limitation='Self-audited planning only; no independent prompt adherence or GS fidelity measurement')
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('suite');p.add_argument('--output',required=True)
    run(**vars(p.parse_args()))
