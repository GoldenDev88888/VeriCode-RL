import time
from .metrics import summarize

def audit(tasks,grader):
    report=[]
    for t in tasks:
        ref=grader.grade(t,t.reference)
        mutants=[grader.grade(t,m) for m in t.mutants]
        report.append({'task_id':t.id,'reference_status':ref.status,'mutants_killed':sum(m.status=='fail' for m in mutants),'mutants_total':len(mutants),'valid':ref.status=='pass' and all(m.status=='fail' for m in mutants)})
    return report

def evaluate(tasks,providers,grader,store,metadata,samples=3,k=1):
    if not 1<=k<=samples: raise ValueError('Require 1 <= k <= samples')
    rid=store.start({**metadata,'samples_per_task':samples,'k':k,'expected_samples':len(tasks)*len(providers)*samples})
    rows=[]
    for provider in providers:
        for task in tasks:
            for i in range(samples):
                started=time.perf_counter(); code=''; gen_ms=0
                try:
                    code=provider.generate(task,i); gen_ms=(time.perf_counter()-started)*1000
                    res=grader.grade(task,code); status=res.status; passed=res.passed; details=list(res.details)
                except Exception as e:
                    status='infra_error'; passed=0; details=[type(e).__name__]
                row={'model':provider.name,'task_id':task.id,'category':task.category,'sample_index':i,'status':status,'passed':passed,'total':len(task.cases),'generation_ms':round(gen_ms,3),'latency_ms':round((time.perf_counter()-started)*1000,3),'details':details,'code':code}
                store.add(rid,row); rows.append(row)
    return rid,summarize(rows,k,samples)
