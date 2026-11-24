from math import comb
import random

def pass_at_k(n,c,k):
    if not 0<=c<=n or not 1<=k<=n: raise ValueError('Invalid n/c/k')
    return 1-comb(n-c,k)/comb(n,k) if n-c>=k else 1.0

def bootstrap_ci(values,seed=42,draws=1000):
    if not values: return None
    rng=random.Random(seed); n=len(values)
    samples=sorted(sum(rng.choices(values,k=n))/n for _ in range(draws))
    return [samples[int(.025*draws)],samples[min(draws-1,int(.975*draws))]]

def summarize(rows,k=1,expected_samples=None):
    groups={}
    for r in rows: groups.setdefault((r['model'],r['task_id']),[]).append(r)
    out={}
    for (model,task),rs in groups.items():
        bucket=out.setdefault(model,{'scores':[],'latency':[],'infra_errors':0,'excluded_tasks':0,'categories':{}})
        bad=sum(r['status']=='infra_error' for r in rs); bucket['infra_errors']+=bad
        if bad or len(rs)<k or (expected_samples is not None and len(rs)!=expected_samples):
            bucket['excluded_tasks']+=1; continue
        score=pass_at_k(len(rs),sum(r['status']=='pass' for r in rs),k)
        bucket['scores'].append(score); bucket['latency'].extend(r['latency_ms'] for r in rs)
        bucket['categories'].setdefault(rs[0]['category'],[]).append(score)
    return {m:{'pass_at_k':sum(v['scores'])/len(v['scores']) if v['scores'] else None,'k':k,'tasks':len(v['scores']),'ci95':bootstrap_ci(v['scores']),'mean_latency_ms':sum(v['latency'])/len(v['latency']) if v['latency'] else None,'infra_errors':v['infra_errors'],'excluded_tasks':v['excluded_tasks'],'categories':{c:sum(s)/len(s) for c,s in v['categories'].items()}} for m,v in out.items()}


def compare_common_tasks(rows, k, expected_samples):
    """Compare models only on the identical set of fully observed valid tasks."""
    groups = {}
    for row in rows:
        groups.setdefault(row['model'], {}).setdefault(row['task_id'], []).append(row)
    eligible = {m: {t for t, rs in tasks.items() if len(rs) == expected_samples and
                    not any(r['status'] == 'infra_error' for r in rs)} for m, tasks in groups.items()}
    common = set.intersection(*eligible.values()) if eligible else set()
    scores = {}
    for model, tasks in groups.items():
        vals = [pass_at_k(expected_samples, sum(r['status']=='pass' for r in tasks[t]), k) for t in sorted(common)]
        scores[model] = sum(vals)/len(vals) if vals else None
    return {'common_task_count': len(common), 'scores': scores}
