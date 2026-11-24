import argparse,json,os
from .tasks import load_tasks,dataset_hash
from .sandbox import DockerGrader,FixtureGrader
from .providers import FixtureProvider,OpenAICompatible
from .store import Store
from .evaluate import audit,evaluate

def main():
    p=argparse.ArgumentParser(prog='coderlab'); p.add_argument('--dataset',default='benchmarks/tasks.json'); p.add_argument('--db',default='runs/experiments.sqlite3')
    sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('audit'); a.add_argument('--fixture',action='store_true'); a.add_argument('--image',default='python:3.11-slim')
    e=sub.add_parser('evaluate'); e.add_argument('--fixture',action='store_true'); e.add_argument('--model',action='append'); e.add_argument('--base-url',default='http://localhost:8001/v1'); e.add_argument('--split',choices=['train','validation','test'],default='test'); e.add_argument('--samples',type=int,default=3); e.add_argument('-k',type=int,default=1); e.add_argument('--image',default='python:3.11-slim')
    n=sub.add_parser('export-verl'); n.add_argument('--output',default='data/coderlab-custom'); n.add_argument('--parquet',action='store_true')
    v=sub.add_parser('audit-native'); v.add_argument('files',nargs='+')
    s=sub.add_parser('serve'); s.add_argument('--port',type=int,default=8000)
    args=p.parse_args()
    if args.cmd=='serve':
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(args.db),host='127.0.0.1',port=args.port); return
    if args.cmd=='export-verl':
        from .native import export_native
        print(json.dumps(export_native(load_tasks(args.dataset),args.output,args.parquet),indent=2)); return
    if args.cmd=='audit-native':
        from .native import audit_native
        result=audit_native(args.files); print(json.dumps(result,indent=2)); raise SystemExit(0 if result['valid'] else 1)
    tasks=load_tasks(args.dataset); grader=FixtureGrader() if args.fixture else DockerGrader(args.image)
    if args.cmd=='audit':
        result=audit(tasks,grader); print(json.dumps({'mode':'fixture-not-real' if args.fixture else 'docker','tasks':result},indent=2)); raise SystemExit(0 if all(r['valid'] for r in result) else 1)
    providers=[FixtureProvider('fixture-reference'),FixtureProvider('fixture-mixed')] if args.fixture else [OpenAICompatible(m,args.base_url,os.getenv('CODERLAB_API_KEY')) for m in (args.model or [])]
    if not providers: p.error('Provide --model or --fixture')
    if len({x.name for x in providers})!=len(providers): p.error('Duplicate model names')
    tasks=[t for t in tasks if t.split==args.split]
    if not tasks: p.error('No tasks in selected split')
    rid,summary=evaluate(tasks,providers,grader,Store(args.db),{'mode':'fixture-not-real' if args.fixture else 'docker','dataset_sha256':dataset_hash(args.dataset),'split':args.split,'image':None if args.fixture else args.image},args.samples,args.k)
    print(json.dumps({'run_id':rid,'summary':summary},indent=2))
if __name__=='__main__': main()
