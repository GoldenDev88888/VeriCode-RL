from pathlib import Path
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
import csv,io
from .store import Store
from .metrics import summarize, compare_common_tasks

def create_app(path='runs/experiments.sqlite3'):
    app=FastAPI(title='CodeR1 Lab',version='0.2.0'); store=Store(path); static=Path(__file__).parent/'static'
    app.mount('/static',StaticFiles(directory=static),name='static')
    @app.get('/')
    def home(): return FileResponse(static/'index.html')
    @app.get('/api/runs')
    def runs(): return store.runs()
    def lookup(rid):
        meta=next((r for r in store.runs() if r['id']==rid),None)
        if not meta: raise HTTPException(404,'Run not found')
        rows=store.rows(rid); return meta,rows
    @app.get('/api/runs/{rid}')
    def detail(rid:str):
        meta,rows=lookup(rid)
        return {'run':meta,'summary':summarize(rows,meta['k'],meta['samples_per_task']),'common_comparison':compare_common_tasks(rows,meta['k'],meta['samples_per_task']),'completed_samples':len(rows),'samples':[{k:v for k,v in r.items() if k!='code'} for r in rows]}
    @app.get('/api/runs/{rid}/export')
    def export(rid:str):
        _,rows=lookup(rid); f=io.StringIO(); keys=['model','task_id','category','sample_index','status','passed','total','latency_ms','generation_ms']; w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore'); w.writeheader()
        for r in rows:
            # Prevent spreadsheet formula injection from model identifiers.
            w.writerow({k:("'"+v if isinstance(v,str) and v.startswith(('=','+','-','@','\t','\r')) else v) for k,v in r.items()})
        return StreamingResponse(iter([f.getvalue()]),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="run-{rid}.csv"'})
    return app
