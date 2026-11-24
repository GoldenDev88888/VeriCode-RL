import json,pytest
from pathlib import Path
from coderlab.tasks import load_tasks
from coderlab.sandbox import FixtureGrader,Result
from coderlab.metrics import pass_at_k,summarize
from coderlab.providers import FixtureProvider,extract_code
from coderlab.evaluate import audit,evaluate
from coderlab.store import Store
from coderlab.api import create_app
from fastapi.testclient import TestClient

DATA=Path(__file__).parents[1]/'benchmarks/tasks.json'

def test_dataset_splits():
    ts=load_tasks(DATA); assert len(ts)==20; assert {t.split for t in ts}=={'train','validation','test'}

def test_reference_and_mutants_trusted_only():
    # These are hand-authored repository fixtures, never externally generated code.
    for t in load_tasks(DATA):
        ns={};exec(t.reference,ns); f=ns[t.entrypoint]
        for c in t.cases: assert f(*c.args)==c.expected, t.id
        for m in t.mutants:
            ns={};exec(m,ns); f=ns[t.entrypoint]; killed=False
            for c in t.cases:
                try: killed |= f(*c.args)!=c.expected
                except Exception: killed=True
            assert killed,t.id

def test_duplicate_rejected(tmp_path):
    rows=json.loads(DATA.read_text()); p=tmp_path/'bad.json';p.write_text(json.dumps(rows+[rows[0]]))
    with pytest.raises(ValueError):load_tasks(p)

@pytest.mark.parametrize('n,c,k,want',[(4,2,1,.5),(4,2,2,5/6),(4,0,2,0),(4,4,3,1)])
def test_pass_at_k(n,c,k,want):assert pass_at_k(n,c,k)==pytest.approx(want)

def test_invalid_pass_at_k():
    with pytest.raises(ValueError):pass_at_k(1,0,2)

def test_extract():assert extract_code('text\n```python\ndef f(): pass\n```')=='def f(): pass'

def test_fixture_audit():assert all(x['valid'] for x in audit(load_tasks(DATA),FixtureGrader()))

def test_infra_exclusion():
    rows=[dict(model='x',task_id='t',category='a',status='infra_error',latency_ms=1),dict(model='x',task_id='t',category='a',status='pass',latency_ms=2)]
    m=summarize(rows)['x'];assert m['pass_at_k'] is None and m['infra_errors']==1 and m['excluded_tasks']==1

def test_pipeline_and_api(tmp_path):
    db=tmp_path/'test.db';store=Store(db); ts=[t for t in load_tasks(DATA) if t.split=='test']; rid,summary=evaluate(ts,[FixtureProvider('fixture-reference'),FixtureProvider('fixture-mixed')],FixtureGrader(),store,{'mode':'fixture-not-real','split':'test','dataset_sha256':'demo'},3,1)
    assert summary['fixture-reference']['pass_at_k']==1;assert summary['fixture-mixed']['pass_at_k']==pytest.approx(2/3);assert len(store.rows(rid))==36
    client=TestClient(create_app(db));assert client.get('/').status_code==200;assert client.get('/static/app.js').status_code==200;assert client.get('/api/runs').json()[0]['id']==rid; d=client.get('/api/runs/'+rid).json();assert d['completed_samples']==36;assert 'code' not in d['samples'][0];assert 'fixture-mixed' in client.get('/api/runs/'+rid+'/export').text;assert client.get('/api/runs/missing').status_code==404

def test_reward():
    from coderlab.reward import execution_reward
    t=load_tasks(DATA)[0]; g=FixtureGrader();assert execution_reward(g,t,t.reference)==1;assert execution_reward(g,t,t.mutants[0])==0
    class Broken:
        def grade(self,*args):return Result('infra_error',0,3)
    with pytest.raises(RuntimeError):execution_reward(Broken(),t,t.reference)

def test_incomplete_excluded():
    rows=[dict(model='x',task_id='t',category='a',status='pass',latency_ms=1)]
    assert summarize(rows,1,3)['x']['pass_at_k'] is None
