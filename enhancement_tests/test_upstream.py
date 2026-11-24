"""Regression tests of actual upstream adapters without CUDA or a live sandbox."""
import asyncio
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import Mock

import pytest
from coderlab.contract import evaluate_completion, extract_completion, validate_oracle, InvalidOracle, RewardInfrastructureError
from coderlab.native import export_native, audit_native
from coderlab.tasks import load_tasks
from coderlab.metrics import compare_common_tasks
from coderlab.process import run_bounded

ROOT = Path(__file__).resolve().parents[1]
REWARD = ROOT/'verl/utils/reward_score/coder1'
TEXT = '<think>reason</think>\n<answer>\n```python\nprint(input())\n```\n</answer>'


def load_adapter(monkeypatch, backend='firejail'):
    # Import the real source package without the unrelated GPU framework's eager imports.
    monkeypatch.setenv('CODER1_EXEC',backend)
    name='isolated_coder1'
    spec=importlib.util.spec_from_file_location(name,REWARD/'__init__.py',submodule_search_locations=[str(REWARD)])
    mod=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,name,mod)
    for suffix in ('firejail_exec','docker_exec','ces_exec','kira_exec','utils'):
        monkeypatch.delitem(sys.modules,name+'.'+suffix,raising=False)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('raw',[{}, {'inputs':[],'outputs':[]},{'inputs':['x'],'outputs':[]},{'inputs':[1],'outputs':['1']},{'functional':''},{'functional':'assert 1','pytest':'assert 1'},'not json'])
def test_bad_oracle(raw):
    with pytest.raises(InvalidOracle):validate_oracle(raw)


def test_oracle_scores_and_format():
    assert extract_completion(TEXT)=='print(input())'
    assert extract_completion(TEXT+'\nextra') is None
    assert extract_completion(TEXT.replace('```python\nprint(input())\n```','```python\nx=1\n```\n```python\nx=2\n```')) is None
    oracle={'inputs':['hi','there'],'outputs':['hi','there']}
    assert evaluate_completion(TEXT,oracle,lambda code,stdin:(True,stdin))==(1.1,'pass',2)
    assert evaluate_completion(TEXT,oracle,lambda code,stdin:(True,'wrong'))==(.1,'fail',2)
    assert evaluate_completion('bad',oracle,lambda *a:None)==(-1.1,'invalid_format',0)


def test_functional_requires_completion_marker():
    assert evaluate_completion(TEXT,{'functional':'assert True'},lambda code:(True,''))[1]=='fail'
    # This fake backend reads the host-inserted marker, not an actual untrusted code execution.
    def success(code):return True,ast.literal_eval(code.splitlines()[-1][6:-1])+'\n'
    assert evaluate_completion(TEXT,{'functional':'assert True'},success)[1]=='pass'


def test_upstream_adapter_and_telemetry(monkeypatch,tmp_path):
    mod=load_adapter(monkeypatch); path=tmp_path/'rewards.jsonl';monkeypatch.setenv('CODER1_REWARD_LOG',str(path))
    monkeypatch.setattr(mod,'code_exec',lambda code,stdin:(True,stdin))
    assert mod.compute_score(TEXT,json.dumps({'inputs':['hello'],'outputs':['hello']}),None)==1.1
    log=json.loads(path.read_text());assert log['status']=='pass' and 'hello' not in path.read_text()
    def broken(*a,**k):raise RewardInfrastructureError('offline')
    monkeypatch.setattr(mod,'code_exec',broken)
    with pytest.raises(RewardInfrastructureError):mod.compute_score(TEXT,{'inputs':['x'],'outputs':['x']})
    assert json.loads(path.read_text().splitlines()[-1])['score'] is None


def test_firejail_flags_and_stdin(monkeypatch):
    load_adapter(monkeypatch);mod=sys.modules['isolated_coder1.firejail_exec']
    monkeypatch.setenv('SECRET_API_KEY','secret');monkeypatch.setattr(mod.shutil,'which',lambda x:'/usr/bin/firejail')
    def fake(cmd,stdin,timeout,env,cwd):
        assert '--net=none' in cmd and 'SECRET_API_KEY' not in env
        assert Path(cmd[-1]).read_text()=='print(input())'
        assert stdin=="quotes ' \n line";return 0,stdin,''
    monkeypatch.setattr(mod,'run_bounded',fake)
    assert mod.code_exec_firejail('print(input())',"quotes ' \n line")[0]
    monkeypatch.setattr(mod.shutil,'which',lambda x:None)
    with pytest.raises(RewardInfrastructureError):mod.code_exec_firejail('pass')


def test_docker_no_shell_and_infrastructure_error(monkeypatch):
    load_adapter(monkeypatch,'docker');mod=sys.modules['isolated_coder1.docker_exec']
    monkeypatch.setattr(mod.shutil,'which',lambda x:'/usr/bin/docker');monkeypatch.setattr(mod.subprocess,'run',Mock())
    def fake(cmd,stdin,timeout):
        assert '--network=none' in cmd and '--read-only' in cmd and '--pull=never' in cmd
        assert stdin=="quotes ' \n line" and 'sh' not in cmd
        return 125,'','daemon unavailable'
    monkeypatch.setattr(mod,'run_bounded',fake)
    with pytest.raises(RewardInfrastructureError):mod.code_exec_docker('print(input())',"quotes ' \n line")
    assert mod.subprocess.run.call_args.args[0][:3]==['docker','rm','-f']


@pytest.mark.parametrize('backend,func',[('ces','remote_code_exec_ces'),('kira','remote_code_exec_kira')])
def test_remote_outage_bounded(monkeypatch,backend,func):
    load_adapter(monkeypatch,backend);mod=sys.modules['isolated_coder1.'+backend+'_exec']
    post=Mock(side_effect=mod.requests.ConnectionError('offline'));monkeypatch.setattr(mod.requests,'post',post);monkeypatch.setattr(mod.time,'sleep',lambda _:None)
    with pytest.raises(RewardInfrastructureError):getattr(mod,func)('pass',timeout=1)
    assert post.call_count==3 and post.call_args.kwargs['timeout']==(3,6)


def test_process_capture_limits():
    # Trusted constant child commands only; no model output is executed on the host.
    assert run_bounded([sys.executable,'-c','print(input())'],'hello',2)[1]=='hello\n'
    assert run_bounded([sys.executable,'-c','import time; time.sleep(10)'],timeout=.1)[2]=='wall_timeout'
    assert run_bounded([sys.executable,'-c','print("x"*200000)'],timeout=2,max_output=1024)[2]=='output_limit'


def test_native_export_and_leakage(tmp_path):
    tasks=load_tasks(ROOT/'benchmarks/tasks.json');manifest=export_native(tasks,tmp_path)
    assert [manifest['splits'][s]['rows'] for s in ('train','validation','test')]==[10,4,6]
    paths=[tmp_path/(s+'.jsonl') for s in ('train','validation','test')]
    assert audit_native(paths)['unique_prompts']==20
    with paths[-1].open('a') as f:f.write(paths[0].read_text().splitlines()[0]+'\n')
    assert audit_native(paths)['invalid']==1


def test_common_task_cohort():
    rows=[dict(model=m,task_id=t,status=s) for m,t,s in [('a','x','pass'),('a','y','pass'),('b','x','fail'),('b','y','infra_error')]]
    assert compare_common_tasks(rows,1,1)=={'common_task_count':1,'scores':{'a':1,'b':0}}


def test_prime_propagates_code_failure(monkeypatch):
    # Test real async function with lightweight stand-ins for unrelated training imports.
    fake_verl=types.ModuleType('verl');fake_verl.DataProto=object
    fake_score=types.ModuleType('verl.utils.reward_score');fake_score._default_compute_score=lambda *a:0
    for n,m in [('verl',fake_verl),('verl.utils.reward_score',fake_score),('torch',types.ModuleType('torch'))]:monkeypatch.setitem(sys.modules,n,m)
    spec=importlib.util.spec_from_file_location('test_prime_upstream',ROOT/'verl/workers/reward_manager/prime.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    def broken(*args):raise RewardInfrastructureError('offline')
    with pytest.raises(RewardInfrastructureError):asyncio.run(mod.parallel_compute_score_async(broken,['c'],['r'],['code'],num_processes=1))
    assert asyncio.run(mod.parallel_compute_score_async(lambda *args:1.1,['c'],['r'],['code'],num_processes=1))==[1.1]
    with pytest.raises(ValueError):asyncio.run(mod.parallel_compute_score_async(lambda *args:0,['c'],[],['code']))


def test_preprocess_stdio():
    # Exercise actual pure function without installing the HF dataset stack.
    source=ast.parse((ROOT/'examples/data_preprocess/coder1.py').read_text());func=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='minimize_stdio')
    ns={};exec(compile(ast.Module(body=[func],type_ignores=[]),'preprocess','exec'),ns)
    f=ns['minimize_stdio'];assert f(['a\r\n'],['b\r\n'])==(['a\n'],['b\n'])
    with pytest.raises(ValueError):f(['x'],[])


@pytest.mark.parametrize('script',['main_grpo.sh','main_ppo.sh'])
def test_launcher_dry_run(script):
    if os.name=='nt':pytest.skip('Bash required')
    env={**os.environ,'GPUS_PER_NODE':'8','CODER1_DRY_RUN':'1'}
    p=subprocess.run(['bash',script],cwd=ROOT,env=env,capture_output=True,text=True)
    assert p.returncode==0 and 'global_batch=' in p.stdout
    p=subprocess.run(['bash',script],cwd=ROOT,env={**env,'GPUS_PER_NODE':'0'},capture_output=True,text=True)
    assert p.returncode!=0
